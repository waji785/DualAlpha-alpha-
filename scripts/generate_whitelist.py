#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加载最终模型，对全市股票回测，生成白名单
用法: python scripts/generate_whitelist.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import numpy as np
import torch
import joblib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import *
from core.data_loader import load_from_cache
from core.model import DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer, Fusion_iTransformer
from core.backtest_engine import run_backtest
from core.metrics import compute_metrics
from utils.common import set_seed
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 配置
MAX_WORKERS = 8                  # 纯读缓存无网络，可开大
WHITELIST_MIN_RETURN = 0.10
WHITELIST_MIN_TRADES = 2
WHITELIST_MAX_DRAWDOWN = 0.40


def _quick_filter(code, start_date):
    """快速预筛：缓存存在 + 目标期内有足够数据 + 非停牌"""
    df = load_from_cache(code)
    if df is None:
        return False
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] >= pd.to_datetime(start_date)]
    if len(df) < SEQ_LEN + 30:   # 至少比序列长一点
        return False
    # 排除长期停牌（最近 60 天成交量全为 0）
    recent = df.tail(60)
    if (recent['Volume'].sum() == 0):
        return False
    return True

def load_unified_model(model_path="model_final.pth",
                       scaler_x_path="scaler_X_final.pkl",
                       scaler_y_path="scaler_Y_final.pkl"):
    """加载模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        logger.error(f"模型 {model_path} 不存在")
        return None, None, None
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    if hasattr(model, 'lstm') and hasattr(model.lstm, 'flatten_parameters'):
        model.lstm.flatten_parameters()
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    logger.info(f"模型加载成功: {model_path}")
    return model, scaler_X, scaler_y

def backtest_single_stock(code, model, scaler_X, scaler_y, start_date):
    """回测单只股票，返回绩效指标"""
    try:
        df = load_from_cache(code)
        if df is None or len(df) < SEQ_LEN + 20:
            return None
        df['Date'] = pd.to_datetime(df['Date'])

        backtest_df = run_backtest(df, model, scaler_X, scaler_y,
                                   start_date=start_date)
        if backtest_df is None or len(backtest_df) < 10:
            return None
        metrics = compute_metrics(backtest_df['Capital'].values)
        trades = (backtest_df['Position'].diff().abs() > 0.01).sum() / 2
        return {
            'code': code,
            'total_return': metrics['total_return'],
            'max_drawdown': metrics['max_drawdown'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'calmar_ratio': metrics['calmar_ratio'],
            'trade_count': int(trades)
        }
    except Exception as e:
        logger.debug(f"{code} 回测异常: {e}")
        return None

def main(model_path="model_final.pth",
         scaler_x="scaler_X_final.pkl", scaler_y="scaler_Y_final.pkl",
         start_date=BACKTEST_START_DATE):
    set_seed(SEED)
    logger.info("=" * 60)
    logger.info(f"回测白名单（模型: {model_path}）")
    logger.info(f"  回测区间: {start_date} ~ {TODAY}")
    logger.info(f"  筛选: 收益 > {WHITELIST_MIN_RETURN*100}%, 交易 >= {WHITELIST_MIN_TRADES}, 回撤 < {WHITELIST_MAX_DRAWDOWN*100}%")
    logger.info("=" * 60)

    # 1. 加载模型
    model, scaler_X, scaler_y = load_unified_model(model_path, scaler_x, scaler_y)
    if model is None:
        return

    # 2. 获取股票列表
    from core.data_loader import get_stock_list
    stock_df = get_stock_list(exclude_st=True, exclude_north=True)
    if stock_df is None:
        logger.error("无法获取股票列表")
        return
    # 预筛：有缓存 + 目标期内数据充足 + 非长期停牌
    codes = [c for c in stock_df['code'].tolist() if _quick_filter(c, start_date)]
    logger.info(f"通过预筛的股票: {len(codes)} 只")

    # 3. 边跑边写 CSV
    out_csv = "final_backtest_results.csv"
    header_written = os.path.exists(out_csv)
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(backtest_single_stock, code, model, scaler_X, scaler_y, start_date): code
            for code in codes
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="回测"):
            res = future.result()
            if res is not None:
                results.append(res)
                row_df = pd.DataFrame([res])
                mode = 'a' if header_written else 'w'
                row_df.to_csv(out_csv, index=False, mode=mode,
                              header=not header_written, encoding='utf-8-sig')
                header_written = True

    if not results and not os.path.exists(out_csv):
        logger.error("无有效回测结果")
        return

    # 从 CSV 读取完整结果（含之前中断恢复的）
    df_res = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(results)
    if results:
        df_res = pd.concat([df_res, pd.DataFrame(results)], ignore_index=True)
    # 统一 code 为字符串再去重（CSV 可能把 code 读成 int）
    df_res['code'] = df_res['code'].astype(str).str.zfill(6)
    if 'name' in df_res.columns:
        df_res = df_res.drop('name', axis=1)
    df_res = df_res.drop_duplicates('code', keep='last')
    df_res.to_csv(out_csv, index=False, encoding='utf-8-sig')

    # 合并股票名称
    name_map = dict(zip(stock_df['code'], stock_df['name']))
    df_res['name'] = df_res['code'].map(name_map)
    cols = ['code', 'name', 'total_return', 'max_drawdown', 'sharpe_ratio', 'calmar_ratio', 'trade_count']
    df_res = df_res[[c for c in cols if c in df_res.columns]]
    df_res = df_res.sort_values('total_return', ascending=False)
    logger.info(f"全部回测结果: {len(df_res)} 只")

    # 4. 筛选白名单
    whitelist = df_res[
        (df_res['total_return'] > WHITELIST_MIN_RETURN) &
        (df_res['trade_count'] >= WHITELIST_MIN_TRADES) &
        (df_res['max_drawdown'] < WHITELIST_MAX_DRAWDOWN)
    ].copy()
    whitelist = whitelist.sort_values('total_return', ascending=False)

    if whitelist.empty:
        logger.warning("无股票满足白名单条件，可降低筛选阈值")
        return

    whitelist.to_csv(WHITELIST_EXTENDED_FILE, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 最终白名单已生成，共 {len(whitelist)} 只，保存至 {WHITELIST_EXTENDED_FILE}")

    # 打印前10名
    print("\n📋 白名单前10名（按收益率排序）:")
    print(whitelist[['code', 'name', 'total_return', 'sharpe_ratio', 'max_drawdown']].head(10).to_string(index=False, float_format="%.3f"))

    # 统计
    print(f"\n📊 白名单统计:")
    print(f"  平均收益率: {whitelist['total_return'].mean()*100:.2f}%")
    print(f"  平均最大回撤: {whitelist['max_drawdown'].mean()*100:.2f}%")
    print(f"  平均夏普: {whitelist['sharpe_ratio'].mean():.3f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model_final.pth")
    p.add_argument("--scaler-x", default="scaler_X_final.pkl")
    p.add_argument("--scaler-y", default="scaler_Y_final.pkl")
    p.add_argument("--start-date", default=BACKTEST_START_DATE, help="回测起始日期")
    args = p.parse_args()
    main(args.model, args.scaler_x, args.scaler_y, start_date=args.start_date)