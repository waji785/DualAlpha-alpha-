#!/usr/bin/env python
# scripts/daily_select.py
"""每日盘后：选股 → 择时信号 → 输出明日操作清单"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from config.settings import *
from core.data_loader import load_from_cache, get_stock_list, get_industry_map
from core.stock_selector import TreeEnsemble
from core.backtest_engine import get_signal_for_day
from utils.logger import setup_logger

logger = setup_logger(__name__)


def daily_select(selector_path="selector_ensemble.pkl",
                 model_path="model_final.pth",
                 top_n=20, output="daily_picks.csv"):
    """盘后选股 + 择时信号"""
    today = pd.Timestamp(TODAY)
    logger.info(f"===== 每日选股 {today.date()} =====")

    # 1. 加载模型
    selector = TreeEnsemble()
    if os.path.exists(selector_path):
        selector.load(selector_path)
    else:
        logger.error(f"选股器不存在: {selector_path}"); return

    is_time_model = os.path.exists(model_path)

    # 2. 获取全市场股票
    stock_df = get_stock_list(exclude_st=True, exclude_north=True)
    codes = stock_df['code'].tolist()
    name_map = dict(zip(stock_df['code'], stock_df['name']))
    ind_map = get_industry_map()

    # 3. 提取截面特征 + 预测
    rows = []
    for code in codes:
        df = load_from_cache(code)
        if df is None or len(df) < SEQ_LEN + 60:
            continue
        df['Date'] = pd.to_datetime(df['Date'])
        pre = df[df['Date'] <= today]
        if len(pre) < SEQ_LEN:
            continue
        feat = pre.iloc[-1][FEATURE_COLS].values.astype(np.float32)
        if np.any(np.isnan(feat)):
            continue
        rows.append((code, feat))

    if not rows:
        logger.error("无可选股票"); return

    codes_list, feats_list = zip(*rows)
    X = np.array(feats_list)
    scores = selector.predict(X)

    # 4. 排序 + 行业过滤（每行业最多3只）
    order = np.argsort(scores)[::-1]
    picks, ind_count = [], {}
    for idx in order:
        code = codes_list[idx]
        ind = ind_map.get(code, '未知')
        if ind_count.get(ind, 0) < 3:
            picks.append(code)
            ind_count[ind] = ind_count.get(ind, 0) + 1
        if len(picks) >= top_n:
            break

    logger.info(f"选出 {len(picks)} 只: {picks}")

    # 5. 择时信号（选出的股票逐一检查）
    results = []
    for code in picks:
        df = load_from_cache(code)
        df['Date'] = pd.to_datetime(df['Date'])
        full = df[df['Date'] <= today]

        row = {
            'code': code,
            'name': name_map.get(code, ''),
            'industry': ind_map.get(code, ''),
            'score': float(scores[codes_list.index(code)]),
            'close': float(full['Close'].iloc[-1]),
            'position': 0.0,
            'action': '观察',
        }

        # 择时模型信号
        if is_time_model and len(full) >= SEQ_LEN + 50:
            try:
                up_prob, trend_ok, vol_ok = get_signal_for_day(
                    full, model_path, SEQ_LEN, BUY_THRESHOLD)
                row['up_prob'] = round(up_prob, 4)
                if trend_ok and vol_ok and up_prob > BUY_THRESHOLD:
                    row['action'] = '买入'
                    # 计算仓位
                    from core.backtest_engine import compute_position_size
                    row['position'] = round(compute_position_size(
                        full, up_prob, BUY_THRESHOLD, MAX_POSITION), 2)
            except Exception as e:
                logger.warning(f"{code} 择时失败: {e}")

        results.append(row)

    # 6. 输出
    df_out = pd.DataFrame(results)
    df_out.to_csv(output, index=False, encoding='utf-8-sig')
    print(df_out.to_string(index=False))

    buys = df_out[df_out['action'] == '买入']
    if len(buys) > 0:
        logger.info(f"明日买入信号: {len(buys)} 只")
        print(f"\n📈 买入清单:")
        for _, r in buys.iterrows():
            print(f"  {r['code']} {r['name']} pos={r['position']:.0%} @ {r['close']:.2f}")
    else:
        logger.info("明日无买入信号")
    logger.info(f"结果保存: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selector", default=os.path.join(OUTPUT_DIR, "selector_ensemble.pkl"))
    p.add_argument("--model", default=os.path.join(OUTPUT_DIR, "model_final.pth"))
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--output", default=os.path.join(OUTPUT_DIR, "daily_picks.csv"))
    args = p.parse_args()
    daily_select(args.selector, args.model, args.top, args.output)
