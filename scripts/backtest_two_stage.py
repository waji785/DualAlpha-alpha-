# scripts/backtest_two_stage.py
"""两阶段回测：树模型选股 + 深度学习择时"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch, joblib
from tqdm import tqdm

from config.settings import *
from core.model import DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer, Fusion_iTransformer
from core.data_loader import load_from_cache, get_stock_list
from core.backtest_engine import run_backtest
from core.metrics import compute_metrics
from core.stock_selector import TreeEnsemble, build_selector_features
from utils.logger import setup_logger

torch.serialization.add_safe_globals([DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer, Fusion_iTransformer])
logger = setup_logger(__name__)


def backtest_two_stage(timing_model_path="model_final.pth",
                       selector_path="selector_ensemble.pkl",
                       scaler_x="scaler_X_final.pkl",
                       scaler_y="scaler_Y_final.pkl",
                       start_date="2025-01-01",
                       top_n=20, initial_capital=100000):
    """
    两阶段回测：
    1. 每月初：树模型选股 Top-N
    2. 每日：深度模型择时（买入/持有/卖出）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 加载择时模型 ----
    logger.info(f"加载择时模型: {timing_model_path}")
    timing_model = torch.load(timing_model_path, map_location=device, weights_only=False)
    timing_model.eval()
    sX = joblib.load(scaler_x)
    sY = joblib.load(scaler_y) if os.path.exists(scaler_y) else None

    # ---- 加载选股模型 ----
    selector = TreeEnsemble()
    if os.path.exists(selector_path):
        selector.load(selector_path)
    else:
        logger.error(f"选股模型不存在: {selector_path}")
        return None

    # ---- 获取股票池 ----
    stock_df = get_stock_list(exclude_st=True, exclude_north=True)
    name_map = dict(zip(stock_df['code'], stock_df['name']))

    # ---- 确定回测月份 ----
    all_codes = stock_df['code'].tolist()
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(TODAY)

    # 找所有月份
    months = pd.date_range(sd, ed, freq='MS')
    logger.info(f"回测区间: {sd.date()} ~ {ed.date()}, {len(months)} 个月")

    # ---- 逐月回测 ----
    total_capital = initial_capital
    capital_history = [{'date': sd, 'capital': initial_capital}]
    portfolio = {}  # {code: holdings}
    monthly_picks_log = []
    cash = initial_capital

    for mi, month_start in enumerate(months):
        month_end = (month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1))
        if month_end > ed:
            month_end = ed

        logger.info(f"\n{'='*50}")
        logger.info(f"月份 {mi+1}/{len(months)}: {month_start.date()} ~ {month_end.date()}")

        # ---- Step 1: 选股 ----
        all_feats = []
        valid_codes = []
        for code in all_codes:
            df = load_from_cache(code)
            if df is None or len(df) < SEQ_LEN + 20:
                continue
            df['Date'] = pd.to_datetime(df['Date'])
            # 取该月之前的数据（无未来信息）
            pre_df = df[df['Date'] < month_start]
            if len(pre_df) < SEQ_LEN:
                continue
            feat = build_selector_features(pre_df)
            if feat is None:
                continue
            all_feats.append(feat)
            valid_codes.append(code)

        if len(all_feats) < top_n:
            logger.warning(f"可选股票不足: {len(all_feats)}")
            continue

        X_sel = np.array(all_feats)
        pred_returns = selector.predict(X_sel)
        sorted_idx = np.argsort(pred_returns)[::-1]
        picks = [valid_codes[i] for i in sorted_idx[:top_n]]

        monthly_picks_log.append({
            'month': month_start.date(),
            'stocks': picks,
            'pred_returns': [pred_returns[i] for i in sorted_idx[:top_n]]
        })
        logger.info(f"选股 Top {top_n}: {picks[:5]}...")

        # ---- Step 2: 清仓旧持仓 ----
        # 卖出上月持有的、不在本月选股中的股票
        to_sell = [c for c in portfolio if c not in picks]
        for code in to_sell:
            df = load_from_cache(code)
            if df is None:
                continue
            # 按月初价格清仓
            df['Date'] = pd.to_datetime(df['Date'])
            row = df[df['Date'] >= month_start].head(1)
            if len(row) == 0:
                continue
            price = float(row['Close'].iloc[0])
            shares = portfolio[code]
            cash += shares * price
            logger.debug(f"清仓 {code} @ {price:.2f}, {shares}股")
            del portfolio[code]

        # ---- Step 3: 择时交易 ----
        per_stock_capital = cash / max(top_n, 1)

        for code in picks:
            df = load_from_cache(code)
            if df is None:
                continue
            df['Date'] = pd.to_datetime(df['Date'])
            # 只取本月数据
            month_df = df[(df['Date'] >= month_start) & (df['Date'] <= month_end)]
            if len(month_df) < SEQ_LEN + 10:
                # 包含该月之前的历史用于序列构建
                month_df = df[df['Date'] <= month_end].tail(SEQ_LEN + 50)
            if len(month_df) < SEQ_LEN + 10:
                continue

            # 运行择时模型
            bt_df = run_backtest(month_df, timing_model, sX, sY,
                                  start_date=str(month_start.date()),
                                  initial_capital=per_stock_capital)
            if bt_df is None:
                continue

            # 根据回测结果更新持仓
            last_row = bt_df.iloc[-1]
            if last_row['Position'] > 0:
                # 持有：按资金比例分配股数
                alloc = per_stock_capital * last_row['Position']
                price = last_row['Close']
                shares = int(alloc / price) if price > 0 else 0
                if shares > 0 and code not in portfolio:
                    cost = shares * price
                    if cost <= cash:
                        cash -= cost
                        portfolio[code] = shares

        # 记录月末资产
        total_asset = cash
        for code, shares in portfolio.items():
            df = load_from_cache(code)
            if df is not None:
                df['Date'] = pd.to_datetime(df['Date'])
                row = df[df['Date'] <= month_end].tail(1)
                if len(row) > 0:
                    total_asset += shares * float(row['Close'].iloc[0])
        capital_history.append({'date': month_end, 'capital': total_asset})
        logger.info(f"月末总资产: {total_asset:,.0f} (现金 {cash:,.0f}, {len(portfolio)} 只持仓)")

    # ---- 结果汇总 ----
    print("\n" + "=" * 60)
    print("📊 两阶段回测结果")
    print(f"  选股模型: {selector_path}")
    print(f"  择时模型: {timing_model_path}")
    print(f"  回测区间: {sd.date()} ~ {ed.date()}")
    print(f"  每期选股: {top_n} 只")

    df_cap = pd.DataFrame(capital_history)
    total_ret = df_cap['capital'].iloc[-1] / initial_capital - 1
    print(f"  总收益: {total_ret*100:+.2f}%")

    # 月收益序列
    months_ret = df_cap['capital'].pct_change().dropna()
    if len(months_ret) > 1:
        ann_ret = (1 + total_ret) ** (12 / len(months_ret)) - 1
        sharpe = months_ret.mean() / months_ret.std() * np.sqrt(12) if months_ret.std() > 0 else 0
        max_dd = (df_cap['capital'] / df_cap['capital'].cummax() - 1).min()
        print(f"  年化收益: {ann_ret*100:.2f}%")
        print(f"  夏普: {sharpe:.3f}")
        print(f"  最大回撤: {max_dd*100:.2f}%")

    print("=" * 60)

    # 保存选股日志
    pd.DataFrame(monthly_picks_log).to_csv("two_stage_picks.csv", index=False)
    logger.info("选股日志: two_stage_picks.csv")

    return df_cap


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--timing-model", default="model_final.pth", help="择时模型")
    p.add_argument("--selector", default="selector_ensemble.pkl", help="选股模型")
    p.add_argument("--scaler-x", default="scaler_X_final.pkl")
    p.add_argument("--scaler-y", default="scaler_Y_final.pkl")
    p.add_argument("--start", default="2025-01-01", help="回测起始日期")
    p.add_argument("--top", type=int, default=20, help="每月选股数")
    args = p.parse_args()
    backtest_two_stage(args.timing_model, args.selector,
                       args.scaler_x, args.scaler_y,
                       args.start, args.top)
