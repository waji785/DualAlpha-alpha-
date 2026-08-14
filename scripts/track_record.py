#!/usr/bin/env python
# scripts/track_record.py
"""计算存档信号的实际收益：每天信号 vs forward_days 后真实涨跌"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import spearmanr

from config.settings import OUTPUT_DIR
from core.data_loader import load_from_cache
from utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_track_record(signal_dir=None, forward_days=20):
    """读取所有存档信号文件，计算实际收益率"""
    if signal_dir is None:
        signal_dir = OUTPUT_DIR

    files = sorted(glob(os.path.join(signal_dir, "signals_*.csv")))
    if not files:
        logger.error(f"未找到信号存档文件 (signals_*.csv) in {signal_dir}")
        return

    logger.info(f"加载 {len(files)} 天信号")

    all_records = []
    for f in files:
        date_str = os.path.basename(f).replace("signals_", "").replace(".csv", "")
        try:
            sig_date = pd.to_datetime(date_str)
        except:
            continue

        df = pd.read_csv(f)
        for _, row in df.iterrows():
            code = str(row['code']).zfill(6)
            up_prob = float(row.get('up_prob', 0))
            signal = row.get('signal', '')
            close = float(row.get('close', 0))

            # 查实际 forward 收益
            stock_df = load_from_cache(code)
            if stock_df is None:
                continue
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            future = stock_df[stock_df['Date'] >= sig_date]
            if len(future) > forward_days:
                actual_close = future['Close'].iloc[forward_days]
            elif len(future) > 0:
                actual_close = future['Close'].iloc[-1]
            else:
                continue
            actual_ret = (actual_close - close) / close if close > 0 else 0

            all_records.append({
                'date': sig_date,
                'code': code,
                'signal': signal,
                'up_prob': up_prob,
                'pred_close': close,
                'actual_close': actual_close,
                'actual_ret': actual_ret,
            })

    if not all_records:
        logger.error("无有效记录")
        return

    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['date'])

    # 逐日统计
    daily = df.groupby('date').agg(
        count=('code', 'count'),
        avg_up_prob=('up_prob', 'mean'),
        avg_actual_ret=('actual_ret', 'mean'),
        buy_count=('signal', lambda x: (x == '买入').sum()),
        buy_actual_ret=('actual_ret', lambda x: x[df.loc[x.index, 'signal'] == '买入'].mean()),
    ).reset_index()

    # 汇总
    total = len(df)
    buy_total = (df['signal'] == '买入').sum()
    hit_rate = (df['actual_ret'] > 0).mean()
    buy_hit = (df[df['signal'] == '买入']['actual_ret'] > 0).mean() if buy_total > 0 else 0
    ic, _ = spearmanr(df['up_prob'], df['actual_ret'])

    print("\n" + "=" * 60)
    print(f"信号跟踪报告 ({forward_days}日预测)")
    print("=" * 60)
    print(f"  统计天数:     {len(daily)}")
    print(f"  总信号数:     {total}")
    print(f"  买入信号:     {buy_total} ({buy_total/total*100:.0f}%)" if total > 0 else "")
    print(f"  整体命中率:   {hit_rate*100:.1f}%")
    print(f"  买入命中率:   {buy_hit*100:.1f}%" if buy_total > 0 else "")
    print(f"  Rank IC:      {ic:.4f}")
    print(f"  日均收益率:   {daily['avg_actual_ret'].mean()*100:.2f}%")
    print(f"  买入日均收益: {daily['buy_actual_ret'].mean()*100:.2f}%" if buy_total > 0 else "")
    print("=" * 60)

    # 按周汇总
    df['week'] = df['date'].dt.isocalendar().week
    weekly = df.groupby('week').agg(
        avg_ret=('actual_ret', 'mean'),
        buy_ret=('actual_ret', lambda x: x[df.loc[x.index, 'signal'] == '买入'].mean()),
        count=('code', 'count'),
    ).reset_index()
    print("\n按周汇总:")
    print(weekly.to_string(index=False))

    # 保存
    summary_path = os.path.join(signal_dir, "track_record_summary.csv")
    daily.to_csv(summary_path, index=False, encoding='utf-8-sig')
    logger.info(f"汇总保存: {summary_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=None, help="信号存档目录")
    p.add_argument("--forward", type=int, default=20, help="前瞻天数")
    args = p.parse_args()
    compute_track_record(args.dir, args.forward)
