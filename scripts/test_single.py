# scripts/test_single.py
"""单股票回测 + 可视化"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch, joblib
import matplotlib.pyplot as plt
from config.settings import *
from core.model import DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer, Fusion_iTransformer
from core.data_loader import load_from_cache
from core.backtest_engine import run_backtest
from core.metrics import compute_metrics

torch.serialization.add_safe_globals([DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer])
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def test_single(code, model_path="model_final.pth",
                scaler_x="scaler_X_final.pkl", scaler_y="scaler_Y_final.pkl",
                start_date=None, initial_capital=100000):
    df = load_from_cache(code)
    if df is None:
        print(f"股票 {code} 无缓存数据")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    sX = joblib.load(scaler_x)
    sY = joblib.load(scaler_y) if os.path.exists(scaler_y) else None

    bt_df, trades = run_backtest(df, model, sX, sY, return_log=True,
                                 start_date=start_date)
    if bt_df is None:
        print("回测失败（数据不足或无信号）")
        return

    m = compute_metrics(bt_df['Capital'].values)
    print("\n" + "=" * 60)
    print(f"📊 {code} 回测结果")
    print(f"  区间: {bt_df['Date'].iloc[0].date()} ~ {bt_df['Date'].iloc[-1].date()}")
    print(f"  总收益: {m['total_return']*100:+.2f}%")
    print(f"  年化:   {m['annual_return']*100:.2f}%")
    print(f"  最大回撤: {m['max_drawdown']*100:.2f}%")
    print(f"  夏普:   {m['sharpe_ratio']:.3f}")
    print(f"  Calmar: {m['calmar_ratio']:.3f}")
    print(f"  胜率:   {m['win_rate']*100:.1f}%")
    print("-" * 60)
    if trades:
        print(f"  交易记录 ({len(trades)} 笔):")
        for t in trades:
            if len(t) >= 4:
                print(f"    {t[0]:10s} {str(t[1])[:10]}  price={t[2]:.2f}  ret={t[3] if t[3] else '':.4}" \
                      if isinstance(t[3], float) else f"    {t[0]:10s} {str(t[1])[:10]}  price={t[2]:.2f}")

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    cap = bt_df['Capital'].values
    axes[0].plot(bt_df['Date'], cap / cap[0] * initial_capital, 'b-', linewidth=1.5)
    axes[0].axhline(initial_capital, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('资金')
    axes[0].set_title(f'{code} 资金曲线')
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(bt_df['Date'], 0, bt_df['Position'], alpha=0.3, color='green')
    axes[1].plot(bt_df['Date'], bt_df['Position'], 'g-', linewidth=1)
    axes[1].set_ylabel('仓位')
    axes[1].set_ylim(0, 1.2)
    axes[1].grid(True, alpha=0.3)

    close = bt_df['Close'].values
    axes[2].plot(bt_df['Date'], close, 'k-', linewidth=1, alpha=0.8)
    axes[2].set_ylabel('价格')
    axes[2].set_xlabel('日期')
    axes[2].grid(True, alpha=0.3)

    # 标记买卖点
    for t in trades:
        if t[0] == '买入':
            idx = (bt_df['Date'] == t[1]).argmax()
            if idx < len(close):
                axes[2].scatter(bt_df['Date'].iloc[idx], close[idx], marker='^',
                               color='red', s=80, zorder=5)
        elif t[0] in ('止盈', '跟踪止盈', '信号卖出', '止损', '减仓'):
            idx = (bt_df['Date'] == t[1]).argmax()
            if idx < len(close):
                axes[2].scatter(bt_df['Date'].iloc[idx], close[idx], marker='v',
                               color='green', s=80, zorder=5)

    plt.tight_layout()
    fig_path = f"test_{code}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n图表: {fig_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--code", required=True, help="股票代码")
    p.add_argument("--model", default="model_final.pth")
    p.add_argument("--scaler-x", default="scaler_X_final.pkl")
    p.add_argument("--scaler-y", default="scaler_Y_final.pkl")
    p.add_argument("--start", default=None, help="回测起始日期")
    args = p.parse_args()
    test_single(args.code, args.model, args.scaler_x, args.scaler_y, args.start)
