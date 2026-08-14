#!/usr/bin/env python
# core/backtest_analyzer.py
"""
回测分析层：对资金曲线做专业报告
支持独立使用（传 CSV）或集成到 batch_backtest
"""
import numpy as np
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_metrics(equity_curve, trade_log=None, benchmark_curve=None, risk_free=0.02):
    """
    从资金曲线计算核心指标

    Args:
        equity_curve: pd.Series 或 array, 每日总资产
        trade_log: list of dicts, 可选
        benchmark_curve: pd.Series, 基准曲线（如沪深 300）
        risk_free: 无风险利率

    Returns:
        dict
    """
    if isinstance(equity_curve, pd.Series):
        ec = equity_curve.values
        dates = equity_curve.index
    else:
        ec = np.asarray(equity_curve)
        dates = None

    if len(ec) < 2:
        return {'error': '数据不足'}

    returns = np.diff(ec) / ec[:-1]
    n_days = len(returns)

    # 基础指标
    total_return = (ec[-1] / ec[0] - 1) if ec[0] > 0 else 0
    ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    peak = np.maximum.accumulate(ec)
    drawdown = (ec - peak) / peak
    max_dd = float(drawdown.min())
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Sortino（只计算下行波动）
    down_returns = returns[returns < 0]
    down_vol = down_returns.std() * np.sqrt(252) if len(down_returns) > 0 else 0
    sortino = (ann_return - risk_free) / down_vol if down_vol > 0 else 0

    # 胜率 & 盈亏比
    win_rate = float(np.mean(returns > 0))
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    # 交易统计
    n_trades = len(trade_log) if trade_log else 0
    trade_wins = sum(1 for t in trade_log if float(t.get('profit', 0)) > 0) if trade_log else 0
    trade_win_rate = trade_wins / n_trades if n_trades > 0 else 0

    # 超额收益 (alpha)
    alpha, beta, ir = np.nan, np.nan, np.nan
    if benchmark_curve is not None:
        if isinstance(benchmark_curve, pd.Series):
            bm = benchmark_curve.reindex(dates).values if dates is not None else benchmark_curve.values
        else:
            bm = np.asarray(benchmark_curve)
        min_len = min(len(returns), len(bm) - 1)
        if min_len > 20:
            bm_ret = np.diff(bm[:min_len+1]) / bm[:min_len]
            ret_aligned = returns[:min_len]
            cov = np.cov(ret_aligned, bm_ret)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
            excess = ret_aligned - bm_ret
            alpha = float(excess.mean()) * 252
            ir = float(excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0

    return {
        'total_return': round(total_return * 100, 2),
        'ann_return': round(ann_return * 100, 2),
        'ann_vol': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 2),
        'win_rate': round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 2),
        'n_trades': n_trades,
        'trade_win_rate': round(trade_win_rate * 100, 1) if n_trades > 0 else 0,
        'alpha': round(alpha, 4) if not np.isnan(alpha) else None,
        'beta': round(beta, 2) if not np.isnan(beta) else None,
        'ir': round(ir, 2) if not np.isnan(ir) else None,
    }


def print_report(metrics, title="回测报告"):
    """打印格式化报告"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    rows = [
        ('累计收益率', f"{metrics.get('total_return', 'N/A')}%"),
        ('年化收益率', f"{metrics.get('ann_return', 'N/A')}%"),
        ('年化波动率', f"{metrics.get('ann_vol', 'N/A')}%"),
        ('夏普比率', f"{metrics.get('sharpe', 'N/A')}"),
        ('索提诺比率', f"{metrics.get('sortino', 'N/A')}"),
        ('最大回撤', f"{metrics.get('max_drawdown', 'N/A')}%"),
        ('卡玛比率', f"{metrics.get('calmar', 'N/A')}"),
        ('日胜率', f"{metrics.get('win_rate', 'N/A')}%"),
        ('盈亏比', f"{metrics.get('profit_factor', 'N/A')}"),
        ('交易次数', f"{metrics.get('n_trades', 'N/A')}"),
        ('交易胜率', f"{metrics.get('trade_win_rate', 'N/A')}%"),
    ]
    if metrics.get('alpha') is not None:
        rows += [
            ('Alpha', f"{metrics['alpha']:.4f}"),
            ('Beta', f"{metrics['beta']}"),
            ('信息比率', f"{metrics['ir']}"),
        ]
    for label, value in rows:
        print(f"  {label:12s} {value:>12s}")
    print("=" * 60)


def monthly_returns_heatmap(equity_curve):
    """月度收益表（不用 matplotlib 也能看）"""
    if not isinstance(equity_curve, pd.Series):
        equity_curve = pd.Series(np.asarray(equity_curve))

    monthly = equity_curve.resample('M').last().pct_change().dropna() * 100
    if len(monthly) == 0:
        return

    df = monthly.to_frame('Return')
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    pivot = df.pivot(index='Year', columns='Month', values='Return')

    print("\n月度收益 (%)")
    print(pivot.round(1).to_string(na_rep='-'))
    return pivot


def analyze_backtest(backtest_df, trade_log=None, benchmark_df=None, title=""):
    """
    一键分析回测结果

    Args:
        backtest_df: run_backtest 返回的 DataFrame (Date, Close, Position, Capital)
        trade_log: 交易列表
        benchmark_df: 基准 DataFrame (Date, Close)
    """
    if backtest_df is None or len(backtest_df) < 2:
        print("回测数据不足")
        return

    equity = backtest_df.set_index('Date')['Capital'] if 'Date' in backtest_df.columns \
             else backtest_df['Capital']

    benchmark = None
    if benchmark_df is not None:
        benchmark = benchmark_df.set_index('Date')['Close'] if 'Date' in benchmark_df \
                    else benchmark_df['Close']

    metrics = compute_metrics(equity, trade_log, benchmark)
    print_report(metrics, title or "回测结果")
    monthly_returns_heatmap(equity)

    return metrics
