import numpy as np

def compute_metrics(capital_series, risk_free_rate=0.025, trading_days=252):
    """计算绩效指标"""
    capital = np.asarray(capital_series)
    if len(capital) < 2:
        return {}
    daily_ret = np.diff(capital) / capital[:-1]
    total_return = (capital[-1] - capital[0]) / capital[0]
    n_days = len(capital)
    annual_return = (1 + total_return) ** (trading_days / n_days) - 1

    peak = np.maximum.accumulate(capital)
    drawdown = (peak - capital) / peak
    max_drawdown = np.max(drawdown)

    excess_ret = daily_ret - risk_free_rate / trading_days
    std = np.std(excess_ret)
    sharpe = np.sqrt(trading_days) * np.mean(excess_ret) / std if std > 1e-8 else 0

    # Calmar: 年化收益 / 最大回撤
    calmar = annual_return / max_drawdown if max_drawdown > 1e-8 else 0

    # Sortino: 只计入下行波动
    downside = excess_ret[excess_ret < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 0
    sortino = np.sqrt(trading_days) * np.mean(excess_ret) / downside_std if downside_std > 1e-8 else 0

    # VaR 95%（历史模拟法）
    var_95 = np.percentile(daily_ret, 5)

    win_days = np.sum(daily_ret > 0)
    win_rate = win_days / len(daily_ret) if len(daily_ret) > 0 else 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'sortino_ratio': sortino,
        'var_95': var_95,
        'win_rate': win_rate,
        'n_days': n_days
    }