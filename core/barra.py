#!/usr/bin/env python
# core/barra.py
"""Barra 风格因子分解（简化版，从现有 OHLCV + 基本面数据计算）"""
import numpy as np
import pandas as pd
from config.settings import *

logger = __import__('utils.logger', fromlist=['setup_logger']).setup_logger(__name__)


def compute_barra_factors(df, index_df=None):
    """
    从单只股票的日线 DataFrame 计算 Barra 风格因子暴露

    Args:
        df: 日线数据（columns 含 Close, Volume, Turn, PeTTM, PbMRQ, Amount）
        index_df: 指数日线（可选，用于 Beta 计算）

    Returns:
        dict of {factor_name: latest_value}
    """
    if df is None or len(df) < 60:
        return {}

    close = df['Close'].values.astype(float)
    volume = df['Volume'].values.astype(float)
    ret = np.diff(close) / close[:-1]  # 日收益率

    factors = {}

    # 1. Size (ln market cap) — 从 Amount 粗略估算: 市值 ≈ Amount / Turnover
    if 'Turn' in df.columns and 'Amount' in df.columns:
        idx = -1
        while idx >= -len(df) and (df['Turn'].iloc[idx] <= 0 or pd.isna(df['Turn'].iloc[idx])):
            idx -= 1
        if idx >= -len(df):
            cap = df['Amount'].iloc[idx] / max(df['Turn'].iloc[idx], 0.001) * 100
            factors['Size'] = np.log(max(cap, 1e8))
        else:
            factors['Size'] = np.log(1e9)  # 默认中等市值

    # 2. Beta (60-day to market)
    if index_df is not None and len(index_df) >= 60:
        idx_ret = index_df['Close'].pct_change().dropna().values[-60:]
        ret_60 = ret[-min(len(ret), len(idx_ret)):]
        idx_60 = idx_ret[-len(ret_60):]
        if len(ret_60) >= 20:
            cov = np.cov(ret_60, idx_60)[0, 1]
            var = np.var(idx_60)
            factors['Beta'] = cov / var if var > 0 else 1.0
        else:
            factors['Beta'] = 1.0
    else:
        factors['Beta'] = 1.0

    # 3. Momentum (20-day return)
    if len(close) >= 21:
        factors['Momentum'] = (close[-1] / close[-21] - 1)

    # 4. Volatility (60-day std of returns)
    factors['Volatility'] = np.std(ret[-60:]) * np.sqrt(252) if len(ret) >= 60 else 0.3

    # 5. Value (B/P = 1/PbMRQ)
    pb = df['PbMRQ'].values[-1] if 'PbMRQ' in df.columns else np.nan
    if not np.isnan(pb) and pb > 0:
        factors['Value'] = 1.0 / pb
    else:
        factors['Value'] = 0.0

    # 6. Leverage — 简化：用 PE 倒数作为盈利收益率
    pe = df['PeTTM'].values[-1] if 'PeTTM' in df.columns else np.nan
    if not np.isnan(pe) and pe > 0:
        factors['Earnings_Yield'] = 1.0 / pe
    else:
        factors['Earnings_Yield'] = 0.03  # 默认 3%

    # 7. Liquidity (20-day avg turnover)
    if 'Turn' in df.columns:
        factors['Liquidity'] = np.nanmean(df['Turn'].values[-20:].astype(float)) / 100
    else:
        factors['Liquidity'] = 0.01

    # 8. Size 分类（用于风格中性化）
    factors['Size_Group'] = 'large' if factors.get('Size', 20) > np.log(5e10) else \
        'mid' if factors['Size'] > np.log(1e10) else 'small'

    return factors


def portfolio_factor_report(positions, df_cache):
    """
    组合层面的 Barra 因子暴露报告

    Args:
        positions: {code: weight} 持仓权重
        df_cache: {code: DataFrame} 缓存日线

    Returns:
        DataFrame: 各因子暴露
    """
    if not positions:
        return pd.DataFrame()

    total_weight = sum(positions.values())
    if total_weight == 0:
        return pd.DataFrame()

    factor_sums = {}
    for code, weight in positions.items():
        df = df_cache.get(code)
        if df is None:
            continue
        factors = compute_barra_factors(df)
        w = weight / total_weight
        for key, val in factors.items():
            if isinstance(val, (int, float, np.floating)):
                factor_sums[key] = factor_sums.get(key, 0) + val * w

    report = pd.DataFrame({
        'Factor': list(factor_sums.keys()),
        'Exposure': [round(v, 4) for v in factor_sums.values()],
    }).sort_values('Exposure', key=abs, ascending=False)

    logger.info(f"组合因子暴露:\n{report.to_string(index=False)}")
    return report


def style_neutralize(X, factor_values, target_factor='Size'):
    """
    单因子中性化：X 对 target 做 OLS 回归取残差

    Args:
        X: (n_samples, n_features) 原始特征
        factor_values: (n_samples,) 因子值
    """
    from sklearn.linear_model import LinearRegression
    if len(factor_values) < 2:
        return X
    reg = LinearRegression().fit(factor_values.reshape(-1, 1), X)
    pred = reg.predict(factor_values.reshape(-1, 1))
    return X - pred


# ---- 快捷入口：持仓组合因子分析 ----
def analyze_portfolio(portfolio_codes, portfolio_weights, all_dfs=None):
    """打印当前组合的 Barra 分解"""
    if all_dfs is None:
        from core.data_loader import load_all_stock_data
        all_dfs = load_all_stock_data()

    df_cache = {c: all_dfs.get(c) for c in portfolio_codes}
    return portfolio_factor_report(
        dict(zip(portfolio_codes, portfolio_weights)), df_cache)
