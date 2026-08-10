#!/usr/bin/env python
# core/portfolio.py
"""组合优化：风险平价 + 置信度加权"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config.settings import *

logger = __import__('utils.logger', fromlist=['setup_logger']).setup_logger(__name__)


def compute_weights(codes, scores, price_history, lookback=60,
                    max_single=0.15, max_total=0.8):
    """
    风险平价权重 + 模型置信度加成

    Args:
        codes: 候选股票代码列表
        scores: {code: {'up_prob': float}} 盘前信号
        price_history: {code: pd.Series} 每只股票的收盘价序列（日线）
        lookback: 协方差回看天数
        max_single: 单只最大权重
        max_total: 总仓位上限

    Returns:
        {code: weight (0~0.15)} 归一化后权重
    """
    if len(codes) == 0:
        return {}

    # 1. 计算日收益率 + 协方差矩阵
    returns = {}
    for code in codes:
        px = price_history.get(code)
        if px is None or len(px) < lookback:
            returns[code] = pd.Series(dtype=float)
            continue
        r = px.pct_change().dropna().tail(lookback)
        if len(r) >= 20:
            returns[code] = r

    valid = [c for c in codes if c in returns and len(returns[c]) >= 20]
    if len(valid) < 2:
        # 单只或不足 → 直接按置信度分配
        return _simple_weights(valid, scores, max_single)

    # 对齐收益率矩阵
    ret_df = pd.DataFrame({c: returns[c] for c in valid}).dropna()
    if len(ret_df) < 20 or ret_df.shape[1] < 2:
        return _simple_weights(valid, scores, max_single)

    cov = ret_df.cov().values       # (n, n)
    vols = np.sqrt(np.diag(cov))    # (n,)

    # 2. 风险平价权重（w_i ∝ 1/vol_i）
    inv_vol = 1.0 / (vols + 1e-8)
    rp_weights = inv_vol / inv_vol.sum()

    # 3. 模型置信度叠加
    up_probs = np.array([scores.get(c, {}).get('up_prob', 0.5) for c in valid])
    # sigmoid 映射：up_prob=0.5 → factor=0.5, up_prob=0.6 → factor=0.73
    confidence = 1.0 / (1.0 + np.exp(-8 * (up_probs - 0.55)))
    final_weights = rp_weights * confidence
    final_weights /= final_weights.sum()

    # 4. 约束：单票 ≤ max_single，总仓位 ≤ max_total
    final_weights = np.clip(final_weights, 0, max_single)
    final_weights /= final_weights.sum()
    final_weights *= max_total

    result = {code: round(float(w), 4) for code, w in zip(valid, final_weights) if w > 0.005}
    logger.debug(f"组合优化: {len(result)}/{len(codes)}只, 权重分布 "
                 f"max={max(result.values()):.2%} sum={sum(result.values()):.2%}")
    return result


def _simple_weights(codes, scores, max_single):
    """单只或不足时，纯按置信度分配"""
    if not codes:
        return {}
    up = [scores.get(c, {}).get('up_prob', 0.5) for c in codes]
    w = np.clip(np.array(up) - 0.45, 0.01, max_single)
    w /= w.sum()
    w *= 0.8
    return {c: round(float(wi), 4) for c, wi in zip(codes, w) if wi > 0.005}


def risk_parity_weights(cov_matrix, max_iter=100):
    """
    精确风险平价（等边际风险贡献）
    每个 asset 的 RC_i = w_i * (Cov @ w)_i / sqrt(w^T Cov w)
    """
    n = cov_matrix.shape[0]
    w0 = np.ones(n) / n

    def obj(w):
        sigma = np.sqrt(w @ cov_matrix @ w)
        rc = w * (cov_matrix @ w) / sigma
        return np.sum((rc - rc.mean()) ** 2)

    cons = [{'type': 'eq', 'fun': lambda x: x.sum() - 1.0}]
    bounds = [(0.005, 0.15) for _ in range(n)]
    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': max_iter, 'ftol': 1e-10})
    return res.x if res.success else w0


# 便捷接口：直接从 data_loader 获取价格序列
def get_price_histories(codes, max_days=120):
    """从缓存读取每只股票的收盘价序列"""
    from core.data_loader import load_from_cache
    result = {}
    for code in codes:
        df = load_from_cache(code)
        if df is None:
            continue
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        close_col = 'Close' if 'Close' in df.columns else 'close'
        result[code] = df[close_col].tail(max_days)
    return result
