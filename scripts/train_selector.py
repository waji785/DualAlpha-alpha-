# scripts/train_selector.py
"""训练树模型选股层"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.settings import *
from core.data_loader import load_from_cache, get_stock_list
from core.stock_selector import TreeEnsemble, build_selector_dataset
from utils.logger import setup_logger

logger = setup_logger(__name__)


def train_selector(output_path="selector_ensemble.pkl",
                   forward_days=20, step_months=3):
    """训练截面选股模型"""
    logger.info("=" * 60)
    logger.info(f"训练树模型选股层 (前瞻{forward_days}天)")
    logger.info("=" * 60)

    # 1. 获取股票列表
    stock_df = get_stock_list(exclude_st=True, exclude_north=True)
    codes = stock_df['code'].tolist()
    logger.info(f"全市场股票: {len(codes)} 只")

    # 2. 加载所有缓存数据
    all_dfs = {}
    failed = 0
    for code in tqdm(codes, desc="加载缓存"):
        df = load_from_cache(code)
        if df is not None and len(df) >= 120:
            df['Date'] = pd.to_datetime(df['Date'])
            all_dfs[code] = df
        else:
            failed += 1
    logger.info(f"加载: {len(all_dfs)} 只有效, {failed} 失败")

    # 3. 构造截面训练数据
    X, y, sw, groups = build_selector_dataset(all_dfs, list(all_dfs.keys()),
                                              forward_days=forward_days,
                                              step_months=step_months)
    logger.info(f"训练数据: X={X.shape}, y={y.shape}")
    logger.info(f"y 分布: mean={y.mean():.4f}, std={y.std():.4f}, "
                f"min={y.min():.4f}, max={y.max():.4f}")

    # 4. 时序划分（按年滚动验证）
    split = int(len(X) * 0.8)
    X_tr, y_tr, sw_tr = X[:split], y[:split], sw[:split]
    X_val, y_val = X[split:], y[split:]

    logger.info(f"训练: {len(X_tr)}, 验证: {len(X_val)} (时序切分)")

    # 5. 训练
    ensemble = TreeEnsemble()
    ensemble.fit(X_tr, y_tr, X_val, y_val, sample_weight=sw_tr, groups=None)

    # 6. 验证（排序目标 = Spearman Rank IC）
    pred = ensemble.predict(X_val)
    from scipy.stats import spearmanr
    sp_corr, _ = spearmanr(pred, y_val)
    logger.info(f"Rank IC (Spearman): {sp_corr:.4f}")

    # 按预测排名分 10 组，看每组的平均真实排名
    n_bins = 10
    order = np.argsort(pred)
    group_size = len(pred) // n_bins
    logger.info("分组验证 (预测排名 → 真实排名均值):")
    for g in range(n_bins):
        g_idx = order[g*group_size:(g+1)*group_size]
        g_real = y_val[g_idx].mean()
        bar = '█' * int(g_real * 60)
        logger.info(f"  组{g+1}: 真实排名均值={g_real:.3f} {bar}")
    top_vs_bottom = y_val[order[-group_size:]].mean() - y_val[order[:group_size]].mean()
    logger.info(f"Top-Bottom 排名差: {top_vs_bottom:.3f}")

    # 7. 保存
    ensemble.save(output_path)
    logger.info(f"选股模型保存: {output_path}")
    return ensemble


if __name__ == "__main__":
    train_selector(os.path.join(OUTPUT_DIR, "selector_ensemble.pkl"))
