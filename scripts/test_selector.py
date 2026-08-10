# scripts/test_selector.py
"""检测选股模型是否正常"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from config.settings import *
from core.data_loader import load_from_cache, get_stock_list
from core.stock_selector import TreeEnsemble, build_selector_dataset
from utils.logger import setup_logger
logger = setup_logger(__name__)


def test_selector(selector_path="selector_ensemble.pkl", top_n=20,
                  forward_days=20, test_date="2025-01-01"):
    """测试选股模型：加载模型 → 选股 → 验证预测与实际收益的排名相关性"""
    print("=" * 60)
    print(f"选股模型测试: {selector_path}")
    print(f"基准日期: {test_date}, 前瞻: {forward_days}天")
    print("=" * 60)

    # 1. 加载模型
    selector = TreeEnsemble()
    if not os.path.exists(selector_path):
        print(f"模型不存在: {selector_path}")
        return
    selector.load(selector_path)
    print("模型加载成功")

    # 2. 获取股票（baostock 挂了就从缓存读取）
    try:
        stock_df = get_stock_list(exclude_st=True, exclude_north=True)
        codes = stock_df['code'].tolist()
        name_map = dict(zip(stock_df['code'], stock_df['name']))
    except Exception:
        from config.settings import CACHE_DIR
        codes = [f.replace('.parquet', '') for f in os.listdir(CACHE_DIR) if f.endswith('.parquet')]
        name_map = {c: c for c in codes}
        logger.warning(f"从缓存读取: {len(codes)} 只")
    test_dt = pd.to_datetime(test_date)

    all_feats, all_codes, all_names = [], [], []
    skipped = 0

    for code in codes:
        df = load_from_cache(code)
        if df is None or len(df) < SEQ_LEN + 60:
            skipped += 1
            continue
        df['Date'] = pd.to_datetime(df['Date'])
        pre = df[df['Date'] < test_dt]
        if len(pre) < SEQ_LEN:
            skipped += 1
            continue
        row = pre.iloc[-1]
        feat = row[FEATURE_COLS].values.astype(np.float32)
        if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
            skipped += 1
            continue
        all_feats.append(feat)
        all_codes.append(code)
        all_names.append(name_map.get(code, ''))

    n_total = len(all_feats)
    print(f"有效股票: {n_total} (跳过: {skipped})")

    if n_total < top_n:
        print(f"不足 {top_n} 只，无法选股")
        return

    # 3. 预测 + 排序
    X = np.array(all_feats)
    predictions = selector.predict(X)
    rank = np.argsort(predictions)[::-1]
    top_idx = rank[:top_n]

    print(f"\n预测统计: mean={predictions.mean():.4f}, std={predictions.std():.4f}, "
          f"min={predictions.min():.4f}, max={predictions.max():.4f}")

    # 4. 展示 Top-N (预测排序分 vs 实际截面排名)
    print(f"\nTop {top_n} (预测分 vs 截面排名, 1.0=最优):")
    print("-" * 55)

    # 计算截面实际排名
    all_actuals = []
    for code in all_codes:
        df = load_from_cache(code); df['Date'] = pd.to_datetime(df['Date'])
        post = df[(df['Date'] >= test_dt)]
        if len(post) > forward_days:
            ret = (post['Close'].iloc[forward_days] - post['Close'].iloc[0]) / post['Close'].iloc[0]
        elif len(post) > 0:
            ret = (post['Close'].iloc[-1] - post['Close'].iloc[0]) / post['Close'].iloc[0]
        else:
            ret = np.nan
        all_actuals.append(ret)
    all_actuals = np.array(all_actuals)
    valid_mask = ~np.isnan(all_actuals)
    from scipy.stats import spearmanr, rankdata
    actual_rank = np.full(len(all_codes), np.nan)
    actual_rank[valid_mask] = rankdata(all_actuals[valid_mask]) / valid_mask.sum()

    for i, idx in enumerate(top_idx):
        code = all_codes[idx]
        name = all_names[idx]
        pred = predictions[idx]
        a_rank = actual_rank[idx]
        bar_pred = '█' * int(pred * 40)
        bar_real = '█' * int(a_rank * 40) if not np.isnan(a_rank) else '?'
        print(f"  {i+1:4d} {code:8s} {name:10s} pred={pred:.3f} real={a_rank:.3f}  {bar_pred}")
        print(f"  {'':4s} {'':8s} {'':10s}         {'':7s}        {bar_real}")

    print("-" * 70)

    # 5. Rank IC
    ic = spearmanr(predictions[valid_mask], actual_rank[valid_mask])[0] if valid_mask.sum() > 30 else 0
    print(f"\nRank IC (Spearman): {ic:.4f} {'✓' if abs(ic)>0.04 else '⚠' if abs(ic)>0.02 else '✗'}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selector", default="selector_ensemble.pkl", help="选股模型路径")
    p.add_argument("--date", default="2025-01-01", help="选股基准日期")
    p.add_argument("--top", type=int, default=20, help="展示前N只")
    p.add_argument("--forward", type=int, default=20, help="前瞻天数")
    args = p.parse_args()
    test_selector(args.selector, args.top, args.forward, args.date)
