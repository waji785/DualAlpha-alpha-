#!/usr/bin/env python
# research/test_alpha101.py
"""Alpha101 因子批量测试 — 独立运行，不影响 core"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from research.factor_lab import FactorLab
from research.alpha101 import ALL_ALPHAS
from research.alpha158 import ALL_ALPHA158
from core.data_loader import load_from_cache

# 测试股票池（主要指数成分股）
TEST_CODES = [
    '000001', '000002', '000858', '002415', '600000', '600036',
    '600519', '600276', '600887', '601012', '601088', '601166',
    '601318', '603259', '688981',
]

def main():
    lab = FactorLab()
    results = {}

    for alpha_name, alpha_func in ALL_ALPHAS.items():
        ics, nans = [], []
        for code in TEST_CODES:
            df = load_from_cache(code)
            if df is None or len(df) < 300:
                continue
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()

            try:
                factor_series = alpha_func(df)
                # 需要 index 对齐
                if isinstance(factor_series, (pd.Series, np.ndarray)):
                    factor_series = pd.Series(factor_series, index=df.index)
                result = lab.test_factor(alpha_name, lambda d: factor_series, df)
                if 'ic' in result:
                    ics.append(result['ic'])
                    nans.append(result.get('nan_rate', 0))
            except Exception as e:
                pass

        if ics:
            results[alpha_name] = {
                'ic_mean': round(np.mean(ics), 4),
                'ic_std': round(np.std(ics), 4),
                'nan_rate': round(np.mean(nans), 4),
                'stocks': len(ics),
            }

    # 排序输出
    print("\n" + "=" * 65)
    print(f"{'Alpha':12s} {'IC均值':>8s} {'IC_std':>8s} {'NaN%':>6s} {'样本':>5s} {'评级':>6s}")
    print("=" * 65)
    for name in sorted(results, key=lambda k: abs(results[k]['ic_mean']), reverse=True):
        r = results[name]
        stars = '★★★' if abs(r['ic_mean']) > 0.03 else \
                '★★' if abs(r['ic_mean']) > 0.02 else '★' if abs(r['ic_mean']) > 0.01 else '-'
        print(f"{name:12s} {r['ic_mean']:+8.4f} {r['ic_std']:8.4f} "
              f"{r['nan_rate']*100:5.1f}% {r['stocks']:5d} {stars:6s}")
    print("=" * 65)

    # 推荐集成的因子
    candidates = [name for name, r in results.items()
                  if abs(r['ic_mean']) > 0.02 and r['nan_rate'] < 0.2]
    if candidates:
        print(f"\n✅ 推荐集成 ({len(candidates)} 个): {', '.join(candidates)}")
        print(f"   加入 config/settings.py 的 FEATURE_COLS 即可")

    # 滚动窗口分析
    print("\n\n滚动窗口分析 (15只股票x10年全量):")
    lab.rolling_window_analysis(ALL_ALPHAS, code_list=TEST_CODES[:15], n_samples=15, window_years=10)

    # Alpha158 测试
    print("\n\n" + "=" * 65)
    print("Alpha158 因子测试")
    print("=" * 65)
    results158 = {}
    for name, func in ALL_ALPHA158.items():
        ics, nans = [], []
        for code in TEST_CODES:
            df = load_from_cache(code)
            if df is None or len(df) < 300: continue
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            try:
                factor_series = func(df)
                if isinstance(factor_series, (pd.Series, np.ndarray)):
                    factor_series = pd.Series(factor_series, index=df.index)
                result = lab.test_factor(name, lambda d: factor_series, df)
                if 'ic' in result:
                    ics.append(result['ic'])
                    nans.append(result.get('nan_rate', 0))
            except: pass
        if ics:
            results158[name] = {'ic_mean': round(np.mean(ics), 4),
                                'ic_std': round(np.std(ics), 4),
                                'stocks': len(ics)}
    for name in sorted(results158, key=lambda k: abs(results158[k]['ic_mean']), reverse=True):
        r = results158[name]
        stars = '★★★' if abs(r['ic_mean'])>0.03 else '★★' if abs(r['ic_mean'])>0.02 else '★' if abs(r['ic_mean'])>0.01 else '-'
        print(f"{name:12s} {r['ic_mean']:+8.4f} {r['ic_std']:8.4f} {r['stocks']:5d} {stars:6s}")
    print("=" * 65)
    candidates158 = [n for n, r in results158.items() if abs(r['ic_mean']) > 0.02]
    if candidates158:
        print(f"\n✅ 推荐集成 ({len(candidates158)} 个): {', '.join(candidates158)}")


if __name__ == "__main__":
    main()
