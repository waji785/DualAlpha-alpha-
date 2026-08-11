#!/usr/bin/env python
# research/factor_lab.py
"""
因子实验室：开发、测试、选择因子
独立于 core 模块，通过验证后手动集成
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import CACHE_DIR, FEATURE_COLS
from core.data_loader import load_from_cache

class FactorLab:
    """因子开发与验证工具"""

    def __init__(self):
        self.registry = {}  # {name: {'ic': float, 'stability': float, 'corr': dict}}

    # ============================================================
    #  单因子测试
    # ============================================================
    def test_factor(self, name, compute_func, df_sample, forward_days=20):
        """
        测试一个因子

        Args:
            name: 因子名称
            compute_func: fn(df) -> pd.Series (idx=日期)
            df_sample: 单只股票的日线 DataFrame
            forward_days: 前瞻天数

        Returns:
            dict: {'ic': float, 'hit_rate': float, 'nan_rate': float}
        """
        try:
            factor = compute_func(df_sample)
        except Exception as e:
            return {'error': str(e)}

        if not isinstance(factor, pd.Series):
            return {'error': 'compute_func 须返回 pd.Series'}

        # 前向收益
        close = df_sample['Close'].values
        future_ret = pd.Series(
            (np.roll(close, -forward_days) - close) / close,
            index=df_sample.index
        )
        future_ret.iloc[-forward_days:] = np.nan

        # 对齐
        aligned = pd.DataFrame({'factor': factor, 'fwd_ret': future_ret}).dropna()
        if len(aligned) < 50:
            return {'error': f'有效样本不足 ({len(aligned)})'}

        nan_rate = (len(df_sample) - len(aligned)) / len(df_sample)

        # 月度截面 IC
        aligned['month'] = aligned.index.to_period('M')
        monthly_ic = []
        for month, grp in aligned.groupby('month'):
            if len(grp) < 10: continue
            ic, _ = spearmanr(grp['factor'], grp['fwd_ret'])
            monthly_ic.append(ic)

        if not monthly_ic:
            return {'error': '月度分组不足'}

        ic_mean = np.mean(monthly_ic)
        ic_std = np.std(monthly_ic)
        stability = np.mean([ic > 0 for ic in monthly_ic])

        result = {
            'ic': round(ic_mean, 4),
            'ic_std': round(ic_std, 4),
            'stability': round(stability, 4),
            'nan_rate': round(nan_rate, 4),
            'months': len(monthly_ic),
            'pass': abs(ic_mean) > 0.02 and stability > 0.55 and nan_rate < 0.2,
        }
        self.registry[name] = result
        return result

    # ============================================================
    #  相关性检查（防冗余）
    # ============================================================
    def check_collinearity(self, new_factor_name, reference_factor_names=None):
        """检查新因子与现有因子的相关性"""
        if reference_factor_names is None:
            reference_factor_names = FEATURE_COLS

        if new_factor_name not in self.registry:
            return {'error': f'{new_factor_name} 未注册（先运行 test_factor）'}

        result = {}
        for ref in reference_factor_names:
            if ref == new_factor_name: continue
            # 简化为名称比较——完整实现需要值对值计算
            result[ref] = np.nan
        return result

    # ============================================================
    #  批量扫描
    # ============================================================
    def scan(self, factor_dict, code_list=None, n_samples=50):
        """
        批量测试多个因子

        Args:
            factor_dict: {name: compute_func}
            n_samples: 抽样股票数
        """
        if code_list is None:
            import os
            code_list = [f.replace('.parquet', '')
                        for f in os.listdir(CACHE_DIR) if f.endswith('.parquet')][:n_samples]

        results = {}
        for name, func in factor_dict.items():
            ics = []
            for code in code_list:
                df = load_from_cache(code)
                if df is None or len(df) < 200: continue
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                r = self.test_factor(name, func, df)
                if 'ic' in r:
                    ics.append(r['ic'])
            if ics:
                results[name] = {
                    'ic_mean': round(np.mean(ics), 4),
                    'ic_std': round(np.std(ics), 4),
                    'stocks_tested': len(ics),
                }
            else:
                results[name] = {'error': '无有效股票'}

        for name, r in results.items():
            print(f"  {name:20s} IC={r.get('ic_mean', 'N/A')}  "
                  f"n={r.get('stocks_tested', 'N/A')}")

        return results

    # ============================================================
    #  因子集成指南
    # ============================================================
    def promote(self, name):
        if name not in self.registry:
            print(f"{name} 未注册")
            return
        r = self.registry[name]
        print(f"\n集成 {name} | IC={r.get('ic')}")
        print(f"  1. features.py 添加计算")
        print(f"  2. settings.py FEATURE_COLS 加入 '{name}'")

    def audit_all_features(self, code_list=None, n_samples=100, forward_days=20):
        """审计 FEATURE_COLS 中所有特征，输出保留/剔除建议"""
        features = list(FEATURE_COLS)
        if code_list is None:
            import os
            code_list = [f.replace('.parquet', '') for f in os.listdir(CACHE_DIR)
                        if f.endswith('.parquet')][:n_samples]

        logger = __import__('utils.logger', fromlist=['setup_logger']).setup_logger(__name__)
        logger.info(f"审计 {len(features)} 特征 x {len(code_list)} 股票")

        feature_ics = {f: [] for f in features}
        for code in code_list:
            df = load_from_cache(code)
            if df is None or len(df) < 200: continue
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            close = df['Close'].values
            fwd = (np.roll(close, -forward_days) - close) / close
            fwd[-forward_days:] = np.nan
            for feat in features:
                if feat not in df.columns: continue
                vals = df[feat].values.astype(float)
                mask = ~(np.isnan(vals) | np.isnan(fwd))
                if mask.sum() < 30: continue
                ic, _ = spearmanr(vals[mask], fwd[mask])
                feature_ics[feat].append(ic)

        results = []
        for feat in features:
            ics = [v for v in feature_ics[feat] if not np.isnan(v)]
            if not ics:
                results.append({'feature': feat, 'ic_mean': 0, 'status': 'NODATA'}); continue
            ic_mean = np.mean(ics); ic_std = np.std(ics)
            stability = np.mean([1 if v > 0 else 0 for v in ics])
            if abs(ic_mean) < 0.005: status = 'DROP'
            elif stability < 0.5: status = 'WEAK'
            elif abs(ic_mean) < 0.015: status = 'WEAK'
            else: status = 'KEEP'
            results.append({'feature': feat, 'ic_mean': round(ic_mean, 4),
                            'ic_std': round(ic_std, 4),
                            'stability': round(stability, 4),
                            'status': status,
                            'n_valid': len(ics)})

        df_out = pd.DataFrame(results).sort_values('ic_mean', key=abs, ascending=False)

        print("\n" + "=" * 72)
        print(f"全量特征审计 ({len(code_list)} 股票, {forward_days}日)")
        print("=" * 72)
        for _, r in df_out.iterrows():
            print(f"{r['feature']:25s} IC={r['ic_mean']:+8.4f}({r.get('ic_std',0):.2f}) "
                  f"s={r.get('stability',0):.0%} n={r.get('n_valid',0)} {r['status']}")

        keep = sum(r['status'] == 'KEEP' for _, r in df_out.iterrows())
        drop = sum(r['status'] == 'DROP' for _, r in df_out.iterrows())
        print(f"\nKEEP:{keep}  DROP:{drop}  WEAK:{len(df_out)-keep-drop}")

        drops = df_out[df_out['status'] == 'DROP']['feature'].tolist()
        if drops:
            print(f"\n建议从 FEATURE_COLS 删除 ({len(drops)}):")
            for f in drops: print(f"  '{f}',")
        return df_out

    def rolling_window_analysis(self, factor_dict, code_list=None, n_samples=30, window_years=1):
        """滚动窗口分析：长期/近期/牛熊/震荡下的因子 IC"""
        if code_list is None:
            import os
            code_list = [f.replace('.parquet', '') for f in os.listdir(CACHE_DIR)
                        if f.endswith('.parquet')][:n_samples]
        logger = __import__('utils.logger', fromlist=['setup_logger']).setup_logger(__name__)
        results = {}
        today = pd.Timestamp.now()
        for name, func in factor_dict.items():
            regimes = {'full': [], 'recent': [], 'bear': [], 'bull': [], 'sideways': []}
            for code in code_list:
                df = load_from_cache(code)
                if df is None or len(df) < 500: continue
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                close = df['Close'].values
                fwd = (np.roll(close, -20) - close) / close; fwd[-20:] = np.nan
                try: factor_vals = pd.Series(func(df), index=df.index).values
                except: continue
                mask = ~(np.isnan(factor_vals) | np.isnan(fwd))
                if mask.sum() < 50: continue
                ic_full, _ = spearmanr(factor_vals[mask], fwd[mask])
                regimes['full'].append(ic_full)
                cutoff = today - pd.DateOffset(years=window_years)
                recent_mask = mask & (df.index >= cutoff)
                if recent_mask.sum() > 30:
                    ic_r, _ = spearmanr(factor_vals[recent_mask], fwd[recent_mask])
                    regimes['recent'].append(ic_r)
                da = pd.DataFrame({'f': factor_vals, 'w': fwd, 'r': df['Close'].pct_change()}, index=df.index).dropna()
                if len(da) < 200: continue
                for yr, g in da.groupby(da.index.year):
                    if len(g) < 30: continue
                    ann = g['r'].mean()*252; ic_y, _ = spearmanr(g['f'], g['w'])
                    if ann > 0.2: regimes['bull'].append(ic_y)
                    elif ann < -0.1: regimes['bear'].append(ic_y)
                    else: regimes['sideways'].append(ic_y)
            results[name] = {r: round(np.mean(ics), 4) if ics else np.nan for r, ics in regimes.items()}
        print("\n" + "=" * 80)
        print(f"{'因子':15s} {'全周期':>8s} {'近期':>8s} {'牛市':>8s} {'熊市':>8s} {'震荡':>8s}  {'状态'}")
        print("=" * 80)
        for name in sorted(results, key=lambda n: abs(results[n].get('full', 0)), reverse=True):
            r = results[name]; f, rn = abs(r.get('full', 0)), abs(r.get('recent', 0))
            s = '无效' if f < 0.01 else ('↑增强' if rn > f*1.2 else ('↓衰减' if rn < f*0.5 else '稳定'))
            print(f"{name:15s} {r.get('full',0):+8.4f} {r.get('recent',0):+8.4f} "
                  f"{r.get('bull',0):+8.4f} {r.get('bear',0):+8.4f} {r.get('sideways',0):+8.4f}  {s}")
        print("=" * 80)
        return results


def audit_features(n_stocks=100, forward=20):
    """独立函数: 审计所有 FEATURE_COLS"""
    lab = FactorLab()
    lab.audit_all_features(n_samples=n_stocks, forward_days=forward)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stocks", type=int, default=100, help="抽样股票数")
    p.add_argument("--forward", type=int, default=20, help="前瞻天数")
    args = p.parse_args()
    audit_features(args.stocks, args.forward)
