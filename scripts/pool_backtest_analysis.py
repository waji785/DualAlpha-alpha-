# scripts/pool_backtest_analysis.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from config.settings import *
from core.model import DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer, Fusion_iTransformer
from core.data_loader import load_from_cache
from core.backtest_engine import run_backtest
from core.metrics import compute_metrics
from utils.common import set_seed
from utils.logger import setup_logger

torch.serialization.add_safe_globals([DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer, Fusion_iTransformer])
logger = setup_logger(__name__)


def load_unified_model(model_path="model_final.pth",
                       scaler_x_path="scaler_X_final.pkl",
                       scaler_y_path="scaler_Y_final.pkl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    if hasattr(model, 'lstm') and hasattr(model.lstm, 'flatten_parameters'):
        model.lstm.flatten_parameters()
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return model, scaler_X, scaler_y


# ============================================================
#  指数数据 & 信息比率
# ============================================================

INDEX_MAP = {'沪深300': 'sh.000300', '中证500': 'sh.000905'}


def fetch_index(bs_symbol, start_date, end_date):
    import baostock as bs
    bs.login()
    rs = bs.query_history_k_data_plus(
        code=bs_symbol, fields="date,close",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2"
    )
    rows = []
    while (rs.error_code == '0') & rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['Date', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df.dropna().sort_values('Date').reset_index(drop=True)


def compute_ir(strat_curve, strat_dates, bench_curve, bench_dates):
    """信息比率：年化超额收益 / 年化跟踪误差"""
    all_d = pd.to_datetime(sorted(set(strat_dates) & set(bench_dates)))
    if len(all_d) < 60:
        return 0
    s = pd.Series(np.diff(strat_curve) / strat_curve[:-1],
                  index=strat_dates[1:]).reindex(all_d).dropna()
    b = pd.Series(np.diff(bench_curve) / bench_curve[:-1],
                  index=bench_dates[1:]).reindex(all_d).dropna()
    common = s.index.intersection(b.index)
    if len(common) < 60:
        return 0
    active = s[common].values - b[common].values
    return np.sqrt(252) * np.mean(active) / np.std(active) if np.std(active) > 1e-8 else 0


# ============================================================
#  动态权重计算
# ============================================================

def compute_dynamic_weights(price_matrix, method='inv_vol', lookback=60, rebalance=20):
    """
    price_matrix: (n_days, n_stocks)
    返回 (n_days, n_stocks) 权重
    """
    n_days, n_stocks = price_matrix.shape
    weights = np.full((n_days, n_stocks), 1.0 / n_stocks)

    for t in range(lookback, n_days, rebalance):
        window = price_matrix[max(0, t - lookback):t]
        if len(window) < 10:
            continue
        rets = np.diff(window, axis=0) / (window[:-1] + 1e-8)
        vol = np.std(rets, axis=0) + 1e-8

        if method == 'inv_vol':
            w = 1.0 / vol
        elif method == 'risk_parity':
            w = 1.0 / (vol ** 2)
        else:
            w = np.ones(n_stocks)

        w = np.nan_to_num(w, nan=0.0, posinf=0.0)
        w = w / w.sum() if w.sum() > 0 else np.ones(n_stocks) / n_stocks
        end = min(t + rebalance, n_days)
        weights[t:end] = w

    return weights


def build_dynamic_portfolio(capital_curves, close_curves, date_arrays,
                            method, initial_capital=100000, rebalance=21):
    all_dates = pd.to_datetime(sorted(set().union(*[set(d) for d in date_arrays])))
    n_days, n_stocks = len(all_dates), len(capital_curves)

    price_mat = np.full((n_days, n_stocks), np.nan)
    cap_mat = np.full((n_days, n_stocks), np.nan)
    for i in range(n_stocks):
        pi = pd.Series(close_curves[i], index=date_arrays[i]).reindex(all_dates, method='ffill')
        price_mat[:, i] = pi.values
        ci = pd.Series(capital_curves[i], index=date_arrays[i])
        first = all_dates.get_loc(date_arrays[i][0])
        ci = ci.reindex(all_dates)
        ci[:first] = initial_capital
        ci = ci.ffill()
        cap_mat[:, i] = ci.values

    weights = compute_dynamic_weights(price_mat, method, rebalance=rebalance)
    cap_norm = cap_mat / cap_mat[0]
    portfolio = (cap_norm * weights).sum(axis=1) * initial_capital
    return all_dates, portfolio


# ============================================================
#  组合回测
# ============================================================

def run_pool_backtest(whitelist_file=WHITELIST_EXTENDED_FILE, max_stocks=30,
                      model_path="model_final.pth",
                      scaler_x="scaler_X_final.pkl",
                      scaler_y="scaler_Y_final.pkl",
                      start_date=BACKTEST_START_DATE):
    if not os.path.exists(whitelist_file):
        logger.error(f"白名单文件 {whitelist_file} 不存在")
        return None

    model, scaler_X, scaler_y = load_unified_model(model_path, scaler_x, scaler_y)
    df_white = pd.read_csv(whitelist_file)
    if max_stocks and len(df_white) > max_stocks:
        df_white = df_white.head(max_stocks)

    initial_capital = 100000
    capital_curves, close_curves, date_arrays = [], [], []
    bench_curves = []
    codes, names, returns = [], [], []

    for idx, row in df_white.iterrows():
        code = str(row['code']).zfill(6)
        name = row.get('name', '')
        logger.info(f"回测 {code} {name} ({idx+1}/{len(df_white)})")
        df = load_from_cache(code)
        if df is None or len(df) < SEQ_LEN + 20:
            continue
        df['Date'] = pd.to_datetime(df['Date'])

        backtest_df = run_backtest(df, model, scaler_X, scaler_y,
                                   start_date=start_date)
        if backtest_df is None:
            continue

        # 前补：start_date ~ 首笔预测之间的空仓期（按日填充）
        first_date = backtest_df['Date'].iloc[0]
        sd = pd.to_datetime(start_date)
        if first_date > sd:
            filler_dates = pd.date_range(sd, first_date - pd.Timedelta(days=1), freq='B')
            filler = pd.DataFrame({
                'Date': filler_dates,
                'Close': backtest_df['Close'].iloc[0],
                'Position': 0,
                'Capital': float(initial_capital)
            })
            backtest_df = pd.concat([filler, backtest_df], ignore_index=True)

        capital_curves.append(backtest_df['Capital'].values)
        close_curves.append(backtest_df['Close'].values)
        date_arrays.append(backtest_df['Date'].values)
        bh = backtest_df['Close'].values / backtest_df['Close'].values[0] * initial_capital
        bench_curves.append(bh)
        ret = backtest_df['Capital'].iloc[-1] / backtest_df['Capital'].iloc[0] - 1
        returns.append(ret)
        codes.append(code)
        names.append(name)

    if not capital_curves:
        logger.error("无有效股票")
        return None

    # 策略组合：方法 × 再平衡频率
    strategies = [
        ('equal',        '等权重',          0),
        ('inv_vol',      '波动率倒数(月)',  21),
        ('inv_vol',      '波动率倒数(季)',  63),
        ('risk_parity',  '风险平价(月)',    21),
        ('risk_parity',  '风险平价(季)',    63),
    ]
    results = {}

    for method, label, rebalance in strategies:
        if method == 'equal':
            all_dates = pd.to_datetime(sorted(set().union(*[set(d) for d in date_arrays])))
            aligned = []
            for curve, d in zip(capital_curves, date_arrays):
                s = pd.Series(curve, index=d).reindex(all_dates, method='ffill')
                aligned.append(s.values)
            curve = np.mean(aligned, axis=0)
        else:
            all_dates, curve = build_dynamic_portfolio(
                capital_curves, close_curves, date_arrays, method, rebalance=rebalance)
        results[label] = (all_dates, curve, compute_metrics(curve))

    # 拉取指数
    start_str = all_dates[0].strftime('%Y-%m-%d')
    end_str = all_dates[-1].strftime('%Y-%m-%d')
    indices = {}
    for name, bs_code in INDEX_MAP.items():
        idx_df = fetch_index(bs_code, start_str, end_str)
        if idx_df is not None:
            idx_df = idx_df[(idx_df['Date'] >= all_dates[0]) & (idx_df['Date'] <= all_dates[-1])]
            indices[name] = idx_df

    # 基准：用沪深300（指数数据已在前面拉取）
    benchmark_name = '沪深300' if '沪深300' in indices else list(indices.keys())[0] if indices else None
    if benchmark_name:
        idx_df = indices[benchmark_name]
        bh_curve = idx_df['Close'].values / idx_df['Close'].values[0] * initial_capital
        bh_dates = idx_df['Date'].values
        bh_metrics = compute_metrics(bh_curve)
    else:
        bh_metrics = {'total_return': 0}

    # 打印
    print("\n" + "=" * 70)
    print("📊 组合回测绩效")
    print("=" * 70)
    print(f"  股票数量: {len(codes)}")
    print(f"  回测区间: {all_dates[0].date()} ~ {all_dates[-1].date()}")
    if benchmark_name:
        print(f"  基准: {benchmark_name} {bh_metrics['total_return']*100:+.2f}%")
    print("-" * 70)
    for label, (dates, curve, m) in results.items():
        ex = m['total_return'] - bh_metrics['total_return']
        print(f"  [{label}]")
        print(f"    总收益:   {m['total_return']*100:+.2f}%  (超额 {ex:+.2%})")
        print(f"    年化:     {m['annual_return']*100:+.2f}%  回撤: {m['max_drawdown']*100:.2f}%")
        print(f"    夏普:     {m['sharpe_ratio']:.3f}  "
              f"Calmar: {m['calmar_ratio']:.3f}  "
              f"Sortino: {m['sortino_ratio']:.3f}")
        print(f"    VaR 95%:  {m['var_95']*100:.2f}%  胜率: {m['win_rate']*100:.1f}%")
        for idx_name, idx_df in indices.items():
            ir = compute_ir(curve, dates, idx_df['Close'].values, idx_df['Date'].values)
            ex_idx = m['total_return'] - (idx_df['Close'].iloc[-1] / idx_df['Close'].iloc[0] - 1)
            print(f"    vs {idx_name}: 超额 {ex_idx:+.2%}  IR={ir:.3f}")
    print("=" * 70)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    base_colors = {'等权重': '#2166ac', 'inv_vol': '#4dac26', 'risk_parity': '#d6604d'}
    style_map = {'月': '-', '季': '--'}

    for label, (dates, curve, _) in results.items():
        method_key = '等权重' if '等权' in label else ('inv_vol' if '波动率' in label else 'risk_parity')
        color = base_colors[method_key]
        ls = style_map.get(label[-3:-1], '-') if method_key != '等权重' else '-'
        ax1.plot(dates, curve / curve[0] * initial_capital,
                 label=label, color=color, linestyle=ls, linewidth=1.5)
    if benchmark_name and benchmark_name in indices:
        idx_b = indices[benchmark_name]
        ax1.plot(idx_b['Date'], idx_b['Close'].values / idx_b['Close'].values[0] * initial_capital,
                 label=f'{benchmark_name}基准', color='gray', linewidth=1, alpha=0.6)
    # 叠加指数
    idx_colors = {'沪深300': '#e41a1c', '中证500': '#377eb8'}
    for idx_name, idx_df in indices.items():
        idx_norm = idx_df['Close'].values / idx_df['Close'].values[0] * initial_capital
        ax1.plot(idx_df['Date'], idx_norm, label=idx_name,
                 color=idx_colors.get(idx_name, 'black'), linewidth=1.2, alpha=0.7)
    ax1.set_title('组合资金曲线')
    ax1.set_ylabel('资金（元）')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    best_label = max(results, key=lambda k: results[k][2]['calmar_ratio'])
    best_dates, best_curve, _ = results[best_label]
    peak = np.maximum.accumulate(best_curve)
    dd = (peak - best_curve) / peak * 100
    ax2.fill_between(best_dates, 0, dd, color='red', alpha=0.3)
    ax2.plot(best_dates, dd, color='red', linewidth=0.5)
    ax2.set_title(f'回撤曲线 ({best_label})')
    ax2.set_ylabel('回撤 (%)')
    ax2.set_xlabel('日期')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pool_curve.png", dpi=150)
    plt.show()

    # 单股票明细
    result_df = pd.DataFrame({
        'code': codes, 'name': names, 'return': returns
    }).sort_values('return', ascending=False)
    result_df.to_csv("pool_returns.csv", index=False, encoding='utf-8-sig')

    print("\n📋 各股票收益:")
    for _, r in result_df.iterrows():
        bar = '█' * max(1, int(r['return'] * 50))
        print(f"  {r['code']} {r['name']:8s} {r['return']*100:+7.2f}%  {bar}")

    return results, result_df


def main(model_path="model_final.pth",
         scaler_x="scaler_X_final.pkl", scaler_y="scaler_Y_final.pkl",
         start_date=BACKTEST_START_DATE):
    run_pool_backtest(max_stocks=10, model_path=model_path,
                      scaler_x=scaler_x, scaler_y=scaler_y,
                      start_date=start_date)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model_final.pth")
    p.add_argument("--scaler-x", default="scaler_X_final.pkl")
    p.add_argument("--scaler-y", default="scaler_Y_final.pkl")
    p.add_argument("--start-date", default=BACKTEST_START_DATE, help="回测起始日期")
    args = p.parse_args()
    main(args.model, args.scaler_x, args.scaler_y, start_date=args.start_date)
