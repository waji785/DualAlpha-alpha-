# scripts/batch_backtest_two_stage.py
"""扩展窗口两阶段回测：每月训练 → 树模型截面数据也按窗口切分"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch, joblib
from datetime import timedelta
from tqdm import tqdm

from config.settings import *
from core.model import DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer, Fusion_iTransformer
from core.data_loader import load_from_cache, get_stock_list, load_all_stock_data
from core.trainer import train_and_save_model
from core.stock_selector import TreeEnsemble, build_selector_dataset
from core.metrics import compute_metrics
from core.backtest_engine import run_backtest
from utils.common import set_seed
from utils.logger import setup_logger

torch.serialization.add_safe_globals([DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer, Fusion_iTransformer])
logger = setup_logger(__name__)

# 窗口配置
WINDOWS = [
    ("2019-01-01", "2019-01-02", "2020-01-01"),
    ("2020-01-01", "2020-01-02", "2021-01-01"),
    ("2021-01-01", "2021-01-02", "2022-01-01"),
    ("2022-01-01", "2022-01-02", "2023-01-01"),
    ("2023-01-01", "2023-01-02", "2024-01-01"),
    ("2024-01-01", "2024-01-02", "2025-01-01"),
    ("2025-01-01", "2025-01-02", "2026-01-01"),
    ("2026-01-01", "2026-01-02", "2026-08-06"),
]

TOP_N = 20
INITIAL_CAPITAL = 100000


def train_selector_on_window(all_dfs, train_end, output_path):
    """在训练截止日期之前的数据上训练选股模型"""
    # 只用 train_end 之前的数据
    window_dfs = {}
    for code, df in all_dfs.items():
        train_df = df[df['Date'] <= pd.to_datetime(train_end)]
        if len(train_df) >= 120:
            window_dfs[code] = train_df

    if len(window_dfs) < 50:
        logger.warning(f"窗口 {train_end}: 有效股票不足 50 只")
        return None

    X, y, sw, _ = build_selector_dataset(window_dfs, list(window_dfs.keys()),
                                          forward_days=20, step_months=3)
    if len(X) < 1000:
        logger.warning(f"窗口 {train_end}: 样本不足 1000")
        return None

    ensemble = TreeEnsemble()
    ensemble.fit(X, y, sample_weight=sw, groups=None)
    ensemble.save(output_path)
    logger.info(f"窗口 {train_end}: 选股模型保存 ({len(X)} 样本)")
    return ensemble


def run_two_stage_window(selector, timing_model, scaler_X, scaler_y,
                         test_start, test_end, all_dfs, stock_df):
    """单窗口两阶段回测"""
    name_map = dict(zip(stock_df['code'], stock_df['name']))
    all_codes = list(all_dfs.keys())
    sd = pd.to_datetime(test_start)
    ed = pd.to_datetime(test_end)
    months = pd.date_range(sd, ed, freq='MS')

    cash = INITIAL_CAPITAL
    portfolio = {}
    capital_history = [{'date': sd, 'capital': INITIAL_CAPITAL}]
    trade_count = 0; select_count = 0  # 统计

    for month_start in months:
        month_end = (month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1))
        if month_end > ed:
            month_end = ed

        # ① 选股（月度日志）
        all_feats, valid_codes = [], []
        n_no, n_short, n_nan = 0, 0, 0
        for code in all_codes:
            df = all_dfs.get(code)
            if df is None: n_no += 1; continue
            pre_df = df[df['Date'] < month_start]
            if len(pre_df) < SEQ_LEN: n_short += 1; continue
            feat = pre_df.iloc[-1][FEATURE_COLS].values.astype(np.float32)
            if np.any(np.isnan(feat)): n_nan += 1; continue
            all_feats.append(feat)
            valid_codes.append(code)

        if len(all_feats) < TOP_N:
            if mi % 3 == 0:
                logger.info(f"  {month_start.date()}: 候选{len(all_feats)} "
                            f"(NaN={n_nan}无缓存={n_no}短史={n_short})")
            continue

        pred_returns = selector.predict(np.array(all_feats))
        picks = [valid_codes[i] for i in np.argsort(pred_returns)[::-1][:TOP_N]]
        # 行业集中度限制：每行业最多3只
        from core.data_loader import get_industry_map
        ind_map = get_industry_map()
        filtered, ind_count = [], {}
        for code in picks:
            ind = ind_map.get(code, '未知')
            if ind_count.get(ind, 0) < 3:
                filtered.append(code)
                ind_count[ind] = ind_count.get(ind, 0) + 1
        picks = filtered
        select_count += len(picks)

        # ② 清仓
        for code in list(portfolio):
            if code not in picks:
                df = all_dfs[code]
                row = df[df['Date'] >= month_start].head(1)
                if len(row) > 0 and portfolio[code] > 0:
                    cash += portfolio[code] * float(row['Close'].iloc[0])
                del portfolio[code]

        # ③ 择时
        per_stock = cash / max(len(picks), 1)
        for code in picks:
            if code in portfolio:
                continue
            df = all_dfs[code]
            month_df = df[(df['Date'] >= month_start) & (df['Date'] <= month_end)]
            if len(month_df) < SEQ_LEN + 10:
                month_df = df[df['Date'] <= month_end].tail(SEQ_LEN + 50)
            if len(month_df) < SEQ_LEN + 10:
                continue

            bt_df = run_backtest(df, timing_model, scaler_X, scaler_y,
                                  initial_capital=per_stock)
            if bt_df is None:
                continue
            last = bt_df.iloc[-1]
            if last['Position'] > 0:
                shares = int(per_stock * last['Position'] / last['Close'])
                if shares > 0:
                    cost = shares * last['Close']
                    if cost <= cash:
                        cash -= cost
                        portfolio[code] = shares
                        trade_count += 1

        # 月末资产
        total = cash
        for code, shares in portfolio.items():
            df = all_dfs[code]
            row = df[df['Date'] <= month_end].tail(1)
            if len(row) > 0 and shares > 0:
                total += shares * float(row['Close'].iloc[0])
        capital_history.append({'date': month_end, 'capital': total})

    return pd.DataFrame(capital_history), trade_count, select_count


def main():
    set_seed(SEED)
    logger.info("=" * 60)
    logger.info("扩展窗口两阶段回测 (树模型选股 + 深度学习择时)")
    logger.info("=" * 60)

    stock_df = get_stock_list(exclude_st=True, exclude_north=True)
    df_all = load_all_stock_data(max_stocks=6000, min_days=200)
    if df_all is None:
        return

    # 构建 code → df 字典
    all_dfs = {}
    for code, group in df_all.groupby('stock_code'):
        group = group.sort_values('Date').reset_index(drop=True)
        group['Date'] = pd.to_datetime(group['Date'])
        all_dfs[code] = group

    all_results = []

    for i, (train_end, test_start, test_end) in enumerate(WINDOWS):
        tag = f"w{i+1}"
        logger.info(f"\n{'='*40}\n窗口 {i+1}: train≤{train_end}, test={test_start}~{test_end}\n{'='*40}")

        # 训练择时模型
        model_path = f"model_two_stage_w{i+1}.pth"
        sx_path = f"scaler_X_two_stage_w{i+1}.pkl"
        sy_path = f"scaler_Y_two_stage_w{i+1}.pkl"

        if os.path.exists(model_path):
            logger.info(f"跳过训练（已有 {model_path})")
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            sX = joblib.load(sx_path)
            sY = joblib.load(sy_path) if os.path.exists(sy_path) else None
        else:
            res = train_and_save_model(df_all.copy(), train_end_date=train_end,
                                       model_save_path=model_path,
                                       scaler_x_path=sx_path, scaler_y_path=sy_path)
            if res is None:
                continue
            model, sX, sY, _ = res

        # 训练选股模型
        sel_path = f"selector_two_stage_w{i+1}.pkl"
        if os.path.exists(sel_path):
            selector = TreeEnsemble()
            selector.load(sel_path)
        else:
            selector = train_selector_on_window(all_dfs, train_end, sel_path)
            if selector is None:
                continue

        # 回测（已有结果则跳过）
        result_file = f"two_stage_w{i+1}_result.csv"
        if os.path.exists(result_file):
            logger.info(f"跳过回测（已有 {result_file})")
            continue

        cap_df, trades, selects = run_two_stage_window(selector, model, sX, sY,
                                       test_start, test_end, all_dfs, stock_df)
        if cap_df is None or len(cap_df) < 2:
            continue

        total_ret = cap_df['capital'].iloc[-1] / INITIAL_CAPITAL - 1
        m_ret = cap_df['capital'].pct_change().dropna()
        ann = (1+total_ret)**(12/len(m_ret)) - 1 if len(m_ret) > 0 else 0
        dd = (cap_df['capital']/cap_df['capital'].cummax()-1).min()
        sharpe = m_ret.mean()/m_ret.std()*np.sqrt(12) if m_ret.std() > 0 else 0

        logger.info(f"窗口 {i+1}: 总收益={total_ret*100:+.2f}% 年化={ann*100:.2f}% "
                    f"回撤={dd*100:.2f}% 夏普={sharpe:.3f} "
                    f"选股={selects} 交易={trades}")

        # 分析层
        try:
            from core.backtest_analyzer import analyze_backtest
            bm_df = load_from_cache('000300')
            analyze_backtest(cap_df, None, bm_df, title=f"窗口 {i+1} ({test_start[:7]}~{test_end[:7]})")
        except Exception as e:
            logger.warning(f"分析层跳过: {e}")

        all_results.append({
            'window': i+1, 'train_end': train_end, 'test': f"{test_start}~{test_end}",
            'total_return': total_ret, 'annual_return': ann,
            'max_drawdown': dd, 'sharpe': sharpe,
            'selects': selects, 'trades': trades, 'months': len(m_ret)
        })
        # 保存单窗口结果标记
        cap_df.to_csv(result_file, index=False)

    # 汇总
    if all_results:
        df_sum = pd.DataFrame(all_results)
        print("\n" + "=" * 60)
        print("📊 全部窗口汇总")
        print(df_sum[['window', 'train_end', 'total_return', 'annual_return',
                       'max_drawdown', 'sharpe']].to_string(
                           formatters={'total_return': '{:+.2%}'.format,
                                       'annual_return': '{:.2%}'.format,
                                       'max_drawdown': '{:.2%}'.format,
                                       'sharpe': '{:.3f}'.format}))
        df_sum.to_csv("two_stage_window_results.csv", index=False)
    else:
        logger.error("无有效窗口结果")


if __name__ == "__main__":
    main()
