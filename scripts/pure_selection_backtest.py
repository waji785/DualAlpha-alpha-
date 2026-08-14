#!/usr/bin/env python
# scripts/pure_selection_backtest.py
"""
纯选股对照回测：选股器选股 → 月度等权持有 → 不做择时
用于定位「选股器本身有没有 alpha」，对比两阶段回测（选股+择时）

复用 batch_backtest_two_stage.py 的窗口配置和选股器缓存：
  - 若 selector_two_stage_wN.pkl 已存在（两阶段回测已训练），直接加载
  - 否则重新训练选股器
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm import tqdm

from config.settings import *
from core.data_loader import load_from_cache, get_stock_list, load_all_stock_data, get_industry_map
from core.stock_selector import TreeEnsemble, build_selector_dataset
from utils.common import set_seed
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 组合优化方法：'equal' 等权 / 'min_variance' 最小方差 / 'risk_parity' 风险平价
# 注意：min_variance/risk_parity 依赖历史协方差，极端市场失效，回测证实负优化，默认 equal
METHOD = 'equal'

# 规则化趋势仓位管理：中证500 MA50 < MA200 时，仓位降到 REGIME_DOWN_MULT
# 回测证实负优化（牛市收益砍半 > 熊市少亏），默认禁用
REGIME_ENABLED = False
REGIME_DOWN_MULT = 0.5

# 行业相对排名标签：True=训练标签改为"行业内排名"（与每行业选1只对齐）
# 回测证实：夏普持平(1.555→1.557)但复合收益砍40%(16→9.65倍)，负优化，默认False
INDUSTRY_RANK = False


def _min_variance_weights(cov):
    """最小方差组合权重"""
    n = cov.shape[0]
    try:
        inv = np.linalg.inv(cov)
        ones = np.ones(n)
        return inv @ ones / (ones @ inv @ ones)
    except np.linalg.LinAlgError:
        return np.ones(n) / n


def _risk_parity_weights(cov):
    """风险平价权重（等边际风险贡献）"""
    from core.portfolio import risk_parity_weights
    try:
        return risk_parity_weights(cov)
    except Exception:
        n = cov.shape[0]
        return np.ones(n) / n


# 窗口配置（与两阶段回测一致）
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
COMMISSION = 0.0003  # 单边佣金（买卖各计一次）

# 止盈止损：持有期间涨超 TAKE_PROFIT 止盈、跌超 STOP_LOSS 止损（0 = 禁用）
# 策略：只止损不止盈（截断亏损、让利润奔跑，牛股不设顶靠月末调仓自然止盈）
TAKE_PROFIT = 0.0
STOP_LOSS = 0.10

# 调仓频率：'MS'=月度 / '2W'=双周（持有14天）
# 回测证实：双周收益砍半(交易成本翻倍+打断趋势+信号20天未兑现)，负优化，默认月度
REBALANCE_FREQ = 'MS'

# 标签持有期（预测未来 N 天收益）：20=原默认 / 30=匹配月度调仓
# 回测证实：20天夏普1.654 vs 30天1.632几乎无差(策略对持有期稳健)，默认20(短期信号信噪比高)
FORWARD_DAYS = 20

# 时间衰减半衰期（天）：365=原默认 / 180=更关注近期样本(A股风格切换快)
# 回测证实：180天负优化(复合18.57→12.41倍，过度关注近期丢长期alpha)，默认365
HALF_LIFE = 365


def train_selector_on_window(all_dfs, train_end, output_path):
    """在 train_end 之前的数据上训练选股器（与两阶段回测一致）"""
    window_dfs = {}
    for code, df in all_dfs.items():
        train_df = df[df['Date'] <= pd.to_datetime(train_end)]
        if len(train_df) >= 120:
            window_dfs[code] = train_df

    if len(window_dfs) < 50:
        logger.warning(f"窗口 {train_end}: 有效股票不足 50 只")
        return None

    X, y, sw, _ = build_selector_dataset(window_dfs, list(window_dfs.keys()),
                                          forward_days=FORWARD_DAYS, step_months=3,
                                          industry_rank=INDUSTRY_RANK,
                                          half_life=HALF_LIFE)
    if len(X) < 1000:
        logger.warning(f"窗口 {train_end}: 样本不足 1000")
        return None

    ensemble = TreeEnsemble()
    ensemble.fit(X, y, sample_weight=sw, groups=None)
    ensemble.save(output_path)
    logger.info(f"窗口 {train_end}: 选股模型保存 ({len(X)} 样本)")
    return ensemble


def run_pure_selection(selector, test_start, test_end, all_dfs):
    """纯选股：月度选 top N，等权持有到下月，再平衡（规则化趋势仓位管理）"""
    ind_map = get_industry_map()
    # 加载中证500 指数数据（趋势仓位判断用）
    index_df = pd.read_parquet(os.path.join(CACHE_DIR, "000905.parquet"))
    index_df['Date'] = pd.to_datetime(index_df['Date'])
    index_df = index_df.sort_values('Date')
    months = pd.date_range(pd.to_datetime(test_start), pd.to_datetime(test_end), freq=REBALANCE_FREQ)
    all_codes = list(all_dfs.keys())
    # 持有天数：双周 14 天，月度约 31 天
    hold_days = 14 if REBALANCE_FREQ == '2W' else 31

    capital = INITIAL_CAPITAL
    history = [{'date': pd.to_datetime(test_start), 'capital': capital}]
    select_count = 0

    for month_start in months:
        month_end = month_start + pd.DateOffset(days=hold_days) - pd.DateOffset(days=1)
        if month_end > pd.to_datetime(test_end):
            month_end = pd.to_datetime(test_end)

        # 规则化趋势仓位：中证500 MA50 vs MA200（趋势向上满仓，向下降仓）
        pre_index = index_df[index_df['Date'] < month_start]
        regime_mult = 1.0
        if REGIME_ENABLED and len(pre_index) >= 200:
            ma50 = float(pre_index['Close'].rolling(50).mean().iloc[-1])
            ma200 = float(pre_index['Close'].rolling(200).mean().iloc[-1])
            regime_mult = 1.0 if ma50 > ma200 else REGIME_DOWN_MULT

        # ① 选股（month_start 之前的数据打分）
        all_feats, valid_codes = [], []
        for code in all_codes:
            df = all_dfs.get(code)
            if df is None: continue
            pre_df = df[df['Date'] < month_start]
            if len(pre_df) < SEQ_LEN: continue
            feat = pre_df.iloc[-1][FEATURE_COLS].values.astype(np.float32)
            if np.any(np.isnan(feat)): continue
            all_feats.append(feat)
            valid_codes.append(code)

        if len(all_feats) < TOP_N:
            continue

        pred = selector.predict(np.array(all_feats))
        score = dict(zip(valid_codes, pred))
        downside = selector.predict_downside(np.array(all_feats))
        risk = dict(zip(valid_codes, downside))

        # P10 风控：原始收益标签排除下行风险(P10<-0.15)，行业排名标签排除行业内弱(P10<0.2)
        DOWNSIDE_THRESHOLD = 0.2 if INDUSTRY_RANK else -0.15
        safe_codes = [c for c in valid_codes if risk[c] > DOWNSIDE_THRESHOLD]
        if len(safe_codes) < TOP_N:
            safe_codes = valid_codes  # 兜底：排除过多时用全部

        # 行业分组
        ind_groups = {}
        for code in safe_codes:
            ind = ind_map.get(code, '未知')
            ind_groups.setdefault(ind, []).append(code)

        # 行业中性化：每个行业选最高分 1 只，跨行业等权分散（避免押注单一行业）
        ind_best = []
        for ind, codes in ind_groups.items():
            best = max(codes, key=lambda c: score[c])
            ind_best.append(best)

        # 按分数选 top N 个行业（每行业 1 只，跨行业分散）
        picks = sorted(ind_best, key=lambda c: score[c], reverse=True)[:TOP_N]
        select_count += len(picks)

        if not picks:
            continue

        # ② 组合优化持有：先算权重（等权/最小方差/风险平价），再加权持有
        weights = {}
        if METHOD in ('min_variance', 'risk_parity'):
            ret_by_code = {}
            for code in picks:
                df = all_dfs[code]
                pre_df = df[df['Date'] < month_start]
                if len(pre_df) >= 40:
                    ret_by_code[code] = pre_df['Close'].pct_change().dropna().tail(40).values
            if len(ret_by_code) >= 3:
                codes = list(ret_by_code.keys())
                min_len = min(len(v) for v in ret_by_code.values())
                mat = np.array([ret_by_code[c][-min_len:] for c in codes])
                cov = np.cov(mat)
                w = _min_variance_weights(cov) if METHOD == 'min_variance' else _risk_parity_weights(cov)
                weights = dict(zip(codes, w))

        month_returns = []  # (code, ret)
        for code in picks:
            df = all_dfs[code]
            buy_row = df[df['Date'] >= month_start].head(1)
            sell_row = df[df['Date'] >= (month_end + pd.Timedelta(days=1))].head(1)
            if len(sell_row) == 0:
                sell_row = df[df['Date'] <= month_end].tail(1)
            if len(buy_row) == 0 or len(sell_row) == 0:
                continue
            buy_p = float(buy_row['Close'].iloc[0])
            sell_p = float(sell_row['Close'].iloc[0])
            if buy_p <= 0 or sell_p <= 0:
                continue
            # 停牌股剔除：卖出日期距买入日期 > 40 天，说明持有期间停牌（复牌暴涨是幸存者偏差）
            buy_date = pd.to_datetime(buy_row['Date'].iloc[0])
            sell_date = pd.to_datetime(sell_row['Date'].iloc[0])
            if (sell_date - buy_date).days > 40:
                continue
            # 止盈止损：持有期间逐日检查，触发即卖出（截断收益）
            if TAKE_PROFIT > 0 or STOP_LOSS > 0:
                hold = df[(df['Date'] >= buy_date) & (df['Date'] <= sell_date)]
                for _, r in hold.iterrows():
                    px = float(r['Close'])
                    if TAKE_PROFIT > 0 and px >= buy_p * (1 + TAKE_PROFIT):
                        sell_p = buy_p * (1 + TAKE_PROFIT)
                        break
                    if STOP_LOSS > 0 and px <= buy_p * (1 - STOP_LOSS):
                        sell_p = buy_p * (1 - STOP_LOSS)
                        break
            ret = sell_p / buy_p - 1
            # 异常暴涨剔除：单月收益 > +100% 视为重组/事件驱动，非选股 alpha
            if ret > 1.0 or ret < -0.6:
                continue
            month_returns.append((code, ret))

        if month_returns:
            codes = [c for c, _ in month_returns]
            rets = [r for _, r in month_returns]
            w = np.array([weights.get(c, 1.0 / len(codes)) for c in codes])
            w = w / w.sum()
            # 加权收益 × 趋势仓位系数 - 交易成本（再平衡：卖旧买新，双边佣金）
            ret = (float(np.dot(w, rets)) - 2 * COMMISSION) * regime_mult
            capital *= (1 + ret)

        history.append({'date': month_end, 'capital': capital})

    return pd.DataFrame(history), select_count


def main():
    set_seed(SEED)
    logger.info("=" * 60)
    logger.info("纯选股对照回测（选股器 → 月度等权持有，不做择时）")
    logger.info("=" * 60)

    stock_df = get_stock_list(exclude_st=True, exclude_north=True)
    df_all = load_all_stock_data(max_stocks=6000, min_days=200)
    if df_all is None:
        return

    all_dfs = {}
    for code, group in df_all.groupby('stock_code'):
        group = group.sort_values('Date').reset_index(drop=True)
        group['Date'] = pd.to_datetime(group['Date'])
        all_dfs[code] = group

    all_results = []

    for i, (train_end, test_start, test_end) in enumerate(WINDOWS):
        tag = f"w{i+1}"
        logger.info(f"\n{'='*40}\n窗口 {i+1}: train≤{train_end}, test={test_start}~{test_end}\n{'='*40}")

        # 复用两阶段回测已训练的选股器，否则训练
        sel_path = f"selector_two_stage_w{i+1}.pkl"
        if os.path.exists(sel_path):
            selector = TreeEnsemble()
            selector.load(sel_path)
            logger.info(f"加载已有选股器 {sel_path}")
        else:
            selector = train_selector_on_window(all_dfs, train_end, sel_path)
            if selector is None:
                continue

        cap_df, selects = run_pure_selection(selector, test_start, test_end, all_dfs)
        if cap_df is None or len(cap_df) < 3:
            continue

        total_ret = cap_df['capital'].iloc[-1] / INITIAL_CAPITAL - 1
        m_ret = cap_df['capital'].pct_change().dropna()
        # 年化/夏普因子：月度 12 周期，双周 26 周期
        ppy = 26 if REBALANCE_FREQ == '2W' else 12
        ann = (1 + total_ret) ** (ppy / len(m_ret)) - 1 if len(m_ret) > 0 else 0
        dd = (cap_df['capital'] / cap_df['capital'].cummax() - 1).min()
        sharpe = m_ret.mean() / m_ret.std() * np.sqrt(ppy) if m_ret.std() > 0 else 0

        logger.info(f"窗口 {i+1}: 总收益={total_ret*100:+.2f}% 年化={ann*100:.2f}% "
                    f"回撤={dd*100:.2f}% 夏普={sharpe:.3f} 选股={selects}")

        all_results.append({
            'window': i + 1, 'train_end': train_end, 'test': f"{test_start}~{test_end}",
            'total_return': total_ret, 'annual_return': ann,
            'max_drawdown': dd, 'sharpe': sharpe, 'selects': selects,
        })

    if all_results:
        df_sum = pd.DataFrame(all_results)
        print("\n" + "=" * 60)
        print("纯选股对照 汇总")
        print(df_sum[['window', 'test', 'total_return', 'annual_return',
                      'max_drawdown', 'sharpe']].to_string(
                          formatters={'total_return': '{:+.2%}'.format,
                                      'annual_return': '{:.2%}'.format,
                                      'max_drawdown': '{:.2%}'.format,
                                      'sharpe': '{:.3f}'.format}))
        df_sum.to_csv("pure_selection_results.csv", index=False)
        print("\n结果已保存: pure_selection_results.csv")
    else:
        logger.error("无有效窗口结果")


if __name__ == "__main__":
    main()
