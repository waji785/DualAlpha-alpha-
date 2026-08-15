#!/usr/bin/env python
# scripts/monthly_live.py
"""月度调仓实盘模拟（复现 pure_selection_backtest.py 的月度+止损逻辑）

每天运行一次，脚本自动判断：
  - 月初（跨月）→ 调仓：卖出旧持仓，选股(P10风控+每行业1只+Top20)，等权买入
  - 持有期间 → 检查止损：跌超 10% 卖出
状态持久化到 output/monthly_state.json，交易记录写入 output/monthly_trade_log.csv
"""
import sys, os, json
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from config.settings import *
from core.data_loader import load_from_cache, get_stock_list, get_industry_map
from core.stock_selector import TreeEnsemble
from utils.logger import setup_logger

logger = setup_logger(__name__)

TOP_N = 20
STOP_LOSS = 0.10          # 止损：跌超 10% 卖出
DOWNSIDE_THRESHOLD = -0.15  # P10 风控阈值（与回测一致）
INITIAL_CAPITAL = 100000
STATE_FILE = os.path.join(OUTPUT_DIR, "monthly_state.json")
TRADE_LOG = os.path.join(OUTPUT_DIR, "monthly_trade_log.csv")


def select_picks(selector, codes, ind_map, today):
    """复现回测选股：P10风控 + 每行业1只 + Top20"""
    rows = []
    for code in codes:
        df = load_from_cache(code)
        if df is None or len(df) < SEQ_LEN + 60:
            continue
        df['Date'] = pd.to_datetime(df['Date'])
        pre = df[df['Date'] <= today]
        if len(pre) < SEQ_LEN:
            continue
        feat = pre.iloc[-1][FEATURE_COLS].values.astype(np.float32)
        if np.any(np.isnan(feat)):
            continue
        rows.append((code, feat))
    if not rows:
        return []
    codes_list = [r[0] for r in rows]
    X = np.array([r[1] for r in rows], dtype=np.float32)
    scores = selector.predict(X)
    downside = selector.predict_downside(X)
    score = dict(zip(codes_list, scores))
    risk = dict(zip(codes_list, downside))

    # P10 下行风控
    safe = [c for c in codes_list if risk[c] > DOWNSIDE_THRESHOLD]
    if len(safe) < TOP_N:
        safe = codes_list

    # 行业中性化：每行业 1 只
    ind_groups = {}
    for c in safe:
        ind_groups.setdefault(ind_map.get(c, '未知'), []).append(c)
    ind_best = [max(cs, key=lambda c: score[c]) for cs in ind_groups.values()]
    picks = sorted(ind_best, key=lambda c: score[c], reverse=True)[:TOP_N]
    return picks


def code_to_ts(code):
    """股票代码转 tushare ts_code"""
    s = str(code).zfill(6)
    return f"{s}.SH" if s[0] == '6' else f"{s}.SZ"


def get_realtime_price(code, trade_date):
    """用 tushare 实时查某股票某天收盘价（失败返回 None）"""
    try:
        import tushare as ts
        from config.settings import TUSHARE_TOKEN
        pro = ts.pro_api(TUSHARE_TOKEN, timeout=10)
        df = pro.daily(ts_code=code_to_ts(code), trade_date=trade_date.strftime('%Y%m%d'))
        if df is not None and len(df) > 0:
            return float(df['close'].iloc[0])
    except Exception:
        pass
    return None


def get_price(code, today):
    """获取某股票实际交易收盘价（tushare 实时，不复权；缓存后复权仅作降级）"""
    # 优先 tushare 实时（实际价格，不复权，与真实交易一致）
    px = get_realtime_price(code, today)
    if px is not None:
        return px
    # 降级：缓存（后复权价，与真实价格有复权偏差，仅 tushare 失败时用）
    df = load_from_cache(code)
    if df is None or len(df) == 0:
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    row = df[df['Date'] <= today]
    if len(row) == 0:
        return None
    return float(row['Close'].iloc[-1])


def log_trade(code, name, action, price, date, detail=""):
    """追加一条交易记录到 CSV"""
    row = pd.DataFrame([{
        'date': str(date.date()), 'code': code, 'name': name,
        'action': action, 'price': round(price, 3), 'detail': detail,
    }])
    if os.path.exists(TRADE_LOG):
        row.to_csv(TRADE_LOG, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        row.to_csv(TRADE_LOG, index=False, encoding='utf-8-sig')


def main():
    today = pd.Timestamp(TODAY)
    logger.info(f"===== 月度实盘 {today.date()} =====")

    # 1. 加载选股器
    selector_path = os.path.join(OUTPUT_DIR, "selector_ensemble.pkl")
    if not os.path.exists(selector_path):
        logger.error(f"选股器不存在: {selector_path}，请先运行 train_selector.py")
        return
    selector = TreeEnsemble()
    selector.load(selector_path)

    # 2. 读状态
    state = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding='utf-8') as f:
            state = json.load(f)
    positions = state.get('positions', {})   # {code: {buy_price, buy_date, name}}
    last_rebalance = state.get('last_rebalance', '')

    # 3. 股票列表 + 行业映射
    stock_df = get_stock_list(exclude_st=True, exclude_north=True)
    if stock_df is None or len(stock_df) == 0:
        logger.error("无法获取股票列表")
        return
    codes = stock_df['code'].tolist()
    name_map = dict(zip(stock_df['code'], stock_df['name']))
    ind_map = get_industry_map()

    # 4. 判断是否调仓日（跨月，即进入新月份）
    is_rebalance = True
    if last_rebalance:
        last_m = pd.Timestamp(last_rebalance).to_period('M')
        if today.to_period('M') == last_m:
            is_rebalance = False

    # 5. 调仓 / 止损
    if is_rebalance:
        logger.info(">>> 调仓日：卖出旧持仓，买入新 Top20")
        for code, pos in list(positions.items()):
            px = get_price(code, today)
            if px is not None:
                ret = (px - pos['buy_price']) / pos['buy_price']
                log_trade(code, pos.get('name', ''), '卖出(调仓)', px, today,
                          detail=f"持有收益 {ret*100:+.2f}%")
            else:
                log_trade(code, pos.get('name', ''), '卖出(调仓)', pos['buy_price'], today, detail="无价格")
        picks = select_picks(selector, codes, ind_map, today)
        if not picks:
            logger.error("选股失败，本次跳过调仓")
            return
        new_positions = {}
        for code in picks:
            px = get_price(code, today)
            if px is None:
                logger.warning(f"{code} 无价格，跳过")
                continue
            new_positions[code] = {'buy_price': px, 'buy_date': str(today.date()),
                                   'name': name_map.get(code, '')}
            log_trade(code, name_map.get(code, ''), '买入', px, today)
        positions = new_positions
        last_rebalance = str(today.date())
        logger.info(f"新持仓 {len(positions)} 只: {list(positions.keys())}")
    else:
        # 持有期：检查止损
        for code, pos in list(positions.items()):
            px = get_price(code, today)
            if px is None:
                continue
            ret = (px - pos['buy_price']) / pos['buy_price']
            if ret <= -STOP_LOSS:
                log_trade(code, pos.get('name', ''), '止损卖出', px, today,
                          detail=f"亏损 {ret*100:+.2f}%")
                del positions[code]
                logger.info(f"止损: {code} {pos.get('name','')} 亏损 {ret*100:.2f}%")

    # 6. 计算当前持仓平均收益 + 写状态
    total_ret = 0.0
    if positions:
        rets = []
        for code, pos in positions.items():
            px = get_price(code, today)
            if px is not None:
                rets.append((px - pos['buy_price']) / pos['buy_price'])
        if rets:
            total_ret = float(np.mean(rets))

    state = {
        'positions': positions,
        'last_rebalance': last_rebalance,
        'last_update': str(today.date()),
        'current_return': round(total_ret, 4),
    }
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    logger.info(f"状态已保存: 持仓 {len(positions)} 只, 当前平均收益 {total_ret*100:+.2f}%")


if __name__ == "__main__":
    main()
