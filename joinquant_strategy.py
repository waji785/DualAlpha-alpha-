# 聚宽模拟盘策略 — 上传到 JoinQuant 策略平台
# 每天开盘前手动上传 today_signals.csv 到研究环境，策略自动读取

import pandas as pd
import numpy as np

def initialize(context):
    """初始化"""
    context.signals = {}
    context.positions_info = {}  # {code: {'cost': float, 'entry_date': str}}
    context.cash = 1000000
    context.max_positions = 10   # 最多同时持有10只
    context.single_position = 0.1  # 单只10%仓位
    context.take_profit = 0.10
    context.stop_loss = -0.08
    g.security = '000300.XSHG'  # 基准
    set_benchmark(g.security)

    # 每天9:25运行：读取信号 + 调仓
    run_daily(load_and_trade, time='9:25')

def load_and_trade(context):
    """读取 today_signals.csv 并执行交易"""
    # 从研究环境读取上传的 CSV
    try:
        df = pd.read_csv('today_signals.csv')
    except:
        log.warning('today_signals.csv 未找到，跳过今日交易')
        return

    buys = df[df['signal'] == '买入'].copy()
    if len(buys) == 0:
        log.info('今日无买入信号')

    # ---- 卖出逻辑 ----
    for code in list(context.portfolio.positions.keys()):
        stock = context.portfolio.positions[code]
        if stock.total_amount == 0:
            continue

        current_price = current_data[code].last_price if code in current_data else stock.price
        cost = context.positions_info.get(code, {}).get('cost', stock.avg_cost)
        profit = (current_price - cost) / cost

        # 止损止盈
        if profit <= context.stop_loss:
            order_target(code, 0)
            log.info(f'止损 {code} @{current_price:.2f} {profit*100:.1f}%')
            if code in context.positions_info:
                del context.positions_info[code]
            continue

        if profit >= context.take_profit:
            order_target(code, 0)
            log.info(f'止盈 {code} @{current_price:.2f} {profit*100:.1f}%')
            if code in context.positions_info:
                del context.positions_info[code]
            continue

        # 不在今日买入列表中 → 减仓
        if code not in buys['code'].values:
            order_target(code, 0)
            log.info(f'信号消失，清仓 {code}')

    # ---- 买入逻辑 ----
    current_positions = len([p for p in context.portfolio.positions.values()
                             if p.total_amount > 0])

    for _, row in buys.iterrows():
        code = format_code(row['code'])
        if code in context.portfolio.positions:
            continue  # 已持有
        if current_positions >= context.max_positions:
            break

        up_prob = float(row.get('up_prob', 0))
        weight = context.single_position * min(1.0, (up_prob - 0.45) * 2.0)
        weight = max(weight, 0.05)  # 最少5%仓位

        order_target_value(code, context.portfolio.total_value * weight)
        context.positions_info[code] = {
            'cost': float(row.get('close', 0)),
            'entry_date': context.current_dt.strftime('%Y-%m-%d'),
        }
        current_positions += 1
        log.info(f'买入 {code} 权重{weight*100:.0f}% up_prob={up_prob:.3f}')


def format_code(code_str):
    """600519 → 600519.XSHG, 000001 → 000001.XSHE"""
    code = str(code_str).zfill(6)
    if code.startswith('6'):
        return f'{code}.XSHG'
    return f'{code}.XSHE'
