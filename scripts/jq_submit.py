#!/usr/bin/env python
# scripts/jq_submit.py
"""
本地运行 → 读取 today_signals.csv → 调用聚宽 API 提交模拟盘订单
需要先 pip install jqdatasdk
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from config.settings import OUTPUT_DIR

# 聚宽账号（用你的替换）
JQ_USER = "你的手机号"
JQ_PASS = "你的密码"

try:
    from jqdatasdk import auth, create_simulated_trade, order, order_target_value, get_position

    auth(JQ_USER, JQ_PASS)

    trade = create_simulated_trade("dualalpha", "1000000", "A股模拟盘")
    signals_path = os.path.join(OUTPUT_DIR, "today_signals.csv")

    if not os.path.exists(signals_path):
        print(f"信号文件不存在: {signals_path}")
        sys.exit(1)

    df = pd.read_csv(signals_path)
    buys = df[df['signal'] == '买入']

    if len(buys) == 0:
        print("今日无买入信号")

    # 清仓不在买入列表的持仓
    positions = get_position(trade)
    for code in positions:
        order_target_value(code, 0, trade)
        print(f"清仓 {code}")

    # 买入
    for _, row in buys.iterrows():
        code = str(row['code']).zfill(6)
        up_prob = float(row.get('up_prob', 0))
        weight = 0.1 * min(1.0, (up_prob - 0.45) * 2.0)
        weight = max(weight, 0.05)
        order_target_value(code, get_portfolio(trade).total_value * weight, trade)
        print(f"买入 {code} {weight*100:.0f}% (up_prob={up_prob:.3f})")

    print("✅ 聚宽模拟盘订单已提交")

except ImportError:
    print("pip install jqdatasdk")
except Exception as e:
    print(f"失败: {e}")
