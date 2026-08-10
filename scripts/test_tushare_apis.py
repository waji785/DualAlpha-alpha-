#!/usr/bin/env python
# scripts/test_tushare_apis.py
"""测试 tushare 三个额外接口的权限和返回"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from config.settings import TUSHARE_TOKEN
import tushare as ts
import pandas as pd

pro = ts.pro_api(TUSHARE_TOKEN)
codes = ['000001.SZ', '600519.SH', '000858.SZ', '600036.SH', '601318.SH']
today = pd.Timestamp.today().strftime('%Y%m%d')

print("=" * 60)
print("测试 tushare 额外数据接口")
print(f"token: {TUSHARE_TOKEN[:8]}...")
print(f"日期: {today}")
print("=" * 60)

for ts_code in codes:
    print(f"\n{'─' * 40}")
    print(f"  {ts_code}")
    
    # 1. hk_hold
    try:
        df = pro.hk_hold(ts_code=ts_code, start_date='20150101', end_date=today)
        if df is not None and len(df) > 0:
            print(f"  hk_hold:        ✅ {len(df)} rows | cols={list(df.columns)}")
            print(f"    last: {df['trade_date'].iloc[-1]} hold_ratio={df['hold_ratio'].iloc[-1]}")
        else:
            print(f"  hk_hold:        ❌ 无数据（可能非港股通标的或无持仓）")
    except Exception as e:
        print(f"  hk_hold:        ❌ {type(e).__name__}: {e}")

    # 2. fina_indicator
    try:
        df = pro.fina_indicator(ts_code=ts_code, start_date='20150101', end_date=today)
        if df is not None and len(df) > 0:
            print(f"  fina_indicator: ✅ {len(df)} rows | roe样例={df['roe'].head(3).tolist()}")
        else:
            print(f"  fina_indicator: ❌ 无数据")
    except Exception as e:
        print(f"  fina_indicator: ❌ {type(e).__name__}: {e}")

    # 3. stk_holdertrade
    try:
        df = pro.stk_holdertrade(ts_code=ts_code, start_date='20150101')
        if df is not None and len(df) > 0:
            print(f"  stk_holdertrade:✅ {len(df)} rows | cols={list(df.columns)[:5]}")
        else:
            print(f"  stk_holdertrade:❌ 无数据")
    except Exception as e:
        print(f"  stk_holdertrade:❌ {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("诊断完毕")
