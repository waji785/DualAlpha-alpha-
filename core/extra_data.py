#!/usr/bin/env python
# core/extra_data.py
"""tushare 额外数据：北向持仓 + 财务指标 + 股东增减持"""
import numpy as np
import pandas as pd
from config.settings import TUSHARE_TOKEN

try:
    import tushare as ts
    _TS_OK = TUSHARE_TOKEN != "your_token_here"
except ImportError:
    _TS_OK = False


def fetch_extra_data(stock_code, start, end):
    """合并北向持仓、财务指标、股东增减持 → 日级别 DataFrame"""
    if not _TS_OK:
        return None

    pro = ts.pro_api(TUSHARE_TOKEN)
    s = stock_code.zfill(6)
    ts_code = f"{s}.{'SH' if s[0] == '6' else 'SZ'}"
    s_d, e_d = start.replace('-', ''), end.replace('-', '')
    extra = {}

    # 1. 北向持仓（日级别）
    try:
        hk = pro.hk_hold(ts_code=ts_code, start_date=s_d, end_date=e_d)
        if hk is not None and len(hk) > 0:
            hk['Date'] = pd.to_datetime(hk['trade_date']).dt.strftime('%Y-%m-%d')
            hk = hk[['Date', 'ratio']].rename(columns={'ratio': 'North_Hold'})
            hk['North_Hold'] = hk['North_Hold'].astype(float)
            extra['hk'] = hk
    except Exception:
        pass

    # 2. 财务指标（季度 → 日级前向填充）
    try:
        fina = pro.fina_indicator(ts_code=ts_code, start_date=s_d[:4] + '0101',
                                   end_date=e_d)
        if fina is not None and len(fina) > 0:
            fina['Date'] = pd.to_datetime(fina['end_date']).dt.strftime('%Y-%m-%d')
            fina = fina[['Date', 'roe', 'grossprofit_margin',
                          'netprofit_margin', 'debt_to_assets']].copy()
            fina.columns = ['Date', 'ROE', 'GrossMargin', 'NetMargin', 'DebtRatio']
            for c in ['ROE', 'GrossMargin', 'NetMargin', 'DebtRatio']:
                fina[c] = fina[c].astype(float)
            dr = pd.date_range(start, end, freq='D')
            base = pd.DataFrame({'Date': dr.strftime('%Y-%m-%d')})
            fina = pd.merge(base, fina, on='Date', how='left')
            for c in ['ROE', 'GrossMargin', 'NetMargin', 'DebtRatio']:
                fina[c] = fina[c].ffill()
            extra['fina'] = fina
    except Exception:
        pass

    # 3. 股东增减持（事件 → 每日信号）
    try:
        hld = pro.stk_holdertrade(ts_code=ts_code, start_date=s_d[:4] + '0101')
        if hld is not None and len(hld) > 0:
            hld['Date'] = pd.to_datetime(hld['ann_date']).dt.strftime('%Y-%m-%d')
            hld['net'] = (hld.get('in_vol', hld.get('in_de', 0)).fillna(0).astype(float)
                          - hld.get('out_vol', hld.get('out_de', 0)).fillna(0).astype(float))
            grp = hld.groupby('Date')['net'].sum().reset_index()
            grp['Insider_Signal'] = grp['net'].apply(
                lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            dr = pd.date_range(start, end, freq='D')
            base = pd.DataFrame({'Date': dr.strftime('%Y-%m-%d')})
            grp = pd.merge(base, grp, on='Date', how='left').fillna(0)
            extra['holder'] = grp[['Date', 'Insider_Signal']]
    except Exception:
        pass

    if not extra:
        return None

    result = list(extra.values())[0]
    for df in list(extra.values())[1:]:
        result = pd.merge(result, df, on='Date', how='outer')
    result = result.sort_values('Date').fillna(0).reset_index(drop=True)
    return result
