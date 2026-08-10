#!/usr/bin/env python
# core/alt_data.py
"""另类数据：北向资金 + 融资融券 + 涨跌停（基于 tushare）"""
import numpy as np
import pandas as pd

logger = __import__('utils.logger', fromlist=['setup_logger']).setup_logger(__name__)


def get_northbound_flow(days=5):
    """
    北向资金（沪深港通）净流向摘要

    Returns:
        dict: {'net_buy': float(亿), 'trend': '流入'/'流出', 'consecutive': int}
    """
    try:
        from config.settings import TUSHARE_TOKEN
        import tushare as ts
        if TUSHARE_TOKEN == "your_token_here":
            return None
        pro = ts.pro_api(TUSHARE_TOKEN)
        df = pro.moneyflow_hsgt(start_date=pd.Timestamp.today().strftime('%Y%m%d'))
        if df is None or len(df) == 0:
            return None
        recent = df.head(days)
        net = recent['north_money'].sum() / 1e8  # 转亿
        consecutive = 0
        for _, r in recent.iterrows():
            if r['north_money'] > 0:
                consecutive += 1
            else:
                break
        return {
            'net_buy': round(net, 2),
            'trend': '流入' if net > 0 else '流出',
            'consecutive': consecutive,
            'sentiment': '偏多' if consecutive >= 3 else (
                '偏空' if net < -50 else '中性'),
        }
    except Exception as e:
        logger.warning(f"北向资金: {e}")
        return None


def get_margin_trend(days=5):
    """
    融资融券趋势

    Returns:
        dict: 融资余额变化趋势
    """
    try:
        from config.settings import TUSHARE_TOKEN
        import tushare as ts
        if TUSHARE_TOKEN == "your_token_here":
            return None
        pro = ts.pro_api(TUSHARE_TOKEN)
        df = pro.margin(start_date=pd.Timestamp.today().strftime('%Y%m%d'))
        if df is None or len(df) == 0:
            return None
        recent = df.sort_values('trade_date', ascending=False).head(days)
        return {
            'rzye': round(float(recent['rzye'].iloc[0]) / 1e8, 2),  # 融资余额(亿)
            'rz_change': f"{ (recent['rzye'].iloc[0] / recent['rzye'].iloc[-1] - 1) * 100:+.1f}%",
            'trend': '加杠杆' if recent['rzye'].iloc[0] > recent['rzye'].iloc[-1] else '降杠杆',
        }
    except Exception as e:
        logger.warning(f"融资融券: {e}")
        return None


def get_stock_limit_info(code):
    """单只股票涨跌停价（用于日内风控）"""
    try:
        from config.settings import TUSHARE_TOKEN
        import tushare as ts
        if TUSHARE_TOKEN == "your_token_here":
            return None
        pro = ts.pro_api(TUSHARE_TOKEN)
        s = code.zfill(6)
        ts_code = f"{s}.SH" if s[0] == '6' else f"{s}.SZ"
        today = pd.Timestamp.today().strftime('%Y%m%d')
        df = pro.stk_limit(ts_code=ts_code, trade_date=today)
        if df is not None and len(df) > 0:
            return {
                'up_limit': float(df['up_limit'].iloc[0]),
                'down_limit': float(df['down_limit'].iloc[0]),
            }
    except Exception:
        pass
    return None


def market_sentiment_summary():
    """
    综合市场情绪摘要

    Returns:
        str: 一句话情绪判断
    """
    north = get_northbound_flow()
    margin = get_margin_trend()

    parts = []
    if north:
        parts.append(f"北向{north['trend']}{north['net_buy']:.0f}亿")
    if margin:
        parts.append(f"融资{margin['trend']}")

    if not parts:
        return "另类数据不可用（检查 tushare token）"

    return " | ".join(parts)


# 集成到每日信号
def enrich_signals(signals_df):
    """
    给 today_signals 加上市场情绪列

    Args:
        signals_df: daily_predict.py 输出的 DataFrame
    Returns:
        DataFrame with added columns
    """
    sentiment = market_sentiment_summary()
    signals_df['market_sentiment'] = sentiment

    # 逐个股票查涨跌停
    limits = {}
    for code in signals_df['code'].unique():
        lim = get_stock_limit_info(str(code).zfill(6))
        if lim:
            limits[code] = lim
    signals_df['up_limit'] = signals_df['code'].map(
        lambda c: limits.get(c, {}).get('up_limit'))
    signals_df['down_limit'] = signals_df['code'].map(
        lambda c: limits.get(c, {}).get('down_limit'))

    return signals_df
