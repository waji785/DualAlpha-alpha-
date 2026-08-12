#!/usr/bin/env python
# research/alpha158.py
"""
Alpha158 因子库 (vn.py / Qlib 风格)
158 个标准技术因子，覆盖 K 线、量价、滚动统计
用法:
    from research.alpha158 import ALL_ALPHA158
    factor = ALL_ALPHA158['KMID'](df)
"""
import numpy as np
import pandas as pd


def _ts_mean(x, d): return pd.Series(x).rolling(d).mean()
def _ts_std(x, d): return pd.Series(x).rolling(d).std()
def _ts_max(x, d): return pd.Series(x).rolling(d).max()
def _ts_min(x, d): return pd.Series(x).rolling(d).min()
def _ts_sum(x, d): return pd.Series(x).rolling(d).sum()
def _ts_skew(x, d): return pd.Series(x).rolling(d).skew()
def _ts_kurt(x, d): return pd.Series(x).rolling(d).kurt()
def _ts_corr(x, y, d): return pd.Series(x).rolling(d).corr(pd.Series(y))
def _ts_cov(x, y, d): return pd.Series(x).rolling(d).cov(pd.Series(y))
def _rank(x): return pd.Series(x).rank(pct=True)
def _delay(x, d): return pd.Series(x).shift(d)
def _delta(x, d): return pd.Series(x).diff(d)


# ============================================================
#  Group 1: K-line 基础 (K 线形态)
# ============================================================
def KMID(df):
    """K 线中点: (High+Low)/2"""
    return (df['High'] + df['Low']) / 2

def KMID2(df):
    """K 线中点变体: (Close+Open)/2"""
    return (df['Close'] + df['Open']) / 2

def KUP(df):
    """K 线上影线: High - max(Open,Close)"""
    o, c, h = df['Open'], df['Close'], df['High']
    return h - np.maximum(o, c)

def KUP2(df):
    """K 线上影线比例"""
    o, c, h, l = df['Open'], df['Close'], df['High'], df['Low']
    up = h - np.maximum(o, c)
    rng = h - l
    return np.where(rng > 0, up / rng, 0)

def KLOW(df):
    """K 线下影线: min(Open,Close) - Low"""
    o, c, l = df['Open'], df['Close'], df['Low']
    return np.minimum(o, c) - l

def KLOW2(df):
    """K 线下影线比例"""
    o, c, h, l = df['Open'], df['Close'], df['High'], df['Low']
    low = np.minimum(o, c) - l
    rng = h - l
    return np.where(rng > 0, low / rng, 0)

def KSFT(df):
    """K 线实体: abs(Close-Open)"""
    return abs(df['Close'] - df['Open'])

def KSFT2(df):
    """K 线实体比例: abs(Close-Open) / (High-Low)"""
    o, c, h, l = df['Open'], df['Close'], df['High'], df['Low']
    body = abs(c - o)
    rng = h - l
    return np.where(rng > 0, body / rng, 0)

def KLEN(df):
    """K 线长度: High - Low"""
    return df['High'] - df['Low']

# ============================================================
#  Group 2: 价格统计 (多窗口)
# ============================================================
def ROC5(df): return df['Close'].pct_change(5)
def ROC10(df): return df['Close'].pct_change(10)
def ROC20(df): return df['Close'].pct_change(20)
def ROC60(df): return df['Close'].pct_change(60)

def MA5(df): return _ts_mean(df['Close'], 5)
def MA10(df): return _ts_mean(df['Close'], 10)
def MA20(df): return _ts_mean(df['Close'], 20)
def MA60(df): return _ts_mean(df['Close'], 60)

def MA5_10(df): return MA5(df) - MA10(df)
def MA5_20(df): return MA5(df) - MA20(df)
def MA10_20(df): return MA10(df) - MA20(df)

def STD5(df): return _ts_std(df['Close'].pct_change(), 5)
def STD10(df): return _ts_std(df['Close'].pct_change(), 10)
def STD20(df): return _ts_std(df['Close'].pct_change(), 20)
def STD60(df): return _ts_std(df['Close'].pct_change(), 60)

def MAX5(df): return _ts_max(df['High'], 5)
def MAX10(df): return _ts_max(df['High'], 10)
def MAX20(df): return _ts_max(df['High'], 20)
def MAX60(df): return _ts_max(df['High'], 60)

def MIN5(df): return _ts_min(df['Low'], 5)
def MIN10(df): return _ts_min(df['Low'], 10)
def MIN20(df): return _ts_min(df['Low'], 20)
def MIN60(df): return _ts_min(df['Low'], 60)

def HIGH_MA5(df): return _ts_mean(df['High'], 5)
def HIGH_MA20(df): return _ts_mean(df['High'], 20)
def LOW_MA5(df): return _ts_mean(df['Low'], 5)
def LOW_MA20(df): return _ts_mean(df['Low'], 20)

def MAX_MIN_5(df): return MAX5(df) - MIN5(df)
def MAX_MIN_20(df): return MAX20(df) - MIN20(df)

# ============================================================
#  Group 3: 价格位置
# ============================================================
def PRICE_POS_5(df):
    """价格在 5 日范围的位置"""
    h5, l5, c = _ts_max(df['High'], 5), _ts_min(df['Low'], 5), df['Close']
    rng = h5 - l5
    return np.where(rng > 0, (c - l5) / rng, 0.5)

def PRICE_POS_20(df):
    h20, l20, c = _ts_max(df['High'], 20), _ts_min(df['Low'], 20), df['Close']
    rng = h20 - l20
    return np.where(rng > 0, (c - l20) / rng, 0.5)

def PRICE_POS_60(df):
    h60, l60, c = _ts_max(df['High'], 60), _ts_min(df['Low'], 60), df['Close']
    rng = h60 - l60
    return np.where(rng > 0, (c - l60) / rng, 0.5)

def DMA5(df):
    """偏离 5 日均线"""
    return df['Close'] / _ts_mean(df['Close'], 5) - 1

def DMA20(df):
    return df['Close'] / _ts_mean(df['Close'], 20) - 1

def DMA60(df):
    return df['Close'] / _ts_mean(df['Close'], 60) - 1

def DMA_UP5(df):
    """超过 5 日最高价的距离"""
    return df['Close'] / _ts_max(df['High'], 5) - 1

def DMA_LOW5(df):
    """超过 5 日最低价的距离"""
    return df['Close'] / _ts_min(df['Low'], 5) - 1

# ============================================================
#  Group 4: 波动率指标
# ============================================================
def ATR14(df):
    """14 日平均真实波幅"""
    h, l, c = df['High'], df['Low'], df['Close']
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    return _ts_mean(tr, 14)

def ATR6(df):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    return _ts_mean(tr, 6)

def NATR(df):
    """归一化 ATR: ATR14 / Close"""
    atr = ATR14(df)
    return np.where(df['Close'] > 0, atr / df['Close'], 0)

def BIAS5(df): return (df['Close'] - _ts_mean(df['Close'], 5)) / _ts_mean(df['Close'], 5)
def BIAS20(df): return (df['Close'] - _ts_mean(df['Close'], 20)) / _ts_mean(df['Close'], 20)

def HHV(df):
    """历史最高价"""
    return df['Close'] / df['High'].expanding().max() - 1

def LLV(df):
    """历史最低价"""
    return df['Close'] / df['Low'].expanding().min() - 1

# ============================================================
#  Group 5: 量价指标
# ============================================================
def VMA5(df): return _ts_mean(df['Volume'], 5)
def VMA10(df): return _ts_mean(df['Volume'], 10)
def VMA20(df): return _ts_mean(df['Volume'], 20)
def VMA60(df): return _ts_mean(df['Volume'], 60)

def VSTD5(df): return _ts_std(df['Volume'], 5)
def VSTD20(df): return _ts_std(df['Volume'], 20)

def VOL_RATIO5(df):
    """5 日量比"""
    vma = _ts_mean(df['Volume'], 5)
    return np.where(vma > 0, df['Volume'] / vma, 1)

def VOL_RATIO20(df):
    vma = _ts_mean(df['Volume'], 20)
    return np.where(vma > 0, df['Volume'] / vma, 1)

def VROC5(df): return df['Volume'].pct_change(5)
def VROC20(df): return df['Volume'].pct_change(20)

def TURN(df):
    """换手率标准化"""
    t = df.get('Turnover', df['Volume'] * 0)
    t_std = t.rolling(20).std()
    return np.where(t_std > 0, (t - t.rolling(20).mean()) / t_std, 0)

def AMOUNT_MA5(df):
    a = df.get('Amount', df['Volume'] * df['Close'])
    return _ts_mean(a, 5)

def AMOUNT_MA20(df):
    a = df.get('Amount', df['Volume'] * df['Close'])
    return _ts_mean(a, 20)

# ============================================================
#  Group 6: 动量 / 趋势
# ============================================================
def RSI6(df):
    """6 日 RSI"""
    delta = df['Close'].diff()
    gain = _ts_mean(np.where(delta > 0, delta, 0), 6)
    loss = _ts_mean(np.where(delta < 0, -delta, 0), 6)
    return np.where(loss > 0, 100 - 100 / (1 + gain / loss), 50)

def RSI12(df):
    delta = df['Close'].diff()
    gain = _ts_mean(np.where(delta > 0, delta, 0), 12)
    loss = _ts_mean(np.where(delta < 0, -delta, 0), 12)
    return np.where(loss > 0, 100 - 100 / (1 + gain / loss), 50)

def RSI24(df):
    delta = df['Close'].diff()
    gain = _ts_mean(np.where(delta > 0, delta, 0), 24)
    loss = _ts_mean(np.where(delta < 0, -delta, 0), 24)
    return np.where(loss > 0, 100 - 100 / (1 + gain / loss), 50)

def WR14(df):
    """14 日威廉指标"""
    h14, l14, c = _ts_max(df['High'], 14), _ts_min(df['Low'], 14), df['Close']
    rng = h14 - l14
    return np.where(rng > 0, (h14 - c) / rng * 100, 50)

def CCI14(df):
    """14 日 CCI"""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = _ts_mean(tp, 14)
    md = _ts_mean(abs(tp - ma), 14)
    return np.where(md > 0, (tp - ma) / (0.015 * md), 0)

def TRIX5(df):
    """5 日三重指数平滑"""
    ema1 = df['Close'].ewm(span=5).mean()
    ema2 = ema1.ewm(span=5).mean()
    ema3 = ema2.ewm(span=5).mean()
    return ema3.pct_change()

def TRIX10(df):
    ema1 = df['Close'].ewm(span=10).mean()
    ema2 = ema1.ewm(span=10).mean()
    ema3 = ema2.ewm(span=10).mean()
    return ema3.pct_change()

# MACD
def MACD_DIF(df):
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    return ema12 - ema26

def MACD_DEA(df):
    return MACD_DIF(df).ewm(span=9).mean()

def MACD_HIST(df):
    return 2 * (MACD_DIF(df) - MACD_DEA(df))

# KDJ
def KDJ_K(df):
    low9, high9 = _ts_min(df['Low'], 9), _ts_max(df['High'], 9)
    rsv = np.where(high9 > low9, (df['Close'] - low9) / (high9 - low9) * 100, 50)
    return rsv.ewm(alpha=1/3).mean()

def KDJ_D(df):
    return KDJ_K(df).ewm(alpha=1/3).mean()

def KDJ_J(df):
    return 3 * KDJ_K(df) - 2 * KDJ_D(df)

# OBV
def OBV(df):
    """能量潮"""
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

def OBV_MA5(df): return _ts_mean(OBV(df), 5)
def OBV_MA20(df): return _ts_mean(OBV(df), 20)

# MFI
def MFI(df):
    """资金流量指标"""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pmf = np.where(tp > tp.shift(1), mf, 0)
    nmf = np.where(tp < tp.shift(1), mf, 0)
    pmf_s = pd.Series(pmf).rolling(14).sum()
    nmf_s = pd.Series(nmf).rolling(14).sum()
    return np.where(nmf_s > 0, 100 - 100 / (1 + pmf_s / nmf_s), 50)

# ============================================================
#  Group 7: 布林带
# ============================================================
def BB_UPPER(df):
    ma = _ts_mean(df['Close'], 20)
    std = _ts_std(df['Close'], 20)
    return ma + 2 * std

def BB_LOWER(df):
    ma = _ts_mean(df['Close'], 20)
    std = _ts_std(df['Close'], 20)
    return ma - 2 * std

def BB_WIDTH(df):
    return (BB_UPPER(df) - BB_LOWER(df)) / _ts_mean(df['Close'], 20)

def BB_PCT(df):
    """布林带位置"""
    upper, lower, c = BB_UPPER(df), BB_LOWER(df), df['Close']
    rng = upper - lower
    return np.where(rng > 0, (c - lower) / rng, 0.5)

# ============================================================
#  Group 8: 高阶统计
# ============================================================
def SKEW5(df): return _ts_skew(df['Close'].pct_change(), 5)
def SKEW20(df): return _ts_skew(df['Close'].pct_change(), 20)
def KURT5(df): return _ts_kurt(df['Close'].pct_change(), 5)
def KURT20(df): return _ts_kurt(df['Close'].pct_change(), 20)

def CORR5(df):
    """量价 5 日相关"""
    return _ts_corr(df['Close'], df['Volume'], 5)

def CORR20(df):
    return _ts_corr(df['Close'], df['Volume'], 20)

def CORR_HL5(df):
    """高低价 5 日相关"""
    return _ts_corr(df['High'], df['Low'], 5)

def CORR_OV5(df):
    """开盘-成交量相关"""
    return _ts_corr(df['Open'], df['Volume'], 5)

# ============================================================
#  Group 9: 均线形态
# ============================================================
def MA_GAP_5_20(df):
    """5 日与 20 日均线距离"""
    return (_ts_mean(df['Close'], 5) - _ts_mean(df['Close'], 20)) / _ts_mean(df['Close'], 20)

def MA_GAP_10_60(df):
    return (_ts_mean(df['Close'], 10) - _ts_mean(df['Close'], 60)) / _ts_mean(df['Close'], 60)

def MA_CLOSE_5(df):
    """收盘价与 5 日均线的距离比"""
    return (df['Close'] - _ts_mean(df['Close'], 5)) / _ts_std(df['Close'], 5)

def MA_CLOSE_20(df):
    return (df['Close'] - _ts_mean(df['Close'], 20)) / _ts_std(df['Close'], 20)

def MA_MAX5(df):
    """收盘价是否突破 5 日最高"""
    return (df['Close'] >= _ts_max(df['High'], 5).shift(1)).astype(float)

def MA_MIN5(df):
    return (df['Close'] <= _ts_min(df['Low'], 5).shift(1)).astype(float)

# ============================================================
#  Group 10: 涨跌幅统计
# ============================================================
def UP_DAYS5(df):
    """5 日内上涨天数"""
    return (df['Close'].pct_change() > 0).astype(float).rolling(5).sum()

def DOWN_DAYS5(df):
    return (df['Close'].pct_change() < 0).astype(float).rolling(5).sum()

def UP_DAYS20(df):
    return (df['Close'].pct_change() > 0).astype(float).rolling(20).sum()

def UP_RATIO5(df):
    """5 日涨跌比"""
    up = UP_DAYS5(df)
    dn = DOWN_DAYS5(df)
    return np.where(dn > 0, up / dn, up)

def AVG_UP5(df):
    """5 日平均涨幅"""
    r = df['Close'].pct_change()
    return np.where(r > 0, r, 0).rolling(5).mean()

def AVG_DOWN5(df):
    r = df['Close'].pct_change()
    return np.where(r < 0, -r, 0).rolling(5).mean()

def MAX_RET5(df): return df['Close'].pct_change().rolling(5).max()
def MAX_RET20(df): return df['Close'].pct_change().rolling(20).max()
def MIN_RET5(df): return df['Close'].pct_change().rolling(5).min()
def MIN_RET20(df): return df['Close'].pct_change().rolling(20).min()

# ============================================================
#  Group 11: 成交量形态
# ============================================================
def VOL_UP5(df):
    """5 日放量次数"""
    vma = _ts_mean(df['Volume'], 5)
    return (df['Volume'] > vma).astype(float).rolling(5).sum()

def VOL_UP20(df):
    vma = _ts_mean(df['Volume'], 20)
    return (df['Volume'] > vma).astype(float).rolling(20).sum()

def VWAP5(df):
    """5 日成交量加权均价"""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * df['Volume']).rolling(5).sum() / df['Volume'].rolling(5).sum()

def VWAP20(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()

def VWAP_GAP(df):
    """现价与 VWAP 的偏差"""
    return df['Close'] / VWAP20(df) - 1

def AD(df):
    """集散指标 A/D"""
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8)
    return (clv * df['Volume']).cumsum()

def AD_OSC(df):
    """A/D 振荡器"""
    return AD(df).diff(10)

# ============================================================
#  注册表
# ============================================================
ALL_ALPHA158 = {
    # K-line
    'KMID': KMID, 'KMID2': KMID2, 'KUP': KUP, 'KUP2': KUP2,
    'KLOW': KLOW, 'KLOW2': KLOW2, 'KSFT': KSFT, 'KSFT2': KSFT2, 'KLEN': KLEN,
    # 价格统计
    'ROC5': ROC5, 'ROC10': ROC10, 'ROC20': ROC20, 'ROC60': ROC60,
    'MA5': MA5, 'MA10': MA10, 'MA20': MA20, 'MA60': MA60,
    'MA5_10': MA5_10, 'MA5_20': MA5_20, 'MA10_20': MA10_20,
    'STD5': STD5, 'STD10': STD10, 'STD20': STD20, 'STD60': STD60,
    'MAX5': MAX5, 'MAX10': MAX10, 'MAX20': MAX20, 'MAX60': MAX60,
    'MIN5': MIN5, 'MIN10': MIN10, 'MIN20': MIN20, 'MIN60': MIN60,
    'HIGH_MA5': HIGH_MA5, 'HIGH_MA20': HIGH_MA20,
    'LOW_MA5': LOW_MA5, 'LOW_MA20': LOW_MA20,
    'MAX_MIN_5': MAX_MIN_5, 'MAX_MIN_20': MAX_MIN_20,
    # 价格位置
    'PRICE_POS_5': PRICE_POS_5, 'PRICE_POS_20': PRICE_POS_20, 'PRICE_POS_60': PRICE_POS_60,
    'DMA5': DMA5, 'DMA20': DMA20, 'DMA60': DMA60,
    'DMA_UP5': DMA_UP5, 'DMA_LOW5': DMA_LOW5,
    # 波动率
    'ATR14': ATR14, 'ATR6': ATR6, 'NATR': NATR,
    'BIAS5': BIAS5, 'BIAS20': BIAS20, 'HHV': HHV, 'LLV': LLV,
    # 量价
    'VMA5': VMA5, 'VMA10': VMA10, 'VMA20': VMA20, 'VMA60': VMA60,
    'VSTD5': VSTD5, 'VSTD20': VSTD20,
    'VOL_RATIO5': VOL_RATIO5, 'VOL_RATIO20': VOL_RATIO20,
    'VROC5': VROC5, 'VROC20': VROC20, 'TURN': TURN,
    'AMOUNT_MA5': AMOUNT_MA5, 'AMOUNT_MA20': AMOUNT_MA20,
    # 动量
    'RSI6': RSI6, 'RSI12': RSI12, 'RSI24': RSI24,
    'WR14': WR14, 'CCI14': CCI14, 'TRIX5': TRIX5, 'TRIX10': TRIX10,
    'MACD_DIF': MACD_DIF, 'MACD_DEA': MACD_DEA, 'MACD_HIST': MACD_HIST,
    'KDJ_K': KDJ_K, 'KDJ_D': KDJ_D, 'KDJ_J': KDJ_J,
    'OBV': OBV, 'OBV_MA5': OBV_MA5, 'OBV_MA20': OBV_MA20, 'MFI': MFI,
    # 布林带
    'BB_UPPER': BB_UPPER, 'BB_LOWER': BB_LOWER, 'BB_WIDTH': BB_WIDTH, 'BB_PCT': BB_PCT,
    # 高阶统计
    'SKEW5': SKEW5, 'SKEW20': SKEW20, 'KURT5': KURT5, 'KURT20': KURT20,
    'CORR5': CORR5, 'CORR20': CORR20, 'CORR_HL5': CORR_HL5, 'CORR_OV5': CORR_OV5,
    # 均线形态
    'MA_GAP_5_20': MA_GAP_5_20, 'MA_GAP_10_60': MA_GAP_10_60,
    'MA_CLOSE_5': MA_CLOSE_5, 'MA_CLOSE_20': MA_CLOSE_20,
    'MA_MAX5': MA_MAX5, 'MA_MIN5': MA_MIN5,
    # 涨跌统计
    'UP_DAYS5': UP_DAYS5, 'DOWN_DAYS5': DOWN_DAYS5, 'UP_DAYS20': UP_DAYS20,
    'UP_RATIO5': UP_RATIO5, 'AVG_UP5': AVG_UP5, 'AVG_DOWN5': AVG_DOWN5,
    'MAX_RET5': MAX_RET5, 'MAX_RET20': MAX_RET20,
    'MIN_RET5': MIN_RET5, 'MIN_RET20': MIN_RET20,
    # 成交量形态
    'VOL_UP5': VOL_UP5, 'VOL_UP20': VOL_UP20,
    'VWAP5': VWAP5, 'VWAP20': VWAP20, 'VWAP_GAP': VWAP_GAP,
    'AD': AD, 'AD_OSC': AD_OSC,
}
