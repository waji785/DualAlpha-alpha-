# core/features.py
import pandas as pd
import numpy as np
from config.settings import FEATURE_COLS

EPS = 1e-8  # 防止除零


# ============================================================
#  基础指标计算函数
# ============================================================

def ema(data, window):
    """指数移动平均"""
    return data.ewm(span=window, adjust=False).mean()


def calculate_rsi(data, window=14):
    """RSI 相对强弱指标"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + EPS)
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(data, window=20, num_std=2):
    middle = data.rolling(window).mean()
    std = data.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return (data - lower) / (upper - lower + EPS), (upper - lower) / (middle + EPS)


def calculate_macd(close, fast=12, slow=26, signal=9):
    """MACD: 返回 DIF, DEA, HIST"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    hist = 2 * (dif - dea)
    return dif, dea, hist


def calculate_kdj(high, low, close, n=9, m1=3, m2=3):
    """KDJ: 返回 K, D, J"""
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rsv = (close - low_n) / (high_n - low_n + EPS) * 100
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calculate_atr(high, low, close, window=14):
    """ATR 真实波幅 & ATRP 百分比"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    atrp = atr / (close + EPS) * 100
    return atr, atrp


def calculate_mfi(high, low, close, volume, window=14):
    """MFI 资金流量指标 (Money Flow Index)"""
    tp = (high + low + close) / 3
    rmf = tp * volume
    tp_diff = tp.diff()
    pos_flow = rmf.where(tp_diff > 0, 0).rolling(window).sum()
    neg_flow = (-rmf.where(tp_diff < 0, 0)).rolling(window).sum()
    mfr = pos_flow / (neg_flow + EPS)
    return 100 - (100 / (1 + mfr))


# ============================================================
#  主特征构造
# ============================================================

def construct_features(df):
    """
    构造全部技术指标 + target（10日后的价格和方向）

    输入必须包含:
        Open, High, Low, Close, Volume, Amount, Turn,
        TradeStatus, PctChg, PeTTM, PbMRQ, PsTTM, PcfNcfTTM
    """
    df = df.copy()
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']

    # ========== ① 动量 / 收益率（3 个）==========
    df['Momentum_5'] = close.pct_change(5)
    df['Momentum_10'] = close.pct_change(10)
    df['Return_1d'] = close.pct_change()

    # ========== ② 波动率（2 个）==========
    df['Volatility_5'] = df['Return_1d'].rolling(5).std()
    df['Volatility_10'] = df['Return_1d'].rolling(10).std()

    # ========== ③ 均线 & 偏离（6 个）==========
    for w in [5, 10, 20]:
        df[f'MA_{w}'] = close.rolling(w).mean()
    df['Price_MA_5_Ratio'] = close / (df['MA_5'] + EPS) - 1
    df['Price_MA_20_Ratio'] = close / (df['MA_20'] + EPS) - 1
    df['MA_5_20_diff'] = df['MA_5'] - df['MA_20']

    # ========== ④ RSI / 布林带（2 个）==========
    df['RSI_14'] = calculate_rsi(close, 14)
    bb_pos, _ = calculate_bollinger_bands(close)
    df['BB_position'] = bb_pos

    # ========== ⑤ 量价关系（3 个）==========
    ma5_vol = volume.rolling(5).mean()
    df['Volume_Ratio'] = volume / (ma5_vol + EPS)
    df['High_Low_Ratio'] = (high - low) / (close + EPS)

    if 'Amount' in df.columns:
        df['Amount_Ratio'] = df['Amount'] / (df['Amount'].rolling(20).mean() + EPS)
    else:
        df['Amount_Ratio'] = 0

    # ========== ⑥ 原始字段标准化（4 个）==========
    if 'PctChg' in df.columns:
        df['PctChg'] = df['PctChg'] / 100.0
    else:
        df['PctChg'] = close.pct_change()

    if 'TradeStatus' in df.columns:
        df['TradeStatus'] = df['TradeStatus'].astype(int)
    else:
        df['TradeStatus'] = 1

    if 'Turn' in df.columns:
        df['Turnover'] = df['Turn'] / 100.0
    else:
        df['Turnover'] = 0

    # 估值字段来自 data_loader 的 enrich_fundamentals，
    # 若无则保留默认 0（不影响计算）

    # ================================================================
    #  🆕 ⑦ MACD（3 个）
    # ================================================================
    dif, dea, hist = calculate_macd(close, fast=12, slow=26, signal=9)
    df['MACD_DIF'] = dif
    df['MACD_DEA'] = dea
    df['MACD_HIST'] = hist

    # ================================================================
    #  🆕 ⑧ KDJ（3 个）
    # ================================================================
    k, d, j = calculate_kdj(high, low, close, n=9, m1=3, m2=3)
    df['KDJ_K'] = k
    df['KDJ_D'] = d
    df['KDJ_J'] = j

    # ================================================================
    #  🆕 ⑨ ATR / ATRP（2 个）
    # ================================================================
    atr, atrp = calculate_atr(high, low, close, window=14)
    df['ATR_14'] = atr
    df['ATRP'] = atrp

    # ================================================================
    #  🆕 ⑩ OBV 斜率（1 个）
    # ================================================================
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['OBV_slope'] = (obv - obv.shift(5)) / (obv.shift(5).abs() + EPS)

    # ================================================================
    #  🆕 ⑪ 连涨 / 连跌天数（2 个）
    # ================================================================
    sign = np.sign(close.diff()).fillna(0)
    df['Consecutive_Up'] = (
        sign.groupby((sign <= 0).cumsum()).cumsum().clip(lower=0)
    )
    sign_neg = (-sign)  # 跌为负 → 翻转后正
    df['Consecutive_Down'] = (
        sign_neg.groupby((sign_neg <= 0).cumsum()).cumsum().clip(lower=0)
    )

    # ================================================================
    #  🆕 ⑫ 价格分位（1 个）
    # ================================================================
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    df['Price_Position_20'] = (close - low_20) / (high_20 - low_20 + EPS)

    # ================================================================
    #  🆕 ⑬ 上下影线比率（2 个）
    # ================================================================
    body_high = np.maximum(df['Open'], close)
    body_low = np.minimum(df['Open'], close)
    df['Upper_Shadow'] = (high - body_high) / (high - low + EPS)
    df['Lower_Shadow'] = (body_low - low) / (high - low + EPS)

    # ================================================================
    #  🆕 ⑭ 量能加速度（1 个）
    # ================================================================
    vol_ma5 = volume.rolling(5).mean()
    vol_ma10 = volume.rolling(10).mean()
    df['Volume_Accel'] = (volume / (vol_ma5 + EPS)) / ((vol_ma5 + EPS) / (vol_ma10 + EPS)) - 1

    # ================================================================
    #  🆕 ⑮ MFI 资金流量指标（1 个）
    # ================================================================
    df['MFI_14'] = calculate_mfi(high, low, close, volume, window=14)

    # ================================================================
    #  🆕 ⑯ 周期特征（6 个）— sin/cos 编码
    # ================================================================
    dow = df['Date'].dt.dayofweek          # 0=Mon ... 6=Sun
    month = df['Date'].dt.month            # 1..12
    quarter = (month - 1) // 3 + 1         # 1..4

    df['Dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['Dow_cos'] = np.cos(2 * np.pi * dow / 7)
    df['Month_sin'] = np.sin(2 * np.pi * month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * month / 12)
    df['Quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
    df['Quarter_cos'] = np.cos(2 * np.pi * quarter / 4)

    # ================================================================
    #  另类数据衍生特征（从 tushare 额外数据列自动计算）
    # ================================================================
    if 'North_Hold' in df.columns:
        df['North_Hold_Chg_5d'] = df['North_Hold'].diff(5)
        df['North_Hold_Chg_20d'] = df['North_Hold'].diff(20)
    else:
        df['North_Hold_Chg_5d'] = 0.0
        df['North_Hold_Chg_20d'] = 0.0

    if 'Insider_Signal' in df.columns:
        df['Insider_Buy_Window'] = df['Insider_Signal'].rolling(60).mean()
    else:
        df['Insider_Buy_Window'] = 0.0

    for c in ['ROE', 'GrossMargin', 'NetMargin', 'DebtRatio', 'North_Hold',
              'Insider_Signal', 'North_Hold_Chg_5d', 'North_Hold_Chg_20d',
              'Insider_Buy_Window']:
        if c not in df.columns:
            df[c] = 0.0

    # ================================================================
    #  Target（2 个，不参与特征）
    # ================================================================
    df['Target_Price'] = close.shift(-10)
    df['Target_Direction'] = (close.shift(-10) > close).astype(int)

    # ================================================================
    #  🆕 ⑲ 波动率缩放（风险调整收益率）
    # ================================================================
    daily_ret = df['Close'].pct_change().fillna(0)
    vol20 = daily_ret.rolling(20).std().clip(lower=0.005).fillna(0.01)

    return_feats = [
        'Momentum_5', 'Momentum_10', 'Return_1d',   # 动量/收益
        'PctChg',                                     # 涨跌幅
        'MACD_DIF', 'MACD_HIST',                      # MACD（价格差）
        'Price_MA_5_Ratio', 'Price_MA_20_Ratio',      # 偏离均线
        'MA_5_20_diff',                                # 均线差
        'KDJ_J',                                       # KDJ 快线
    ]
    for f in return_feats:
        if f in df.columns:
            df[f] = df[f] / vol20

    # ================================================================
    #  🆕 ⑳ 交叉特征（树模型用）
    # ================================================================
    df['Mom_Vol'] = df['Momentum_10'] / (df['Volatility_10'] + EPS)
    df['RSI_Vol'] = df['RSI_14'] * df['Volume_Ratio']

    # ========== 清理 ==========
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


def clean_data(df):
    """清理异常值并确保类型正确"""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    if 'TradeStatus' in df.columns:
        df['TradeStatus'] = df['TradeStatus'].astype(int).clip(0, 1)
    return df
