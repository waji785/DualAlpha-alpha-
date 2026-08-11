#!/usr/bin/env python
# research/alpha101.py
"""
Alpha101 因子库 (WorldQuant)
基于论文《101 Formulaic Alphas》实现常用因子
"""
import numpy as np
import pandas as pd


# ============================================================
#  基础算子
# ============================================================
def _ts_sum(x, d):
    return pd.Series(x).rolling(d).sum()

def _ts_mean(x, d):
    return pd.Series(x).rolling(d).mean()

def _ts_std(x, d):
    return pd.Series(x).rolling(d).std()

def _ts_max(x, d):
    return pd.Series(x).rolling(d).max()

def _ts_min(x, d):
    return pd.Series(x).rolling(d).min()

def _ts_argmax(x, d):
    x_s = pd.Series(x)
    return x_s.rolling(d).apply(lambda s: s.values.argmax(), raw=True)

def _ts_argmin(x, d):
    x_s = pd.Series(x)
    return x_s.rolling(d).apply(lambda s: s.values.argmin(), raw=True)

def _rank(x):
    x_s = pd.Series(x)
    return x_s.rank(pct=True)

def _delay(x, d):
    return pd.Series(x).shift(d)

def _delta(x, d):
    return pd.Series(x).diff(d)

def _signed_power(x, a):
    return np.sign(x) * (np.abs(x) ** a)

def _scale(x, a=1.0):
    """归一化：sum(abs(x)) = a"""
    s = np.nansum(np.abs(x))
    if s > 0:
        return x / s * a
    return x

def _correlation(x, y, d):
    return pd.Series(x).rolling(d).corr(pd.Series(y))

def _covariance(x, y, d):
    return pd.Series(x).rolling(d).cov(pd.Series(y))

def _decay_linear(x, d):
    x_s = pd.Series(x)
    w = np.arange(1, d+1) / np.arange(1, d+1).sum()
    return x_s.rolling(d).apply(lambda s: (s * w).sum(), raw=True)

def _ind_neutralize(x, g):
    """行业中性化（简化版：组内减均值）"""
    return pd.Series(x).groupby(g).transform(lambda s: s - s.mean())


# ============================================================
#  Alpha 因子 (选取代表性的 15 个)
# ============================================================
def alpha001(df):
    """Alpha#1: 超跌反弹信号"""
    close = df['Close']
    ret = close.pct_change()
    cond = (ret < 0).astype(float) * close.pct_change(20).rolling(20).std().fillna(0) + \
           (ret >= 0).astype(float) * close
    return _rank(_ts_argmax(_signed_power(cond, 2), 5)) - 0.5


def alpha002(df):
    """Alpha#2: 量价背离 — 成交量变化 vs 日内涨幅"""
    log_vol = np.log(df['Volume'] + 1)
    intraday_ret = (df['Close'] - df['Open']) / df['Open']
    return -1 * _correlation(_rank(_delta(log_vol, 2)), _rank(intraday_ret), 6)


def alpha003(df):
    """Alpha#3: 开盘价 vs 成交量负相关"""
    return -1 * _correlation(_rank(df['Open'].fillna(method='ffill')),
                              _rank(df['Volume']), 10)


def alpha004(df):
    """Alpha#4: 持续新低的反弹预期"""
    return -1 * _ts_max(_rank(df['Low']).rolling(9).apply(
        lambda s: s.iloc[-1] if len(s) == 9 else np.nan, raw=True), 9)

def _ts_rank(x, d): return pd.Series(x).rolling(d).apply(lambda s: s.rank().iloc[-1] / len(s))
def _vwap(df, d=20): return ((df['High']+df['Low']+df['Close'])/3*df['Volume']).rolling(d).sum()/df['Volume'].rolling(d).sum()
def _adv(df, d=20): return df['Volume'].rolling(d).mean()

def alpha005(df):
    o, c = df['Open'], df['Close']; vwap10 = _vwap(df, 10)
    return _rank((o - _ts_sum(vwap10, 10)/10)) * (-1 * abs(_rank(c - vwap10)))

def alpha007(df):
    c, v = df['Close'], df['Volume']; adv20 = _adv(df, 20)
    return -1 * _ts_max(_correlation(_rank((c.pct_change()>0).astype(float)), _rank(adv20), 5), 3) * np.sign(_delta(c, 7))

def alpha008(df):
    o, v = df['Open'], df['Volume']; adv5 = _adv(df, 5)
    return -1 * _rank((_ts_sum(o,5)*_ts_sum(v,5)) - _delay(_ts_sum(o,5)*_ts_sum(v,5), 10))

def alpha010(df):
    c = df['Close']; x = _signed_power(c.pct_change() < 0, 1)
    return _rank(_ts_max(x, 5) - _ts_min(x, 5))

def alpha011(df):
    c, v = df['Close'], df['Volume']; vwap5 = _vwap(df, 5)
    return (_rank(_ts_max(vwap5-c, 3)) + _rank(_ts_min(vwap5-c, 3))) * _rank(_delta(v, 3))


def alpha006(df):
    """Alpha#6: 开盘价 vs 成交量负相关（更短窗口）"""
    return -1 * _correlation(df['Open'], df['Volume'], 10)


def alpha009(df):
    """Alpha#9: 短期动量回调"""
    close = df['Close']
    delta_close = _delta(close, 1)
    cond_min = _ts_min(delta_close, 5) > 0
    cond_max = _ts_max(delta_close, 5) < 0
    ret = close.pct_change()
    out = pd.Series(0.0, index=df.index)
    out[cond_min] = delta_close[cond_min]
    out[cond_max] = delta_close[cond_max] if cond_max.any() else out
    return out.rolling(5).apply(lambda s: s.iloc[-1] if len(s) == 5 else np.nan)


def alpha012(df):
    """Alpha#12: 成交量加速"""
    return np.sign(_delta(df['Volume'], 1)) * (-1 * _delta(df['Close'], 1))


def alpha013(df):
    """Alpha#13: 量价平滑背离"""
    return -1 * _rank(_covariance(_rank(df['Close']), _rank(df['Volume']), 5))


def alpha015(df):
    """Alpha#15: 高开买压"""
    high = df['High']; close = df['Close']; volume = df['Volume']
    return -1 * _ts_sum(_rank(_correlation(_rank(high), _rank(volume), 3)), 3)


def alpha017(df):
    """Alpha#17: 价差均值回复"""
    close = df['Close']
    return -1 * (_rank(_ts_max(close, 10).rolling(20).apply(
        lambda s: s.iloc[-1], raw=True) - _delay(close, 1)) *
                 _rank(_correlation(close, df['Volume'], 10)))


def alpha020(df):
    """Alpha#20: 开盘缺口回补"""
    open_, high, low, close = df['Open'], df['High'], df['Low'], df['Close']
    return -1 * _rank(open_ - _delay(high, 1)) * \
           _rank(open_ - _delay(close, 1)) * \
           _rank(open_ - _delay(low, 1))


def alpha028(df):
    """Alpha#28: 高低点扩张"""
    high, low, close, volume = df['High'], df['Low'], df['Close'], df['Volume']
    adv20 = volume.rolling(20).mean()
    return _scale(
        (_correlation(adv20, low, 5) + ((high + low) / 2 - close)))


def alpha032(df):
    """Alpha#32: 量价突破"""
    close, volume = df['Close'], df['Volume']
    return _scale(
        _ts_sum(close.pct_change() * volume, 5) / volume.rolling(20).mean())


def alpha041(df):
    """Alpha#41: VWAP 偏离"""
    high, low, close, volume = df['High'], df['Low'], df['Close'], df['Volume']
    vwap = ((high + low + close) / 3 * volume).rolling(20).sum() / volume.rolling(20).sum()
    return _rank((vwap - close).rolling(5).max()) ** 2


def alpha101(df):
    """Alpha#101: 高开缺口回补"""
    return (_rank(df['Close'] - df['Open']) -
            _rank(_correlation(df['Close'], df['Volume'].rolling(30).mean(), 20)))

def alpha026(df): c,v=df['Close'],df['Volume']; return _ts_sum(_correlation(v.rolling(120).mean(),c,20),2)
def alpha029(df): c,v=df['Close'],df['Volume']; r=c.pct_change(); return _rank(_ts_min(_signed_power(_rank(_ts_sum(r,6)),2),5)-_ts_sum(_signed_power(_rank(_ts_sum(r,20)),3),2))*_rank(_correlation(_rank(v),_rank(c),5))
def alpha031(df): c=df['Close']; adv15=_adv(df,15); return _rank(_decay_linear(-1*_rank(_delta(c,10)),10))+_rank(-1*_delta(c,3))*np.sign(_correlation(adv15,c,12))
def alpha036(df): o,c,v=df['Open'],df['Close'],df['Volume']; adv7=_adv(df,7); return _rank(_correlation(1-_rank(o),_rank(v),7))*_rank(_decay_linear(_rank(_correlation(_ts_rank(c,8),_ts_rank(adv7,8),5)),10))
def alpha047(df): h,l,c,v=df['High'],df['Low'],df['Close'],df['Volume']; vwap5=_vwap(df,5); return _rank(1/c)*v/_adv(df,5)*(h*(h-c)/(h-l+1e-8)-vwap5)
def alpha050(df): v=df['Volume']; adv20=_adv(df,20); return -1*_ts_max(_rank(_correlation(_rank(v),_rank(adv20),5)),5)
def alpha055(df): h,l,c=df['High'],df['Low'],df['Close']; hcp=(h-c)/c; return -1*_rank(-hcp*_rank(h)*_rank(l))
def alpha056(df): c,v=df['Close'],df['Volume']; return -1*_rank(_ts_sum(c/_ts_mean(c,10),10))*_rank(_ts_sum(v/_ts_mean(v,10),10))
def alpha066(df): o,c=df['Open'],df['Close']; return -1*_rank(c.pct_change(10))*_rank(o-c)
def alpha082(df): c,v=df['Close'],df['Volume']; return -1*_rank(_ts_sum(c.pct_change(),10))*_rank(_ts_sum(v,10))
def alpha018(df): c=df['Close']; return -1*_rank(_ts_std(abs(c-_ts_mean(c,10)),5))*_rank(c.pct_change())
def alpha019(df): c=df['Close']; return -1*np.sign(_delta(c,10))*(1+_rank(_ts_sum(c.pct_change(),10)))
def alpha021(df): c,v=df['Close'],df['Volume']; adv20=_adv(df,20); cond=_ts_sum(c.pct_change(),2)>0; x=pd.Series(np.nan,index=df.index); x[cond]=_ts_sum(v/adv20,8)[cond]; x[~cond]=-1; return x.rolling(20).mean()
def alpha022(df): h,v=df['High'],df['Volume']; return -1*_delta(_correlation(h,v,5),5)*_rank(_ts_std(df['Close'].pct_change(),20))
def alpha024(df): r=df['Close'].pct_change(); return _rank(_ts_sum(r,5))-_rank(_ts_mean(r,20))
def alpha030(df): c=df['Close']; r=c.pct_change(); return (1-_rank(_ts_std(r,2)/_ts_std(r,5)))*(1-_rank(_delta(c,1)))
def alpha033(df): return _rank(-1*(1-(df['Open']/df['Close']))**2)
def alpha034(df): c,v=df['Close'],df['Volume']; x=c.pct_change()/v; return _rank(1-_rank(_ts_std(x,2)/_ts_std(x,5)))
def alpha037(df): o,c=df['Open'],df['Close']; return _rank(_correlation(_delay(o-c,1),c,10))*_rank((o-c)/o)
def alpha039(df): c,v=df['Close'],df['Volume']; adv10=_adv(df,10); return -1*_rank(_delta(c,10))*(1-_rank(_correlation(v,adv10,10)))
def alpha016(df): return -1*_rank(_covariance(_rank(df['High']),_rank(df['Volume']),5))
def alpha025(df): h=df['High']; adv5=_adv(df,5); return _rank(-1*_ts_max(_correlation(_rank(h),_rank(adv5),5),5))
def alpha035(df): c=df['Close']; adv15=_adv(df,15); return _ts_rank(_ts_rank(df['Volume']*_correlation(adv15,c,17),20),7)
def alpha038(df): h,o=df['High'],df['Open']; return -1*_rank(_ts_rank(o,10))*_rank(_delta(_delta(o,1),1))*_rank(_ts_rank((h-o)/o,10))
def alpha040(df): h,v=df['High'],df['Volume']; return -1*_rank(_ts_std(h,10))*_correlation(h,v,10)
def alpha043(df): h,c=df['High'],df['Close']; return -1*_rank(_ts_sum(_delta(_rank(1/c),1),20)/_ts_sum(h,20))
def alpha044(df): h=df['High']; adv5=_adv(df,5); return -1*_correlation(h,adv5,10)*_ts_sum(_rank(h-_ts_mean(h,20)),5)
def alpha045(df): c=df['Close']; adv5=_adv(df,5); return -1*_rank(_ts_sum(_delta(c,5),5))*_rank(_correlation(c,adv5,5))
def alpha048(df): c=df['Close']; s=np.sign(_delta(c,1)+_delta(c,-1)); return -1*_rank(s+_delay(s,1))
def alpha051(df): c=df['Close']; return -1*(_rank(_ts_sum(_delta(c,1),20))*_rank(c/_ts_mean(c,20)))
def alpha014(df): o,v=df['Open'],df['Volume']; return -1*_rank(_delta(v,3))*_correlation(o,v,10)
def alpha042(df): h,v=df['High'],df['Volume']; return -1*_rank(_ts_std(h,10))*_correlation(h,v,10)
def alpha052(df): h=df['High']; adv5=_adv(df,5); return _ts_sum(-1*_delta(_ts_min(h,2),5)*_rank(_correlation(h,adv5,5)),3)
def alpha057(df): r=df['Close'].pct_change(); return -1*_rank(_ts_sum(r,10)/_ts_sum(_ts_sum(r,2),3))
def alpha058(df): c,v=df['Close'],df['Volume']; return -1*_rank(_ts_sum(c/_ts_mean(c,10),10))*_rank(_correlation(c,v,10))
def alpha059(df): c=df['Close']; return -1*_rank(_ts_sum(c/_ts_mean(c,10),10))*_rank(_ts_sum(c/_ts_mean(c,20),10))
def alpha060(df): r,v=df['Close'].pct_change(),df['Volume']; return -1*_rank(_ts_sum(r,5)/_ts_sum(r,20))*_rank(_ts_sum(r*v,5)/_ts_sum(r*v,20))
def alpha061(df): v=df['Volume']; adv20=_adv(df,20); return -1*_rank(_ts_max(_rank(_correlation(v,adv20,5)),5))
def alpha062(df): h=df['High']; adv5=_adv(df,5); return -1*_rank(_correlation(h,adv5,10))*_rank(_ts_sum(h,10))
def alpha063(df): r=df['Close'].pct_change(); return -1*_rank(_ts_sum(r,20))*_rank(r.rolling(20).std())
def alpha064(df): c,v=df['Close'],df['Volume']; return -1*_rank(_ts_sum(c.pct_change(),10))*_rank(_correlation(c,v,10))
def alpha065(df): l=df['Low']; adv10=_adv(df,10); return -1*_rank(_ts_sum(l.pct_change(),10))*_rank(_correlation(l,adv10,10))
def alpha067(df): r=df['Close'].pct_change(); return -1*_rank(_ts_sum(r,20))*_rank(r.rolling(20).std())
def alpha068(df): h,c=df['High'],df['Close']; adv10=_adv(df,10); return -1*_rank(_ts_sum((h-c)/c,10))*_rank(_correlation(c,adv10,10))
def alpha069(df): r=df['Close'].pct_change(); return -1*_rank(_ts_sum(r,20))*_rank(r.rolling(20).std())
def alpha070(df): h,c,v=df['High'],df['Close'],df['Volume']; return -1*_rank(_ts_sum((h-c)/c,20))*_rank(_correlation(h,v,10))
def alpha071(df): o,c,v=df['Open'],df['Close'],df['Volume']; return -1*_rank(_ts_sum((o-c)/o,20))*_rank(_correlation(o,v,10))
def alpha072(df): l,v=df['Low'],df['Volume']; return -1*_rank(_ts_sum(_delta(l,1),10))*_rank(_correlation(l,v,10))
def alpha073(df): o,c=df['Open'],df['Close']; adv10=_adv(df,10); return -1*_rank(_decay_linear(_delta(o,10),10))*_rank(_decay_linear(_correlation(c,adv10,10),3))
def alpha074(df): c,v=df['Close'],df['Volume']; return -1*_rank(_correlation(c,_adv(df,30),15))*_rank(_correlation(c,v,20))
def alpha075(df): c,v=df['Close'],df['Volume']; return -1*_rank(_ts_sum(c.pct_change(),20)*_correlation(c,v,20))
def alpha076(df): o,v=df['Open'],df['Volume']; return -1*_rank(_ts_sum(_delta(o,1),10))*_rank(_correlation(o,v,10))
def alpha078(df): c,v=df['Close'],df['Volume']; adv5=_adv(df,5); return -1*_rank(_ts_sum(c.pct_change(),5))*_rank(v/adv5)
def alpha079(df): c=df['Close']; adv20=_adv(df,20); r=c.pct_change(); return -1*_rank(_ts_sum(r,20))*_rank(_ts_sum(r,10))*_rank(_correlation(c,adv20,20))
def alpha080(df): c,v=df['Close'],df['Volume']; adv5=_adv(df,5); return -1*_rank(_ts_sum(c.pct_change()*v/adv5,5))
def alpha081(df): r,v=df['Close'].pct_change(),df['Volume']; return -1*_rank(_correlation(r,_ts_sum(v,5),10))
def alpha084(df): o,c=df['Open'],df['Close']; return -1*_rank(_ts_sum((o-c)/o,10))*_rank(_correlation(o,c,20))
def alpha085(df): c,v=df['Close'],df['Volume']; r=c.pct_change(); return -1*_rank(_ts_sum(r*v,20))*_rank(_correlation(c,v,20))
def alpha086(df): h,l,c=df['High'],df['Low'],df['Close']; return -1*_rank((h-l)/c)*_rank(_ts_sum(c.pct_change(),20))
def alpha087(df): o,c=df['Open'],df['Close']; return -1*_rank(o-_delay(df['High'],5))*_rank(_ts_sum(c.pct_change(5),10))
def alpha088(df): o,c=df['Open'],df['Close']; return -1*_rank(_ts_sum(o-c,10))*_rank(_correlation(o,c,10))
def alpha089(df): o,c=df['Open'],df['Close']; return -1*_rank(_ts_sum(o-c,5)*_correlation(_rank(o),_rank(c),5))
def alpha090(df): o,c=df['Open'],df['Close']; return -1*_rank(_ts_sum(o-c,10)*_correlation(_rank(o),_rank(c),10))
def alpha091(df): h,l=df['High'],df['Low']; return -1*_rank(_ts_sum((h-l)/h,10))
def alpha094(df): r=df['Close'].pct_change(); return -1*_rank(_ts_sum(r,5)*_ts_sum(r,20))*_rank(r.rolling(20).std())
def alpha095(df): o,c=df['Open'],df['Close']; return -1*_rank(_ts_sum(o-c,20)*_ts_sum(o-c,10))*_rank(_correlation(o,c,20))
def alpha096(df): c,v=df['Close'],df['Volume']; r=c.pct_change(); return -1*_rank(_ts_sum(r*v,10))*_rank(_correlation(c,v,10))
def alpha097(df): c,v=df['Close'],df['Volume']; return -1*_rank(_ts_sum(c.pct_change(),10)*_correlation(c,v,20))
def alpha100(df): c,v=df['Close'],df['Volume']; return -1*_rank(_ts_sum(c.pct_change(),20))*_rank(_correlation(c,v,10))


# ============================================================
#  因子注册表
def alpha023(df): h,c=df['High'],df['Close']; adv20=_adv(df,20); return -1*_ts_max(_correlation(_rank(h),_rank(adv20),5),3)*(_rank(_ts_std(c.pct_change(),2))**2)
def alpha027(df): r=df['Close'].pct_change(); return _rank(_ts_sum(r,2)/_ts_std(r,20)-_ts_sum(r,5)/_ts_std(r,5))
def alpha046(df): c=df['Close']; return _rank(c/_ts_mean(c,10)-1)-_rank(c/_ts_mean(c,3)-1)
def alpha049(df): c=df['Close']; return -1*(_rank(_ts_sum(_delta(c,1),20))*_rank(c/_ts_mean(c,20)))
def alpha053(df): c=df['Close']; return -1*_delta((c-_ts_min(c,12))/(_ts_max(c,12)-_ts_min(c,12)+1e-8),5)
def alpha054(df): o,c,h=df['Open'],df['Close'],df['High']; return -1*(o-c)*(_rank(_ts_std(h,40))**2)
def alpha077(df): o,l=df['Open'],df['Low']; adv10=_adv(df,10); return -1*_rank(_ts_sum((o-l)/o,10))*_rank(_correlation(o,adv10,10))
def alpha083(df): h,l,c=df['High'],df['Low'],df['Close']; return -1*_rank((h-l)/c)*_rank(_ts_sum(c.pct_change(),10))
def alpha092(df): o,c=df['Open'],df['Close']; return -1*_rank(_ts_sum(o-c,30))*_rank(_ts_sum(o-c,10))
def alpha098(df): o,c=df['Open'],df['Close']; return -1*_rank(_ts_sum(o-c,10))*_rank(o.pct_change().rolling(10).std())
def alpha091(df): h,l=df['High'],df['Low']; return -1*_rank(_ts_sum((h-l)/h,10))
def alpha093(df): o,c=df['Open'],df['Close']; return -1*_rank(_ts_sum((o-_delay(o,10))/o,10))*_rank(_correlation(o,c,20))
def alpha099(df): h,o=df['High'],df['Open']; return -1*_rank(_ts_sum((h-o)/o,10))
# ============================================================
ALL_ALPHAS = {k: v for k, v in {
    'Alpha001': alpha001, 'Alpha002': alpha002, 'Alpha003': alpha003,
    'Alpha004': alpha004, 'Alpha005': alpha005, 'Alpha006': alpha006,
    'Alpha007': alpha007, 'Alpha008': alpha008, 'Alpha009': alpha009,
    'Alpha010': alpha010, 'Alpha011': alpha011, 'Alpha012': alpha012,
    'Alpha013': alpha013, 'Alpha015': alpha015, 'Alpha017': alpha017,
    'Alpha020': alpha020, 'Alpha028': alpha028, 'Alpha032': alpha032,
    'Alpha041': alpha041, 'Alpha101': alpha101,
    'Alpha023': alpha023, 'Alpha027': alpha027, 'Alpha046': alpha046,
    'Alpha049': alpha049, 'Alpha053': alpha053, 'Alpha054': alpha054,
    'Alpha077': alpha077, 'Alpha083': alpha083, 'Alpha092': alpha092,
    'Alpha098': alpha098,
    'Alpha018': alpha018, 'Alpha019': alpha019, 'Alpha021': alpha021,
    'Alpha022': alpha022, 'Alpha024': alpha024, 'Alpha030': alpha030,
    'Alpha033': alpha033, 'Alpha034': alpha034, 'Alpha037': alpha037,
    'Alpha039': alpha039,
    'Alpha026': alpha026, 'Alpha029': alpha029, 'Alpha031': alpha031,
    'Alpha036': alpha036, 'Alpha047': alpha047, 'Alpha050': alpha050,
    'Alpha055': alpha055, 'Alpha056': alpha056, 'Alpha066': alpha066,
    'Alpha082': alpha082,
    'Alpha018': alpha018, 'Alpha019': alpha019, 'Alpha021': alpha021,
    'Alpha022': alpha022, 'Alpha024': alpha024, 'Alpha030': alpha030,
    'Alpha033': alpha033, 'Alpha034': alpha034, 'Alpha037': alpha037,
    'Alpha039': alpha039, 'Alpha016': alpha016, 'Alpha025': alpha025,
    'Alpha035': alpha035, 'Alpha038': alpha038, 'Alpha040': alpha040,
    'Alpha043': alpha043, 'Alpha044': alpha044, 'Alpha045': alpha045,
    'Alpha048': alpha048, 'Alpha051': alpha051,
    'Alpha014': alpha014, 'Alpha042': alpha042, 'Alpha052': alpha052,
    'Alpha057': alpha057, 'Alpha058': alpha058, 'Alpha059': alpha059,
    'Alpha060': alpha060, 'Alpha061': alpha061, 'Alpha062': alpha062,
    'Alpha063': alpha063, 'Alpha064': alpha064, 'Alpha065': alpha065,
    'Alpha067': alpha067, 'Alpha068': alpha068, 'Alpha069': alpha069,
    'Alpha070': alpha070, 'Alpha071': alpha071, 'Alpha072': alpha072,
    'Alpha073': alpha073, 'Alpha074': alpha074, 'Alpha075': alpha075,
    'Alpha076': alpha076, 'Alpha078': alpha078, 'Alpha079': alpha079,
    'Alpha080': alpha080, 'Alpha081': alpha081, 'Alpha084': alpha084,
    'Alpha085': alpha085, 'Alpha086': alpha086, 'Alpha087': alpha087,
    'Alpha088': alpha088, 'Alpha089': alpha089, 'Alpha090': alpha090,
    'Alpha091': alpha091, 'Alpha094': alpha094, 'Alpha095': alpha095,
    'Alpha096': alpha096, 'Alpha097': alpha097, 'Alpha100': alpha100,
}.items() if v is not None}
