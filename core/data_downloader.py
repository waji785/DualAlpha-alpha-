# core/data_downloader.py
"""
A 股数据下载模块（baostock 唯一数据源）

功能：
    - 全量下载：OHLCV + PE/PB/PS/PCF 一次返回
    - 增量更新：仅下载缓存中每只股票的新增交易日
    - 数据校验：OHLC 一致性 / 停牌 / 异常值 / NaN
    - 错误收集

使用方式：
    from core.data_downloader import DataDownloader
    dl = DataDownloader()
    dl.run_full_download()
    dl.run_incremental_update()
    dl.print_report()
"""
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from config.settings import CACHE_DIR, FEATURE_COLS, TODAY
from core.features import construct_features, clean_data
from utils.logger import setup_logger

logger = setup_logger(__name__)

def _rebuild_one_stock(args):
    """模块级函数：重建单只股票特征"""
    cache_dir, code, start, end = args

    from core.features import construct_features, clean_data
    cache_file = os.path.join(cache_dir, f"{code}.parquet")
    if not os.path.exists(cache_file): return code, False
    import datetime as _dt
    mtime = _dt.datetime.fromtimestamp(os.path.getmtime(cache_file))
    if mtime > _dt.datetime.now() - _dt.timedelta(minutes=30):
        return code, 'skip'
    df = pd.read_parquet(cache_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = construct_features(df)
    df = clean_data(df)
    if len(df) > 50 and df['Close'].notna().sum() > 50:
        df.to_parquet(cache_file, index=False)
        return code, True
    return code, False

# ---- baostock / tushare 登录 ----
import baostock as bs
try:
    import tushare as ts
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False

_logged_in = False


def _login():
    global _logged_in
    if not _logged_in:
        lg = bs.login()
        if lg.error_code != '0':
            logger.error(f"baostock 登录失败: {lg.error_msg}")
            return False
        _logged_in = True
    return True


def _code_to_bs(stock_code):
    code = stock_code.zfill(6)
    if code.startswith('6'):
        return f"sh.{code}"
    if code.startswith(('0', '3')):
        return f"sz.{code}"
    return None


_BS_FIELDS = ("date,open,high,low,close,preclose,volume,amount,"
              "adjustflag,turn,tradestatus,pctChg,"
              "peTTM,pbMRQ,psTTM,pcfNcfTTM,isST")

# 列映射
_RENAME_MAP = {
    'date': 'Date', 'open': 'Open', 'high': 'High',
    'low': 'Low', 'close': 'Close', 'preclose': 'PreClose',
    'volume': 'Volume', 'amount': 'Amount',
    'turn': 'Turn', 'tradestatus': 'TradeStatus',
    'pctChg': 'PctChg',
    'peTTM': 'PeTTM', 'pbMRQ': 'PbMRQ',
    'psTTM': 'PsTTM', 'pcfNcfTTM': 'PcfNcfTTM',
    'isST': 'isST',
}

# ---- Tushare 下载 ----
def _ts_download(stock_code, start, end):
    import time
    time.sleep(0.4)  # 限速：单线程下每只股票约 0.4s，避免触发 tushare 限流
    """tushare 日线 + 基本面"""
    from config.settings import TUSHARE_TOKEN
    if not _TS_AVAILABLE or TUSHARE_TOKEN == "your_token_here":
        return None
    pro = ts.pro_api(TUSHARE_TOKEN, timeout=10)
    s = stock_code.zfill(6)
    ts_code = f"{s}.SH" if s[0] == '6' else f"{s}.SZ"
    s_date = start.replace('-', '')
    e_date = end.replace('-', '')
    try:
        df1 = pro.daily(ts_code=ts_code, start_date=s_date, end_date=e_date)
        if df1 is None or len(df1) == 0:
            import time; time.sleep(0.05)  # 限速退避
            return None
        # 复权因子（后复权 = 价格 × adj_factor）
        try:
            df_adj = pro.adj_factor(ts_code=ts_code, start_date=s_date, end_date=e_date)
            if df_adj is not None and len(df_adj) > 0:
                df1 = df1.merge(df_adj[['trade_date', 'adj_factor']], on='trade_date', how='left')
                af = df1['adj_factor'].fillna(1.0)
                for c in ['open', 'high', 'low', 'close', 'pre_close']:
                    df1[c] = (df1[c] * af).round(4)
        except Exception:
            pass
        df2 = pro.daily_basic(ts_code=ts_code, start_date=s_date, end_date=e_date)
        cols = ['trade_date', 'pe_ttm', 'pb', 'ps_ttm', 'turnover_rate']
        if df2 is not None and len(df2) > 0:
            df = df1.merge(df2[cols], on='trade_date', how='left')
        else:
            df = df1.copy()
            for c in ['pe_ttm', 'pb', 'ps_ttm', 'turnover_rate']:
                df[c] = np.nan
        df['date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        df['preclose'] = df['pre_close']
        df['volume'] = df['vol']
        df['pctChg'] = df['pct_chg']
        df['peTTM'] = df['pe_ttm']
        df['pbMRQ'] = df['pb']
        df['psTTM'] = df['ps_ttm']
        df['PcfNcfTTM'] = np.nan
        df['turn'] = df['turnover_rate']
        df['tradestatus'] = 1.0
        df['isST'] = 0
        df['amount'] = df['amount'].fillna(0.0)
        return df[['date', 'open', 'high', 'low', 'close', 'preclose',
                    'volume', 'amount', 'turn', 'tradestatus',
                    'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'PcfNcfTTM', 'isST']]
    except Exception as e:
        logger.warning(f"tushare {stock_code}: {e}")
        return None

_FLOAT_COLS = ['Open', 'High', 'Low', 'Close', 'PreClose',
               'Volume', 'Amount', 'Turn', 'PctChg',
               'PeTTM', 'PbMRQ', 'PsTTM', 'PcfNcfTTM']


# ============================================================
#  DataDownloader
# ============================================================

class DataDownloader:
    def __init__(self, start="2015-01-01", end=TODAY, cache_dir=None):
        self.start = start
        self.end = end
        self.cache_dir = cache_dir or CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.stats = {
            'success': 0, 'failed': 0, 'skipped': 0,
            'validated_ok': 0, 'validated_warn': 0,
        }
        self.errors = []
        self.fundamental_cache = None  # {ts_code: 季度财务 DataFrame}

    # ============================================================
    #  Stock List (baostock)
    # ============================================================

    def get_stock_list(self, exclude_st=True, exclude_north=True):
        # 优先 tushare
        try:
            from config.settings import TUSHARE_TOKEN
            if _TS_AVAILABLE and TUSHARE_TOKEN != "your_token_here":
                pro = ts.pro_api(TUSHARE_TOKEN, timeout=10)
                df = pro.stock_basic(exchange='', list_status='L',
                                     fields='ts_code,symbol,name,area,industry,list_date')
                if df is not None and len(df) > 0:
                    df = df[~df['name'].str.contains('ST', na=False)]
                    df = df[~df['symbol'].str.startswith(('8', '9', '4'))]  # 排除北交所/新三板
                    df['code'] = df['symbol'].str.zfill(6)
                    return df[['code', 'name', 'industry']]
        except Exception as e:
            logger.warning(f"tushare stock list: {e}")

        if not _login():
            return None
        try:
            rs = bs.query_stock_basic()
            if rs.error_code != '0':
                logger.error(f"获取股票列表失败: {rs.error_msg}")
                return None
            rows = []
            while (rs.error_code == '0') & rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
            df.rename(columns={'code': 'raw_code', 'code_name': 'name'}, inplace=True)

            if 'type' in df.columns:
                df = df[df['type'] == '1'].copy()
            if 'status' in df.columns:
                df = df[df['status'] == '1'].copy()

            df['code'] = df['raw_code'].str.replace('sh.', '', regex=False)
            df['code'] = df['code'].str.replace('sz.', '', regex=False)
            df['code'] = df['code'].str.zfill(6)

            if exclude_north:
                north = ('920','430','830','870','871','872',
                         '873','874','875','876','877','878','879')
                df = df[~df['code'].str.startswith(north)]
            if exclude_st:
                before = len(df)
                df = df[~df['name'].str.contains('ST|\\*ST', na=False, case=False)]
                logger.info(f"过滤 ST {before - len(df)} 只")

            df = df[['code', 'name']].drop_duplicates('code').reset_index(drop=True)
            logger.info(f"股票列表: {len(df)} 只")
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return None

    # ============================================================
    #  Single Stock Download
    # ============================================================

    def _download_raw(self, stock_code, start=None, end=None):
        """
        baostock 日线下载，返回包含 OHLCV + PE/PB/PS/PCF 的 DataFrame。
        失败返回 None，错误信息存入 self._last_error。
        """
        import socket
        socket.setdefaulttimeout(15)  # baostock 超时保护（避免卡住下载）
        bs_symbol = _code_to_bs(stock_code)
        if bs_symbol is None:
            self._last_error = "代码无法转为 baostock 格式"
            return None
        if not _login():
            self._last_error = "baostock 登录失败"
            return None
        start = start or self.start
        end = end or self.end
        try:
            rs = bs.query_history_k_data_plus(
                code=bs_symbol, fields=_BS_FIELDS,
                start_date=start, end_date=end,
                frequency="d", adjustflag="2"
            )
            if rs.error_code != '0':
                self._last_error = rs.error_msg
                return None
            rows = []
            while (rs.error_code == '0') & rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                self._last_error = "无数据返回"
                return None
            df = pd.DataFrame(rows, columns=rs.fields)
            df.rename(columns=_RENAME_MAP, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

            for c in _FLOAT_COLS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            if 'TradeStatus' in df.columns:
                df['TradeStatus'] = pd.to_numeric(df['TradeStatus'], errors='coerce').fillna(1).astype(int)
            for vcol in ['PeTTM', 'PbMRQ', 'PsTTM', 'PcfNcfTTM']:
                if vcol in df.columns:
                    df[vcol] = df[vcol].fillna(0.0)
            # Volume NaN → 0（baostock 停牌日留空）
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].fillna(0.0)
            return df
        except Exception as e:
            self._last_error = str(e)
            return None

    def download_and_save(self, stock_code, start=None, end=None):
        """下载 → 校验 → 特征 → 缓存（仅 tushare，失败重试一次后跳过）"""
        df = _ts_download(stock_code, start or self.start, end or self.end)
        if df is None:
            # tushare 偶发超时，重试一次；不再回退 baostock（baostock 会卡死）
            df = _ts_download(stock_code, start or self.start, end or self.end)
        if df is not None:
            df.rename(columns=_RENAME_MAP, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            for c in _FLOAT_COLS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            if 'TradeStatus' in df.columns:
                df['TradeStatus'] = pd.to_numeric(df['TradeStatus'], errors='coerce').fillna(1).astype(int)
            if 'isST' in df.columns:
                df['isST'] = pd.to_numeric(df['isST'], errors='coerce').fillna(0).astype(int)
        if df is None:
            self.stats['failed'] += 1
            reason = getattr(self, '_last_error', '未知错误')
            self.errors.append((stock_code, reason))
            logger.warning(f"❌ 下载失败: {stock_code} — {reason}")
            return False

        # 校验
        warnings = self.validate_data(df, stock_code)
        if warnings:
            self.stats['validated_warn'] += 1
        else:
            self.stats['validated_ok'] += 1

        # ---- 另类数据 merge（财务 + 融资融券 + 龙虎榜 + 大宗交易）----
        df = self._merge_alt_data(df, stock_code)

        # 特征
        df = construct_features(df)
        df = clean_data(df)

        cache_file = os.path.join(self.cache_dir, f"{stock_code}.parquet")
        df.to_parquet(cache_file, index=False)
        self.stats['success'] += 1
        return True

    def _merge_alt_data(self, df, stock_code):
        """merge 财务 + 融资融券 + 龙虎榜 + 大宗交易（全量下载和增量更新共用）"""
        s = stock_code.zfill(6)
        ts_code = f"{s}.SH" if s[0] == '6' else f"{s}.SZ"
        # 财务指标（ann_date 前视修复）
        if self.fundamental_cache:
            fund = self.fundamental_cache.get(ts_code)
            if fund is not None and len(fund) > 0:
                f = fund.copy()
                if 'ann_date' in f.columns:
                    f['Date'] = pd.to_datetime(f['ann_date'].fillna(f['end_date']))
                else:
                    f['Date'] = pd.to_datetime(f['end_date'])
                f = f[['Date', 'roe', 'grossprofit_margin', 'netprofit_margin', 'debt_to_assets']].copy()
                f.columns = ['Date', 'ROE', 'GrossMargin', 'NetMargin', 'DebtRatio']
                f = f.sort_values('Date').drop_duplicates('Date', keep='last')
                df = df.merge(f, on='Date', how='left')
                for c in ['ROE', 'GrossMargin', 'NetMargin', 'DebtRatio']:
                    df[c] = df[c].ffill().fillna(0.0)
        # 融资融券
        m = self.margin_cache.get(ts_code) if getattr(self, 'margin_cache', None) else None
        if m is not None and len(m) > 0:
            mm = m.copy()
            mm['Date'] = pd.to_datetime(mm['trade_date'])
            mm = mm.sort_values('Date').drop_duplicates('Date', keep='last')
            df = df.merge(mm[['Date', 'rzye']], on='Date', how='left')
            bal = df['rzye'].ffill()
            df['Margin_Chg_5d'] = bal.pct_change(5).fillna(0.0)
            df['Margin_Chg_20d'] = bal.pct_change(20).fillna(0.0)
            df.drop(columns=['rzye'], inplace=True)
        else:
            df['Margin_Chg_5d'] = 0.0
            df['Margin_Chg_20d'] = 0.0
        # 龙虎榜
        lhb = self.lhb_cache.get(ts_code) if getattr(self, 'lhb_cache', None) else None
        if lhb is not None and len(lhb) > 0:
            ll = lhb.copy()
            ll['Date'] = pd.to_datetime(ll['trade_date'])
            ll = ll.sort_values('Date').drop_duplicates('Date', keep='last')
            df = df.merge(ll[['Date', 'net_buy']], on='Date', how='left')
            df['LHB_Net_Buy_20d'] = df['net_buy'].fillna(0.0).rolling(20, min_periods=1).sum()
            df.drop(columns=['net_buy'], inplace=True)
        else:
            df['LHB_Net_Buy_20d'] = 0.0
        # 大宗交易
        blk = self.block_cache.get(ts_code) if getattr(self, 'block_cache', None) else None
        if blk is not None and len(blk) > 0:
            bb = blk.copy()
            bb['Date'] = pd.to_datetime(bb['trade_date'])
            bb = bb.sort_values('Date').drop_duplicates('Date', keep='last')
            df = df.merge(bb[['Date', 'amount']], on='Date', how='left')
            df['Block_Amount_20d'] = df['amount'].fillna(0.0).rolling(20, min_periods=1).sum()
            df.drop(columns=['amount'], inplace=True)
        else:
            df['Block_Amount_20d'] = 0.0
        return df

    # ============================================================
    #  Validation
    # ============================================================

    def validate_data(self, df, stock_code):
        warnings = []
        if df is None or len(df) < 10:
            warnings.append("数据不足 10 条")
            return warnings

        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if any(c not in df.columns for c in required):
            warnings.append(f"缺失关键列")
            return warnings

        o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
        if (h < o).any() or (h < c).any():
            warnings.append("High < Open/Close")
        if (l > o).any() or (l > c).any():
            warnings.append("Low > Open/Close")
        if (h < l).any():
            warnings.append("High < Low")
        for col in required:
            if (df[col] < 0).any():
                warnings.append(f"{col} 含负值")

        # 停牌（60 天窗口，正常现象，降级为 debug）
        flat = (c.diff().abs() < 1e-6) & (df['Volume'].diff().abs() < 1e-6)
        if (flat.rolling(60).sum() >= 60).any():
            warnings.append("疑似停牌")

        # Volume NaN（baostock 停牌日留空）
        if df['Volume'].isna().any():
            df['Volume'] = df['Volume'].fillna(0.0)
            if (df['Volume'] == 0).sum() > len(df) * 0.3:
                warnings.append("成交量缺失 >30%")

        if warnings:
            for w in warnings:
                logger.warning(f"[校验] {stock_code}: {w}")
        return warnings

    # ============================================================
    #  Batch
    # ============================================================

    def _cache_last_date(self, stock_code):
        cache_file = os.path.join(self.cache_dir, f"{stock_code}.parquet")
        if not os.path.exists(cache_file):
            return None
        try:
            df = pd.read_parquet(cache_file, columns=['Date'])
            if df.empty:
                return None
            last = pd.to_datetime(df['Date'].max())
            return None if pd.isna(last) else last
        except Exception:
            return None

    def _load_fundamentals(self, codes):
        """加载财务指标：优先磁盘缓存，否则串行下载（逐股票 + 限速），返回 {ts_code: 季度财务 DataFrame}"""
        import time
        from config.settings import TUSHARE_TOKEN
        if not _TS_AVAILABLE or TUSHARE_TOKEN == "your_token_here":
            logger.warning("tushare 不可用，跳过财务数据")
            return {}

        # 磁盘缓存：下载一次后持久化，重开命令行直接加载
        cache_file = os.path.join(self.cache_dir, "_fundamental_cache.parquet")
        if os.path.exists(cache_file):
            try:
                full = pd.read_parquet(cache_file)
                cache = {}
                for ts_code, grp in full.groupby('ts_code'):
                    cache[ts_code] = grp.sort_values('end_date')
                logger.info(f"财务指标从缓存加载: {len(cache)} 只")
                return cache
            except Exception:
                pass

        pro = ts.pro_api(TUSHARE_TOKEN, timeout=10)
        s_d = self.start.replace('-', '')[:4] + '0101'
        e_d = self.end.replace('-', '')
        cache = {}
        fields = 'ts_code,end_date,ann_date,roe,grossprofit_margin,netprofit_margin,debt_to_assets'
        for code in tqdm(codes, desc='下载财务指标'):
            s = code.zfill(6)
            ts_code = f"{s}.SH" if s[0] == '6' else f"{s}.SZ"
            try:
                df = pro.fina_indicator(ts_code=ts_code, start_date=s_d, end_date=e_d, fields=fields)
                if df is not None and len(df) > 0:
                    cache[ts_code] = df
            except Exception:
                pass
            time.sleep(0.55)  # 限速约 100 次/分钟

        # 持久化到磁盘，避免重开命令行后重复下载
        if cache:
            try:
                full = pd.concat(cache.values(), ignore_index=True)
                full.to_parquet(cache_file, index=False)
                logger.info(f"财务指标已缓存到磁盘: {cache_file}")
            except Exception:
                pass

        logger.info(f"财务指标加载: {len(cache)}/{len(codes)} 只")
        return cache

    def _load_margin(self):
        """加载融资融券数据（增量更新：缓存存在则下载最新几天合并），返回 dict{ts_code: DataFrame}"""
        from config.settings import TUSHARE_TOKEN
        if not _TS_AVAILABLE or TUSHARE_TOKEN == "your_token_here":
            logger.warning("tushare 不可用，跳过融资融券数据")
            return {}
        cache_file = os.path.join(self.cache_dir, "_margin_cache.parquet")
        full = None
        if os.path.exists(cache_file):
            try:
                full = pd.read_parquet(cache_file)
            except Exception:
                full = None
        # 增量起始日期：缓存最新日期 + 1 天，否则从年初
        if full is not None and len(full) > 0:
            last_date = str(full['trade_date'].max())
            start_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y%m%d')
        else:
            start_date = self.start.replace('-', '')[:4] + '0101'
        end_date = self.end.replace('-', '')
        # 增量下载（仅下载缓存之后的新交易日）
        pro = ts.pro_api(TUSHARE_TOKEN, timeout=10)
        cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
        trade_dates = cal['cal_date'].tolist()
        if trade_dates:
            frames = []
            for td in tqdm(trade_dates, desc='下载融资融券'):
                try:
                    df = pro.margin_detail(trade_date=td)
                    if df is not None and len(df) > 0:
                        frames.append(df[['trade_date', 'ts_code', 'rzye']])
                except Exception:
                    pass
                time.sleep(0.3)
            if frames:
                new_df = pd.concat(frames, ignore_index=True)
                full = pd.concat([full, new_df], ignore_index=True).drop_duplicates(
                    ['ts_code', 'trade_date'], keep='last') if full is not None else new_df
                full.to_parquet(cache_file, index=False)
                logger.info(f"融资融券已缓存: {len(full)} 行")
        if full is None:
            return {}
        margin_dict = {}
        for ts_code, grp in full.groupby('ts_code'):
            margin_dict[ts_code] = grp[['trade_date', 'rzye']].sort_values('trade_date')
        logger.info(f"融资融券加载: {len(margin_dict)} 只")
        return margin_dict

    def _load_lhb(self):
        """下载龙虎榜机构净买入（增量更新），返回 dict{ts_code: DataFrame(trade_date, net_buy)}"""
        from config.settings import TUSHARE_TOKEN
        if not _TS_AVAILABLE or TUSHARE_TOKEN == "your_token_here":
            return {}
        cache_file = os.path.join(self.cache_dir, "_lhb_cache.parquet")
        full = None
        if os.path.exists(cache_file):
            try:
                full = pd.read_parquet(cache_file)
            except Exception:
                full = None
        if full is not None and len(full) > 0:
            last_date = str(full['trade_date'].max())
            start_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y%m%d')
        else:
            start_date = self.start.replace('-', '')[:4] + '0101'
        end_date = self.end.replace('-', '')
        pro = ts.pro_api(TUSHARE_TOKEN, timeout=10)
        cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
        trade_dates = cal['cal_date'].tolist()
        if trade_dates:
            frames = []
            for td in tqdm(trade_dates, desc='下载龙虎榜'):
                try:
                    df = pro.top_inst(trade_date=td)
                    if df is not None and len(df) > 0:
                        agg = df.groupby('ts_code')['net_buy'].sum().reset_index()
                        agg['trade_date'] = td
                        frames.append(agg)
                except Exception:
                    pass
                time.sleep(0.3)
            if frames:
                new_df = pd.concat(frames, ignore_index=True)
                full = pd.concat([full, new_df], ignore_index=True).drop_duplicates(
                    ['ts_code', 'trade_date'], keep='last') if full is not None else new_df
                full.to_parquet(cache_file, index=False)
                logger.info(f"龙虎榜已缓存: {len(full)} 行")
        if full is None:
            return {}
        lhb_dict = {}
        for ts_code, grp in full.groupby('ts_code'):
            lhb_dict[ts_code] = grp[['trade_date', 'net_buy']].sort_values('trade_date')
        logger.info(f"龙虎榜加载: {len(lhb_dict)} 只")
        return lhb_dict

    def _load_block(self):
        """下载大宗交易金额（增量更新），返回 dict{ts_code: DataFrame(trade_date, amount)}"""
        from config.settings import TUSHARE_TOKEN
        if not _TS_AVAILABLE or TUSHARE_TOKEN == "your_token_here":
            return {}
        cache_file = os.path.join(self.cache_dir, "_block_cache.parquet")
        full = None
        if os.path.exists(cache_file):
            try:
                full = pd.read_parquet(cache_file)
            except Exception:
                full = None
        if full is not None and len(full) > 0:
            last_date = str(full['trade_date'].max())
            start_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y%m%d')
        else:
            start_date = self.start.replace('-', '')[:4] + '0101'
        end_date = self.end.replace('-', '')
        pro = ts.pro_api(TUSHARE_TOKEN, timeout=10)
        cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
        trade_dates = cal['cal_date'].tolist()
        if trade_dates:
            frames = []
            for td in tqdm(trade_dates, desc='下载大宗交易'):
                try:
                    df = pro.block_trade(trade_date=td)
                    if df is not None and len(df) > 0:
                        agg = df.groupby('ts_code')['amount'].sum().reset_index()
                        agg['trade_date'] = td
                        frames.append(agg)
                except Exception:
                    pass
                time.sleep(0.3)
            if frames:
                new_df = pd.concat(frames, ignore_index=True)
                full = pd.concat([full, new_df], ignore_index=True).drop_duplicates(
                    ['ts_code', 'trade_date'], keep='last') if full is not None else new_df
                full.to_parquet(cache_file, index=False)
                logger.info(f"大宗交易已缓存: {len(full)} 行")
        if full is None:
            return {}
        block_dict = {}
        for ts_code, grp in full.groupby('ts_code'):
            block_dict[ts_code] = grp[['trade_date', 'amount']].sort_values('trade_date')
        logger.info(f"大宗交易加载: {len(block_dict)} 只")
        return block_dict

    def run_full_download(self, codes=None):
        if codes is None:
            stock_df = self.get_stock_list(exclude_st=True, exclude_north=True)
            if stock_df is None:
                logger.error("无法获取股票列表")
                return
            codes = stock_df['code'].tolist()

        # 跳过已缓存且新鲜的股票（3 天容差，baostock 无当日数据）
        end_dt = pd.to_datetime(self.end)
        cutoff = end_dt - pd.Timedelta(days=3)
        to_download = []
        for code in codes:
            last = self._cache_last_date(code)
            if last is not None and last >= cutoff:
                self.stats['skipped'] += 1
            else:
                to_download.append(code)

        logger.info(f"全量下载: {len(to_download)} 需下载, "
                    f"{self.stats['skipped']} 已缓存 ({self.start} ~ {self.end})")

        # 批量加载财务指标 + 融资融券 + 龙虎榜 + 大宗交易（供 download_and_save merge）
        if to_download:
            self.fundamental_cache = self._load_fundamentals(to_download)
            self.margin_cache = self._load_margin()
            self.lhb_cache = self._load_lhb()
            self.block_cache = self._load_block()
        else:
            self.margin_cache = {}
            self.lhb_cache = {}
            self.block_cache = {}

        from concurrent.futures import ThreadPoolExecutor, as_completed
        # 单线程 + 限速：tushare 2000 积分限速约 100 次/分钟，多线程会触发限流
        N_WORKERS = 1
        logger.info(f"串行下载: {N_WORKERS} 线程（避免限流）")
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(self.download_and_save, code, self.start, self.end): code
                       for code in to_download}
            for f in tqdm(as_completed(futures), total=len(futures), desc="下载"):
                f.result()
        logger.info("全量下载完成")
        self._print_latest_date()

    def _print_latest_date(self):
        """打印缓存数据的最新日期"""
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.parquet')]
        if not files: return
        dates = []
        for f in files[:50]:  # 抽样 50 只
            try:
                df = pd.read_parquet(os.path.join(self.cache_dir, f))
                if 'Date' in df.columns:
                    dates.append(pd.to_datetime(df['Date'].max()))
            except: pass
        if dates:
            latest = max(dates).strftime('%Y-%m-%d')
            logger.info(f"数据最新日期: {latest}")

    def run_incremental_update(self, codes=None, force_rebuild=False):
        """增量更新，force_rebuild=True 时跳过日期检查，全部重建特征"""
        if codes is None:
            codes = [f.replace('.parquet', '')
                     for f in os.listdir(self.cache_dir) if f.endswith('.parquet')]
            logger.info(f"扫描到 {len(codes)} 只已缓存股票")

        cutoff = pd.to_datetime(self.end)  # 缓存未到今天就更新
        need_update = []
        rebuild_only = []  # 不需下载但需重建特征的
        for code in codes:
            last = self._cache_last_date(code)
            if force_rebuild:
                rebuild_only.append(code)
                continue
            if last is not None and not pd.isna(last) and last >= cutoff:
                self.stats['skipped'] += 1
                continue
            if last is None or pd.isna(last):
                next_date = self.start
            else:
                next_date = (last + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            need_update.append((code, next_date))

        logger.info(f"需更新: {len(need_update)}, 重建: {len(rebuild_only)}")
        if not need_update and not rebuild_only:
            logger.info("所有股票已是最新")
            return

        # 加载另类数据 cache（供增量更新 merge，避免抹掉财务/融资融券/龙虎榜/大宗交易因子）
        if need_update:
            self.fundamental_cache = self._load_fundamentals([c for c, _ in need_update])
            self.margin_cache = self._load_margin()
            self.lhb_cache = self._load_lhb()
            self.block_cache = self._load_block()

        # 增量下载截止日：17:00 后用今天（数据已发布），否则昨天
        now = pd.Timestamp.now()
        if now.hour >= 17:
            download_end = pd.to_datetime(self.end).strftime('%Y-%m-%d')
        else:
            download_end = (pd.to_datetime(self.end) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        for code, from_date in tqdm(need_update, desc="增量更新"):
            if from_date > download_end:
                self.stats['skipped'] += 1; continue
            new_df = _ts_download(code, from_date, download_end)
            if new_df is not None:
                new_df.rename(columns=_RENAME_MAP, inplace=True)
                new_df['Date'] = pd.to_datetime(new_df['Date'])
            else:
                new_df = self._download_raw(code, from_date, download_end)
            if new_df is None or new_df.empty:
                self.stats['failed'] += 1
                reason = getattr(self, '_last_error', '未知')
                self.errors.append((code, f"{reason} ({from_date}~{self.end})"))
                logger.warning(f"❌ 增量失败: {code} — {reason}")
                continue

            cache_file = os.path.join(self.cache_dir, f"{code}.parquet")
            raw_cols = ['Date'] + _FLOAT_COLS + ['TradeStatus', 'isST']
            raw_cols = [c for c in raw_cols if c in new_df.columns]

            if os.path.exists(cache_file):
                old = pd.read_parquet(cache_file)
                old['Date'] = pd.to_datetime(old['Date'])
                old_raw = old[[c for c in raw_cols if c in old.columns]].copy()
                combined = pd.concat([old_raw, new_df[raw_cols]], ignore_index=True)
            else:
                combined = new_df[raw_cols].copy()

            combined = combined.drop_duplicates('Date', keep='last')
            combined = combined.sort_values('Date').reset_index(drop=True)
            if 'isST' in combined.columns:
                combined['isST'] = pd.to_numeric(combined['isST'], errors='coerce').fillna(0).astype(int)
            if 'TradeStatus' in combined.columns:
                combined['TradeStatus'] = pd.to_numeric(combined['TradeStatus'], errors='coerce').fillna(1).astype(int)

            warnings = self.validate_data(combined, code)
            if warnings:
                self.stats['validated_warn'] += 1
            else:
                self.stats['validated_ok'] += 1

            # merge 另类数据（财务+融资融券+龙虎榜+大宗交易），避免增量更新抹掉这些因子
            combined = self._merge_alt_data(combined, code)
            combined = construct_features(combined)
            combined = clean_data(combined)
            combined.to_parquet(cache_file, index=False)
            self.stats['success'] += 1
            time.sleep(0.05)

        # ---- 强制重建 ----
        if rebuild_only:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            logger.info(f"特征重建: {len(rebuild_only)} 只 (4线程)")

            with ThreadPoolExecutor(max_workers=4) as pool:
                tasks = [(self.cache_dir, c, self.start, self.end) for c in rebuild_only]
                futures = {pool.submit(_rebuild_one_stock, t): t[1] for t in tasks}
                for f in tqdm(as_completed(futures), total=len(futures), desc="重建特征"):
                    code, ok = f.result()
                    if ok is True: self.stats['success'] += 1
                    elif ok == 'skip': self.stats['skipped'] += 1
                    else: self.stats['failed'] += 1
            for code, reason in self.errors[:10]:
                print(f"    {code}: {reason}")
        print("=" * 40)
