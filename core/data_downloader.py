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
    """tushare 日线 + 基本面"""
    from config.settings import TUSHARE_TOKEN
    if not _TS_AVAILABLE or TUSHARE_TOKEN == "your_token_here":
        return None
    pro = ts.pro_api(TUSHARE_TOKEN)
    s = stock_code.zfill(6)
    ts_code = f"{s}.SH" if s[0] == '6' else f"{s}.SZ"
    s_date = start.replace('-', '')
    e_date = end.replace('-', '')
    try:
        df1 = pro.daily(ts_code=ts_code, start_date=s_date, end_date=e_date)
        if df1 is None or len(df1) == 0:
            return None
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
    def __init__(self, start="2018-01-01", end=TODAY, cache_dir=None):
        self.start = start
        self.end = end
        self.cache_dir = cache_dir or CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.stats = {
            'success': 0, 'failed': 0, 'skipped': 0,
            'validated_ok': 0, 'validated_warn': 0,
        }
        self.errors = []

    # ============================================================
    #  Stock List (baostock)
    # ============================================================

    def get_stock_list(self, exclude_st=True, exclude_north=True):
        # 优先 tushare
        try:
            from config.settings import TUSHARE_TOKEN
            if _TS_AVAILABLE and TUSHARE_TOKEN != "your_token_here":
                pro = ts.pro_api(TUSHARE_TOKEN)
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
        """下载 → 校验 → 特征 → 缓存（优先 tushare，回退 baostock）"""
        df = _ts_download(stock_code, start or self.start, end or self.end)
        if df is None:
            df = self._download_raw(stock_code, start, end)
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

        # 特征
        df = construct_features(df)
        df = clean_data(df)

        # ---- 额外数据（北向/财务/股东）----
        try:
            from core.extra_data import fetch_extra_data
            extra = fetch_extra_data(stock_code, self.start, self.end)
            if extra is not None and len(extra) > 0:
                df = pd.merge(df, extra, on='Date', how='left')
                df.fillna({col: 0.0 for col in extra.columns if col != 'Date'}, inplace=True)
        except Exception:
            logger.debug(f"额外数据失败 {stock_code}")

        cache_file = os.path.join(self.cache_dir, f"{stock_code}.parquet")
        df.to_parquet(cache_file, index=False)
        self.stats['success'] += 1
        return True

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
            return pd.to_datetime(df['Date'].max())
        except Exception:
            return None

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

        from concurrent.futures import ThreadPoolExecutor, as_completed
        N_WORKERS = min(4, max(1, len(to_download) // 1000 + 1))
        logger.info(f"并行下载: {N_WORKERS} 线程")
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(self.download_and_save, code, self.start, self.end): code
                       for code in to_download}
            for f in tqdm(as_completed(futures), total=len(futures), desc="下载"):
                f.result()
        logger.info("全量下载完成")

    def run_incremental_update(self, codes=None, force_rebuild=False):
        """增量更新，force_rebuild=True 时跳过日期检查，全部重建特征"""
        if codes is None:
            codes = [f.replace('.parquet', '')
                     for f in os.listdir(self.cache_dir) if f.endswith('.parquet')]
            logger.info(f"扫描到 {len(codes)} 只已缓存股票")

        cutoff = pd.to_datetime(self.end) - pd.Timedelta(days=3)
        need_update = []
        rebuild_only = []  # 不需下载但需重建特征的
        for code in codes:
            last = self._cache_last_date(code)
            if last is not None and last >= cutoff:
                if force_rebuild:
                    rebuild_only.append(code)
                    continue
                else:
                    self.stats['skipped'] += 1
                    continue
            next_date = (last + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            need_update.append((code, next_date))

        logger.info(f"需更新: {len(need_update)}, 重建: {len(rebuild_only)}")
        if not need_update and not rebuild_only:
            logger.info("所有股票已是最新")
            return

        for code, from_date in tqdm(need_update, desc="增量更新"):
            new_df = _ts_download(code, from_date, self.end)
            if new_df is None:
                new_df = self._download_raw(code, from_date, self.end)
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

            combined = construct_features(combined)
            combined = clean_data(combined)
            combined.to_parquet(cache_file, index=False)
            self.stats['success'] += 1
            time.sleep(0.05)

        # ---- 强制重建：只重建特征不下载 ----
        if rebuild_only:
            logger.info(f"特征重建: {len(rebuild_only)} 只 (无数据下载)")
            for code in tqdm(rebuild_only, desc="重建特征"):
                cache_file = os.path.join(self.cache_dir, f"{code}.parquet")
                if not os.path.exists(cache_file):
                    continue
                df = pd.read_parquet(cache_file)
                df['Date'] = pd.to_datetime(df['Date'])
                raw_cols = ['Date'] + _FLOAT_COLS + ['TradeStatus', 'isST']
                raw_cols = [c for c in raw_cols if c in df.columns]
                df = df[raw_cols].copy()
                df = construct_features(df)
                df = clean_data(df)
                if not self.validate_data(df, code):
                    df.to_parquet(cache_file, index=False)
                    self.stats['success'] += 1
                else:
                    self.stats['failed'] += 1

        logger.info("增量更新完成")

    # ============================================================
    #  Report
    # ============================================================

    def print_report(self):
        total = sum(v for k, v in self.stats.items() if k in ('success', 'failed', 'skipped'))
        print("\n" + "=" * 40)
        print("📊 下载报告")
        print("=" * 40)
        print(f"  总计:      {total}")
        print(f"  成功:      {self.stats['success']}")
        print(f"  失败:      {self.stats['failed']}")
        print(f"  跳过:      {self.stats['skipped']}")
        print(f"  校验通过:  {self.stats['validated_ok']}")
        print(f"  校验警告:  {self.stats['validated_warn']}")
        if self.errors:
            print(f"\n  失败 ({len(self.errors)}):")
            for code, reason in self.errors[:10]:
                print(f"    {code}: {reason}")
        print("=" * 40)
