# core/data_loader.py
"""
缓存读取模块：从本地 parquet 加载已下载的数据。
回测 / 训练时使用此模块，不会触发任何网络下载。

数据下载请使用 core/data_downloader.py 或 scripts/download_data.py
"""
import os
import pandas as pd
import numpy as np

from config.settings import CACHE_DIR, FEATURE_COLS
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 北交所 & 三板前缀（用于过滤）
_NORTH_PREFIXES = ('920', '430', '830', '870', '871', '872',
                   '873', '874', '875', '876', '877', '878', '879')


# ============================================================
#  缓存读写
# ============================================================

def get_cache_path(stock_code):
    return os.path.join(CACHE_DIR, f"{stock_code}.parquet")


def load_from_cache(stock_code):
    """从缓存加载单只股票的特征数据（含 Target），失败返回 None"""
    cache_file = get_cache_path(stock_code)
    if not os.path.exists(cache_file):
        return None
    try:
        df = pd.read_parquet(cache_file)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        required_cols = set(['Date'] + FEATURE_COLS + ['Target_Price', 'Target_Direction'])
        if required_cols.issubset(set(df.columns)):
            return df
        else:
            logger.warning(f"{stock_code} 缓存缺少特征列，请重新下载")
            return None
    except Exception as e:
        logger.error(f"读取缓存失败 {stock_code}: {e}")
        return None


def save_to_cache(stock_code, df):
    """保存数据到缓存（供 data_downloader 使用）"""
    try:
        df.to_parquet(get_cache_path(stock_code), index=False)
    except Exception as e:
        logger.error(f"保存缓存失败 {stock_code}: {e}")


# ============================================================
#  股票列表（baostock，单次轻量 API）
# ============================================================

def get_stock_list(exclude_st=True, exclude_north=True):
    """
    获取 A 股股票列表：优先本地缓存，失败回退 tushare/baostock。
    """
    # 优先本地缓存
    if os.path.exists(CACHE_DIR):
        cache_files = [f.replace('.parquet', '') for f in os.listdir(CACHE_DIR)
                       if f.endswith('.parquet') and not f.startswith('_')]
        if len(cache_files) > 100:
            df = pd.DataFrame({'code': cache_files, 'name': cache_files})
            # 尝试从 tushare 获取真实名称（失败则保留 code）
            try:
                from config.settings import TUSHARE_TOKEN
                import tushare as ts
                if TUSHARE_TOKEN != "your_token_here":
                    pro = ts.pro_api(TUSHARE_TOKEN)
                    stocks = pro.stock_basic(exchange='', list_status='L',
                                             fields='symbol,name')
                    if stocks is not None and len(stocks) > 0:
                        stocks['code'] = stocks['symbol'].str.zfill(6)
                        name_map = dict(zip(stocks['code'], stocks['name'].fillna('')))
                        df['name'] = df['code'].map(name_map).fillna(df['code'])
            except Exception:
                pass
            logger.info(f"股票列表从缓存: {len(df)} 只")
            return df

    import baostock as bs
    try:
        bs.login()
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
            df = df[~df['code'].str.startswith(_NORTH_PREFIXES)]
        if exclude_st:
            df = df[~df['name'].str.contains('ST|\\*ST', na=False, case=False)]

        df = df[['code', 'name']].drop_duplicates('code').reset_index(drop=True)
        bs.logout()

        if len(df) < 1000:
            logger.warning(f"baostock 仅返回 {len(df)} 只（疑似断连），从缓存扫描...")
            cache_codes = [f.replace('.parquet', '')
                          for f in os.listdir(CACHE_DIR) if f.endswith('.parquet') and not f.startswith('_')]
            if len(cache_codes) > len(df):
                df = pd.DataFrame({'code': cache_codes, 'name': ''})
            logger.info(f"缓存目录: {len(df)} 只")

        logger.info(f"A 股列表: {len(df)} 只（已过滤 ST/北交所）")
        return df
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return None


# ============================================================
#  批量加载
# ============================================================

def load_all_stock_data(max_stocks=None, min_days=100,
                        exclude_st=True):
    """
    从缓存批量加载全市股票特征数据。
    - max_stocks:  限制加载数量（None = 全部）
    - min_days:    最短数据天数要求
    - exclude_st:  是否用 get_stock_list 过滤 ST（轻量 API）

    注意：此函数不下载任何数据。缺失缓存的股票会被静默跳过。
    """
    # 获取股票列表用于过滤
    stock_list = get_stock_list(exclude_st=exclude_st)
    if stock_list is None:
        logger.error("无法获取股票列表")
        return None

    valid_codes = set(stock_list['code'].tolist())
    all_codes = sorted(valid_codes)

    if max_stocks:
        all_codes = all_codes[:max_stocks]
        logger.info(f"限制为前 {max_stocks} 只")

    loaded = 0
    skipped = 0
    all_dfs = []

    for code in all_codes:
        df = load_from_cache(code)
        if df is not None and len(df) >= min_days:
            df['stock_code'] = code
            all_dfs.append(df)
            loaded += 1
        else:
            skipped += 1

    logger.info(f"从缓存加载: {loaded} 只, 跳过（无缓存或数据不足）: {skipped} 只")

    if not all_dfs:
        logger.error("未加载到任何符合条件的股票，请先运行 scripts/download_data.py 下载数据")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(['stock_code', 'Date']).reset_index(drop=True)
    logger.info(f"最终加载 {loaded} 只股票，总样本数 {len(combined)}")
    return combined


# ============================================================
#  单股票快捷加载（test.py 等使用）
# ============================================================

def load_stock_data(stock_code):
    """加载单只股票的特征数据（仅缓存，不下载）"""
    code = stock_code.zfill(6)
    df = load_from_cache(code)
    if df is None:
        logger.error(f"{code} 无缓存数据，请先运行 scripts/download_data.py --code {code}")
        return None
    return df


# ============================================================
#  行业分类
# ============================================================

_industry_map = None


def get_industry_map():
    """获取股票→行业映射（优先 tushare）"""
    global _industry_map
    if _industry_map is not None:
        return _industry_map
    # 尝试 tushare
    try:
        from config.settings import TUSHARE_TOKEN
        import tushare as ts
        if TUSHARE_TOKEN != "your_token_here":
            pro = ts.pro_api(TUSHARE_TOKEN)
            df = pro.stock_basic(exchange='', list_status='L',
                                 fields='symbol,industry')
            if df is not None and len(df) > 0:
                df['code'] = df['symbol'].str.zfill(6)
                _industry_map = dict(zip(df['code'], df['industry'].fillna('未知')))
                return _industry_map
    except Exception:
        pass
    # 回退 baostock
    import baostock as bs
    bs.login()
    rs = bs.query_stock_industry()
    rows = []
    while (rs.error_code == '0') & rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(rows, columns=rs.fields)
    # code: sh.600000 → 600000, industry: 行业名称
    df['code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
    _industry_map = dict(zip(df['code'], df['industry']))
    logger.info(f"行业分类加载: {len(_industry_map)} 只")
    return _industry_map


# ============================================================
#  风格中性化
# ============================================================

class StyleNeutralizer:
    """行业中性化：每只股票的特征减去同行业均值，除以同行业标准差"""
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.industry_stats = {}  # {industry: (mean_vec, std_vec)}

    def fit(self, features, codes):
        """
        features: (n_samples, n_feat)
        codes: 每行对应的股票代码列表
        """
        ind_map = get_industry_map()
        # 按行业分组计算均值/标准差
        df = pd.DataFrame(features)
        df['_code'] = codes
        df['_ind'] = df['_code'].map(ind_map).fillna('未知')
        for ind, group in df.groupby('_ind'):
            feats = group.drop(columns=['_code', '_ind']).values
            self.industry_stats[ind] = (feats.mean(axis=0), feats.std(axis=0) + self.eps)
        return self

    def transform(self, features, codes):
        """
        features: (n_samples, n_feat)
        codes: list of stock codes
        """
        ind_map = get_industry_map()
        out = features.copy()
        for i, code in enumerate(codes):
            ind = ind_map.get(code, '未知')
            if ind in self.industry_stats:
                mu, sigma = self.industry_stats[ind]
                out[i] = (features[i] - mu) / sigma
        return out

    def fit_transform(self, features, codes):
        return self.fit(features, codes).transform(features, codes)
