# scripts/rebuild_financial_features.py
"""重建财务因子（ann_date 消除前视）+ 融资融券因子，不重新下载股票价格数据"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np
from tqdm import tqdm
from config.settings import CACHE_DIR
from core.data_downloader import DataDownloader
from utils.logger import setup_logger

logger = setup_logger(__name__)

FIN_COLS = ['ROE', 'GrossMargin', 'NetMargin', 'DebtRatio']
FIN_FIELDS = ['roe', 'grossprofit_margin', 'netprofit_margin', 'debt_to_assets']
MARGIN_COLS = ['Margin_Chg_5d', 'Margin_Chg_20d']
LHB_COLS = ['LHB_Net_Buy_20d']
BLOCK_COLS = ['Block_Amount_20d']


def main():
    # 1. 股票代码列表（跳过 _fundamental_cache 等下划线开头的文件）
    codes = [f.replace('.parquet', '') for f in os.listdir(CACHE_DIR)
             if f.endswith('.parquet') and not f.startswith('_')]
    logger.info(f"股票缓存: {len(codes)} 只")

    dl = DataDownloader()

    # 2. 删除旧财务缓存 + 融资融券缓存，重新下载（ann_date + rzye）
    for name in ['_fundamental_cache.parquet', '_margin_cache.parquet',
                 '_lhb_cache.parquet', '_block_cache.parquet']:
        cache_file = os.path.join(CACHE_DIR, name)
        if os.path.exists(cache_file):
            os.remove(cache_file)
            logger.info(f"已删除旧缓存: {name}")

    fundamental_cache = dl._load_fundamentals(codes)
    logger.info(f"财务指标重下完成: {len(fundamental_cache)} 只")

    margin_cache = dl._load_margin()
    logger.info(f"融资融券重下完成: {len(margin_cache)} 只")

    lhb_cache = dl._load_lhb()
    logger.info(f"龙虎榜重下完成: {len(lhb_cache)} 只")

    block_cache = dl._load_block()
    logger.info(f"大宗交易重下完成: {len(block_cache)} 只")

    # 3. 遍历股票缓存，重建财务因子 + 融资融券因子
    rebuilt = 0
    for code in tqdm(codes, desc='重建因子'):
        path = os.path.join(CACHE_DIR, f'{code}.parquet')
        df = pd.read_parquet(path)

        # 删除旧财务因子 + 旧融资融券因子 + 旧龙虎榜/大宗交易因子
        for c in FIN_COLS + MARGIN_COLS + LHB_COLS + BLOCK_COLS:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

        s = code.zfill(6)
        ts_code = f"{s}.SH" if s[0] == '6' else f"{s}.SZ"

        # ---- 财务因子（ann_date 版，前视修复）----
        fund = fundamental_cache.get(ts_code)
        if fund is not None and len(fund) > 0:
            f = fund.copy()
            if 'ann_date' in f.columns:
                f['Date'] = pd.to_datetime(f['ann_date'].fillna(f['end_date']))
            else:
                f['Date'] = pd.to_datetime(f['end_date'])
            f = f[['Date'] + FIN_FIELDS].copy()
            f.columns = ['Date'] + FIN_COLS
            f = f.sort_values('Date').drop_duplicates('Date', keep='last')
            df = df.merge(f, on='Date', how='left')
            for c in FIN_COLS:
                df[c] = df[c].ffill().fillna(0.0)
        else:
            for c in FIN_COLS:
                df[c] = 0.0

        # ---- 融资融券因子（融资余额变化率）----
        m = margin_cache.get(ts_code)
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

        # ---- 龙虎榜机构净买入（20天累计）----
        lhb = lhb_cache.get(ts_code)
        if lhb is not None and len(lhb) > 0:
            ll = lhb.copy()
            ll['Date'] = pd.to_datetime(ll['trade_date'])
            ll = ll.sort_values('Date').drop_duplicates('Date', keep='last')
            df = df.merge(ll[['Date', 'net_buy']], on='Date', how='left')
            df['LHB_Net_Buy_20d'] = df['net_buy'].fillna(0.0).rolling(20, min_periods=1).sum()
            df.drop(columns=['net_buy'], inplace=True)
        else:
            df['LHB_Net_Buy_20d'] = 0.0

        # ---- 大宗交易金额（20天累计）----
        blk = block_cache.get(ts_code)
        if blk is not None and len(blk) > 0:
            bb = blk.copy()
            bb['Date'] = pd.to_datetime(bb['trade_date'])
            bb = bb.sort_values('Date').drop_duplicates('Date', keep='last')
            df = df.merge(bb[['Date', 'amount']], on='Date', how='left')
            df['Block_Amount_20d'] = df['amount'].fillna(0.0).rolling(20, min_periods=1).sum()
            df.drop(columns=['amount'], inplace=True)
        else:
            df['Block_Amount_20d'] = 0.0

        df.to_parquet(path, index=False)
        rebuilt += 1

    logger.info(f"因子重建完成: {rebuilt} 只")
    sample = pd.read_parquet(os.path.join(CACHE_DIR, '600519.parquet'))
    for c in FIN_COLS + MARGIN_COLS + LHB_COLS + BLOCK_COLS:
        logger.info(f"验证 {c} 非零: {(sample[c] != 0).sum()}/{len(sample)}")


if __name__ == '__main__':
    main()
