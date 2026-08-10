#!/usr/bin/env python
# scripts/download_data.py
"""
数据下载 CLI：全量下载 / 增量更新 / 单股票下载 / 数据校验

用法:
    python scripts/download_data.py                     # 增量更新（默认）
    python scripts/download_data.py --full              # 全量下载全部 A 股
    python scripts/download_data.py --full --max 100    # 全量下载前 100 只
    python scripts/download_data.py --code 600000       # 下载单只股票
    python scripts/download_data.py --validate          # 校验已缓存数据
    python scripts/download_data.py --force             # 只重建特征，不重新下载（快）
"""
import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
from config.settings import TODAY
from core.data_downloader import DataDownloader
from core.data_loader import load_from_cache
from utils.logger import setup_logger

logger = setup_logger(__name__)


def cmd_incremental(force_rebuild=False):
    dl = DataDownloader(end=TODAY)
    dl.run_incremental_update(force_rebuild=force_rebuild)
    dl.print_report()


def cmd_full(max_stocks, start="2018-01-01"):
    dl = DataDownloader(start=start, end=TODAY)
    stock_df = dl.get_stock_list(exclude_st=True, exclude_north=True)
    if stock_df is None:
        print("❌ 无法获取股票列表")
        return
    codes = stock_df['code'].tolist()
    if max_stocks:
        codes = codes[:max_stocks]
    print(f"即将下载 {len(codes)} 只股票")
    dl.run_full_download(codes=codes)
    dl.print_report()


def cmd_single(code):
    dl = DataDownloader(end=TODAY)
    print(f"下载 {code} ...")
    ok = dl.download_and_save(code)
    if ok:
        df = load_from_cache(code)
        if df is not None:
            print(f"✅ {code}: {len(df)} 条, {df['Date'].min().date()} ~ {df['Date'].max().date()}")
            pe_valid = (df['PeTTM'] > 0).sum()
            print(f"   PE 有效: {pe_valid}/{len(df)}, 特征数: {len(df.columns)}")
        else:
            print(f"⚠️  {code} 下载成功但缓存读取失败")
    else:
        print(f"❌ {code} 下载失败")
    dl.print_report()


def cmd_validate():
    import os as _os
    from config.settings import CACHE_DIR
    codes = [f.replace('.parquet', '') for f in _os.listdir(CACHE_DIR) if f.endswith('.parquet')]

    dl = DataDownloader()
    print(f"校验 {len(codes)} 只股票...")
    for code in codes:
        df = load_from_cache(code)
        if df is None:
            dl.stats['failed'] += 1
            dl.errors.append((code, "缓存不可读"))
            continue
        warnings = dl.validate_data(df, code)
        if warnings:
            dl.stats['validated_warn'] += 1
        else:
            dl.stats['validated_ok'] += 1
        dl.stats['success'] += 1
    dl.print_report()


def main():
    parser = argparse.ArgumentParser(description="A 股数据下载工具")
    parser.add_argument("--full", action="store_true", help="全量下载（默认增量更新）")
    parser.add_argument("--code", type=str, default=None, help="下载单只股票")
    parser.add_argument("--validate", action="store_true", help="校验缓存数据")
    parser.add_argument("--max", type=int, default=None, dest="max_stocks", help="限制下载数量")
    parser.add_argument("--start", type=str, default="2018-01-01", help="起始日期")
    parser.add_argument("--force", action="store_true", help="强制全部重建特征（无数据下载）")
    args = parser.parse_args()

    if args.code:
        cmd_single(args.code)
    elif args.validate:
        cmd_validate()
    elif args.full:
        cmd_full(max_stocks=args.max_stocks, start=args.start)
    else:
        cmd_incremental(force_rebuild=args.force)


if __name__ == "__main__":
    main()
