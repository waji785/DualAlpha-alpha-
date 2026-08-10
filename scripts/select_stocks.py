# scripts/select_stocks.py
"""用最终模型对未来选股：取每只股票最后 90 天 → 预测上涨概率 → 排序"""
import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np
import torch
import joblib
from tqdm import tqdm

from config.settings import *
from core.model import DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer, Fusion_iTransformer
from core.data_loader import load_from_cache, get_stock_list
from utils.common import set_seed
from utils.logger import setup_logger

logger = setup_logger(__name__)


def select_stocks(top_n=20, model_path="model_final.pth"):
    """加载最终模型，对所有有缓存的股票做预测，返回前 top_n 只"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    if not os.path.exists(model_path):
        logger.error(f"模型 {model_path} 不存在")
        return None
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    if hasattr(model, 'lstm') and hasattr(model.lstm, 'flatten_parameters'):
        model.lstm.flatten_parameters()
    scaler_X = joblib.load("scaler_X_final.pkl")
    logger.info(f"模型加载成功: {model_path}")

    # 股票列表
    stock_df = get_stock_list(exclude_st=True, exclude_north=True)
    if stock_df is None:
        return None
    codes = [c for c in stock_df['code'].tolist() if load_from_cache(c) is not None]
    logger.info(f"有缓存的股票: {len(codes)} 只")

    results = []
    for code in tqdm(codes, desc="预测"):
        df = load_from_cache(code)
        if df is None or len(df) < SEQ_LEN + 10:
            continue
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # 取最后 SEQ_LEN 天特征
        feat = df[FEATURE_COLS].values[-SEQ_LEN:]
        if len(feat) < SEQ_LEN:
            continue
        feat_scaled = scaler_X.transform(feat).astype(np.float32)
        x = torch.from_numpy(feat_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            _, cls_out = model(x)
            prob = torch.softmax(cls_out, dim=1).squeeze().cpu().numpy()
            up_prob = float(prob[1])

        # 附加信息
        last_close = float(df['Close'].iloc[-1])
        name = stock_df.loc[stock_df['code'] == code, 'name'].values
        name = name[0] if len(name) > 0 else ''

        results.append({
            'code': code,
            'name': name,
            'up_prob': up_prob,
            'close': last_close,
            'date': str(df['Date'].iloc[-1].date()),
        })

    if not results:
        logger.error("无有效预测结果")
        return None

    df_out = pd.DataFrame(results).sort_values('up_prob', ascending=False)
    df_out = df_out.head(top_n).reset_index(drop=True)

    print(f"\n📈 未来选股 Top {top_n}（按上涨概率排序）:")
    print("=" * 60)
    for i, row in df_out.iterrows():
        bar = '█' * int(row['up_prob'] * 20)
        print(f"  {i+1:2d}. {row['code']} {row['name']:8s}  "
              f"up={row['up_prob']:.3f}  close={row['close']:7.2f}  {bar}")
    print("=" * 60)

    df_out.to_csv("stock_picks.csv", index=False, encoding='utf-8-sig')
    logger.info(f"结果已保存: stock_picks.csv")
    return df_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20, help="选股数量")
    parser.add_argument("--models", type=str, nargs='+',
                        default=["model_final.pth"],
                        help="多个模型路径，up_prob 取平均")
    args = parser.parse_args()

    # 加载多个模型并集成
    all_probs = []
    for model_path in args.models:
        print(f"\n加载模型: {model_path}")
        result = select_stocks(top_n=args.top, model_path=model_path)
        if result is not None:
            all_probs.append(result.set_index('code')['up_prob'])

    if len(all_probs) > 1:
        print(f"\n🔀 集成 {len(args.models)} 个模型 (up_prob 取平均)...")
        ensemble = pd.concat(all_probs, axis=1).mean(axis=1)
        df_ens = ensemble.sort_values(ascending=False).head(args.top).reset_index()
        df_ens.columns = ['code', 'up_prob']
        # 合并名称
        from core.data_loader import get_stock_list
        stock_df = get_stock_list()
        name_map = dict(zip(stock_df['code'], stock_df['name']))
        df_ens['name'] = df_ens['code'].map(name_map)
        print("\n📈 集成选股 Top {args.top}:")
        for i, row in df_ens.iterrows():
            bar = '█' * int(row['up_prob'] * 20)
            print(f"  {i+1:2d}. {row['code']} {row['name']:8s}  up={row['up_prob']:.3f}  {bar}")
        df_ens.to_csv("stock_picks_ensemble.csv", index=False, encoding='utf-8-sig')
