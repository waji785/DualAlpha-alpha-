#!/usr/bin/env python
# scripts/daily_predict.py
"""
盘前运行：对 daily_picks.csv 每只股票跑择时模型 → 输出今日信号
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd
import torch
import joblib

from config.settings import *
from core.data_loader import load_from_cache
from core.model import create_model
from utils.logger import setup_logger

logger = setup_logger(__name__)


def run_predictions(watchlist_csv="daily_picks.csv",
                    model_path="model_final.pth",
                    output="today_signals.csv"):
    """盘前打分：up_prob + 多周期预测 + 趋势/量确认"""
    if not os.path.exists(watchlist_csv):
        logger.error(f"{watchlist_csv} 不存在，先运行 daily_select.py")
        return

    df_wl = pd.read_csv(watchlist_csv)
    codes = [str(c).zfill(6) for c in df_wl['code']]

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(
        MODEL_TYPE, INPUT_DIM, ITRANSFORMER_D_MODEL,
        ITRANSFORMER_NHEAD, ITRANSFORMER_NUM_LAYERS,
        ITRANSFORMER_DIM_FF, ITRANSFORMER_DROPOUT,
        USE_REVIN
    ).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.state_dict(), strict=False)
    model.eval()
    sX, sY = joblib.load('scaler_X_final.pkl'), joblib.load('scaler_Y_final.pkl')
    logger.info(f"模型加载: {model_path}")

    results = []
    for _, row in df_wl.iterrows():
        code = str(row['code']).zfill(6)
        df = load_from_cache(code)
        if df is None:
            logger.warning(f"{code} 无缓存")
            results.append({'code': code, 'signal': '无数据'})
            continue

        df['Date'] = pd.to_datetime(df['Date'])
        close_col = 'Close' if 'Close' in df.columns else 'close'
        close = df[close_col].values.astype(np.float32)

        if len(df) < SEQ_LEN + 200:
            results.append({'code': code, 'signal': '数据不足'})
            continue

        # 特征提取
        feat_raw = df[FEATURE_COLS].values[-SEQ_LEN:].astype(np.float32)
        feat = sX.transform(feat_raw)
        x = torch.FloatTensor(feat).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_reg, pred_dir = model(x)
            prob = torch.softmax(pred_dir, dim=1).squeeze().cpu().numpy()
            pred_ret = sY.inverse_transform(
                pred_reg.cpu().numpy()
            ).squeeze()  # [5d, 10d, 20d]

        up_prob = float(prob[1])
        returns_5d = float(pred_ret[0])
        returns_10d = float(pred_ret[1])
        returns_20d = float(pred_ret[2])

        last_close = float(close[-1])

        # 趋势过滤
        ma50 = pd.Series(close).rolling(50).mean().iloc[-1]
        ma200 = pd.Series(close).rolling(200).mean().iloc[-1]
        price_ma200_ratio = (last_close - ma200) / ma200 if ma200 > 0 else 0
        trend_ok = bool(ma50 > ma200 and price_ma200_ratio > 0.03)

        # 量确认（日线级别）
        vol_arr = df['Volume'].values[-30:]
        ma_vol = np.mean(vol_arr[:-1])
        vol_ok = bool(vol_arr[-1] > ma_vol * 0.8)

        # 目标价
        target_price = round(last_close * (1 + returns_20d), 2)
        entry_range = (round(last_close * 0.995, 2), round(last_close * 1.005, 2))

        # 信号决策
        regime_buy = BUY_THRESHOLD - 0.04 if price_ma200_ratio > 0 else BUY_THRESHOLD + 0.03
        if trend_ok and vol_ok and up_prob > regime_buy:
            signal = '买入'
        elif trend_ok and up_prob > BUY_THRESHOLD:
            signal = '关注'
        else:
            signal = '观望'

        results.append({
            'code': code,
            'name': row.get('name', ''),
            'close': last_close,
            'signal': signal,
            'up_prob': round(up_prob, 4),
            'ret_5d': f"{returns_5d*100:+.1f}%",
            'ret_10d': f"{returns_10d*100:+.1f}%",
            'ret_20d': f"{returns_20d*100:+.1f}%",
            'target_price': target_price,
            'entry_low': entry_range[0],
            'entry_high': entry_range[1],
            'trend_ok': trend_ok,
            'vol_ok': vol_ok,
            'ma200_ratio': f"{price_ma200_ratio*100:+.1f}%",
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output, index=False, encoding='utf-8-sig')
    print(df_out[['code', 'name', 'signal', 'up_prob', 'ret_20d', 'target_price']].to_string(index=False))

    buys = df_out[df_out['signal'] == '买入']
    # 组合优化：风险平价 + 置信度加权
    if len(buys) >= 2:
        from core.portfolio import compute_weights, get_price_histories
        price_hist = get_price_histories([str(c).zfill(6) for c in buys['code']])
        scores_dict = {
            str(r['code']).zfill(6): {'up_prob': r['up_prob']}
            for _, r in buys.iterrows()
        }
        opt_w = compute_weights(
            [str(c).zfill(6) for c in buys['code']], scores_dict, price_hist)
        # 写回权重
        df_out['portfolio_weight'] = 0.0
        for code, w in opt_w.items():
            mask = df_out['code'].astype(str).str.zfill(6) == code
            df_out.loc[mask, 'portfolio_weight'] = w
        df_out.loc[~df_out['code'].astype(str).str.zfill(6).isin(opt_w), 'portfolio_weight'] = 0
    else:
        df_out['portfolio_weight'] = 0.0

    buys = df_out[df_out['signal'] == '买入']
    logger.info(f"今日信号: 买入{len(buys)}只 关注{len(df_out[df_out['signal']=='关注'])}只 观望{len(df_out[df_out['signal']=='观望'])}只")
    logger.info(f"保存: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--watchlist", default="daily_picks.csv")
    p.add_argument("--model", default="model_final.pth")
    p.add_argument("--output", default="today_signals.csv")
    args = p.parse_args()
    run_predictions(args.watchlist, args.model, args.output)
