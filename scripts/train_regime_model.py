#!/usr/bin/env python
# scripts/train_regime_model.py
"""训练大盘仓位模型：iTransformer 预测中证500未来5/10/20天收益 → 映射仓位系数"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from config.settings import *
from core.model import create_model
from utils.logger import setup_logger

logger = setup_logger(__name__)


def build_dataset(df, feat_cols, seq_len=SEQ_LEN):
    """构造训练样本：90天窗口 → 未来5/10/20天收益"""
    X_list, y_list = [], []
    for i in range(seq_len, len(df) - HORIZON_DAYS[-1]):
        X = df.iloc[i - seq_len:i][feat_cols].values.astype(np.float32)
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            continue
        c0 = float(df.iloc[i]['Close'])
        if c0 <= 0:
            continue
        y = [float(df.iloc[i + d]['Close']) / c0 - 1 for d in HORIZON_DAYS]
        X_list.append(X)
        y_list.append(y)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_regime_model():
    logger.info("=" * 60)
    logger.info("训练大盘仓位模型（iTransformer → 中证500 未来收益）")
    logger.info("=" * 60)

    # 1. 加载中证500 指数数据
    df = pd.read_parquet(os.path.join(CACHE_DIR, "000905.parquet"))
    df = df.sort_values('Date').reset_index(drop=True)
    logger.info(f"中证500: {len(df)} 天 ({df['Date'].min().date()} ~ {df['Date'].max().date()})")

    # 2. 构造样本
    X, y = build_dataset(df, FEATURE_COLS)
    logger.info(f"样本: X={X.shape}, y={y.shape}")
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    logger.info(f"训练 {len(X_tr)}, 验证 {len(X_val)} (时序切分)")

    # 3. 建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(MODEL_TYPE, INPUT_DIM, ITRANSFORMER_D_MODEL, ITRANSFORMER_NHEAD,
                         ITRANSFORMER_NUM_LAYERS, ITRANSFORMER_DIM_FF, ITRANSFORMER_DROPOUT,
                         USE_REVIN).to(device)
    logger.info(f"模型: {MODEL_TYPE}, 参数 {sum(p.numel() for p in model.parameters()):,}, 设备 {device}")

    # 4. 训练（纯回归，只用 reg_head）
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.SmoothL1Loss()
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)

    bs = 128
    EPOCHS = 40
    best_val = float('inf')
    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(len(X_tr))
        total = 0.0
        for s in range(0, len(X_tr), bs):
            b = idx[s:s + bs]
            xb, yb = X_tr_t[b], y_tr_t[b]
            reg, _ = model(xb)
            loss = loss_fn(reg, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        tr_loss = total / len(X_tr)

        # 验证
        model.eval()
        with torch.no_grad():
            reg_val, _ = model(X_val_t)
            val_loss = loss_fn(reg_val, torch.FloatTensor(y_val).to(device)).item()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            logger.info(f"Epoch {ep}: tr={tr_loss:.5f} val={val_loss:.5f}")

    # 5. 验证：20天收益预测的 IC 和方向准确率
    model.eval()
    with torch.no_grad():
        reg_val, _ = model(X_val_t)
        pred20 = reg_val[:, -1].cpu().numpy()  # 20天收益预测
    from scipy.stats import spearmanr
    ic, _ = spearmanr(pred20, y_val[:, -1])
    acc = float((np.sign(pred20) == np.sign(y_val[:, -1])).mean())
    logger.info(f"20天收益预测: Rank IC={ic:.4f}, 方向准确率={acc:.4f}")

    # 6. 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "regime_model.pth")
    torch.save({'model_state_dict': model.state_dict(),
                'input_dim': INPUT_DIM, 'seq_len': SEQ_LEN}, path)
    logger.info(f"仓位模型保存: {path}")
    return model


if __name__ == '__main__':
    train_regime_model()
