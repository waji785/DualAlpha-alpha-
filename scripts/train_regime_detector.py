#!/usr/bin/env python
# scripts/train_regime_detector.py
"""训练市场状态检测器（GAN-GRU）"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.regime_detector import RegimeDetector
from core.data_loader import load_from_cache
from config.settings import CACHE_DIR, OUTPUT_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)


def prepare_sequences(df, seq_len=90):
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[cols].values.astype(np.float32)
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    seqs = []
    for i in range(0, len(data) - seq_len, 20):
        seqs.append(data[i:i+seq_len])
    return np.array(seqs)


def main(epochs=30, n_stocks=50):
    logger.info(f"训练市场状态检测器 ({n_stocks} 只股票, {epochs} 轮)")

    # 加载训练数据
    codes = [f.replace('.parquet', '') for f in os.listdir(CACHE_DIR)
             if f.endswith('.parquet')][:n_stocks]

    all_seqs = []
    for code in codes:
        try:
            df = load_from_cache(code)
            if df is not None and len(df) > 200:
                df = df.sort_values('Date')
                seqs = prepare_sequences(df)
                if len(seqs) > 0:
                    all_seqs.append(seqs)
        except: pass

    if len(all_seqs) == 0:
        logger.error("训练数据不足")
        return

    X = np.concatenate(all_seqs, axis=0)
    X = torch.FloatTensor(X)
    logger.info(f"训练样本: {len(X)} 序列")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegimeDetector().to(device)
    opt_g = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.generator.parameters()), lr=1e-4)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4)

    for epoch in range(epochs):
        idx = torch.randperm(len(X))[:512]
        x = X[idx].to(device)

        # 训练判别器
        noise = torch.randn(len(x), 32, device=device)
        _, real_score, fake_score, gen_seq = model(x, noise)
        d_loss = F.binary_cross_entropy(real_score, torch.ones_like(real_score) * 0.9) + \
                 F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score) * 0.1)
        opt_d.zero_grad()
        d_loss.backward(retain_graph=True)
        opt_d.step()

        # 训练生成器 + 编码器
        noise2 = torch.randn(len(x), 32, device=device)
        _, real_score2, fake_score2, _ = model(x, noise2)
        g_loss = F.binary_cross_entropy(fake_score2, torch.ones_like(fake_score2) * 0.9) + \
                 0.1 * F.mse_loss(gen_seq, x)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        if epoch % 5 == 0:
            logger.info(f"  Epoch {epoch:3d}  D={d_loss.item():.4f}  G={g_loss.item():.4f}")

    # 保存
    save_path = os.path.join(OUTPUT_DIR, "regime_detector.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"模型已保存: {save_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--stocks", type=int, default=50)
    args = p.parse_args()
    main(args.epochs, args.stocks)
