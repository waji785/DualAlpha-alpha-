#!/usr/bin/env python
# core/regime_detector.py
"""
市场状态检测：GAN-GRU 重建误差判断市场是否变天
模型训练: scripts/train_regime_detector.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]


class Generator(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64, seq_len=90, n_variates=5):
        super().__init__()
        self.seq_len = seq_len
        self.n_variates = n_variates
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len * n_variates),
        )

    def forward(self, latent, hidden):
        z = torch.cat([latent, hidden], dim=-1)
        return torch.tanh(self.fc(z).view(-1, self.seq_len, self.n_variates))


class Discriminator(nn.Module):
    def __init__(self, seq_len=90, n_variates=5, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(n_variates, hidden_dim, 2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


class RegimeDetector(nn.Module):
    """GAN-GRU 市场状态检测器"""

    def __init__(self, n_variates=5, hidden_dim=64, seq_len=90):
        super().__init__()
        self.encoder = GRUEncoder(n_variates, hidden_dim)
        self.generator = Generator(latent_dim=32, hidden_dim=hidden_dim,
                                    seq_len=seq_len, n_variates=n_variates)
        self.discriminator = Discriminator(seq_len=seq_len, n_variates=n_variates,
                                            hidden_dim=hidden_dim)
        self.factor_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, noise=None):
        h = self.encoder(x)
        if noise is None:
            noise = torch.randn(x.size(0), 32, device=x.device)
        gen_seq = self.generator(noise, h)
        real_score = self.discriminator(x)
        fake_score = self.discriminator(gen_seq)
        factor = torch.tanh(self.factor_head(h))
        return factor, real_score, fake_score, gen_seq

    def get_reconstruction_error(self, x):
        h = self.encoder(x)
        noise = torch.randn(x.size(0), 32, device=x.device)
        recon = self.generator(noise, h)
        return F.mse_loss(recon, x, reduction='none').mean(dim=[1, 2])

    def detect(self, x, threshold=0.5):
        """
        检测市场状态
        Args:
            x: (1, 90, 5) 张量 [Open,High,Low,Close,Volume]
            threshold: 重建误差阈值
        Returns:
            tuple: ('normal'|'regime_shift', error_value)
        """
        err = self.get_reconstruction_error(x).item()
        return ('regime_shift' if err > threshold else 'normal', err)


def load_regime_detector(path="output/regime_detector.pth"):
    """加载预训练的市场状态检测器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegimeDetector().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        return model
    return None


def detect_market_regime(df, model=None, model_path="output/regime_detector.pth"):
    """
    用沪深 300 等指数判断当前市场状态
    Args:
        df: 日线 DataFrame (需 Open,High,Low,Close,Volume 列, 至少 90 行)
        model: 预加载的模型，为 None 则自动加载
    Returns:
        dict: {'status': 'normal'|'regime_shift', 'error': float}
    """
    if model is None:
        model = load_regime_detector(model_path)
    if model is None:
        return {'status': 'unknown', 'error': 0}

    device = next(model.parameters()).device
    df = df.sort_values('Date')
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[cols].tail(90).values.astype(np.float32)
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    x = torch.FloatTensor(data).unsqueeze(0).to(device)

    status, err = model.detect(x)
    return {'status': status, 'error': round(err, 4)}
