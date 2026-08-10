# core/model.py
import torch
import torch.nn as nn
import math
from config.settings import NUM_HORIZONS, HORIZON_DAYS, SEQ_LEN


# ============================================================
#  RevIN — Reversible Instance Normalization
# ============================================================

class RevIN(nn.Module):
    """
    可逆实例归一化（Kim et al., ICLR 2022）
    对每条序列独立做 z-score 归一化，消除不同股票间的量纲差异。
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.affine_weight = None
            self.affine_bias = None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False) + self.eps
        x = (x - mean) / std
        if self.affine_weight is not None:
            x = x * self.affine_weight + self.affine_bias
        return x


# ============================================================
#  LSTM 双头模型
# ============================================================

class DualLSTM(nn.Module):
    """原始 LSTM 双头模型（保留用于 AB 对照）"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3,
                 use_revin=False):
        super().__init__()
        self.revin = RevIN(input_size) if use_revin else None
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.cls_head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        if self.revin is not None:
            x = self.revin(x)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.reg_head(last_out), self.cls_head(last_out)


# ============================================================
#  iTransformer（Inverted Transformer）
# ============================================================

class iTransformer(nn.Module):
    """
    Inverted Transformer（Liu et al., ICLR 2024）

    核心思路：把每个特征当作一个 token ，对特征做 self-attention。
    与标准 Transformer 的区别：
        标准:  (B, time, feat) → embed time → attend across time
        iTransformer: (B, feat, time) → project time → attend across feat

    每个 "variate token" 包含该特征的全部 90 天历史，
    self-attention 捕捉特征间的交叉关系。

    架构：
        Input          (B, seq_len, n_variates)
        → RevIN        (可选)
        → Transpose    (B, n_variates, seq_len)
        → VariateProj  (B, n_variates, d_model)   # Linear(seq_len, d_model)
        → Encoder × N  (B, n_variates, d_model)   # 跨特征 attention
        → Mean Pool    (B, d_model)
        → Dropout
        → Reg head     (B, 1)
        → Cls head     (B, 2)
    """
    def __init__(self, n_variates, seq_len, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=512, dropout=0.5,
                 use_revin=False):
        super().__init__()
        self.seq_len = seq_len
        self.revin = RevIN(n_variates) if use_revin else None

        # 时序投影：把每支特征的 seq_len 天压缩为 d_model 维
        self.variate_proj = nn.Linear(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = nn.Linear(d_model, 1)
        self.reg_dropout = nn.Dropout(dropout + 0.1)   # 回归高 dropout
        self.cls_dropout = nn.Dropout(dropout - 0.1)   # 分类低 dropout
        self.reg_head = nn.Linear(d_model, NUM_HORIZONS)
        self.cls_head = nn.Linear(d_model, 2)
        self.log_var_reg = nn.Parameter(torch.zeros(NUM_HORIZONS))
        self.log_var_cls = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x, return_attn=False):
        # x: (B, seq_len, n_variates)
        if self.revin is not None:
            x = self.revin(x)
        x = x.transpose(1, 2)                  # (B, n_variates, seq_len)
        x = self.variate_proj(x)               # (B, n_variates, d_model)
        x = self.encoder(x)                    # (B, n_variates, d_model)
        # 注意力池化（向后兼容：旧模型无 attn_pool 则用 mean）
        if hasattr(self, 'attn_pool'):
            scores = torch.softmax(self.attn_pool(x), dim=1)  # (B, n_variates, 1)
            x = (x * scores).sum(dim=1)                      # (B, d_model)
        else:
            x = x.mean(dim=1)
            scores = torch.ones(x.shape[1], 1, device=x.device) / x.shape[1]
        # 独立 dropout（兼容旧模型无 reg_dropout 则用 self.dropout）
        if hasattr(self, 'reg_dropout'):
            reg_x = self.reg_dropout(x)
            cls_x = self.cls_dropout(x)
        elif hasattr(self, 'dropout'):
            reg_x = cls_x = self.dropout(x)
        else:
            reg_x = cls_x = x
        out = self.reg_head(reg_x), self.cls_head(cls_x)
        if return_attn:
            return out, scores.squeeze(-1)     # (B, n_variates)
        return out


# ============================================================
#  GRU + iTransformer 混合模型
# ============================================================

class GRU_iTransformer(nn.Module):
    """
    GRU 提取短期趋势 → iTransformer 做全局决策

    架构：
        Input (B, 90, 47)
        ├─ GRU: 取最后 20 天 → (B, 16) 短期趋势因子
        │       expand → (B, 90, 16)
        ├─ 拼接 → (B, 90, 63)
        └─ iTransformer: 63 variates → reg + cls
    """
    def __init__(self, n_variates, seq_len, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=512, dropout=0.5,
                 use_revin=False, gru_seq=20, gru_hidden=16):
        super().__init__()
        self.gru_seq = gru_seq
        self.gru = nn.GRU(input_size=n_variates, hidden_size=gru_hidden,
                          num_layers=1, batch_first=True)

        # iTransformer 接收原始特征 + GRU 因子
        n_combined = n_variates + gru_hidden
        self.itrans = iTransformer(n_variates=n_combined, seq_len=seq_len,
                                   d_model=d_model, nhead=nhead,
                                   num_layers=num_layers,
                                   dim_feedforward=dim_feedforward,
                                   dropout=dropout, use_revin=use_revin)

    def forward(self, x, return_attn=False):
        # x: (B, 90, 47)
        # ① GRU: 取最后 gru_seq 天
        x_gru = x[:, -self.gru_seq:, :]          # (B, 20, 47)
        _, h_n = self.gru(x_gru)                  # h_n: (1, B, 16)
        short_term = h_n.squeeze(0)               # (B, 16)
        # expand to all timesteps
        short_term = short_term.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, 90, 16)

        # ② 拼接 → iTransformer
        x_combined = torch.cat([x, short_term], dim=-1)  # (B, 90, 63)
        return self.itrans(x_combined, return_attn=return_attn)


# ============================================================
#  N-BEATS + iTransformer 混合模型
# ============================================================

class NBeatsBlock(nn.Module):
    """N-BEATS 基础块：全连接 → backcast + trend"""
    def __init__(self, seq_len, hidden=256, trend_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(seq_len, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.backcast = nn.Linear(hidden, seq_len)
        self.trend = nn.Linear(hidden, trend_dim)

    def forward(self, x):
        # x: (B, seq_len)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        h = torch.relu(self.fc4(h))
        return self.backcast(h), self.trend(h)


class NBeats(nn.Module):
    """N-BEATS: 多栈残差连接，提取长期趋势因子"""
    def __init__(self, seq_len, n_stacks=2, hidden=256, trend_dim=16):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(seq_len, hidden, trend_dim) for _ in range(n_stacks)
        ])

    def forward(self, x):
        # x: (B, seq_len) — 单变量价格序列
        residual = x
        trend_total = 0
        for block in self.blocks:
            backcast, trend = block(residual)
            residual = residual - backcast
            trend_total = trend_total + trend
        return trend_total  # (B, trend_dim)


class NBeats_iTransformer(nn.Module):
    """
    N-BEATS 提取长期趋势因子 → iTransformer 做全局决策

    架构：
        Input (B, 90, 47)
        ├─ N-BEATS(Close 价格) → (B, 16) 长期趋势因子
        │       expand → (B, 90, 16)
        ├─ 拼接 → (B, 90, 63)
        └─ iTransformer: 63 variates → reg + cls
    """
    def __init__(self, n_variates, seq_len, close_idx=16,
                 d_model=128, nhead=8, num_layers=4, dim_feedforward=512,
                 dropout=0.5, use_revin=False,
                 nbeats_stacks=2, nbeats_hidden=256, nbeats_dim=16):
        super().__init__()
        self.close_idx = close_idx
        self.nbeats = NBeats(seq_len, n_stacks=nbeats_stacks,
                             hidden=nbeats_hidden, trend_dim=nbeats_dim)

        n_combined = n_variates + nbeats_dim
        self.itrans = iTransformer(n_variates=n_combined, seq_len=seq_len,
                                   d_model=d_model, nhead=nhead,
                                   num_layers=num_layers,
                                   dim_feedforward=dim_feedforward,
                                   dropout=dropout, use_revin=use_revin)

    def forward(self, x, return_attn=False):
        # x: (B, 90, 47)
        close = x[:, :, self.close_idx]            # (B, 90)
        trend = self.nbeats(close)                  # (B, 16)
        trend = trend.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, 90, 16)
        x_combined = torch.cat([x, trend], dim=-1)  # (B, 90, 63)
        return self.itrans(x_combined, return_attn=return_attn)


# ============================================================
#  GRU + N-BEATS + iTransformer 深度融合
# ============================================================

class Fusion_iTransformer(nn.Module):
    """
    GRU 提取短期动量 + N-BEATS 提取长期趋势 → 拼接 → iTransformer

    架构：
        Input (B, 90, 47)
        ├─ GRU(last 20 days) → (B, 16) expand → (B, 90, 16)
        ├─ N-BEATS(Close)    → (B, 16) expand → (B, 90, 16)
        ├─ concat → (B, 90, 79)
        └─ iTransformer(79 variates) → reg + cls
    """
    def __init__(self, n_variates, seq_len, close_idx=16,
                 d_model=128, nhead=8, num_layers=4, dim_feedforward=512,
                 dropout=0.5, use_revin=False,
                 gru_seq=20, gru_hidden=16,
                 nbeats_stacks=2, nbeats_hidden=256, nbeats_dim=16):
        super().__init__()
        self.close_idx = close_idx
        self.gru_seq = gru_seq

        self.gru = nn.GRU(input_size=n_variates, hidden_size=gru_hidden,
                          num_layers=1, batch_first=True)
        self.nbeats = NBeats(seq_len, n_stacks=nbeats_stacks,
                             hidden=nbeats_hidden, trend_dim=nbeats_dim)

        n_combined = n_variates + gru_hidden + nbeats_dim
        self.itrans = iTransformer(n_variates=n_combined, seq_len=seq_len,
                                   d_model=d_model, nhead=nhead,
                                   num_layers=num_layers,
                                   dim_feedforward=dim_feedforward,
                                   dropout=dropout, use_revin=use_revin)

    def forward(self, x, return_attn=False):
        # GRU 短期
        x_gru = x[:, -self.gru_seq:, :]
        _, h_n = self.gru(x_gru)
        short = h_n.squeeze(0).unsqueeze(1).expand(-1, x.size(1), -1)

        # N-BEATS 长期
        close = x[:, :, self.close_idx]
        trend = self.nbeats(close).unsqueeze(1).expand(-1, x.size(1), -1)

        x_combined = torch.cat([x, short, trend], dim=-1)
        return self.itrans(x_combined, return_attn=return_attn)


def create_model(model_type, n_variates, d_model, nhead, num_layers,
                 dim_feedforward, dropout, use_revin, seq_len=SEQ_LEN,
                 **kwargs):
    """模型工厂函数"""
    if model_type == "itransformer":
        return iTransformer(n_variates=n_variates, seq_len=seq_len,
                           d_model=d_model, nhead=nhead, num_layers=num_layers,
                           dim_feedforward=dim_feedforward,
                           dropout=dropout, use_revin=use_revin)
    elif model_type == "gru_itrans":
        return GRU_iTransformer(n_variates=n_variates, seq_len=seq_len,
                                d_model=d_model, nhead=nhead, num_layers=num_layers,
                                dim_feedforward=dim_feedforward,
                                dropout=dropout, use_revin=use_revin)
    elif model_type == "nbeats_itrans":
        return NBeats_iTransformer(n_variates=n_variates, seq_len=seq_len,
                                    d_model=d_model, nhead=nhead, num_layers=num_layers,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout, use_revin=use_revin)
    elif model_type == "fusion_itrans":
        return Fusion_iTransformer(n_variates=n_variates, seq_len=seq_len,
                                    d_model=d_model, nhead=nhead, num_layers=num_layers,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout, use_revin=use_revin)
    else:
        raise ValueError(f"未知 MODEL_TYPE: {model_type}")
