# scripts/viz_attention.py
"""加载模型，对样本股票做一次 forward，绘制特征注意力热力图"""
import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from config.settings import *
from core.model import DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer, Fusion_iTransformer
from core.data_loader import load_from_cache, get_stock_list

torch.serialization.add_safe_globals([DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer])

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def viz_attention(model_path="model_final.pth", scaler_x_path="scaler_X_final.pkl",
                  code=None, top_n=20):
    """加载模型，对指定股票（或随机选一只）做注意力可视化"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    scaler_X = joblib.load(scaler_x_path)

    if code:
        df = load_from_cache(code)
        if df is None:
            print(f"股票 {code} 无缓存")
            return
    else:
        stock_df = get_stock_list()
        for c in stock_df['code'].sample(50):
            df = load_from_cache(c)
            if df is not None and len(df) >= SEQ_LEN + 20:
                code = c
                break
        if df is None:
            print("无可用缓存股票")
            return

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    feat = df[FEATURE_COLS].values[-SEQ_LEN:].astype(np.float32)
    feat_scaled = scaler_X.transform(feat)
    x = torch.from_numpy(feat_scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        (reg, cls), attn = model(x, return_attn=True)
        up_prob = torch.softmax(cls, dim=1)[0, 1].item()
        attn_weights = attn[0].cpu().numpy()  # (n_variates,)

    # Sort by attention weight
    sorted_idx = np.argsort(attn_weights)[::-1]
    top_feats = [(FEATURE_COLS[i], attn_weights[i]) for i in sorted_idx[:top_n]]

    print(f"\n📊 {code} in={df['Date'].iloc[-1].date()} up_prob={up_prob:.3f}")
    print("=" * 60)
    print(f"  {'特征':25s} 权重")
    print("-" * 40)
    for i, (feat, w) in enumerate(top_feats):
        bar = '█' * int(w * 80)
        print(f"  {i+1:2d}. {feat:25s} {w:.4f}  {bar}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: top features bar chart
    names, weights = zip(*top_feats)
    axes[0].barh(range(top_n), weights[::-1], color='steelblue')
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(names[::-1], fontsize=9)
    axes[0].set_xlabel('Attention Weight')
    axes[0].set_title(f'{code} Feature Attention (up={up_prob:.3f})')
    axes[0].invert_yaxis()

    # Right: full heatmap of all features over last 30 days
    n_show = 15
    show_idx = sorted_idx[:n_show]
    show_data = feat_scaled[-30:, show_idx].T
    im = axes[1].imshow(show_data, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1].set_yticks(range(n_show))
    axes[1].set_yticklabels([FEATURE_COLS[i] for i in show_idx], fontsize=8)
    axes[1].set_xlabel('Last 30 days')
    axes[1].set_title('Top Feature Values (scaled)')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    fig_path = f"attention_{code}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {fig_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model_final.pth")
    p.add_argument("--code", default=None, help="股票代码，不指定则随机")
    p.add_argument("--scaler-x", default="scaler_X_final.pkl")
    p.add_argument("--top", type=int, default=20, help="展示前 N 个特征")
    args = p.parse_args()
    viz_attention(args.model, args.scaler_x, args.code, args.top)
