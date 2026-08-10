# scripts/debug_train_test.py
"""单股票训练+回测 / 两阶段调试"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from config.settings import *
from core.model import DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer, Fusion_iTransformer
from core.data_loader import load_from_cache, get_stock_list
from core.backtest_engine import run_backtest
from core.metrics import compute_metrics
from core.stock_selector import TreeEnsemble, build_selector_dataset
from utils.common import create_sequences
from utils.logger import setup_logger

logger = setup_logger(__name__)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
#  模式1：单模型训练+回测
# ============================================================

def debug_train_test(code="000001", model_type="itransformer",
                     train_ratio=0.7, epochs=20, lr=1e-4):
    logger.info(f"调试: {code} | {model_type} | {epochs} epochs")
    df = load_from_cache(code)
    if df is None or len(df) < SEQ_LEN + 50:
        logger.error(f"数据不足"); return
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    feat = df[FEATURE_COLS].values.astype(np.float32)
    tp = df['Target_Price'].values.astype(np.float32)
    td = df['Target_Direction'].values.astype(np.int8)

    split = int(len(df) * train_ratio)
    scaler_X = StandardScaler().fit(feat[:split])
    scaler_y = StandardScaler().fit(tp[:split].reshape(-1, 1))
    feat_s = scaler_X.transform(feat).astype(np.float32)
    tp_s = scaler_y.transform(tp.reshape(-1, 1)).ravel().astype(np.float32)

    X, yp, yd = create_sequences(feat_s, tp_s, td, SEQ_LEN)
    n_train = int(len(X) * train_ratio)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    close_idx = FEATURE_COLS.index('Close')
    models = {
        "itransformer": iTransformer(len(FEATURE_COLS), SEQ_LEN, d_model=64, nhead=4, num_layers=2, dropout=0.3),
        "gru_itrans": GRU_iTransformer(len(FEATURE_COLS), SEQ_LEN, d_model=64, nhead=4, num_layers=2, dropout=0.3),
        "nbeats_itrans": NBeats_iTransformer(len(FEATURE_COLS), SEQ_LEN, close_idx=close_idx, d_model=64, nhead=4, num_layers=2, dropout=0.3),
        "fusion_itrans": Fusion_iTransformer(len(FEATURE_COLS), SEQ_LEN, close_idx=close_idx, d_model=64, nhead=4, num_layers=2, dropout=0.3),
    }
    model = models.get(model_type, DualLSTM(len(FEATURE_COLS), 64, 2, dropout=0.3)).to(device)
    logger.info(f"参数: {sum(p.numel() for p in model.parameters()):,}")

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    cr, cc = nn.MSELoss(), nn.CrossEntropyLoss()
    tX = torch.tensor(X[:n_train]).to(device)
    tYp = torch.tensor(yp[:n_train]).reshape(-1,1).to(device)
    tYd = torch.tensor(yd[:n_train]).long().to(device)
    vX = torch.tensor(X[n_train:]).to(device)
    vYp = torch.tensor(yp[n_train:]).reshape(-1,1).to(device)
    vYd = torch.tensor(yd[n_train:]).long().to(device)

    history = {'loss': [], 'val_loss': [], 'val_acc': []}
    for e in range(epochs):
        model.train(); opt.zero_grad()
        pr, pd_ = model(tX)
        loss = cr(pr[:,1:2], tYp) + cc(pd_, tYd)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp, vd_ = model(vX)
            v_loss = cr(vp[:,1:2], vYp) + cc(vd_, vYd)
            acc = (vd_.argmax(1)==vYd).float().mean().item()
        history['loss'].append(loss.item())
        history['val_loss'].append(v_loss.item()); history['val_acc'].append(acc)
        if (e+1)%5==0: logger.info(f"Epoch {e+1}: loss={loss.item():.4f} val={v_loss.item():.4f} acc={acc:.3f}")

    bt_df = run_backtest(df, model, scaler_X, scaler_y)
    if bt_df is None: logger.warning("回测无信号"); return

    m = compute_metrics(bt_df['Capital'].values)
    print(f"\n📊 {code}: 收益={m['total_return']*100:+.2f}% 夏普={m['sharpe_ratio']:.3f} 回撤={m['max_drawdown']*100:.2f}%")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0,0].plot(history['loss'], label='train'); axes[0,0].plot(history['val_loss'], label='val')
    axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(history['val_acc'], 'g-'); axes[0,1].set_title('Val Acc'); axes[0,1].grid(True, alpha=0.3); axes[0,1].set_ylim(0,1)
    axes[1,0].plot(bt_df['Date'], bt_df['Capital']/bt_df['Capital'].iloc[0]*bt_df['Capital'].iloc[0])
    axes[1,0].set_title('资金曲线'); axes[1,0].grid(True, alpha=0.3)
    axes[1,1].fill_between(bt_df['Date'], 0, bt_df['Position'], alpha=0.3, color='g')
    axes[1,1].plot(bt_df['Date'], bt_df['Position'], 'g-')
    axes[1,1].set_title('仓位'); axes[1,1].grid(True, alpha=0.3); axes[1,1].set_ylim(0,1.2)
    plt.tight_layout(); plt.savefig(f"debug_{code}.png", dpi=120, bbox_inches='tight')
    logger.info("✅ 通过")
    return bt_df, m


# ============================================================
#  模式2：两阶段集成调试
# ============================================================

def debug_two_stage(epochs=15, n_stocks=20):
    logger.info(f"两阶段调试: {n_stocks}只, {epochs} epochs")

    stock_df = get_stock_list()
    codes = stock_df['code'].sample(min(n_stocks*3, 300)).tolist()
    all_dfs = {}
    for code in codes:
        df = load_from_cache(code)
        if df is not None and len(df) >= 300:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            all_dfs[code] = df
            if len(all_dfs) >= n_stocks: break
    logger.info(f"加载 {len(all_dfs)} 只")

    # ---- 训练选股器 ----
    X_sel, y_sel, sw_sel, _ = build_selector_dataset(all_dfs, list(all_dfs.keys()), forward_days=10, step_months=1)
    if len(X_sel) < 100: logger.error(f"截面样本不足 ({len(X_sel)})"); return
    logger.info(f"截面样本: {len(X_sel)}, y: mean={y_sel.mean():.4f} std={y_sel.std():.4f}")

    selector = TreeEnsemble()
    selector.fit(X_sel, y_sel, sample_weight=sw_sel)
    logger.info("选股器训练完成")

    # ---- 训练择时模型 ----
    code = list(all_dfs.keys())[0]
    df = all_dfs[code]
    feat = df[FEATURE_COLS].values.astype(np.float32)
    tp = df['Target_Price'].values.astype(np.float32); td = df['Target_Direction'].values.astype(np.int8)
    split = int(len(df)*0.7)
    sX = StandardScaler().fit(feat[:split]); sY = StandardScaler().fit(tp[:split].reshape(-1,1))
    feat_s = sX.transform(feat).astype(np.float32)
    tp_s = sY.transform(tp.reshape(-1,1)).ravel().astype(np.float32)
    X, yp, yd = create_sequences(feat_s, tp_s, td, SEQ_LEN)
    n_tr = int(len(X)*0.7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Fusion_iTransformer(len(FEATURE_COLS), SEQ_LEN, close_idx=FEATURE_COLS.index('Close'),
                                 d_model=64, nhead=4, num_layers=2, dropout=0.3).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    tX = torch.tensor(X[:n_tr]).to(device); tYp=torch.tensor(yp[:n_tr]).reshape(-1,1).to(device); tYd=torch.tensor(yd[:n_tr]).long().to(device)
    for e in range(epochs):
        model.train(); opt.zero_grad()
        pr, pd_ = model(tX)
        (nn.MSELoss()(pr[:,1:2], tYp)+nn.CrossEntropyLoss()(pd_, tYd)).backward(); opt.step()
    logger.info(f"择时模型训练完成 ({code})")

    # ---- 模拟选股 ----
    pool_f, pool_c = [], []
    for c in list(all_dfs.keys()):
        row = all_dfs[c].iloc[-1]; f = row[FEATURE_COLS].values.astype(np.float32)
        if not np.any(np.isnan(f)): pool_f.append(f); pool_c.append(c)
    preds = selector.predict(np.array(pool_f))
    top3 = [pool_c[i] for i in np.argsort(preds)[::-1][:3]]
    logger.info(f"Top3: {top3}")

    # ---- 择时回测 ----
    bt_df = run_backtest(all_dfs[top3[0]], model, sX, sY)
    if bt_df is not None:
        m = compute_metrics(bt_df['Capital'].values)
        print(f"\n📊 两阶段 {top3[0]}: 收益={m['total_return']*100:+.2f}% 夏普={m['sharpe_ratio']:.3f}")
    logger.info("✅ 两阶段通过")
    return selector, model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--code", default="000001")
    p.add_argument("--model", default="itransformer",
                   choices=["lstm","itransformer","gru_itrans","nbeats_itrans","fusion_itrans"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--two-stage", action="store_true", help="两阶段调试")
    p.add_argument("--n-stocks", type=int, default=20)
    args = p.parse_args()
    if args.two_stage:
        debug_two_stage(epochs=args.epochs, n_stocks=args.n_stocks)
    else:
        debug_train_test(args.code, args.model, epochs=args.epochs)
