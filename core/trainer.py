# core/trainer.py
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from config.settings import *
from core.model import DualLSTM, iTransformer, GRU_iTransformer, NBeats_iTransformer, Fusion_iTransformer
from core.data_loader import get_stock_list
from utils.common import create_sequences, set_seed
from utils.logger import setup_logger

logger = setup_logger(__name__)

MAX_SAMPLES_PER_EPOCH = 400000
MAX_VAL_SAMPLES = 100000


def train_and_save_model(df, train_end_date=None, model_save_path=MODEL_PATH,
                         scaler_x_path=SCALER_X_PATH, scaler_y_path=SCALER_Y_PATH,
                         val_ratio=0.2, seed=42, exclude_st=True):
    """训练全市模型并保存（存 raw 特征，epoch 内实时建序列，滚动窗口全量历史）"""
    set_seed(seed)

    if df is None or len(df) < 100:
        logger.error("数据不足")
        return None, None, None, None

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    if exclude_st and 'stock_code' in df.columns:
        try:
            stock_list = get_stock_list(exclude_st=True, exclude_north=True)
            if stock_list is not None:
                valid_codes = set(stock_list['code'].tolist())
                before = df['stock_code'].nunique()
                df = df[df['stock_code'].isin(valid_codes)].copy()
                after = df['stock_code'].nunique()
                if after < before:
                    logger.info(f"已剔除 ST/退市 {before-after} 只, 剩余 {after} 只")
        except Exception as e:
            logger.warning(f"ST 过滤跳过: {e}")

    if train_end_date is None:
        max_date = df['Date'].max()
        train_end_date = (max_date - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    train_end = pd.to_datetime(train_end_date)

    train_df = df[df['Date'] <= train_end]
    del df
    if len(train_df) < SEQ_LEN * 2:
        logger.error(f"训练数据不足 (需 ≥{SEQ_LEN*2})")
        return None, None, None, None

    rng = np.random.RandomState(seed)

    # ---- 存 raw 特征（不建序列）----
    meta = []  # [(feat, tp, td, seq_count, code), ...]
    stock_ends = []  # unused, kept for compat
    n_total = 0

    for code, group in train_df.groupby('stock_code'):
        group = group.sort_values('Date').reset_index(drop=True)
        if len(group) < SEQ_LEN + 1:
            continue
        feat = group[FEATURE_COLS].values.astype(np.float32)
        close = group['Close'].values.astype(np.float32)
        # 多周期目标：5/10/20 日收益率（裁剪极端值防过拟合）
        tp_mat = np.stack([
            np.clip((np.roll(close, -h) - close) / close, -1.0, 1.0)
            for h in HORIZON_DAYS
        ], axis=1).astype(np.float32)  # (n_days, 3)
        td = group['Target_Direction'].values.astype(np.int8)
        seq_n = len(feat) - SEQ_LEN + 1
        if seq_n <= 0:
            continue
        meta.append((feat, tp_mat, td, seq_n, code))
        n_total += seq_n

    del train_df

    if not meta:
        logger.error("无训练序列")
        return None, None, None, None

    n_stocks = len(meta)
    logger.info(f"raw 特征加载: {n_total} 潜在序列, {n_stocks} 只股票")

    # ---- 标准化器：partial_fit 全量 raw ----
    scaler_X = StandardScaler()
    for i in range(0, n_stocks, 200):
        chunk = np.concatenate([m[0] for m in meta[i:i+200]], axis=0)
        scaler_X.partial_fit(chunk)
    # 标准化 raw 特征
    for i in range(n_stocks):
        f, tp, td, n, code = meta[i]
        meta[i] = (scaler_X.transform(f).astype(np.float32), tp, td, n, code)
    logger.info(f"scaler_X 拟合 + 标准化完成 ({n_stocks} 只)")

    # ---- 风格中性化（行业 z-score） ----
    if USE_STYLE_NEUTRAL:
        from core.data_loader import StyleNeutralizer
        neutralizer = StyleNeutralizer()
        # 取每只股票所有天数的特征拼成 (n_total_days, n_feat)，拟合
        all_feats = np.concatenate([m[0] for m in meta], axis=0)
        all_codes = []
        for m in meta:
            all_codes.extend([m[4]] * len(m[0]))
        neutralizer.fit(all_feats, all_codes)
        # 逐股票 transform
        for i in range(n_stocks):
            f, tp, td, n, code = meta[i]
            neutralized = neutralizer.transform(f, [code] * len(f))
            meta[i] = (neutralized.astype(np.float32), tp, td, n, code)
        logger.info(f"风格中性化完成 ({n_stocks} 只)")

    # ---- 验证集：独立抽样，实时建序列 ----
    val_idx = list(rng.choice(n_stocks, size=min(n_stocks, 200), replace=False))
    val_set = set(val_idx)
    train_pool = [i for i in range(n_stocks) if i not in val_set]

    Xv, ypv, ydv = [], [], []
    for i in val_idx:
        f, tp, td, _, _ = meta[i]
        x, yp, yd = create_sequences(f, tp, td, SEQ_LEN)
        Xv.append(x); ypv.append(yp); ydv.append(yd)
    X_val = np.concatenate(Xv, axis=0)
    yp_val = np.concatenate(ypv, axis=0)
    yd_val = np.concatenate(ydv, axis=0)
    del Xv, ypv, ydv
    n_val = min(len(X_val), MAX_VAL_SAMPLES)
    X_val, yp_val, yd_val = X_val[:n_val], yp_val[:n_val], yd_val[:n_val]
    _, seq_len, n_feat = X_val.shape

    scaler_y = StandardScaler().fit(yp_val)
    yp_val_s = scaler_y.transform(yp_val)

    val_dataset = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(yp_val_s.astype(np.float32)),   # (n, 3)
        torch.from_numpy(yd_val.copy()).long())
    del X_val, yp_val, yd_val, yp_val_s
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    logger.info(f"验证集: {n_val} 序列 ({len(val_idx)} 只, 独立)")

    # ---- 模型 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODEL_TYPE == "fusion_itrans":
        close_idx = FEATURE_COLS.index('Close')
        model = Fusion_iTransformer(n_variates=n_feat, seq_len=seq_len,
                                    close_idx=close_idx,
                                    d_model=ITRANSFORMER_D_MODEL, nhead=ITRANSFORMER_NHEAD,
                                    num_layers=ITRANSFORMER_NUM_LAYERS,
                                    dim_feedforward=ITRANSFORMER_DIM_FF,
                                    dropout=ITRANSFORMER_DROPOUT, use_revin=USE_REVIN,
                                    gru_seq=GRU_SEQ_LEN, gru_hidden=GRU_HIDDEN,
                                    nbeats_stacks=NBEATS_STACKS,
                                    nbeats_hidden=NBEATS_HIDDEN,
                                    nbeats_dim=NBEATS_DIM).to(device)
        logger.info(f"使用 Fusion iTransformer: GRU({GRU_HIDDEN}) + N-BEATS({NBEATS_DIM})")
    elif MODEL_TYPE == "nbeats_itrans":
        close_idx = FEATURE_COLS.index('Close')
        model = NBeats_iTransformer(n_variates=n_feat, seq_len=seq_len,
                                    close_idx=close_idx,
                                    d_model=ITRANSFORMER_D_MODEL, nhead=ITRANSFORMER_NHEAD,
                                    num_layers=ITRANSFORMER_NUM_LAYERS,
                                    dim_feedforward=ITRANSFORMER_DIM_FF,
                                    dropout=ITRANSFORMER_DROPOUT, use_revin=USE_REVIN,
                                    nbeats_stacks=NBEATS_STACKS,
                                    nbeats_hidden=NBEATS_HIDDEN,
                                    nbeats_dim=NBEATS_DIM).to(device)
        logger.info(f"使用 N-BEATS+iTransformer: stacks={NBEATS_STACKS}, "
                    f"hidden={NBEATS_HIDDEN}, dim={NBEATS_DIM}")
    elif MODEL_TYPE == "gru_itrans":
        model = GRU_iTransformer(n_variates=n_feat, seq_len=seq_len,
                                 d_model=ITRANSFORMER_D_MODEL, nhead=ITRANSFORMER_NHEAD,
                                 num_layers=ITRANSFORMER_NUM_LAYERS,
                                 dim_feedforward=ITRANSFORMER_DIM_FF,
                                 dropout=ITRANSFORMER_DROPOUT, use_revin=USE_REVIN,
                                 gru_seq=GRU_SEQ_LEN, gru_hidden=GRU_HIDDEN).to(device)
        logger.info(f"使用 GRU+iTransformer: gru_seq={GRU_SEQ_LEN}, gru_hidden={GRU_HIDDEN}")
    elif MODEL_TYPE == "itransformer":
        model = iTransformer(n_variates=n_feat, seq_len=seq_len,
                             d_model=ITRANSFORMER_D_MODEL, nhead=ITRANSFORMER_NHEAD,
                             num_layers=ITRANSFORMER_NUM_LAYERS,
                             dim_feedforward=ITRANSFORMER_DIM_FF,
                             dropout=ITRANSFORMER_DROPOUT, use_revin=USE_REVIN).to(device)
    else:
        model = DualLSTM(input_size=n_feat, hidden_size=HIDDEN_SIZE,
                         num_layers=NUM_LAYERS, dropout=DROPOUT,
                         use_revin=USE_REVIN).to(device)

    # 两阶段训练：先分类后回归
    phase = 1  # 1=只分类, 2=分类+回归
    if CLASSIFICATION_FIRST:
        for name, param in model.named_parameters():
            if 'reg_head' in name or 'log_var_reg' in name:
                param.requires_grad = False
        logger.info("阶段1: 冻结回归头，仅训练分类")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    crit_reg = nn.SmoothL1Loss()
    crit_cls = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # EMA 影子模型
    ema_model = None
    if EMA_DECAY > 0:
        import copy
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad = False

    # 余弦退火
    scheduler = None
    if USE_COSINE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=1, eta_min=LEARNING_RATE * 0.01)

    # ---- 训练循环 ----
    best_loss, patience_counter = float('inf'), 0
    import heapq
    top_loss = []        # [(loss, epoch, state), ...] top-5 by val_loss
    best_acc = 0.0; best_acc_epoch = 0
    global_step = 0

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ---- 训练循环（每 epoch 实时建序列）----
    for epoch in range(EPOCHS):
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        epoch_rng = np.random.RandomState(seed + epoch * 1000)
        order = epoch_rng.permutation(train_pool)
        # 时间衰减采样：近期股票更多被选中
        if len(stock_ends) == len(train_pool):
            ends_ts = np.array([pd.Timestamp(d).timestamp() for d in stock_ends])
            tw = np.exp(-(ends_ts.max() - ends_ts) / (365.25 * 24 * 3600))
            tw /= tw.sum()
            order = np.random.choice(train_pool, size=min(len(train_pool), 2000),
                                     replace=True, p=tw)
            order = epoch_rng.permutation(order)

        Xt, ypt, ydt = [], [], []
        ep_count = 0
        for idx in order:
            f, tp, td, _, _ = meta[idx]
            x, yp, yd = create_sequences(f, tp, td, SEQ_LEN)
            remain = MAX_SAMPLES_PER_EPOCH - ep_count
            if len(x) > remain:
                x, yp, yd = x[:remain], yp[:remain], yd[:remain]
            Xt.append(x); ypt.append(yp); ydt.append(yd)
            ep_count += len(x)
            if ep_count >= MAX_SAMPLES_PER_EPOCH:
                break

        X_tr = np.concatenate(Xt, axis=0)
        yp_tr = np.concatenate(ypt, axis=0)
        yd_tr = np.concatenate(ydt, axis=0)
        del Xt, ypt, ydt

        # 类别权重（根据本 epoch 样本的涨跌比例）
        classes, counts = np.unique(yd_tr, return_counts=True)
        cls_weights = torch.tensor(
            [len(yd_tr) / (len(classes) * max(c, 1)) for c in counts],
            dtype=torch.float32).to(device)
        crit_cls = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=LABEL_SMOOTHING)
        n_tr = len(X_tr)
        yp_tr_s = scaler_y.transform(yp_tr)  # (n, 3) 多周期

        tr_dataset = TensorDataset(
            torch.from_numpy(X_tr.astype(np.float32)),
            torch.from_numpy(yp_tr_s.astype(np.float32)),   # (n, 3)
            torch.from_numpy(yd_tr.copy()).long())
        del X_tr, yp_tr, yp_tr_s, yd_tr
        tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)

        logger.info(f"Epoch {epoch+1}: {n_tr} 样本, {len(tr_loader)} batches")
        model.train()
        tr_loss = 0.0
        optimizer.zero_grad()
        for i, (bx, bp, bd) in enumerate(tr_loader):
            bx, bp, bd = bx.to(device), bp.to(device), bd.to(device)
            pr, pd_ = model(bx)
            # 两阶段 / 不确定性加权
            if CLASSIFICATION_FIRST and phase == 1:
                loss = crit_cls(pd_, bd) / GRAD_ACCUM_STEPS
            elif hasattr(model, 'log_var_reg'):
                log_reg = torch.clamp(model.log_var_reg, -3.0, 3.0)
                log_cls = torch.clamp(model.log_var_cls, -3.0, 3.0)
                prec_reg = torch.exp(-log_reg)
                prec_cls = torch.exp(-log_cls)
                huber_h = F.smooth_l1_loss(pr, bp, reduction='none').mean(dim=0)
                # 阶段2: 仅对高置信度样本算回归损失
                if CLASSIFICATION_FIRST and phase == 2:
                    cls_prob = torch.softmax(pd_, dim=1)
                    mask = cls_prob.max(dim=1).values > PHASE2_CONFIDENCE
                    if mask.sum() > 0:
                        huber_h = F.smooth_l1_loss(pr[mask], bp[mask], reduction='none').mean(dim=0)
                    else:
                        # 全批次无高置信样本 → 跳过回归损失
                        loss = crit_cls(pd_, bd) * prec_cls + log_cls.squeeze() + 1.0
                        loss = loss / GRAD_ACCUM_STEPS
                        loss.backward(); tr_loss += loss.item() * GRAD_ACCUM_STEPS
                        continue
                loss_reg = (huber_h * prec_reg).sum() + log_reg.sum() + NUM_HORIZONS
                loss_cls = crit_cls(pd_, bd) * prec_cls + log_cls.squeeze() + 1.0
                loss = (loss_reg + loss_cls) / GRAD_ACCUM_STEPS
            else:
                loss_reg = crit_reg(pr, bp)
                loss_cls = crit_cls(pd_, bd)
                loss = (loss_reg + loss_cls) / GRAD_ACCUM_STEPS
            loss.backward()
            tr_loss += loss.item() * GRAD_ACCUM_STEPS

            if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(tr_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                # EMA 更新
                if ema_model is not None:
                    with torch.no_grad():
                        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                            ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1 - EMA_DECAY)
                if global_step < WARMUP_STEPS:
                    for pg in optimizer.param_groups:
                        pg['lr'] = LEARNING_RATE * (global_step + 1) / WARMUP_STEPS
                global_step += 1

        model.eval()
        vl_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for bx, bp, bd in val_loader:
                bx, bp, bd = bx.to(device), bp.to(device), bd.to(device)
                pr, pd_ = model(bx)
                if CLASSIFICATION_FIRST and phase == 1:
                    vl_loss += crit_cls(pd_, bd).item()
                elif hasattr(model, 'log_var_reg'):
                    vl_h = F.smooth_l1_loss(pr, bp, reduction='none').mean(dim=0)
                    vlr = model.log_var_reg.clamp(-3.0, 3.0)
                    vl_reg = (vl_h * torch.exp(-vlr) + vlr + 1.0).sum()
                    vlc = model.log_var_cls.clamp(-3.0, 3.0)
                    vl_cls = crit_cls(pd_, bd) * torch.exp(-vlc) + vlc.squeeze() + 1.0
                    vl_loss += (vl_reg + vl_cls).item()
                else:
                    vl_loss += (crit_reg(pr, bp) + crit_cls(pd_, bd)).item()
                _, preds = torch.max(pd_, 1)
                correct += (preds == bd).sum().item()
                total += bd.size(0)
        avg_tr = tr_loss / len(tr_loader)
        avg_vl = vl_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        if val_acc >= best_acc:
            best_acc = val_acc; best_acc_epoch = epoch + 1
        if avg_vl < best_loss:
            best_loss, patience_counter = avg_vl, 0
        else:
            patience_counter += 1

        # 两阶段切换：分类准确率达目标 → 解冻回归头
        if CLASSIFICATION_FIRST and phase == 1 and (
                val_acc >= PHASE1_ACC_TARGET or epoch >= 20):
            phase_trigger = "acc" if val_acc >= PHASE1_ACC_TARGET else "timeout"
            phase = 2
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
            if scheduler is not None:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=T_0, T_mult=1, eta_min=PHASE2_LR * 0.01)
            logger.info(f"→ 阶段2: 解冻回归头 (lr={PHASE2_LR}, trigger={phase_trigger})")
            best_loss, patience_counter = float('inf'), 0  # 重置（阶段2损失量纲不同）

        state = model.state_dict().copy()
        heapq.heappush(top_loss, (avg_vl, epoch, state))
        if len(top_loss) > 5:
            heapq.heappop(top_loss)

        # 余弦退火 step
        if scheduler is not None:
            scheduler.step()

        marker = "★" if patience_counter == 0 else ""
        if patience_counter == 0 or (epoch+1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}: tr={avg_tr:.4f} vl={avg_vl:.4f} acc={val_acc:.4f} {marker}")
        if patience_counter >= PATIENCE:
            logger.info(f"早停于 epoch {epoch+1}")
            break

    # ---- 最终模型选择 ----
    final_epoch = epoch + 1  # 早停点
    if ema_model is not None:
        model.load_state_dict(ema_model.state_dict())
        logger.info(f"加载 EMA 模型 (decay={EMA_DECAY})")
    else:
        # 只保留训练后半段 + acc 未下降的 checkpoint
        half_point = max(1, final_epoch // 2)
        acc_cutoff = best_acc_epoch if best_acc_epoch > 0 else final_epoch
        filtered = [(ep, s) for _, ep, s in top_loss
                    if ep >= half_point and ep <= acc_cutoff]
        # 过滤后不足 2 个 → 退化为单一最优 loss 模型
        if len(filtered) < 2:
            _, _, best_single = top_loss[0]
            model.load_state_dict(best_single)
            logger.info(f"单最优模型 (filtered={len(filtered)}, epoch={top_loss[0][1]}, best_acc={best_acc:.3f})")
        else:
            avg_state = filtered[0][1]
            for key in avg_state:
                avg_state[key] = sum(s[key].float() for _, s in filtered) / len(filtered)
            model.load_state_dict(avg_state)
            logger.info(f"权重平均: loss-top{len(filtered)} (epoch {half_point}~{acc_cutoff}, best_acc={best_acc:.3f})")

    model.cpu()
    torch.save(model, model_save_path)
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    logger.info(f"模型已保存: {model_save_path}")
    return model, scaler_X, scaler_y, None
