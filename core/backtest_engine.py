# core/backtest_engine.py
import pandas as pd
import numpy as np
import torch
from config.settings import (
    FEATURE_COLS, SEQ_LEN,
    BUY_THRESHOLD, SELL_THRESHOLD,
    STOP_LOSS, TAKE_PROFIT, MAX_POSITION, MIN_VOLATILITY,
    COMMISSION_RATE, MIN_COMMISSION, STAMP_DUTY_RATE, SLIPPAGE
)
from utils.common import create_sequences
from utils.logger import setup_logger

logger = setup_logger(__name__)

def calc_trade_cost(price, shares, is_buy, commission_rate=COMMISSION_RATE,
                    min_commission=MIN_COMMISSION, stamp_duty_rate=STAMP_DUTY_RATE,
                    slippage=SLIPPAGE):
    """
    计算交易成本（佣金、印花税、滑点）
    返回: (总成本, 实际成交价格)
    """
    if shares <= 0 or price <= 0:
        return 0, price

    if is_buy:
        exec_price = price * (1 + slippage)
    else:
        exec_price = price * (1 - slippage)

    turnover = exec_price * shares
    commission = max(turnover * commission_rate, min_commission)
    stamp_duty = turnover * stamp_duty_rate if not is_buy else 0.0
    total_cost = commission + stamp_duty
    return total_cost, exec_price

def run_backtest(df, model, scaler_X, scaler_y, initial_capital=100000,
                 return_log=False, start_date=None, end_date=None):
    """
    统一回测函数：使用已训练好的模型对指定股票数据进行回测
    支持交易成本，返回资金曲线和交易明细
    """
    if df is None or len(df) < SEQ_LEN + 10:
        return None

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]
    if len(df) < SEQ_LEN + 10:
        return None

    if 'Target_Price' not in df.columns:
        from core.features import construct_features, clean_data
        df = construct_features(df)
        df = clean_data(df)

    scaled = scaler_X.transform(df[FEATURE_COLS].values)
    X, _, _ = create_sequences(scaled, df['Target_Price'].values,
                               df['Target_Direction'].values, seq_len=SEQ_LEN)
    if len(X) == 0:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # LSTM 特化：flatten_parameters 优化；Transformer 无此方法，安全跳过
    if hasattr(model, 'lstm') and hasattr(model.lstm, 'flatten_parameters'):
        model.lstm.flatten_parameters()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dates = df['Date'].values[SEQ_LEN:]
    close_prices = df['Close'].values[SEQ_LEN:]

    ma_200 = df['Close'].rolling(200).mean().values[SEQ_LEN:]
    ma_vol = df['Volume'].rolling(20).mean().values[SEQ_LEN:]  # 量能均值
    # 修正：True Range ATR（非 High-Low 极差）
    high_arr = df['High'].values[SEQ_LEN:]
    low_arr = df['Low'].values[SEQ_LEN:]
    pc_arr = df['Close'].shift(1).values[SEQ_LEN:]
    tr = np.maximum(high_arr - low_arr,
                    np.maximum(np.abs(high_arr - pc_arr),
                               np.abs(low_arr - pc_arr)))
    tr = np.nan_to_num(tr, nan=0)
    atr_pct = pd.Series(tr).rolling(14, min_periods=1).mean().values / close_prices * 100

    capital = float(initial_capital)
    holdings = 0.0
    entry_price = 0.0
    entry_idx = 0
    highest_since_entry = 0.0
    min_hold_remaining = 0   # 最低持仓剩余天数
    consecutive_up = 0       # 信号连续天数
    trade_log = []
    positions = []
    asset_history = []  # 逐日总资产

    with torch.no_grad():
        for i in range(len(X_tensor)):
            current_price = close_prices[i]
            x_sample = X_tensor[i].unsqueeze(0)
            _, dir_logits = model(x_sample)
            prob = torch.softmax(dir_logits, dim=1).squeeze().cpu().numpy()
            up_prob = prob[1]

            # 趋势过滤
            if i < len(ma_200) and ma_200[i] > 0:
                price_ma200_ratio = (current_price - ma_200[i]) / ma_200[i]
            else:
                price_ma200_ratio = 0
            regime_buy = BUY_THRESHOLD - 0.04 if price_ma200_ratio > 0 else BUY_THRESHOLD + 0.03
            ma50 = df['Close'].rolling(50).mean().values[SEQ_LEN+i] if (SEQ_LEN+i) < len(df) else current_price
            ma200 = df['Close'].rolling(200).mean().values[SEQ_LEN+i] if (SEQ_LEN+i) < len(df) else current_price
            trend_up = ma50 > ma200
            allow_long = price_ma200_ratio > 0.03  # 价格在年线上方3%
            vol_factor = min(atr_pct[i] / MIN_VOLATILITY, 1.0) if i < len(atr_pct) else 0.5

            # 月度波动率仓位管理
            today = pd.Timestamp(dates[i])
            if i == 0 or today.month != pd.Timestamp(dates[i-1]).month:
                lookback = min(i, 20)
                if lookback >= 5:
                    daily_rets = np.diff(close_prices[i-lookback:i+1]) / close_prices[i-lookback:i]
                    month_vol = np.std(daily_rets) * np.sqrt(252)
                    position_factor = 0.5 if month_vol > 0.30 else 1.0
                else:
                    position_factor = 1.0
            effective_max_pos = MAX_POSITION * position_factor * vol_factor
            # 市场状态：牛市 1.0×仓位上限，熊市 0.5×
            regime_mult = 1.0 if price_ma200_ratio > 0 else 0.5
            effective_max_pos *= regime_mult

            # 量能确认 + ATR 动态止损
            vol_ok = (i < len(ma_vol) and ma_vol[i] > 0 and
                      df['Volume'].values[SEQ_LEN+i] > ma_vol[i] * 0.8)
            dynamic_stop = max(STOP_LOSS, -atr_pct[i] * 2.0 / 100) if i < len(atr_pct) and atr_pct[i] > 0 else STOP_LOSS

            total_asset = capital + holdings * current_price
            current_pos_ratio = (holdings * current_price) / total_asset if total_asset > 0 else 0

            # --- 卖出逻辑 ---
            if holdings > 0:
                profit = (current_price - entry_price) / entry_price if entry_price > 0 else 0
                hold_days = i - entry_idx

                # 最低持仓 10 天：跳过硬止损和 SELL_THRESHOLD（跟踪止盈保留）
                if hold_days < 10:
                    if profit >= TAKE_PROFIT:
                        # 止盈放行
                        pass
                    elif profit <= STOP_LOSS * 2:
                        # 严重亏损放行
                        pass
                    else:
                        positions.append(current_pos_ratio)
                        asset_history.append(total_asset)
                        continue

                # 跟踪止损：记录最高价，回落触发
                highest_since_entry = max(highest_since_entry, current_price)
                if profit > 0.10:
                    trail = max(0.05, (highest_since_entry - entry_price) / entry_price * 0.3)
                    if current_price < highest_since_entry * (1 - trail):
                        cost, exec_price = calc_trade_cost(current_price, holdings, is_buy=False)
                        capital += holdings * exec_price - cost
                        trade_log.append(('跟踪止盈', dates[i], exec_price, (exec_price/entry_price-1)))
                        holdings = 0; entry_price = 0; highest_since_entry = 0
                        positions.append(0)
                        asset_history.append(capital)
                        continue

                # 硬止损/止盈
                if profit >= TAKE_PROFIT:
                    cost, exec_price = calc_trade_cost(current_price, holdings, is_buy=False)
                    capital += holdings * exec_price - cost
                    trade_log.append(('止盈', dates[i], exec_price, profit))
                    holdings = 0
                    entry_price = 0; highest_since_entry = 0
                    positions.append(0)
                    asset_history.append(capital)
                    continue
                elif profit <= dynamic_stop:
                    cost, exec_price = calc_trade_cost(current_price, holdings, is_buy=False)
                    capital += holdings * exec_price - cost
                    trade_log.append(('止损', dates[i], exec_price, profit))
                    holdings = 0
                    entry_price = 0
                    positions.append(0)
                    asset_history.append(capital)
                    continue

                # 持有超过60天且盈利<5%，减仓一半
                if hold_days > 60 and profit < 0.05:
                    sell_shares = holdings * 0.5
                    cost, exec_price = calc_trade_cost(current_price, sell_shares, is_buy=False)
                    capital += sell_shares * exec_price - cost
                    holdings -= sell_shares
                    trade_log.append(('减仓', dates[i], exec_price, profit))
                    total_asset = capital + holdings * current_price
                    current_pos_ratio = (holdings * current_price) / total_asset if total_asset > 0 else 0

            # --- SELL_THRESHOLD 卖出（信号衰减）---
            if holdings > 0 and up_prob * np.exp(-hold_days / 30) < SELL_THRESHOLD:
                cost, exec_price = calc_trade_cost(current_price, holdings, is_buy=False)
                capital += holdings * exec_price - cost
                trade_log.append(('信号卖出', dates[i], exec_price, (exec_price/entry_price-1)))
                holdings = 0; entry_price = 0; highest_since_entry = 0
                positions.append(0)
                asset_history.append(capital)
                continue

            # --- 买入逻辑 ---
            # 信号延续：需要连续 2 天上穿阈值才买入
            if up_prob > regime_buy:
                consecutive_up += 1
            else:
                consecutive_up = 0

            if holdings == 0 and allow_long and vol_ok and consecutive_up >= 2 and vol_factor > 0.3:
                position_ratio = min((up_prob - 0.45) * 1.0, effective_max_pos)
                position_ratio = max(position_ratio, 0.1)
                buy_amount = capital * position_ratio
                cost, exec_price = calc_trade_cost(current_price, 1, is_buy=True)  # 估算单位成本
                max_shares = int((buy_amount - cost) / exec_price) if exec_price > 0 else 0
                if max_shares > 0:
                    cost_actual, exec_price_actual = calc_trade_cost(current_price, max_shares, is_buy=True)
                    total_cost = max_shares * exec_price_actual + cost_actual
                    if total_cost <= capital:
                        holdings = max_shares
                        entry_price = exec_price_actual
                        entry_idx = i
                        highest_since_entry = exec_price_actual
                        capital -= total_cost
                        trade_log.append(('买入', dates[i], exec_price_actual, None, position_ratio))
                        total_asset = capital + holdings * current_price
                        current_pos_ratio = (holdings * current_price) / total_asset if total_asset > 0 else 0

            positions.append(current_pos_ratio)
            asset_history.append(total_asset)

    if not positions:
        return None

    # 强制长度一致
    target_len = len(positions)
    dates = dates[:target_len]
    close_prices = close_prices[:target_len]

    # 资金曲线：实际逐日总资产（含手续费印花税滑点）
    capital_curve = np.array(asset_history, dtype=float)
    if len(capital_curve) < target_len:
        pad = np.full(target_len - len(capital_curve),
                      capital_curve[-1] if len(capital_curve) > 0 else float(initial_capital))
        capital_curve = np.concatenate([capital_curve, pad])
    capital_curve = capital_curve[:target_len]

    # 断言长度一致
    assert len(dates) == len(close_prices) == len(positions) == len(capital_curve), \
        f"长度不一致: dates={len(dates)}, close={len(close_prices)}, positions={len(positions)}, capital={len(capital_curve)}"

    backtest_df = pd.DataFrame({
        'Date': dates,
        'Close': close_prices,
        'Position': positions,
        'Capital': capital_curve
    })

    if return_log:
        return backtest_df, trade_log
    else:
        return backtest_df


def get_signal_for_day(df, model_path, seq_len, buy_threshold):
    """盘后用：取最新一天信号的 up_prob, trend_ok, vol_ok"""
    import torch
    import joblib
    from config.settings import INPUT_DIM, ITRANSFORMER_D_MODEL, ITRANSFORMER_NHEAD, \
        ITRANSFORMER_NUM_LAYERS, ITRANSFORMER_DIM_FF, ITRANSFORMER_DROPOUT, \
        USE_REVIN, MODEL_TYPE
    from core.model import create_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(MODEL_TYPE, INPUT_DIM, ITRANSFORMER_D_MODEL,
                         ITRANSFORMER_NHEAD, ITRANSFORMER_NUM_LAYERS,
                         ITRANSFORMER_DIM_FF, ITRANSFORMER_DROPOUT,
                         USE_REVIN).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.state_dict(), strict=False)
    model.eval()
    sX = joblib.load(os.path.join(OUTPUT_DIR, 'scaler_X_final.pkl'))
    sY = joblib.load(os.path.join(OUTPUT_DIR, 'scaler_Y_final.pkl'))

    close = df['Close'].values[-seq_len-50:].astype(np.float32)
    feat_raw = df[FEATURE_COLS].values[-seq_len-50:].astype(np.float32)
    feat = sX.transform(feat_raw)

    x = torch.FloatTensor(feat[-seq_len:]).unsqueeze(0).to(device)
    with torch.no_grad():
        _, dir_logits = model(x)
        prob = torch.softmax(dir_logits, dim=1).squeeze().cpu().numpy()
    up_prob = float(prob[1])

    current_price = close[-1]
    ma200 = pd.Series(close).rolling(200).mean().values[-1] if len(close) >= 200 else current_price
    price_ma200_ratio = (current_price - ma200) / ma200 if ma200 > 0 else 0
    ma50 = pd.Series(close).rolling(50).mean().values[-1]
    trend_ok = bool(ma50 > ma200 and price_ma200_ratio > 0.03)

    vol_arr = df['Volume'].values[-(seq_len+20):]
    ma_vol = pd.Series(vol_arr).rolling(20).mean().values[-1]
    vol_ok = bool(vol_arr[-1] > ma_vol * 0.8 if ma_vol > 0 else False)

    return up_prob, trend_ok, vol_ok


def compute_position_size(df, up_prob, buy_threshold, max_position):
    """返回建议仓位比例 [0, max_position]"""
    return round(min((up_prob - 0.45) * 1.0, max_position), 2)