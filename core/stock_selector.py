# core/stock_selector.py
"""树模型集成选股层：XGBoost + LightGBM + CatBoost（任一可用即可，sklearn GBR 兜底）"""
import os, joblib
import numpy as np
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TreeEnsemble:
    """三模型集成，自动跳过未安装的库"""
    def __init__(self):
        self.xgb = None
        self.lgb = None
        self.cat = None
        self.gbr = None
        self._model_weights = {}
        self.trained = False

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, groups=None):
        # ---- XGBoost ----
        try:
            import xgboost as xgb
            self.xgb = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=1,
                random_state=42, n_jobs=-1, verbosity=0)
            logger.info("训练 XGBoost...")
            self.xgb.fit(X, y, sample_weight=sample_weight)
        except ImportError:
            logger.info("XGBoost 未安装 (pip install xgboost)")

        # ---- LightGBM ----
        try:
            import lightgbm as lgb
            if groups is not None:
                self.lgb = lgb.LGBMRanker(
                    objective='lambdarank', n_estimators=300,
                    max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=1, reg_lambda=1,
                    random_state=42, n_jobs=-1, verbosity=-1)
                logger.info("训练 LightGBM LambdaMART...")
                self.lgb.fit(X, y, group=groups, sample_weight=sample_weight)
            else:
                self.lgb = lgb.LGBMRegressor(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=1, reg_lambda=1,
                    random_state=42, n_jobs=-1, verbosity=-1)
                logger.info("训练 LightGBM...")
                self.lgb.fit(X, y, sample_weight=sample_weight)
        except ImportError:
            logger.info("LightGBM 未安装")

        # ---- CatBoost ----
        try:
            from catboost import CatBoostRegressor
            self.cat = CatBoostRegressor(
                iterations=300, depth=6, learning_rate=0.05,
                random_seed=42, verbose=False, allow_writing_files=False)
            logger.info("训练 CatBoost...")
            self.cat.fit(X, y, sample_weight=sample_weight)
        except ImportError:
            logger.info("CatBoost 未安装 (pip install catboost)")

        # sklearn 兜底
        if self.xgb is None and self.lgb is None and self.cat is None:
            from sklearn.ensemble import GradientBoostingRegressor
            self.gbr = GradientBoostingRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
            logger.info("使用 sklearn GradientBoosting 兜底")
            self.gbr.fit(X, y, sample_weight=sample_weight)

        self._model_weights = self._compute_ic_weights(X_val, y_val)
        self.trained = True
        logger.info("树模型训练完成")

    def _compute_ic_weights(self, X_val, y_val):
        weights = {}
        if X_val is None or y_val is None or len(y_val) < 50:
            return weights
        from scipy.stats import spearmanr
        models = {'xgb': self.xgb, 'lgb': self.lgb, 'cat': self.cat, 'gbr': self.gbr}
        ics = {}
        for name, m in models.items():
            if m is not None:
                ic, _ = spearmanr(m.predict(X_val), y_val)
                ics[name] = max(abs(ic), 0.02)
        if ics:
            total = sum(ics.values())
            weights = {k: v/total for k, v in ics.items()}
            logger.info(f"IC权重: { {k:round(v,3) for k,v in weights.items()} }")
        return weights

    def predict(self, X):
        preds, w = [], self._model_weights
        for name, m in [('xgb', self.xgb), ('lgb', self.lgb), ('cat', self.cat), ('gbr', self.gbr)]:
            if m is not None:
                preds.append(m.predict(X) * w.get(name, 1.0))
        if not preds:
            raise RuntimeError("无可用模型")
        return np.sum(preds, axis=0) if w else np.mean(preds, axis=0)

    def save(self, path):
        joblib.dump({'xgb': self.xgb, 'lgb': self.lgb, 'cat': self.cat, 'gbr': self.gbr}, path)
        logger.info(f"选择器模型保存: {path}")

    def load(self, path):
        data = joblib.load(path)
        self.xgb = data.get('xgb')
        self.lgb = data.get('lgb')
        self.cat = data.get('cat')
        self.gbr = data.get('gbr')
        self._model_weights = data.get('weights', {})
        self.trained = True
        logger.info(f"选择器模型加载: {path}")


def build_selector_features(df, feat_cols=None):
    if df is None or len(df) == 0:
        return None
    if feat_cols is None:
        from config.settings import FEATURE_COLS
        feat_cols = FEATURE_COLS
    return df.iloc[-1][feat_cols].values.astype(np.float32)


def build_selector_dataset(all_dfs, stock_codes, feat_cols=None,
                           forward_days=20, step_months=1):
    if feat_cols is None:
        from config.settings import FEATURE_COLS
        feat_cols = FEATURE_COLS

    X_list, y_list, dates_list, months_list = [], [], [], []
    for code in stock_codes:
        df = all_dfs.get(code)
        if df is None or len(df) < 60:
            continue
        df = df.sort_values('Date').reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.to_period('M')
        for month in df['Month'].unique()[::step_months]:
            month_df = df[df['Month'] == month]
            if len(month_df) < 1:
                continue
            idx = month_df.index[-1]
            if idx + forward_days >= len(df):
                continue
            feat = df.loc[idx, feat_cols].values.astype(np.float32)
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                continue
            future_close = df.loc[idx + forward_days, 'Close']
            current_close = df.loc[idx, 'Close']
            ret = (future_close - current_close) / current_close
            X_list.append(feat); y_list.append(ret)
            dates_list.append(df.loc[idx, 'Date'])
            months_list.append(month)

    if not X_list:
        return np.array([]), np.array([]), np.array([])

    # 截面排序：同月内 rank → 百分位[0,1]
    df_tmp = pd.DataFrame({'month': months_list, 'ret': y_list})
    df_tmp['y'] = df_tmp.groupby('month')['ret'].rank(pct=True)
    y_rank = df_tmp['y'].values.astype(np.float32)

    latest = max(dates_list)
    weights = np.exp(-np.array([(latest - d).days for d in dates_list]) / 365)
    weights = weights / weights.mean()
    # LambdaMART 分组：按月连续排列，groups 为每月样本数
    df_tmp['idx'] = range(len(months_list))
    groups = df_tmp.groupby('month').size().values
    logger.info(f"截面数据集: {len(X_list)} 样本 (排序目标)")
    return np.array(X_list), y_rank, weights, groups


def compute_time_weights(dates, ref_date=None):
    """根据参考日期计算时间衰减权重（训练时 ref_date = 训练集最晚日期）"""
    if ref_date is None:
        ref_date = max(dates)
    w = np.exp(-np.array([(ref_date - d).days for d in dates]) / 365)
    return w / w.mean()
