# core/stock_selector.py
"""树模型选股层：多模型融合（P50 期望收益）+ LightGBM 分位数（P10/P90 风控）"""
import os, joblib
import numpy as np
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 堆叠集成开关：False=IC加权融合(回测最优)，True=MLP+Ridge堆叠(回测证实过拟合负优化)
STACKING_ENABLED = False


class TreeEnsemble:
    """融合模型输出 P50 期望收益（选股），LightGBM 分位数输出 P10/P90（风控）"""
    def __init__(self):
        # 融合模型（P50 期望收益）
        self.xgb = None
        self.lgb = None
        self.cat = None
        self.gbr = None
        self.mlp = None            # 深度学习（堆叠第一层，多样性）
        self.meta = None           # 堆叠第二层元模型
        self._mlp_scaler = None    # MLP 特征标准化器
        self._model_weights = {}
        # 分位数模型（P10 下行风险 / P90 上行潜力）
        self.lgb_p10 = None
        self.lgb_p90 = None
        self.trained = False

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, groups=None):
        import lightgbm as lgb

        # ---- 融合模型：P50 期望收益（普通回归）----
        try:
            import xgboost as xgb
            self.xgb = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=1,
                random_state=42, n_jobs=-1, verbosity=0)
            logger.info("训练 XGBoost (P50)...")
            self.xgb.fit(X, y, sample_weight=sample_weight)
        except ImportError:
            logger.info("XGBoost 未安装")

        try:
            self.lgb = lgb.LGBMRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1, reg_lambda=1,
                random_state=42, n_jobs=-1, verbosity=-1)
            logger.info("训练 LightGBM (P50)...")
            self.lgb.fit(X, y, sample_weight=sample_weight)
        except ImportError:
            logger.info("LightGBM 未安装")

        try:
            from catboost import CatBoostRegressor
            self.cat = CatBoostRegressor(
                iterations=300, depth=6, learning_rate=0.05,
                random_seed=42, verbose=False, allow_writing_files=False)
            logger.info("训练 CatBoost (P50)...")
            self.cat.fit(X, y, sample_weight=sample_weight)
        except ImportError:
            logger.info("CatBoost 未安装")

        if self.xgb is None and self.lgb is None and self.cat is None:
            from sklearn.ensemble import GradientBoostingRegressor
            self.gbr = GradientBoostingRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
            logger.info("使用 sklearn GradientBoosting 兜底")
            self.gbr.fit(X, y, sample_weight=sample_weight)

        if STACKING_ENABLED:
            # ---- MLP（深度学习，堆叠第一层的多样性）----
            try:
                from sklearn.neural_network import MLPRegressor
                from sklearn.preprocessing import StandardScaler
                self._mlp_scaler = StandardScaler()
                X_scaled = self._mlp_scaler.fit_transform(X)
                self.mlp = MLPRegressor(
                    hidden_layer_sizes=(256, 128), activation='relu',
                    max_iter=200, early_stopping=True, validation_fraction=0.1,
                    random_state=42)
                logger.info("训练 MLP (P50)...")
                self.mlp.fit(X_scaled, y)
            except Exception as e:
                logger.info(f"MLP 训练失败: {e}")

            # ---- 堆叠元模型（第二层）：用第一层在验证集上的预测训练 Ridge ----
            self.meta = None
            if X_val is not None and y_val is not None and len(y_val) >= 50:
                base_preds = self._base_predictions(X_val)
                if len(base_preds) >= 2:
                    from sklearn.linear_model import Ridge
                    self.meta = Ridge(alpha=1.0)
                    self.meta.fit(np.column_stack(base_preds), y_val)
                    logger.info(f"堆叠元模型: Ridge({len(base_preds)}个基学习器)")

        self._model_weights = self._compute_ic_weights(X_val, y_val)

        # ---- 分位数模型：P10 / P90（LightGBM quantile）----
        self.lgb_p10 = lgb.LGBMRegressor(
            objective='quantile', alpha=0.1,
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1, reg_lambda=1,
            random_state=42, n_jobs=-1, verbosity=-1)
        self.lgb_p90 = lgb.LGBMRegressor(
            objective='quantile', alpha=0.9,
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1, reg_lambda=1,
            random_state=42, n_jobs=-1, verbosity=-1)
        logger.info("训练 LightGBM 分位数 (P10/P90)...")
        self.lgb_p10.fit(X, y, sample_weight=sample_weight)
        self.lgb_p90.fit(X, y, sample_weight=sample_weight)

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

    def _base_predictions(self, X):
        """第一层各模型的预测列表（用于堆叠元模型）"""
        preds = []
        for m in [self.xgb, self.lgb, self.cat, self.gbr]:
            if m is not None:
                preds.append(m.predict(X))
        if self.mlp is not None and self._mlp_scaler is not None:
            preds.append(self.mlp.predict(self._mlp_scaler.transform(X)))
        return preds

    def predict(self, X):
        """输出 P50 期望收益（堆叠融合或 IC 加权，越大越好）"""
        if self.meta is not None:
            base_preds = self._base_predictions(X)
            if len(base_preds) >= 2:
                return self.meta.predict(np.column_stack(base_preds))
        preds, w = [], self._model_weights
        for name, m in [('xgb', self.xgb), ('lgb', self.lgb), ('cat', self.cat), ('gbr', self.gbr)]:
            if m is not None:
                preds.append(m.predict(X) * w.get(name, 1.0))
        if not preds:
            raise RuntimeError("无可用模型")
        return np.sum(preds, axis=0) if w else np.mean(preds, axis=0)

    def predict_downside(self, X):
        """输出 P10 下行风险（越低越危险，用于风控）"""
        if self.lgb_p10 is None:
            return np.full(len(X), 0.0)
        return self.lgb_p10.predict(X)

    def predict_upside(self, X):
        """输出 P90 上行潜力（用于止盈参考）"""
        if self.lgb_p90 is None:
            return self.predict(X)
        return self.lgb_p90.predict(X)

    def predict_interval(self, X):
        """输出 (P50, P10, P90) 三元组"""
        return (self.predict(X), self.predict_downside(X), self.predict_upside(X))

    def save(self, path):
        joblib.dump({'xgb': self.xgb, 'lgb': self.lgb, 'cat': self.cat, 'gbr': self.gbr,
                     'mlp': self.mlp, 'meta': self.meta,
                     '_mlp_scaler': self._mlp_scaler,
                     'lgb_p10': self.lgb_p10, 'lgb_p90': self.lgb_p90,
                     'weights': self._model_weights}, path)
        logger.info(f"选择器模型保存: {path}")

    def load(self, path):
        data = joblib.load(path)
        self.xgb = data.get('xgb')
        self.lgb = data.get('lgb')
        self.cat = data.get('cat')
        self.gbr = data.get('gbr')
        self.mlp = data.get('mlp')
        self.meta = data.get('meta')
        self._mlp_scaler = data.get('_mlp_scaler')
        self.lgb_p10 = data.get('lgb_p10')
        self.lgb_p90 = data.get('lgb_p90')
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
                           forward_days=20, step_months=1, industry_rank=False,
                           half_life=365):
    if feat_cols is None:
        from config.settings import FEATURE_COLS
        feat_cols = FEATURE_COLS

    # 行业映射（industry_rank=True 时加载，用于行业内排名）
    ind_map = None
    if industry_rank:
        from core.data_loader import get_industry_map
        ind_map = get_industry_map()

    X_list, y_list, dates_list, months_list, ind_list = [], [], [], [], []
    for code in stock_codes:
        df = all_dfs.get(code)
        if df is None or len(df) < 60:
            continue
        df = df.sort_values('Date').reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.to_period('M')
        ind = ind_map.get(code, '未知') if ind_map else '未知'
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
            ind_list.append(ind)

    if not X_list:
        return np.array([]), np.array([]), np.array([])

    # 标签：原始收益率（分位数回归）or 行业内排名（行业相对强弱）
    if industry_rank:
        df_tmp = pd.DataFrame({'month': months_list, 'ind': ind_list, 'ret': y_list})
        df_tmp['y'] = df_tmp.groupby(['month', 'ind'])['ret'].rank(pct=True)
        y = df_tmp['y'].values.astype(np.float32)
        logger.info(f"截面数据集: {len(X_list)} 样本 (行业相对排名目标)")
    else:
        y = np.array(y_list, dtype=np.float32)
        logger.info(f"截面数据集: {len(X_list)} 样本 (分位数回归目标)")

    latest = max(dates_list)
    weights = np.exp(-np.array([(latest - d).days for d in dates_list]) / half_life)
    weights = weights / weights.mean()
    groups = pd.DataFrame({'month': months_list}).groupby('month').size().values
    return np.array(X_list), y, weights, groups


def compute_time_weights(dates, ref_date=None):
    """根据参考日期计算时间衰减权重（训练时 ref_date = 训练集最晚日期）"""
    if ref_date is None:
        ref_date = max(dates)
    w = np.exp(-np.array([(ref_date - d).days for d in dates]) / 365)
    return w / w.mean()
