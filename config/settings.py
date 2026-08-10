# config/settings.py
# # 盘后（15:30）
# python scripts/download_data.py
# python scripts/daily_select.py          # 选股 → output/daily_picks.csv
# python scripts/daily_predict.py         # 择时 → output/today_signals.csv

# # 盘中（次日 9:25）
# python scripts/sim_live.py              # 模拟监控
import os
import datetime

# ------------------- 时间范围 -------------------
TRAIN_END_DATE = "2025-12-31"
BACKTEST_START_DATE = "2025-01-01"
TODAY = datetime.datetime.now().strftime("%Y-%m-%d")   # 必须定义
TUSHARE_TOKEN = ""    # tushare pro token

# ------------------- 路径 -------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "stock_data_cache")  # 必须定义
OUTPUT_DIR = os.path.join(BASE_DIR, "output")           # 统一输出目录
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
SCALER_X_PATH = os.path.join(BASE_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_y.pkl")
WHITELIST_FILE = os.path.join(BASE_DIR, "whitelist.csv")
WHITELIST_EXTENDED_FILE = os.path.join(BASE_DIR, "whitelist_extended.csv")

# ------------------- 特征列（59 个） -------------------
FEATURE_COLS = [
    # 动量 / 收益率（3）
    'Momentum_5', 'Momentum_10', 'Return_1d',
    # 波动率（2）
    'Volatility_5', 'Volatility_10',
    # 均线 & 偏离（6）
    'MA_5', 'MA_10', 'MA_20',
    'Price_MA_5_Ratio', 'Price_MA_20_Ratio',
    'MA_5_20_diff',
    # 超买超卖（2）
    'RSI_14', 'BB_position',
    # 量价关系（3）
    'Volume_Ratio', 'High_Low_Ratio', 'Amount_Ratio',
    # 原始字段标准化（5）
    'Close', 'Volume', 'PctChg', 'TradeStatus', 'Turnover',
    # 估值（4）— 由 data_loader.enrich_fundamentals 获取真实数据
    'PeTTM', 'PbMRQ', 'PsTTM', 'PcfNcfTTM',
    # MACD（3）
    'MACD_DIF', 'MACD_DEA', 'MACD_HIST',
    # KDJ（3）
    'KDJ_K', 'KDJ_D', 'KDJ_J',
    # ATR（2）
    'ATR_14', 'ATRP',
    # OBV 斜率（1）
    'OBV_slope',
    # 连涨连跌（2）
    'Consecutive_Up', 'Consecutive_Down',
    # 价格分位（1）
    'Price_Position_20',
    # K 线影线（2）
    'Upper_Shadow', 'Lower_Shadow',
    # 量能加速度（1）
    'Volume_Accel',
    # 资金流量（1）
    'MFI_14',
    # 🆕 周期特征（6）— sin/cos 编码
    'Dow_sin', 'Dow_cos',          # 星期几
    'Month_sin', 'Month_cos',      # 月份
    'Quarter_sin', 'Quarter_cos',  # 季度
    # 交叉特征（2）
    'Mom_Vol', 'RSI_Vol',
    # 🆕 另类数据特征（6）— tushare 接口，缺失则填充 0
    'North_Hold',                   # 北向持仓比例
    'ROE', 'GrossMargin',           # 基本面质量
    'NetMargin', 'DebtRatio',
    'Insider_Signal',               # 股东增减持信号（-1/0/+1）
    # 🆕 衍生特征（3）— 自动计算
    'North_Hold_Chg_5d',            # 北向 5 日变化
    'North_Hold_Chg_20d',           # 北向 20 日变化
    'Insider_Buy_Window',           # 股东 60 日净买入
]

INPUT_DIM = len(FEATURE_COLS)   # 自动跟随特征数变化（当前 58）

# ------------------- 模型选择 -------------------
# "lstm" / "itransformer" / "gru_itrans" / "nbeats_itrans" / "fusion_itrans"
MODEL_TYPE = "itransformer"

# ------------------- GRU + iTransformer 超参数 -------------------
GRU_SEQ_LEN = 20       # GRU 的短期回溯窗口（天）
GRU_HIDDEN = 16        # GRU 输出的短期趋势因子维度

# ------------------- N-BEATS + iTransformer 超参数 -------------------
NBEATS_STACKS = 2      # N-BEATS 栈数
NBEATS_HIDDEN = 256    # N-BEATS 隐藏层维度
NBEATS_DIM = 16        # N-BEATS 输出的长期趋势因子维度

# ------------------- 模型超参数（共用）-------------------
SEQ_LEN = 90

# ------------------- LSTM 超参数 -------------------
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3

# ------------------- iTransformer 超参数 -------------------
ITRANSFORMER_D_MODEL = 128       # 嵌入维度（47 特征 → 128 维，nhead=8 时每头 16 维）
ITRANSFORMER_NHEAD = 8           # 注意力头数
ITRANSFORMER_NUM_LAYERS = 4      # Encoder 层数
ITRANSFORMER_DIM_FF = 512        # 前馈网络维度（4× d_model）
ITRANSFORMER_DROPOUT = 0.5       # 高 dropout 配合 4 层防过拟合

# ------------------- RevIN -------------------
USE_REVIN = True                 # 可逆实例归一化（消除股票间量纲差异）

# ------------------- 学习率预热 -------------------
WARMUP_STEPS = 3000              # 线性预热步数（从 0 → LR）

# ------------------- 训练参数 -------------------
BATCH_SIZE = 128
MAX_SAMPLES_PER_EPOCH = 200000   # 每 epoch 最多 20 万样本（8G VRAM 上限）
EPOCHS = 100
GRAD_ACCUM_STEPS = 2

# ---- 数据 ----
MAX_SAMPLES_PER_EPOCH = 200000
EPOCHS = 100
PATIENCE = 25
REGRESSION_WEIGHT = 0.1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
GRAD_ACCUM_STEPS = 4     # 梯度累积（等效 batch=512，防 8GB 显存 OOM）
USE_STYLE_NEUTRAL = True # 行业中性化（减去同行业均值/标准差）

# ------------------- 正则化 / 增强 -------------------
EMA_DECAY = 0.999         # EMA 权重衰减（0=不启用）
LABEL_SMOOTHING = 0.1     # 标签平滑（0=不启用）
USE_COSINE_SCHEDULER = True  # 余弦退火学习率
T_0 = 20                   # 余弦重启周期（epoch）— 拉长避免过早重启

# ------------------- 两阶段训练 -------------------
CLASSIFICATION_FIRST = True   # 先分类后回归
PHASE1_ACC_TARGET = 0.75      # 阶段1目标准确率
PHASE2_LR = 1e-5              # 阶段2回归头学习率
PHASE2_CONFIDENCE = 0.6       # 阶段2置信度阈值（只对高置信样本算回归损失）

# ------------------- 多周期预测 -------------------
NUM_HORIZONS = 3            # 同时预测的周期数
HORIZON_DAYS = [5, 10, 20]  # 预测天数

# ------------------- 回测参数 -------------------
BUY_THRESHOLD = 0.38    # 降低门槛（多周期模型输出更保守）
SELL_THRESHOLD = 0.42
STOP_LOSS = -0.08
TAKE_PROFIT = 0.20
MAX_POSITION = 0.6
MIN_VOLATILITY = 1.0

# ------------------- 回撤控制 -------------------
DRAWDOWN_THRESHOLD = 0.025
RECOVERY_RATIO = 0.25

# ------------------- 白名单筛选条件 -------------------
WHITELIST_MIN_RETURN = 0.15
WHITELIST_MIN_TRADES = 2
WHITELIST_MAX_DRAWDOWN = 0.65

# ------------------- 日志配置 -------------------
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "logs", "quant.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ------------------- 交易成本参数 -------------------
COMMISSION_RATE = 0.00025        # 佣金费率（万2.5）
MIN_COMMISSION = 5.0             # 最低佣金（元）
STAMP_DUTY_RATE = 0.001          # 印花税率（仅卖出，千1）
SLIPPAGE = 0.001                 # 滑点（买卖各0.1%）

# ------------------- 扩展窗口交叉验证参数 -------------------
TRAIN_START_DATE = "2018-01-01"        # 训练起始日期（固定）
WINDOW_LENGTH_YEARS = 3                # 每个窗口的训练集长度（年）
TEST_LENGTH_YEARS = 1                  # 每个窗口的测试集长度（年）
NUM_WINDOWS = 1                        # 窗口数量（最多不超过数据总长度）


# ----- 配置（覆盖 settings.py 默认值） -----
MAX_WORKERS = 1
TRAIN_STOCKS = 10          # 参与训练的股票数量
TEST_STOCKS = 6000            # 回测股票数（None 表示全部）
WHITELIST_MIN_RETURN = 0.15
WHITELIST_MIN_TRADES = 2
WHITELIST_MAX_DRAWDOWN = 0.65

SEED = 42

# 本地覆盖（不上传 Git）
try:
    from config.local_settings import *
except ImportError:
    pass