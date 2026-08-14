# research/factor_dev.ipynb 使用说明
#
# 在项目根目录启动 Jupyter:
#   jupyter notebook
# 然后打开 research/factor_dev_template.py 作为起点

"""因子开发模板 —— 复制到新笔记本中开始实验"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from research.factor_lab import FactorLab
from core.data_loader import load_from_cache

# ===== 1. 加载实验室 =====
lab = FactorLab()

# ===== 2. 加载测试股票 =====
df = load_from_cache('000001')  # 平安银行
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').sort_index()
print(f"测试数据: {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")

# ===== 3. 定义新因子 =====
def my_new_factor(df):
    """
    你的因子逻辑写在这里
    返回: pd.Series, index 和 df.index 一致
    """
    # 示例: 20日动量 / 20日波动率
    ret = df['Close'].pct_change(20)
    vol = df['Close'].pct_change().rolling(20).std()
    return ret / (vol + 1e-8)

# ===== 4. 测试 =====
result = lab.test_factor('MomentumDivVol', my_new_factor, df)
print(result)

# ===== 5. 批量测试（50 只股票） =====
# results = lab.scan({'MomentumDivVol': my_new_factor}, n_samples=50)

# ===== 6. 通过后集成 =====
# lab.promote('MomentumDivVol')
