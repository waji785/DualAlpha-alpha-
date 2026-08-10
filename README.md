# DualAlpha — A股双阶段量化交易系统

**选股（树模型集成） + 择时（iTransformer） → 模拟盘全自动交易**

## 架构

```
tushare 日线 (2015-2026, 5000+ 只)
  │
  ├→ 58 维因子计算（量价 + 基本面 + 北向 + 股东行为）
  │
  ├→ TreeEnsemble (XGBoost + LightGBM + CatBoost)
  │     └→ 截面排序 → Top-N 股票池
  │
  └→ iTransformer (58 特征 × 90 天)
        └→ 两阶段训练（分类优先 → 回归微调）
              └→ up_prob 择时信号
                    │
                    └→ 风险平价 + 市场自适应 → 最终仓位
```

## 核心特性

- **58 维因子**：量价（33）、基本面（6）、北向资金（3）、股东行为（2）、周期编码（6）、交叉特征（2） + 衍生
- **两阶段训练**：阶段 1 冻结回归头直训分类 → acc≥75% 解冻回归头微调
- **市场自适应**：牛市低门槛 + 满仓、熊市高门槛 + 半仓
- **风险平价**：协方差矩阵 + 置信度加权 → 每只股票权重
- **8 窗口滚动回测**：2019-2026，逐窗口验证模型衰减周期
- **T+1 模拟盘**：交易费率、涨跌停、行业集中度限制，全自动日志

## 快速开始

```bash
# 1. 安装依赖
pip install tushare torch lightgbm xgboost catboost pandas numpy scikit-learn joblib

# 2. 配置 token（config/local_settings.py）
TUSHARE_TOKEN = "你的token"

# 3. 下载数据
python scripts/download_data.py --full --start 2015-01-01

# 4. 训练模型
python scripts/train_final_model.py       # 时序（~8h GPU）
python scripts/train_selector.py           # 选股（~2min）

# 5. 每日盘后
python scripts/download_data.py             # 增量更新
python scripts/daily_select.py              # 选股
python scripts/daily_predict.py             # 择时打分

# 6. 次日盘中
python scripts/sim_live.py                  # 模拟监控
```

## 回测结果（中低频策略，8 窗口滚动）

| 窗口 | 测试年份 | 收益率 | 交易笔数 | 市场环境 |
|------|---------|--------|---------|---------|
| W1   | 2019    | -3.55% | 17      | 牛市（训熊测牛） |
| W2   | 2020    | +1.19% | 29      | 大幅波动 |
| W3   | 2021    | -0.75% | 11      | 横盘震荡 |
| W4   | 2022    | -5.10% | 33      | 大熊市（跑赢指数 16%） |
| W7   | 2025    | -3.59% | 32      | 回升中 |
| W8   | 2026    | -1.10% | 18      | 趋近收敛 |

模型对市场结构变化的适应窗口约 2-3 年，建议每 6 个月重训。

## 技术栈

`Python` `PyTorch` `LightGBM` `XGBoost` `CatBoost` `Pandas` `NumPy` `Tushare`

## 免责声明

本项目仅供研究和学习使用，不构成任何投资建议。历史回测收益不代表未来表现。
