# DualAlpha —— A股多因子量化选股系统

一个基于**树模型集成 + 多源另类数据**的 A 股量化选股系统。核心是"截面选股"：每月从全市场选出 20 只股票等权持有，配合止损控制尾部风险。

## 核心特性

- **多源另类数据因子**：财务指标（ROE/毛利率/净利率/负债率）+ 融资融券 + 龙虎榜机构净买入 + 大宗交易 + 动量 + Alpha158 技术因子，共 55 个特征
- **分位数回归选股器**：LightGBM 分位数回归输出 P50（期望收益）/ P10（下行风险）/ P90（上行潜力），XGBoost/LightGBM/CatBoost 三模型 IC 加权融合
- **P10 下行风控 + 行业中性化**：排除下行风险大的股票，每行业仅选 1 只，天然分散
- **截断亏损、让利润奔跑**：持有期跌超 10% 止损，上涨不设顶（靠月度调仓自然止盈）
- **滚动窗口交叉验证**：8 个年度窗口（2019-2026）样本外验证，避免过拟合

## 最终策略配置

```
选股器：LightGBM 分位数回归 + XGBoost/LightGBM/CatBoost IC加权融合（55特征）
标签：  未来 20 天收益
选股：  P10风控(排除P10<-15%) + 行业中性化(每行业1只) + Top20
持有：  等权月度再平衡 + 止损10%（只止损不止盈）
```

## 回测结果（8窗口滚动验证，2019-2026）

| 窗口 | 年份 | 市场环境 | 总收益 | 最大回撤 | 夏普 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 1 | 2019 | 普涨 | +99.44% | -5.88% | 1.930 |
| 2 | 2020 | 疫情波动 | +46.20% | -6.97% | 1.809 |
| 3 | 2021 | 结构牛 | +63.88% | -6.27% | 2.381 |
| 4 | 2022 | 大熊市 | +22.91% | -8.97% | 0.965 |
| 5 | 2023 | 震荡 | +29.82% | -9.89% | 1.305 |
| 6 | 2024 | 政策牛 | +37.17% | -10.06% | 0.983 |
| 7 | 2025 | 稳健上行 | +50.35% | -7.08% | 2.729 |
| 8 | 2026(至8月) | 分化 | +18.07% | -6.95% | 1.127 |

**核心指标**：平均夏普 **1.654**，7.5 年复合收益 **18.57 倍**（年化约 44%）。

**关键亮点**：2022 年大熊市和 2026 年科技暴跌中依然正收益——龙虎榜/大宗交易因子识别"资金护盘"，止损截断尾部风险。

## 技术栈

| 类别 | 技术 |
| :--: | :--: |
| 选股模型 | LightGBM / XGBoost / CatBoost（分位数回归 + IC加权融合）|
| 择时模型 | PyTorch iTransformer（RevIN + 注意力池化，双头回归/分类）|
| 数据源 | Tushare（复权数据 + 另类数据 API）|
| 数据处理 | Pandas / NumPy / Scikit-learn / scipy |
| 模型持久化 | joblib / torch |
| 回测引擎 | 自定义滚动窗口回测（含佣金/印花税/止损/停牌剔除）|

## 项目结构

```text
.
├── config/settings.py           # 全局配置（FEATURE_COLS、阈值、路径）
├── core/
│   ├── data_downloader.py       # Tushare 数据下载 + 复权 + 另类数据缓存
│   ├── features.py              # 特征构造（技术指标 + Alpha158 + 另类数据）
│   ├── stock_selector.py        # 树模型选股器（分位数回归 + IC加权融合）
│   ├── model.py                 # iTransformer 择时模型
│   ├── trainer.py               # 择时模型训练（AMP + 两阶段）
│   ├── backtest_engine.py       # 两阶段回测引擎
│   └── ...
├── research/                    # 因子研究实验室（独立于 core）
│   ├── factor_lab.py            # 因子 IC 审计 / 滚动窗口分析
│   └── alpha101.py              # WorldQuant Alpha101 因子库
├── scripts/
│   ├── download_data.py         # 数据下载（--full / --incremental / --force）
│   ├── train_selector.py        # 训练选股器
│   ├── pure_selection_backtest.py # 纯选股对照回测（8窗口）
│   ├── daily_select.py          # 盘后选股（选股器打分）
│   ├── daily_predict.py         # 盘前择时推断（可选，事件驱动那套）
│   └── monthly_live.py          # 月度实盘（跨月调仓 + 持有期止损，复现回测逻辑）
└── output/                      # 模型与结果统一输出目录
```

## 运行流程

### 1. 数据下载 + 特征重建

```bash
python scripts/download_data.py --full        # 全量下载（tushare）
python scripts/rebuild_financial_features.py  # 重建财务/融资融券/龙虎榜/大宗交易因子
```

### 2. 因子筛选（可选）

```bash
python research/factor_lab.py --stocks 100 --forward 20
```

### 3. 训练选股器 + 回测验证

```bash
python scripts/train_selector.py              # 训练最终选股器 → output/selector_ensemble.pkl
python scripts/pure_selection_backtest.py     # 8窗口滚动回测
```

### 4. 每日实盘流程（月度调仓，复现回测逻辑）

```bash
python scripts/download_data.py   # 每月最后一天增量更新数据（盘后）
python scripts/train_selector.py		#重训选股器
python scripts/monthly_live.py                 # 每日盘后运行，模拟止损
```

`monthly_live.py` 每天运行一次，自动判断：
- **跨月** → 调仓：卖出旧持仓 → 选股(P10风控+每行业1只+Top20) → 等权买入
- **持有期** → 检查止损：持仓跌超 10% 自动卖出

状态持久化到 `output/monthly_state.json`，交易流水写入 `output/monthly_trade_log.csv`（即"真实信号 vs 实际走势"的记录来源）。

## 回测优化结论（重要）

本项目在 8 窗口样本外回测中系统性验证了多个优化方向：

| 有效（正贡献）| 无效（负优化/无差）|
| :--: | :--: |
| 龙虎榜/大宗交易因子（夏普 1.295→1.555）| 行业轮动、最小方差、趋势仓位 |
| 止损10%（夏普 1.555→1.654）| 堆叠集成(MLP+Ridge)、行业排名标签 |
| 前视偏差修复（财务ann_date/停牌股）| 双周调仓、止盈、持仓数量、时间衰减180天 |

**核心洞察**：多源数据因子 + 简单等权 + 止损，优于任何"更聪明"的策略/模型叠加——简单就是最好。

## 交流

暂冻代码备考，后续更新实盘模拟，如果有任何交流，请邮件acsorceress@gmail.com

## 免责声明

**⚠️ 本项目仅供量化研究与学习使用，不构成任何投资建议。实盘交易需谨慎，风险自负。**

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
