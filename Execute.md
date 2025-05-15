# LSTM量化交易预测模型使用说明

## 1. 项目概述

本项目是一个基于长短期记忆网络(LSTM)的量化交易预测模型，旨在预测股票价格的涨跌方向并生成交易信号。模型结合了丰富的技术指标、主成分分析(PCA)和优化的深度学习架构，将股票预测从传统价格回归转变为方向分类问题，提高了预测准确率。

## 2. 环境要求

- Python 3.8+
- 依赖库：
  ```
  numpy
  pandas
  matplotlib
  seaborn
  tensorflow>=2.6.0
  scikit-learn
  yfinance
  scipy
  ```

## 3. 安装方法

```bash
# 创建虚拟环境(可选)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn yfinance scipy

# 解决可能的SSL证书问题(Windows系统)
pip install pip-system-certs
pip install python-certifi-win32
```
### Tips：下载Yahoo的数据源，可能会遇到SSL证书问题，可以使用 pip install pip-system-certs解决（Windows系统下，还需要 pip install python-certifi-win32）。

## 4. 使用方法

### 基本运行

```python
# 导入模块
from prediction import predict

# 使用默认参数(苹果股票2021-2023)
results = predict()

# 自定义参数
results = predict(
    ticker="MSFT",                # 股票代码
    start_date="2021-01-01",      # 开始日期
    end_date="2023-12-31",        # 结束日期
    load_existing=False           # 是否加载已有模型
)
```

### 命令行运行

```bash
# 直接运行脚本
python prediction.py
```

## 5. 主要功能

模型工作流程包括以下步骤：

1. **数据获取**：使用yfinance下载指定股票的历史数据
2. **数据预处理**：
   - 异常值检测与处理
   - 计算技术指标(MA、RSI、KDJ、布林带等)
   - PCA降维减少特征相关性
3. **数据分割**：划分训练集(70%)、验证集(15%)和测试集(15%)
4. **模型构建与训练**：
   - 双向LSTM网络架构
   - 早停机制和动态学习率调整
   - 高比例Dropout防止过拟合
5. **生成交易信号**：
   - 上涨概率>0.52：买入
   - 上涨概率<0.48：卖出
   - 其他情况：持有
6. **回测评估**：
   - 计算收益率、夏普比率、最大回撤等指标
   - 可视化结果(ROC曲线、混淆矩阵、资产价值变化)

## 6. 输出说明

程序输出包括：

1. **模型保存**：训练好的模型保存在`predict_output`目录
2. **性能指标**：
   - 测试集准确率
   - ROC曲线下面积(AUC)
   - 精确率、召回率、F1分数
   - 混淆矩阵
3. **回测结果**：
   - 总收益率和年化收益率
   - 年化波动率
   - 夏普比率
   - 最大回撤
   - 交易次数统计
4. **可视化图表**：
   - 混淆矩阵(`confusion_matrix.png`)
   - 预测结果与交易信号(`{ticker}_predictions_and_signals.png`)
   - 回测资产价值变化(`{ticker}_backtest_results.png`)
   - ROC曲线(`roc_curve.png`)
   - 训练历史(`training_history.png`)

## 7. 注意事项

1. **数据质量**：确保选择的股票在指定时间范围内有足够的交易数据
2. **计算资源**：模型训练可能需要较大的计算资源，特别是处理长时间序列时
3. **风险控制**：模型预测仅供参考，实际交易中应结合风险管理策略
4. **参数调整**：可以尝试调整以下参数以获得更好的性能：
   - `look_back`：历史回看天数
   - `threshold`：交易信号生成阈值
   - LSTM网络结构(层数、单元数等)
5. **模型局限性**：
   - 在高波动市场中表现可能不稳定
   - 策略回测显示的较高回撤率需要注意

## 8. 示例结果

以可口可乐(KO)股票为例，模型在测试集上达到了以下性能：

- 方向预测准确率：57.14%
- 年化收益率：17.71%
- 夏普比率：7.60
- 总交易次数：39
