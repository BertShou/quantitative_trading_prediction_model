# 项目 TODO list
## 各组员参考[README.md](README.md)后，选择自己的研究方向并填写，在相应的checkbox中打钩
- 组长：寿毅宁
  - [x] 研究方向：**优化LSTM网络结构与正则化策略以提升股票方向分类模型的泛化能力**
  - [ ] 论文已完成
- 组员：缪怡然
  - [ ] 研究方向：
  - [ ] 论文已完成
- 组员：詹水霞
  - [ ] 研究方向：
  - [ ] 论文已完成
- 组员：张宪铮
  - [ ] 研究方向：
  - [ ] 论文已完成
- 组员：方建业
  - [ ] 研究方向：
  - [ ] 论文已完成

## 参考论文点：

1.  **从价格回归到方向分类：LSTM模型在股票预测应用中的范式转变与性能评估**
    *   **核心内容**：深入探讨金融时间序列预测中，为何从直接预测价格（回归问题）转向预测价格变动方向（分类问题）能带来显著的性能提升。
    *   **具体研究点**：
        *   分析初始回归模型（`优化过程.txt`中提及的准确率49.06%时期）的局限性，如对趋势不敏感、易陷入局部均值等问题。
        *   详细阐述`README.md`中提到的优化过程，特别是“从回归到分类转变”的技术细节：标签如何定义（例如 `(df['price_change'] > 0).astype(int)`）、损失函数的选择（二元交叉熵）、输出层激活（Sigmoid）。
        *   对比分析转变前后在准确率、AUC、F1分数以及最终回测策略盈利能力上的差异。
        *   讨论将连续的价格预测问题离散化为方向预测的理论依据和挑战。

2.  **金融时间序列特征工程的深度探索及其对LSTM股票方向预测模型性能的影响**
    *   **核心内容**：基于`README.md`中详述的特征工程（MA, RSI, Bollinger, KDJ, MACD, PCA等），评估现有特征组合的有效性，并探索更优的特征构建与选择方案。
    *   **具体研究点**：
        *   对`README.md`中提及的各项技术指标的贡献度进行敏感性分析或特征重要性评估。
        *   深入分析PCA降维（保留5个主成分，解释94%方差）的利弊，例如是否存在信息损失，以及不使用PCA或使用其他非线性降维方法（如Autoencoders）的效果。
            *   选择当前这些特征因子的原因？PCA降维时，为什么要选择那6个主要因子
        *   响应`README.md`未来方向中“引入宏观经济指标和市场情绪数据”的建议，研究如何整合这些外部数据，并评估其对模型预测能力的提升。
        *   探索更复杂的特征交互，或针对特定市场状态（如高波动、趋势市）构建动态特征。

3.  **优化LSTM网络结构与正则化策略以提升股票方向分类模型的泛化能力**
    *   **核心内容**：针对`README.md`中给出的最终模型结构（双向LSTM(32) -> Dropout(0.4) -> LSTM(16) -> Dropout(0.4) -> Dense(8) -> Dropout(0.3) -> Dense(1, sigmoid)），分析其设计思想并探索进一步优化的可能性。
    *   **具体研究点**：
        *   分析双向LSTM层、Dropout高比例（0.4）、简化网络深度、移除注意力机制（如`README.md` 3.2.3所述）等决策对模型性能（特别是防止过拟合，提升57.14%准确率）的影响。
        *   实验不同的LSTM单元数、层数、Dropout率组合，以及其他正则化方法（如L1/L2正则化）。
        *   探讨其他可能适用于此分类任务的先进RNN变体（如GRU）或注意力机制的重新引入（可能以更轻量级的方式）。
        *   研究`look_back`参数（当前代码中为10，但`优化过程.txt`曾提及60）对模型捕捉不同长度依赖关系的影响。

4.  **基于LSTM预测概率的股票交易信号生成机制与动态阈值优化研究**
    *   **核心内容**：聚焦于`generate_signal`函数中`threshold=0.52`的设定，并探索更智能的信号生成和阈值调整方法。
    *   **具体研究点**：
        *   分析为何选择0.52而非严格的0.5作为阈值，以及该选择对交易频率、精确率、召回率的影响。
        *   实现`README.md`未来方向中“开发自适应阈值机制，根据市场波动性调整交易信号”的设想，例如，阈值可以与近期波动率、预测概率的置信度等因素挂钩。
        *   研究更复杂的信号过滤机制，如连续多个信号确认、结合交易量放大信号等。
        *   评估不同信号生成策略对回测结果中各项指标（特别是夏普比率和最大回撤）的敏感性。

5.  **针对LSTM股票预测模型高最大回撤问题的风险管理策略集成与有效性评估**
    *   **核心内容**：直接解决`README.md`中指出的关键局限性——高达87.16%的最大回撤和113.02%的年化波动率，研究如何通过集成风险管理模块来改善策略的风险调整后收益。
    *   **具体研究点**：
        *   分析导致当前策略高回撤的具体交易序列和市场条件。
        *   在`backtest_strategy`函数中设计并实现多种风险控制手段，例如：
            *   不同类型的止损策略（固定百分比、基于ATR的动态止损、时间止损）。
            *   仓位管理规则（如固定分数法、基于波动率的仓位调整）。
            *   考虑交易成本和滑点对策略的影响。
        *   量化评估这些风险管理策略在降低最大回撤、平滑收益曲线方面的效果，并分析其对总收益率和夏普比率可能产生的权衡。

6.  **集成学习方法在提升LSTM股票方向预测稳定性与准确率中的应用探索**
    *   **核心内容**：根据`README.md`未来方向中“探索集成学习方法”，研究如何通过结合多个LSTM模型或与其他类型模型集成来进一步提升预测性能。
    *   **具体研究点**：
        *   设计并实验基于LSTM的集成策略，如Bagging（训练多个模型在不同数据子集上）、Boosting（用LSTM的输出作为更高级模型的输入特征）、模型平均（对多个略有差异的LSTM模型的概率输出进行平均或加权平均）。
        *   评估集成模型相对于单一优化模型的准确率、AUC、F1分数以及回测表现的改进程度。
        *   分析集成学习在降低模型方差、提高预测鲁棒性方面的潜力，尤其是在面对不同市场环境时。

7.  **LSTM模型在不同股票资产和市场周期下的泛化能力与适应性研究**
    *   **核心内容**：响应`README.md`未来方向“测试更多资产类别，验证模型的通用性”，系统评估当前模型（针对KO股票优化）在不同场景下的表现。
    *   **具体研究点**：
        *   将现有模型应用于不同行业、不同市值、不同波动特性的其他股票。
        *   分析模型在不同宏观经济周期或市场情绪主导的阶段（如牛市、熊市、高波动期、低波动期）的表现差异。
        *   探讨模型的超参数（如`look_back`、特征集、交易阈值）是否需要针对不同资产或市场环境进行重新校准。
        *   研究模型表现与特定股票/市场特征（如流动性、趋势性、均值回归特性）之间的关系。
