# 导入必要的库
import os  # 导入os，用于文件和目录操作

import matplotlib.pyplot as plt  # 导入matplotlib，用于数据可视化
import numpy as np  # 导入numpy库，用于数值计算
import pandas as pd  # 导入pandas，用于数据处理
import seaborn as sns  # 导入seaborn，用于统计数据可视化
import tensorflow as tf  # 导入tensorflow，深度学习框架
import yfinance as yf  # 导入yfinance库，用于获取股票历史数据
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # 导入Keras回调函数，用于模型训练控制
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Concatenate, Layer  # 导入Keras的LSTM、Dense和Dropout层，用于构建神经网络
from keras.models import Sequential, load_model, Model  # 导入Keras的Sequential模型和加载模型函数
from scipy import stats  # 导入stats，用于异常值检测
from sklearn.decomposition import PCA  # 导入PCA，用于降维
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score  # 导入混淆矩阵和评估指标
from sklearn.metrics import roc_curve, auc  # 导入ROC曲线和AUC计算函数
from sklearn.preprocessing import MinMaxScaler  # 导入MinMaxScaler，用于数据归一化

# 创建输出目录
OUTPUT_DIR = 'predict_output'  # 定义输出目录名称
if not os.path.exists(OUTPUT_DIR):  # 如果目录不存在
    os.makedirs(OUTPUT_DIR)  # 创建目录

# 设置matplotlib支持中文显示
# 使用系统中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体系列
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义常量
CURRENCY_SYMBOL = '$'  # 使用美元符号
INITIAL_CASH = 10000.0  # 初始资金
SHARES_PER_TRADE = 100  # 每次交易的股票数量
RISK_FREE_RATE = 0.02  # 无风险利率
BUY_THRESHOLD = 1.02  # 买入阈值（预测价格比当前价格高2%）
SELL_THRESHOLD = 0.98  # 卖出阈值（预测价格比当前价格低2%）

# 默认配置
DEFAULT_TICKER = "AAPL"  # 默认股票代码
DEFAULT_START_DATE = "2021-01-01"  # 默认开始日期
DEFAULT_END_DATE = "2023-12-31"  # 默认结束日期


def predict(ticker=DEFAULT_TICKER, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE, load_existing=False):
    """
    主函数入口：训练模型或加载现有模型，并进行预测评估

    参数:
    ticker - 股票代码
    start_date - 开始日期
    end_date - 结束日期
    load_existing - 是否加载现有模型

    返回:
    回测结果DataFrame
    """
    # 如果是新的股票或时间范围，且不使用已有模型，则重新训练
    global model  # 使用全局变量

    if load_existing:  # 如果要加载现有模型
        # 尝试加载模型和数据
        model_path = os.path.join(OUTPUT_DIR, f'{ticker}_{start_date}_{end_date}_model.keras')  # 模型路径
        if os.path.exists(model_path):  # 如果模型文件存在
            model = load_model(model_path)  # 加载模型
            print(f"已加载现有模型: {model_path}")  # 打印加载成功消息
            # 评估加载的模型
            return evaluate(ticker, start_date, end_date)
        else:  # 如果模型文件不存在
            print(f"未找到模型: {model_path}，将重新训练")  # 打印未找到模型消息
            return train_and_predict(ticker, start_date, end_date)  # 重新训练
    else:  # 如果不加载现有模型
        # 如果未初始化必要的变量，则调用训练函数
        if 'model' not in globals() or model is None:  # 如果模型未初始化
            return train_and_predict(ticker, start_date, end_date)  # 训练模型
        else:
            # 如果模型已经存在，直接评估
            return evaluate(ticker, start_date, end_date)


def train_and_predict(ticker=DEFAULT_TICKER, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """
    训练模型并进行预测

    参数:
    ticker - 股票代码
    start_date - 开始日期 (格式: YYYY-MM-DD)
    end_date - 结束日期 (格式: YYYY-MM-DD)

    返回:
    回测结果DataFrame
    """
    global df, df_processed, pca_df, train_df, val_df, test_df, train_pca, val_pca, test_pca, model, scaler  # 使用全局变量

    print(f"=== 使用参数 ===")  # 打印参数信息
    print(f"股票代码: {ticker}")  # 打印股票代码
    print(f"时间范围: {start_date} 至 {end_date}")  # 打印时间范围

    # 1. 下载股票数据
    df = download_stock_data(ticker, start_date, end_date)  # 下载历史股票数据

    # 2. 数据预处理
    df_processed = preprocess_data(df)  # 预处理数据
    df_processed['ma'] = df_processed['ma20']  # 保持原有代码兼容性

    # 3. PCA降维
    pca_df, pca_model, pca_scaler = apply_pca(df_processed)  # 应用PCA降维

    # 4. 数据分割
    (train_df, val_df, test_df), (train_pca, val_pca, test_pca) = split_data(df_processed, pca_df)  # 分割数据集

    # 5. 准备训练数据
    X_train, y_train, scaler = prepare_lstm_data(train_df, train_pca)  # 准备训练数据
    X_val, y_val, _ = prepare_lstm_data(val_df, val_pca)  # 准备验证数据

    # 6. 构建并训练模型
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))  # 构建模型
    model.summary()  # 打印模型结构

    # 定义模型保存路径
    model_path = os.path.join(OUTPUT_DIR, f'{ticker}_{start_date}_{end_date}_model.keras')  # 设置保存路径

    # 定义回调函数
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),  # 早停，监控验证准确率，15轮无改善则停止
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),  # 模型检查点，保存最佳模型
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)  # 学习率调整，当验证损失不再改善时降低学习率
    ]

    # 训练模型
    history = model.fit(
        X_train, y_train,  # 训练数据和标签
        epochs=100,  # 最大训练轮数
        batch_size=32,  # 批次大小
        verbose=1,  # 显示进度条
        validation_data=(X_val, y_val),  # 验证数据
        callbacks=callbacks  # 回调函数列表
    )

    # 7. 可视化训练结果
    plot_training_history(history)  # 绘制训练历史

    # 8. 在测试集上评估
    backtest_results = evaluate(ticker, start_date, end_date)  # 进行评估和回测

    return backtest_results  # 返回回测结果


# 1. 使用yfinance下载历史数据
def download_stock_data(ticker, start_date, end_date):
    """
    下载指定股票和时间范围的历史数据
    
    参数:
    ticker - 股票代码（例如"AAPL"）
    start_date - 开始日期（格式：YYYY-MM-DD）
    end_date - 结束日期（格式：YYYY-MM-DD）
    
    返回:
    pandas DataFrame对象，包含历史股价数据
    """
    ticker_obj = yf.Ticker(ticker)  # 创建一个股票Ticker对象
    df = ticker_obj.history(start=start_date, end=end_date)  # 下载指定时间范围的数据

    if df.empty:  # 如果数据为空
        raise ValueError(f"未能获取到股票 {ticker} 在 {start_date} 至 {end_date} 期间的数据")  # 抛出错误

    print(f"成功下载股票 {ticker} 在 {start_date} 至 {end_date} 期间的数据，共 {len(df)} 条记录")  # 打印成功信息
    return df  # 返回数据框


# 2. 数据预处理和清洗
def preprocess_data(df):
    """
    预处理股票数据，包括异常值检测、去噪和特征工程

    参数:
    df - 原始股票数据DataFrame

    返回:
    处理后的DataFrame，包含各类技术指标
    """
    # 保存原始数据副本
    df_processed = df.copy()  # 复制原始数据，避免修改原始数据

    # 检测并处理收盘价异常值（使用Z分数方法）
    z_scores = stats.zscore(df_processed['Close'])  # 计算收盘价的Z分数
    abs_z_scores = np.abs(z_scores)  # 取Z分数绝对值
    filtered_entries = (abs_z_scores < 3)  # 过滤Z分数绝对值大于3的数据点（异常值）

    # 打印检测到的异常值数量
    outliers_count = len(filtered_entries) - np.sum(filtered_entries)  # 计算异常值数量
    print(f"检测到 {outliers_count} 个异常值")  # 打印异常值数量

    # 处理异常值 - 使用移动中位数替换
    median_window = 5  # 设定中位数窗口大小
    df_processed.loc[~filtered_entries, 'Close'] = df_processed['Close'].rolling(
        window=median_window, center=True).median().loc[~filtered_entries]  # 用移动中位数替换异常值

    # 填充可能的NaN值
    df_processed = df_processed.ffill().bfill()  # 前向填充再后向填充，确保没有NaN值

    # 计算技术指标
    # 移动平均线
    df_processed['ma5'] = df_processed['Close'].rolling(window=5).mean()  # 5日均线
    df_processed['ma10'] = df_processed['Close'].rolling(window=10).mean()  # 10日均线
    df_processed['ma20'] = df_processed['Close'].rolling(window=20).mean()  # 20日均线
    df_processed['ma50'] = df_processed['Close'].rolling(window=50).mean()  # 50日均线

    # 相对强弱指标 (RSI)
    delta = df_processed['Close'].diff()  # 计算价格变化
    gain = delta.where(delta > 0, 0)  # 获取正向变化（涨）
    loss = -delta.where(delta < 0, 0)  # 获取负向变化（跌）的绝对值
    avg_gain = gain.rolling(window=14).mean()  # 计算14日平均涨幅
    avg_loss = loss.rolling(window=14).mean()  # 计算14日平均跌幅
    rs = avg_gain / avg_loss  # 计算相对强度
    df_processed['rsi'] = 100 - (100 / (1 + rs))  # 计算RSI指标

    # 交易量变化
    df_processed['volume_change'] = df_processed['Volume'].pct_change()  # 交易量变化率
    df_processed['volume_ma10'] = df_processed['Volume'].rolling(window=10).mean()  # 10日交易量均线
    df_processed['volume_ratio'] = df_processed['Volume'] / df_processed['volume_ma10']  # 交易量比率

    # 波动率指标
    df_processed['volatility'] = df_processed['Close'].rolling(window=20).std()  # 20日收盘价标准差
    df_processed['volatility_10'] = df_processed['Close'].rolling(window=10).std()  # 10日收盘价标准差

    # 添加更多金融特有特征
    df_processed['price_momentum'] = df_processed['Close'].pct_change(5)  # 5日价格动量
    df_processed['price_momentum_10'] = df_processed['Close'].pct_change(10)  # 10日价格动量

    # 布林带
    df_processed['bollinger_upper'] = df_processed['ma20'] + 2 * df_processed['volatility']  # 布林带上轨
    df_processed['bollinger_lower'] = df_processed['ma20'] - 2 * df_processed['volatility']  # 布林带下轨
    df_processed['bollinger_width'] = (df_processed['bollinger_upper'] - df_processed['bollinger_lower']) / df_processed[
        'ma20']  # 布林带宽度

    # MACD
    df_processed['macd'] = df_processed['ma20'] - df_processed['ma50']  # MACD简化版

    # 趋势指标
    df_processed['ma_trend'] = (df_processed['ma5'] - df_processed['ma20']) / df_processed['ma20']  # 短期均线与长期均线偏离

    # 价格与均线之间的关系
    df_processed['price_to_ma20'] = df_processed['Close'] / df_processed['ma20'] - 1  # 价格相对20日均线偏离
    df_processed['price_to_ma50'] = df_processed['Close'] / df_processed['ma50'] - 1  # 价格相对50日均线偏离

    # 加入方向变化指标 - 过去几天价格方向
    df_processed['direction_1d'] = np.sign(df_processed['Close'].diff(1))  # 1日价格变化方向
    df_processed['direction_2d'] = np.sign(df_processed['Close'].diff(2))  # 2日价格变化方向
    df_processed['direction_3d'] = np.sign(df_processed['Close'].diff(3))  # 3日价格变化方向

    # 计算KDJ指标
    low_min = df_processed['Low'].rolling(window=14).min()  # 14日最低价
    high_max = df_processed['High'].rolling(window=14).max()  # 14日最高价

    df_processed['k_percent'] = 100 * ((df_processed['Close'] - low_min) / (high_max - low_min))  # 计算K值
    df_processed['k_percent'] = df_processed['k_percent'].rolling(window=3).mean()  # 平滑K值
    df_processed['d_percent'] = df_processed['k_percent'].rolling(window=3).mean()  # 计算D值
    df_processed['j_percent'] = 3 * df_processed['k_percent'] - 2 * df_processed['d_percent']  # 计算J值

    # 填充计算技术指标产生的NaN值
    df_processed = df_processed.bfill()  # 后向填充

    # 返回处理后的DataFrame
    return df_processed  # 返回处理后的数据


# 3. 使用PCA进行特征降维
def apply_pca(df, n_components=5):
    """
    应用PCA降维，提取主要特征
    
    参数:
    df - 预处理后的数据框
    n_components - 希望保留的主成分个数
    
    返回:
    pca_df - PCA降维后的特征DataFrame
    pca - 训练好的PCA模型
    scaler - 用于标准化的缩放器
    """
    # 选择用于PCA的特征 - 使用所有相关技术指标
    features = [  # 定义用于PCA的特征列表
        'Close', 'ma5', 'ma10', 'ma20', 'ma50', 'rsi', 'volatility', 'volatility_10',
        'price_momentum', 'price_momentum_10', 'bollinger_width', 'macd', 'ma_trend',
        'price_to_ma20', 'price_to_ma50', 'volume_ratio',
        'k_percent', 'd_percent', 'j_percent'
    ]

    # 标准化特征
    scaler = MinMaxScaler()  # 创建MinMaxScaler实例
    scaled_features = scaler.fit_transform(df[features])  # 对特征进行标准化处理

    # 应用PCA
    pca = PCA(n_components=n_components)  # 创建PCA实例，设置主成分个数
    pca_result = pca.fit_transform(scaled_features)  # 进行PCA降维

    # 创建包含PCA结果的DataFrame
    pca_df = pd.DataFrame(  # 将PCA结果转换为DataFrame
        data=pca_result,
        columns=[f'PC{i + 1}' for i in range(n_components)],  # 设置列名为PC1, PC2...
        index=df.index  # 保持与原始数据相同的索引
    )

    # 打印解释方差比
    print("PCA解释方差比: ", pca.explained_variance_ratio_)  # 打印各主成分的方差贡献率

    return pca_df, pca, scaler  # 返回PCA结果、PCA模型和缩放器


# 5. 准备LSTM训练数据
def prepare_lstm_data(df, pca_df, look_back=10):
    """
    准备LSTM模型的训练数据，包括序列化、标签生成和归一化
    
    参数:
    df - 预处理后的数据框
    pca_df - PCA降维后的特征数据框
    look_back - 回看天数，即用多少天的历史数据预测未来

    返回:
    X - 特征序列
    y - 标签（涨跌方向）
    scaler - 特征缩放器
    """
    # 复制数据框避免警告
    df = df.copy()  # 创建数据副本防止SettingWithCopyWarning

    # 计算价格变化率和方向标签
    df['price_change'] = df['Close'].pct_change(1)  # 计算每日价格变化百分比
    # 分类标签: 1表示上涨，0表示下跌
    df['direction'] = (df['price_change'] > 0).astype(int)  # 上涨标记为1，下跌标记为0

    # 使用PCA特征和其他关键特征
    selected_features = ['Close', 'price_change', 'rsi', 'volatility', 'ma_trend', 'price_to_ma20']  # 选择关键特征
    all_features = pd.concat([df[selected_features], pca_df], axis=1)  # 合并选定特征和PCA特征

    # 确保没有NaN值
    all_features = all_features.fillna(0)  # 将所有NaN值替换为0

    # 归一化特征
    scaler = MinMaxScaler(feature_range=(0, 1))  # 创建归一化器，将数据缩放到0-1范围
    scaled_data = scaler.fit_transform(all_features)  # 对特征进行归一化处理

    X, y = [], []  # 创建空列表存储输入特征X和目标值y
    for i in range(look_back, len(df) - 1):  # -1因为我们需要有未来1天的数据
        X.append(scaled_data[i - look_back:i])  # 添加过去look_back天的数据作为特征
        y.append(df['direction'].iloc[i])  # 添加当天的涨跌方向作为标签

    X = np.array(X)  # 将列表转换为numpy数组
    y = np.array(y)  # 将列表转换为numpy数组

    return X, y, scaler  # 返回处理后的数据和归一化器


# 4. 数据分割，将数据集分为训练集、验证集和测试集
def split_data(df, pca_df=None, train_ratio=0.7, val_ratio=0.15):
    """
    将数据分割为训练集、验证集和测试集
    
    参数:
    df - 预处理后的数据框
    pca_df - PCA降维后的特征
    train_ratio - 训练集比例
    val_ratio - 验证集比例
    
    返回:
    (train_df, val_df, test_df) - 分割后的数据框元组
    (train_pca, val_pca, test_pca) - 分割后的PCA特征元组
    """
    # 去除前30行，因为计算技术指标时可能有NaN值
    df = df.iloc[30:].copy()  # 去除前30行并创建副本
    if pca_df is not None:  # 如果PCA特征存在
        pca_df = pca_df.iloc[30:].copy()  # 同样去除前30行

    # 获取训练集、验证集、测试集的分割索引
    n = len(df)  # 获取数据总长度
    train_size = int(n * train_ratio)  # 计算训练集大小
    val_size = int(n * val_ratio)  # 计算验证集大小

    # 分割数据
    train_df = df.iloc[:train_size]  # 提取训练集
    val_df = df.iloc[train_size:train_size + val_size]  # 提取验证集
    test_df = df.iloc[train_size + val_size:]  # 提取测试集

    # 如果有PCA特征，也对其进行分割
    if pca_df is not None:
        train_pca = pca_df.iloc[:train_size]  # 分割PCA训练集
        val_pca = pca_df.iloc[train_size:train_size + val_size]  # 分割PCA验证集
        test_pca = pca_df.iloc[train_size + val_size:]  # 分割PCA测试集
        return (train_df, val_df, test_df), (train_pca, val_pca, test_pca)
    else:
        return (train_df, val_df, test_df), (None, None, None)


# 自定义注意力层
class AttentionLayer(Layer):
    """
    自定义注意力机制层，用于关注时间序列中的重要部分
    """

    def __init__(self, **kwargs):
        """初始化函数"""
        super(AttentionLayer, self).__init__(**kwargs)  # 调用父类初始化

    def build(self, input_shape):
        """构建层参数"""
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")  # 注意力权重矩阵
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")  # 注意力偏置
        super(AttentionLayer, self).build(input_shape)  # 调用父类构建方法

    def call(self, x):
        """前向传播"""
        # 计算注意力权重
        e = tf.tanh(tf.matmul(x, self.W) + self.b)  # 使用tanh激活权重

        # 对权重进行softmax归一化
        a = tf.nn.softmax(e, axis=1)  # 在时间轴上进行softmax

        # 应用注意力权重到输入序列
        output = x * tf.expand_dims(a, -1)  # 权重乘以输入

        # 沿着时间轴进行加权和
        return tf.reduce_sum(output, axis=1)  # 返回加权和

    def compute_output_shape(self, input_shape):
        """计算输出形状"""
        return input_shape[0], input_shape[-1]  # 返回批大小和特征维度


# 6. 构建优化的分类模型 - 使用双向LSTM但简化结构
def build_model(input_shape):
    """
    构建股票价格涨跌预测模型
    
    参数:
    input_shape - 输入数据的形状 (seq_length, features)
    
    返回:
    model - 构建好的Keras模型
    """
    # 简化模型，减少过拟合
    model = Sequential()  # 创建序列模型

    # 第一层 Bidirectional LSTM，保留中间状态
    model.add(Bidirectional(LSTM(32, return_sequences=True),  # 32个LSTM单元，返回序列
                            input_shape=input_shape))  # 设置输入形状
    model.add(Dropout(0.4))  # 添加Dropout防止过拟合，丢弃率为0.4

    # 第二层 LSTM
    model.add(LSTM(16))  # 16个LSTM单元，不返回序列
    model.add(Dropout(0.4))  # 添加Dropout，丢弃率为0.4

    # 全连接层
    model.add(Dense(8, activation='relu'))  # 8个神经元的全连接层，使用ReLU激活
    model.add(Dropout(0.3))  # 添加Dropout，丢弃率为0.3

    # 输出层 - 二分类
    model.add(Dense(1, activation='sigmoid'))  # 1个神经元的输出层，使用sigmoid激活，输出0-1之间的概率

    # 使用二元交叉熵损失和准确率度量
    model.compile(optimizer='adam',  # 使用Adam优化器
                  loss='binary_crossentropy',  # 二元交叉熵损失函数
                  metrics=['accuracy'])  # 使用准确率评估模型

    return model  # 返回构建好的模型


# 7. 可视化训练结果
def plot_training_history(history):
    """
    可视化训练历史，展示损失和准确率变化

    参数:
    history - 模型训练历史记录对象
    """
    plt.figure(figsize=(12, 10))  # 创建12x10英寸的图像

    # 绘制训练损失和验证损失
    plt.subplot(2, 1, 1)  # 创建子图1
    plt.plot(history.history['loss'], label='训练损失')  # 绘制训练损失曲线
    plt.plot(history.history['val_loss'], label='验证损失')  # 绘制验证损失曲线
    plt.title('模型损失')  # 设置标题
    plt.ylabel('损失')  # 设置y轴标签
    plt.xlabel('Epoch')  # 设置x轴标签
    plt.legend()  # 显示图例

    # 绘制训练准确率和验证准确率
    plt.subplot(2, 1, 2)  # 创建子图2
    plt.plot(history.history['accuracy'], label='训练准确率')  # 绘制训练准确率曲线
    plt.plot(history.history['val_accuracy'], label='验证准确率')  # 绘制验证准确率曲线
    plt.title('模型准确率')  # 设置标题
    plt.ylabel('准确率')  # 设置y轴标签
    plt.xlabel('Epoch')  # 设置x轴标签
    plt.legend()  # 显示图例

    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))  # 保存图像
    plt.close()  # 关闭图像


# 8. 在测试集上评估
def evaluate(ticker=DEFAULT_TICKER, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """
    在测试集上评估模型、可视化结果并生成回测报告

    参数:
    ticker - 股票代码
    start_date - 开始日期
    end_date - 结束日期

    返回:
    回测结果DataFrame
    """
    global test_df, test_pca, model  # 使用全局变量

    # 第一步：准备测试数据
    X_test, y_test, test_scaler = prepare_lstm_data(test_df, test_pca)  # 准备测试数据

    # 第二步：计算模型性能指标
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)  # 评估模型
    print(f"测试损失: {test_loss}, 测试准确率: {test_acc * 100:.2f}%")  # 打印测试结果

    # 第三步：绘制ROC曲线并计算AUC
    auc_value = plot_roc_curve(model, X_test, y_test)  # 绘制ROC曲线并计算AUC
    print(f"ROC曲线下面积(AUC): {auc_value:.4f}")  # 打印AUC值

    # 第四步：计算并显示混淆矩阵
    y_pred = (model.predict(X_test) > 0.5).astype(int)  # 获取预测标签
    cm = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵

    # 打印混淆矩阵
    print("混淆矩阵:")  # 打印标题
    print(f"真实下跌 预测下跌: {cm[0][0]}")  # 打印真阴性
    print(f"真实下跌 预测上涨: {cm[0][1]}")  # 打印假阳性
    print(f"真实上涨 预测下跌: {cm[1][0]}")  # 打印假阴性
    print(f"真实上涨 预测上涨: {cm[1][1]}")  # 打印真阳性

    # 第五步：计算精确率、召回率和F1分数
    precision = precision_score(y_test, y_pred)  # 计算精确率
    recall = recall_score(y_test, y_pred)  # 计算召回率
    f1 = f1_score(y_test, y_pred)  # 计算F1分数

    print(f"精确率: {precision:.4f}")  # 打印精确率
    print(f"召回率: {recall:.4f}")  # 打印召回率
    print(f"F1分数: {f1:.4f}")  # 打印F1分数

    # 第六步：可视化混淆矩阵
    plt.figure(figsize=(8, 6))  # 创建图像
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  # 绘制热图
                xticklabels=['预测下跌', '预测上涨'],  # 设置x轴标签
                yticklabels=['真实下跌', '真实上涨'])  # 设置y轴标签
    plt.title('股票涨跌预测混淆矩阵')  # 设置标题
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))  # 保存图像
    plt.close()  # 关闭图像

    # 第七步：对测试集进行逐日预测并生成交易信号
    predictions = []  # 预测概率列表
    signals = []  # 信号列表
    dates = []  # 日期列表
    actual_directions = []  # 实际方向列表

    look_back = X_test.shape[1]  # 获取模型的输入窗口大小

    for i in range(len(test_df) - 1):  # -1 因为我们需要有第二天的实际数据来计算准确率
        if i < look_back:  # 跳过前look_back天，因为需要历史数据来预测
            continue

        # 获取当前真实价格和移动平均线
        current_date = test_df.index[i]  # 获取当前日期
        current_price = test_df['Close'].iloc[i]  # 获取当前价格
        ma = test_df['ma'].iloc[i]  # 获取移动平均线价格

        # 准备输入序列
        if i >= look_back:
            # 获取测试序列
            test_seq = np.reshape(X_test[i - look_back],
                                  (1, X_test[i - look_back].shape[0], X_test[i - look_back].shape[1]))  # 重塑为批次形式

            # 预测上涨概率
            predicted_prob = model.predict(test_seq)[0][0]  # 预测上涨概率

            # 生成交易信号
            signal = generate_signal(predicted_prob)  # 根据概率生成信号

            # 获取实际方向
            next_price = test_df['Close'].iloc[i + 1]  # 获取下一天价格
            actual_direction = 1 if next_price > current_price else 0  # 计算实际涨跌方向
            actual_directions.append(actual_direction)  # 添加到实际方向列表

            # 格式化日期（移除时区信息）
            formatted_date = pd.to_datetime(current_date).strftime('%Y-%m-%d')  # 格式化日期

            dates.append(current_date)  # 添加日期
            predictions.append(predicted_prob)  # 添加预测概率
            signals.append(signal)  # 添加信号

            # 打印结果
            print(
                f"日期: {formatted_date}, 当前价格: {current_price:.2f}, 预测上涨概率: {predicted_prob:.4f}, MA: {ma:.2f}, 信号: {signal}")  # 打印单日预测结果

    # 第八步：计算预测方向准确率
    direction_accuracy = 0
    if len(actual_directions) > 0 and len(predictions) > 0:  # 如果有足够的预测和实际值
        correct_directions = 0  # 正确预测计数器
        for i in range(len(predictions)):  # 遍历所有预测
            prediction = 1 if predictions[i] > 0.5 else 0  # 将概率转换为二元方向
            if prediction == actual_directions[i]:  # 如果预测方向正确
                correct_directions += 1  # 增加正确计数
        direction_accuracy = (correct_directions / len(predictions)) * 100  # 计算准确率百分比

    print(f"预测方向准确率: {direction_accuracy:.2f}%")  # 打印方向准确率

    # 第九步：可视化预测结果
    visualize_predictions(dates, test_df['Close'].iloc[look_back:len(test_df) - 1].values,
                          predictions, signals, ticker)  # 可视化预测

    # 第十步：执行回测并评估策略
    backtest_results = backtest_strategy(dates, test_df['Close'].iloc[look_back:len(test_df) - 1].values, signals)  # 执行回测

    # 第十一步：计算风险收益指标
    calculate_performance_metrics(backtest_results, ticker, start_date, end_date)  # 计算性能指标

    # 第十二步：可视化回测结果
    visualize_backtest_results(backtest_results, ticker)  # 可视化回测结果

    return backtest_results  # 返回回测结果


def plot_roc_curve(model, X_test, y_test):
    """
    绘制ROC曲线和计算AUC值

    参数:
    model - 训练好的模型
    X_test - 测试集特征
    y_test - 测试集标签

    返回:
    auc_value - 计算的AUC值
    """
    predictions = model.predict(X_test).flatten()  # 获取预测概率并展平

    fpr, tpr, thresholds = roc_curve(y_test, predictions)  # 计算ROC曲线点
    auc_value = auc(fpr, tpr)  # 计算AUC值

    plt.figure(figsize=(8, 6))  # 创建8x6英寸的图像
    plt.plot(fpr, tpr, 'b', label=f'AUC = {auc_value:.3f}')  # 绘制ROC曲线
    plt.plot([0, 1], [0, 1], 'r--', label='随机猜测')  # 绘制随机猜测基准线
    plt.xlim([0, 1])  # 设置x轴范围
    plt.ylim([0, 1])  # 设置y轴范围
    plt.xlabel('假阳性率 (FPR)')  # 设置x轴标签
    plt.ylabel('真阳性率 (TPR)')  # 设置y轴标签
    plt.title('ROC曲线')  # 设置标题
    plt.legend(loc='lower right')  # 在右下角显示图例
    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'))  # 保存图像
    plt.close()  # 关闭图像

    return auc_value  # 返回AUC值


def generate_signal(prediction, threshold=0.55):
    """
    根据预测的上涨概率生成交易信号

    参数:
    prediction - 模型预测的上涨概率
    threshold - 决策阈值，高于此值买入，低于(1-threshold)卖出

    返回:
    交易信号: 'buy', 'sell' 或 'hold'
    """
    if prediction > threshold:  # 如果上涨概率高于阈值
        return 'buy'  # 返回买入信号
    elif prediction < (1 - threshold):  # 如果上涨概率低于(1-阈值)
        return 'sell'  # 返回卖出信号
    else:  # 如果在阈值区间内
        return 'hold'  # 返回持有信号


def backtest_strategy(dates, prices, signals, initial_cash=INITIAL_CASH, shares_per_trade=SHARES_PER_TRADE):
    """
    对交易策略进行回测，模拟交易过程并计算收益
    
    参数:
    dates - 交易日期列表
    prices - 资产价格列表
    signals - 交易信号列表 ('buy', 'sell', 'hold')
    initial_cash - 初始现金金额
    shares_per_trade - 每次交易的股票数量
    
    返回:
    包含回测结果的DataFrame
    """
    # 创建结果DataFrame
    results = pd.DataFrame({  # 创建回测结果数据框
        'Date': dates,  # 日期列
        'Price': prices,  # 价格列
        'Signal': signals  # 信号列
    })

    # 初始化交易状态
    cash = initial_cash  # 初始化现金
    shares = 0  # 初始化持股数量
    trades = []  # 初始化交易记录列表

    # 记录每日资产状态
    results['Cash'] = 0.0  # 现金列
    results['Shares'] = 0  # 持股数量列
    results['ShareValue'] = 0.0  # 持股价值列
    results['TotalValue'] = 0.0  # 总资产价值列
    results['Returns'] = 0.0  # 收益率列

    # 模拟交易
    for i in range(len(results)):  # 遍历每一个交易日
        current_date = results['Date'].iloc[i]  # 获取当前日期
        current_price = results['Price'].iloc[i]  # 获取当前价格
        signal = results['Signal'].iloc[i]  # 获取当前信号

        # 格式化日期（移除时区信息）
        formatted_date = pd.to_datetime(current_date).strftime('%Y-%m-%d')  # 格式化日期

        # 执行交易
        if signal == 'buy' and cash >= current_price * shares_per_trade:  # 如果信号为买入且现金足够
            # 买入操作
            purchase_amount = current_price * shares_per_trade  # 计算购买金额
            cash -= purchase_amount  # 扣除现金
            shares += shares_per_trade  # 增加持股数量
            trades.append({  # 记录交易
                'Date': current_date,  # 交易日期
                'Type': 'Buy',  # 交易类型
                'Price': current_price,  # 交易价格
                'Shares': shares_per_trade,  # 交易股数
                'Amount': purchase_amount,  # 交易金额
                'Cash': cash,  # 交易后现金
                'Holdings': shares  # 交易后持股
            })
            print(
                f"买入: {formatted_date} - {shares_per_trade}股 @ {CURRENCY_SYMBOL}{current_price:.2f} = {CURRENCY_SYMBOL}{purchase_amount:.2f}")  # 打印买入信息

        elif signal == 'sell' and shares >= shares_per_trade:  # 如果信号为卖出且有足够的股票
            # 卖出操作
            sale_amount = current_price * shares_per_trade  # 计算销售金额
            cash += sale_amount  # 增加现金
            shares -= shares_per_trade  # 减少持股数量
            trades.append({  # 记录交易
                'Date': current_date,  # 交易日期
                'Type': 'Sell',  # 交易类型
                'Price': current_price,  # 交易价格
                'Shares': shares_per_trade,  # 交易股数
                'Amount': sale_amount,  # 交易金额
                'Cash': cash,  # 交易后现金
                'Holdings': shares  # 交易后持股
            })
            print(
                f"卖出: {formatted_date} - {shares_per_trade}股 @ {CURRENCY_SYMBOL}{current_price:.2f} = {CURRENCY_SYMBOL}{sale_amount:.2f}")  # 打印卖出信息

        # 更新每日状态
        share_value = shares * current_price  # 计算持股价值
        total_value = cash + share_value  # 计算总资产价值

        results.loc[results.index[i], 'Cash'] = cash  # 更新现金
        results.loc[results.index[i], 'Shares'] = shares  # 更新持股数量
        results.loc[results.index[i], 'ShareValue'] = share_value  # 更新持股价值
        results.loc[results.index[i], 'TotalValue'] = total_value  # 更新总资产价值

    # 计算日收益率
    results['Returns'] = results['TotalValue'].pct_change().fillna(0)  # 计算每日收益率

    # 计算累积收益
    results['CumulativeReturns'] = (1 + results['Returns']).cumprod() - 1  # 计算累积收益率

    # 创建交易记录DataFrame
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()  # 将交易记录转换为DataFrame

    # 添加交易记录到结果中
    results.attrs['trades'] = trades_df  # 将交易记录添加到结果的属性中
    results.attrs['initial_cash'] = initial_cash  # 记录初始现金
    results.attrs['final_value'] = results['TotalValue'].iloc[-1]  # 记录最终价值

    return results  # 返回回测结果


def calculate_performance_metrics(results, ticker, start_date, end_date):
    """
    计算策略的风险收益指标
    
    参数:
    results - 回测结果DataFrame
    ticker - 股票代码
    start_date - 开始日期
    end_date - 结束日期
    
    返回:
    指标字典，包含总收益率、年化收益率等
    """
    # 提取关键数据
    initial_value = results.attrs['initial_cash']  # 获取初始资金
    final_value = results.attrs['final_value']  # 获取最终资产
    returns = results['Returns']  # 获取收益率序列
    cum_returns = results['CumulativeReturns']  # 获取累积收益率序列

    # 计算总收益率
    total_return = (final_value / initial_value) - 1  # 计算总收益率

    # 计算交易天数和年化因子
    trading_days = len(results)  # 获取交易天数
    years = max(trading_days / 252, 0.01)  # 计算交易年数，假设一年约有252个交易日，确保不会太小

    # 计算年化收益率
    annualized_return = (1 + total_return) ** (1 / years) - 1  # 计算年化收益率

    # 计算风险指标
    daily_returns_std = returns.std()  # 计算日收益率标准差
    annualized_volatility = daily_returns_std * np.sqrt(252)  # 计算年化波动率

    # 计算夏普比率 (假设无风险利率为0.02)
    daily_risk_free = ((1 + RISK_FREE_RATE) ** (1 / 252)) - 1  # 计算日无风险收益率

    # 安全计算夏普比率，避免除以零
    if daily_returns_std > 0:  # 如果有波动性
        sharpe_ratio = (returns.mean() - daily_risk_free) / daily_returns_std * np.sqrt(252)  # 计算夏普比率
    else:  # 如果没有波动性
        sharpe_ratio = 0  # 将夏普比率设为0

    # 计算最大回撤
    if not cum_returns.empty:  # 如果有累积收益率数据
        cumulative_max = cum_returns.cummax()  # 计算累积收益率的历史最大值
        drawdown = cumulative_max - cum_returns  # 计算回撤
        max_drawdown = drawdown.max() if not drawdown.empty else 0  # 计算最大回撤
    else:  # 如果没有累积收益率数据
        max_drawdown = 0  # 将最大回撤设为0

    # 计算交易次数
    trades_df = results.attrs.get('trades', pd.DataFrame())  # 获取交易记录
    num_trades = len(trades_df)  # 计算交易总次数
    num_buys = len(trades_df[trades_df['Type'] == 'Buy']) if not trades_df.empty else 0  # 计算买入次数
    num_sells = len(trades_df[trades_df['Type'] == 'Sell']) if not trades_df.empty else 0  # 计算卖出次数

    # 获取起止日期（格式化为YYYY-MM-DD）
    result_start_date = pd.to_datetime(results['Date'].iloc[0]).strftime('%Y-%m-%d') if not results.empty else "N/A"  # 格式化开始日期
    result_end_date = pd.to_datetime(results['Date'].iloc[-1]).strftime('%Y-%m-%d') if not results.empty else "N/A"  # 格式化结束日期

    # 打印结果
    print(f"\n===== {ticker} 策略绩效指标 =====")  # 打印标题
    print(f"交易时间段: {result_start_date} 至 {result_end_date}")  # 打印交易时间段
    print(f"初始资金: {CURRENCY_SYMBOL}{initial_value:.2f}")  # 打印初始资金
    print(f"最终资产: {CURRENCY_SYMBOL}{final_value:.2f}")  # 打印最终资产
    print(f"总收益率: {total_return:.2%}")  # 打印总收益率
    print(f"年化收益率: {annualized_return:.2%}")  # 打印年化收益率
    print(f"年化波动率: {annualized_volatility:.2%}")  # 打印年化波动率
    print(f"夏普比率: {sharpe_ratio:.2f}")  # 打印夏普比率
    print(f"最大回撤: {max_drawdown:.2%}")  # 打印最大回撤
    print(f"交易次数: {num_trades} (买入: {num_buys}, 卖出: {num_sells})")  # 打印交易次数

    # 保存指标到文件
    metrics_file = os.path.join(OUTPUT_DIR, f'{ticker}_{start_date}_{end_date}_performance_metrics.txt')  # 设置保存路径
    with open(metrics_file, 'w', encoding='utf-8') as f:  # 打开文件
        f.write(f"===== {ticker} 量化交易策略绩效指标 =====\n")  # 写入标题
        f.write(f"数据范围: {start_date} 至 {end_date}\n")  # 写入数据范围
        f.write(f"交易时间段: {result_start_date} 至 {result_end_date}\n")  # 写入交易时间段
        f.write(f"初始资金: {CURRENCY_SYMBOL}{initial_value:.2f}\n")  # 写入初始资金
        f.write(f"最终资产: {CURRENCY_SYMBOL}{final_value:.2f}\n")  # 写入最终资产
        f.write(f"总收益率: {total_return:.2%}\n")  # 写入总收益率
        f.write(f"年化收益率: {annualized_return:.2%}\n")  # 写入年化收益率
        f.write(f"年化波动率: {annualized_volatility:.2%}\n")  # 写入年化波动率
        f.write(f"夏普比率: {sharpe_ratio:.2f}\n")  # 写入夏普比率
        f.write(f"最大回撤: {max_drawdown:.2%}\n")  # 写入最大回撤
        f.write(f"交易次数: {num_trades} (买入: {num_buys}, 卖出: {num_sells})\n")  # 写入交易次数

    # 返回指标字典
    return {  # 返回包含各指标的字典
        'total_return': total_return,  # 总收益率
        'annualized_return': annualized_return,  # 年化收益率
        'annualized_volatility': annualized_volatility,  # 年化波动率
        'sharpe_ratio': sharpe_ratio,  # 夏普比率
        'max_drawdown': max_drawdown,  # 最大回撤
        'num_trades': num_trades  # 交易次数
    }


def visualize_backtest_results(results, ticker):
    """
    可视化回测结果，包括资产价值变化、收益率和回撤
    
    参数:
    results - 回测结果DataFrame
    ticker - 股票代码
    """
    # 创建多子图
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [2, 1, 1]})  # 创建3个子图，按比例设置高度

    # 1. 绘制资产价值和持仓变化
    ax1 = axes[0]  # 获取第一个子图
    ax1.plot(results['Date'], results['TotalValue'], label='总资产价值', color='blue', linewidth=2)  # 绘制总资产价值曲线
    ax1.set_title(f'{ticker} 策略资产价值变化')  # 设置标题
    ax1.set_ylabel(f'资产价值 ({CURRENCY_SYMBOL})')  # 设置y轴标签
    ax1.legend(loc='upper left')  # 在左上角显示图例
    ax1.grid(True)  # 显示网格

    # 添加持仓信息到第二个Y轴
    ax1_shares = ax1.twinx()  # 创建第二个y轴
    ax1_shares.fill_between(results['Date'], 0, results['Shares'], alpha=0.3, color='green', label='持股数量')  # 以填充区域显示持股数量
    ax1_shares.set_ylabel('持股数量')  # 设置y轴标签
    ax1_shares.legend(loc='upper right')  # 在右上角显示图例

    # 标记交易点
    trades_df = results.attrs.get('trades', pd.DataFrame())  # 获取交易记录
    if not trades_df.empty:  # 如果有交易记录
        buy_trades = trades_df[trades_df['Type'] == 'Buy']  # 获取买入交易
        sell_trades = trades_df[trades_df['Type'] == 'Sell']  # 获取卖出交易

        # 买入点
        if not buy_trades.empty:  # 如果有买入交易
            for _, trade in buy_trades.iterrows():  # 遍历每次买入交易
                trade_date = trade['Date']  # 获取交易日期
                # 找到对应日期在results中的位置
                idx = results[results['Date'] == trade_date].index  # 获取交易日期在结果中的索引
                if len(idx) > 0:  # 如果找到了日期
                    idx = idx[0]  # 获取第一个匹配的索引
                    value = results.loc[idx, 'TotalValue']  # 获取该日期的总资产价值
                    ax1.scatter([trade_date], [value], marker='^', color='green', s=100)  # 标记买入点

            # 添加标签（只添加一次）
            ax1.scatter([], [], marker='^', color='green', s=100, label='买入信号')  # 添加买入信号图例

        # 卖出点
        if not sell_trades.empty:  # 如果有卖出交易
            for _, trade in sell_trades.iterrows():  # 遍历每次卖出交易
                trade_date = trade['Date']  # 获取交易日期
                # 找到对应日期在results中的位置
                idx = results[results['Date'] == trade_date].index  # 获取交易日期在结果中的索引
                if len(idx) > 0:  # 如果找到了日期
                    idx = idx[0]  # 获取第一个匹配的索引
                    value = results.loc[idx, 'TotalValue']  # 获取该日期的总资产价值
                    ax1.scatter([trade_date], [value], marker='v', color='red', s=100)  # 标记卖出点

            # 添加标签（只添加一次）
            ax1.scatter([], [], marker='v', color='red', s=100, label='卖出信号')  # 添加卖出信号图例

    # 更新图例
    handles, labels = ax1.get_legend_handles_labels()  # 获取所有图例
    ax1.legend(handles, labels, loc='upper left')  # 在左上角显示所有图例

    # 2. 绘制累积收益率
    ax2 = axes[1]  # 获取第二个子图
    ax2.plot(results['Date'], results['CumulativeReturns'] * 100, label='策略累积收益率', color='green',
             linewidth=2)  # 绘制策略累积收益率曲线

    # 计算并绘制买入持有策略的收益率
    buy_hold_returns = (results['Price'] / results['Price'].iloc[0]) - 1  # 计算买入持有策略的收益率
    ax2.plot(results['Date'], buy_hold_returns * 100, label='买入持有策略收益率', color='gray', linestyle='--',
             linewidth=1.5)  # 绘制买入持有策略收益率曲线

    ax2.set_title('策略累积收益率')  # 设置标题
    ax2.set_ylabel('累积收益率 (%)')  # 设置y轴标签
    ax2.legend()  # 显示图例
    ax2.grid(True)  # 显示网格

    # 3. 绘制回撤
    ax3 = axes[2]  # 获取第三个子图
    cumulative_returns = results['CumulativeReturns']  # 获取累积收益率
    cumulative_max = cumulative_returns.cummax()  # 计算累积收益率的历史最大值
    drawdown = (cumulative_max - cumulative_returns) * 100  # 计算回撤百分比

    ax3.fill_between(results['Date'], 0, drawdown, color='red', alpha=0.3)  # 以填充区域显示回撤
    ax3.plot(results['Date'], drawdown, color='red', label='回撤 (%)')  # 绘制回撤曲线
    ax3.set_title('策略回撤')  # 设置标题
    ax3.set_xlabel('日期')  # 设置x轴标签
    ax3.set_ylabel('回撤 (%)')  # 设置y轴标签
    ax3.grid(True)  # 显示网格
    ax3.legend()  # 显示图例

    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(os.path.join(OUTPUT_DIR, f'{ticker}_backtest_results.png'))  # 保存图像
    plt.close()  # 关闭图像


def calculate_prediction_accuracy(actual, predicted):
    """
    计算预测方向的准确率
    
    参数:
    actual - 实际价格序列
    predicted - 预测价格序列
    
    返回:
    准确率百分比
    """
    actual_direction = np.diff(actual) > 0  # 计算实际价格变化方向，上涨为True
    predicted_direction = np.diff(predicted) > 0  # 计算预测价格变化方向，上涨为True

    # 计算方向预测的准确率
    accuracy = np.mean(actual_direction == predicted_direction) * 100  # 计算方向预测的准确率百分比
    return accuracy  # 返回准确率


def visualize_predictions(dates, prices, predictions, signals, ticker):
    """
    可视化预测结果和交易信号
    
    参数:
    dates - 日期列表
    prices - 价格列表
    predictions - 预测上涨概率列表
    signals - 交易信号列表
    ticker - 股票代码
    """
    plt.figure(figsize=(14, 7))  # 创建图像

    # 绘制价格和上涨概率
    fig, ax1 = plt.subplots(figsize=(14, 7))  # 创建子图

    # 绘制实际价格
    ax1.plot(dates, prices, label='实际价格', color='blue')  # 绘制实际价格曲线
    ax1.set_xlabel('日期')  # 设置x轴标签
    ax1.set_ylabel('价格', color='blue')  # 设置y轴标签
    ax1.tick_params(axis='y', labelcolor='blue')  # 设置y轴刻度标签颜色

    # 创建另一个Y轴用于上涨概率
    ax2 = ax1.twinx()  # 创建第二个y轴
    ax2.plot(dates, predictions, label='上涨概率', color='red', linestyle='--')  # 绘制上涨概率曲线
    ax2.set_ylabel('上涨概率', color='red')  # 设置y轴标签
    ax2.tick_params(axis='y', labelcolor='red')  # 设置y轴刻度标签颜色
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)  # 添加0.5概率水平线

    # 标记交易信号
    buy_indices = [i for i, signal in enumerate(signals) if signal == 'buy']  # 获取所有买入信号的索引
    sell_indices = [i for i, signal in enumerate(signals) if signal == 'sell']  # 获取所有卖出信号的索引

    if buy_indices:  # 如果有买入信号
        buy_dates = [dates[i] for i in buy_indices]  # 获取所有买入日期
        buy_prices = [prices[i] for i in buy_indices]  # 获取所有买入价格
        ax1.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='买入信号')  # 标记买入点

    if sell_indices:  # 如果有卖出信号
        sell_dates = [dates[i] for i in sell_indices]  # 获取所有卖出日期
        sell_prices = [prices[i] for i in sell_indices]  # 获取所有卖出价格
        ax1.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='卖出信号')  # 标记卖出点

    plt.title(f'{ticker} 股票价格预测与交易信号')  # 设置标题

    # 合并两个图例
    lines1, labels1 = ax1.get_legend_handles_labels()  # 获取第一个轴的图例
    lines2, labels2 = ax2.get_legend_handles_labels()  # 获取第二个轴的图例
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')  # 合并图例并显示

    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(os.path.join(OUTPUT_DIR, f'{ticker}_predictions_and_signals.png'))  # 保存图像
    plt.close()  # 关闭图像


if __name__ == '__main__':
    """主函数入口"""
    # 可以通过命令行参数或配置文件修改这些默认值
    ticker = "KO"  # 默认使用可口可乐股票
    start_date = "2021-01-01"  # 默认起始日期
    end_date = "2023-12-31"  # 默认结束日期
    # 使用默认参数（苹果股票，2021-2023年）
    # predict()
    results = predict(ticker, start_date, end_date, load_existing=False)
    # 自定义股票和日期
    # predict("AAPL", "2021-01-01", "2023-12-31")
    # predict("MSFT", "2021-01-01", "2023-12-31")  # 微软
    # predict("GOOGL", "2021-01-01", "2023-12-31")  # 谷歌

    # 打印最终结果概要
    if results is not None:
        final_value = results.attrs['final_value']
        initial_cash = results.attrs['initial_cash']
        total_return = (final_value / initial_cash - 1) * 100
        print(f"\n==== 预测总结 ====")
        print(f"股票: {ticker}")
        print(f"时间段: {start_date} 至 {end_date}")
        print(f"初始资金: ${initial_cash:.2f}")
        print(f"最终资产: ${final_value:.2f}")
        print(f"总收益率: {total_return:.2f}%")
    else:
        print("预测未完成或出现错误。")
