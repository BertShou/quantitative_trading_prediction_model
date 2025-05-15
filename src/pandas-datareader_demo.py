# 引用网页[2][6]的API参数规则
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

# 设置时间范围（网页2示例）
start = datetime(2020, 1, 1)
end = datetime.now()

# 获取苹果公司历史数据（网页2美股代码规则）
stock = web.DataReader("AAPL", "yahoo", start, end)

# 获取上证指数历史数据（网页6深沪股票规则）
# stock_sh = web.DataReader("000001.SS", "yahoo", start, end)

print(stock.head())