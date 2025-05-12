import yfinance as yf

# 获取苹果公司历史数据（需非中国大陆IP）
data = yf.download("AAPL", start="2023-01-01", end="2025-05-03")
print(data.head())