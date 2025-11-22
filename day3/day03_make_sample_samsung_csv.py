import pandas as pd
import numpy as np

# 날짜 생성 (약 5년치)
dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="B")  # 평일만

# 가격 시뮬레이션 (랜덤 워크)
np.random.seed(42)
price = 60000 + np.cumsum(np.random.randn(len(dates)) * 300)

# OHLC 생성
df = pd.DataFrame({
    "Date": dates,
    "Open": price + np.random.randn(len(dates)) * 50,
    "High": price + np.abs(np.random.randn(len(dates)) * 70),
    "Low": price - np.abs(np.random.randn(len(dates)) * 70),
    "Close": price + np.random.randn(len(dates)) * 50,
    "Volume": np.random.randint(5e5, 5e6, size=len(dates)),
})

df.to_csv("samsung_daily.csv", index=False)

print("samsung_daily.csv 생성 완료!")
print(df.head())
