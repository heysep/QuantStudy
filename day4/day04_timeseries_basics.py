import pandas as pd

df = pd.read_csv("samsung_daily.csv", parse_dates=["Date"])
df = df.set_index("Date")

print(df.index)          # DatetimeIndex
print(df.index.dtype)    # datetime64[ns]

df.loc["2023"]          # 2023년 전체
df.loc["2023-03"]       # 2023년 3월
df.loc["2023-03-15"]    # 특정 날짜
df.loc["2023-06-01":"2023-06-30"]   # 날짜 범위

# 주간 데이터 생성
weekly = df.resample("W").agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
})
print(weekly.head())

# 월간 데이터 생성
monthly = df.resample("M").agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
})
print(monthly.head())

# 어제 종가
df["Close_shift1"] = df["Close"].shift(1)

# 오늘 종가 - 어제 종가 (일일 변화폭)
df["Close_diff1"] = df["Close"].diff(1)

# 5일 전 종가
df["Close_shift5"] = df["Close"].shift(5)

# 5일 차이
df["Close_diff5"] = df["Close"].diff(5)

# 5일 이동평균선
df["MA5"] = df["Close"].rolling(window=5).mean()

# 20일 이동평균선
df["MA20"] = df["Close"].rolling(window=20).mean()


import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(df["Close"], label="Close")
plt.plot(df["MA5"], label="MA5 (5-day)")
plt.plot(df["MA20"], label="MA20 (20-day)")

plt.legend()
plt.title("Close Price with MA5 and MA20")
plt.grid(True)
plt.show()

