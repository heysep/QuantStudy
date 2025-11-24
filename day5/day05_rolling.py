# Day 5: Rolling 연습
# rolling으로 할수 있는것들 : 이동평균선, 이동표준편차, 최근 n일 최대값/최솟값값 등
# 이동평균선 : 최근 n일 종가의 평균값
# 이동표준편차 : 최근 n일 종가의 표준편차
# 최근 n일 최대값 : 최근 n일 종가의 최대값
# 최근 n일 최솟값 : 최근 n일 종가의 최솟값

import pandas as pd

df = pd.read_csv("samsung_daily.csv", parse_dates=["Date"])
df = df.set_index("Date")

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

# 50일 이동 평균선
df["MA50"] = df["Close"].rolling(window=50).mean()

# 100일 이동 평균선
df["MA100"] = df["Close"].rolling(window=100).mean()

# 50일선과 100일선의 이격도
df["MA50_MA100_diff"] = df["MA50"] - df["MA100"]

# 차트 표준편차
df["std"] = df["Close"].rolling(window=20).std()

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(df["Close"], label="Close")
plt.plot(df["MA5"], label="MA5 (5-day)")
plt.plot(df["MA20"], label="MA20 (20-day)")
plt.plot(df["MA50"], label="MA50 (50-day)")
plt.plot(df["MA100"], label="MA100 (100-day)")
plt.plot(df["MA50_MA100_diff"], label="MA50 - MA100")
plt.plot(df["std"], label="std (20-day)")
plt.legend()
plt.title("Close Price with MA5, MA20, MA50, MA100")
plt.grid(True)
plt.show()

