import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. 데이터 불러오기
# ------------------------------------------------
df = pd.read_csv("samsung_daily.csv", parse_dates=["Date"])
df = df.set_index("Date")

# ------------------------------------------------
# 2. 수익률 계산
# ------------------------------------------------
df["return_simple"] = df["Close"].pct_change()              # 단순수익률
df["return_log"] = np.log(df["Close"] / df["Close"].shift(1))  # 로그수익률

# ------------------------------------------------
# 3. 변동성 계산
# ------------------------------------------------
df["vol_20"] = df["return_log"].rolling(20).std() * np.sqrt(252)  # 연간 변동성
df["vol_5"] = df["return_log"].rolling(5).std() * np.sqrt(252)

# ------------------------------------------------
# 4. 누적수익률
# ------------------------------------------------
df["cumulative"] = (1 + df["return_simple"]).cumprod()

# ------------------------------------------------
# 5. 시각화
# ------------------------------------------------

# 누적수익률 시각화
plt.figure(figsize=(12, 6))
plt.plot(df["cumulative"], label="Cumulative Return")
plt.title("Samsung Cumulative Return")
plt.legend()
plt.show()

# 변동성 시각화
plt.figure(figsize=(12, 6))
plt.plot(df["vol_20"], label="20-day Volatility")
plt.plot(df["vol_5"], label="5-day Volatility")
plt.title("Volatility")
plt.legend()
plt.show()

# ------------------------------------------------
# 6. 샤프지수 계산
# ------------------------------------------------
risk_free = 0.02   # 연 2% 기준
daily_risk_free = risk_free / 252

mean_return = df["return_log"].mean()
volatility = df["return_log"].std()

sharpe_ratio = (mean_return - daily_risk_free) / volatility * np.sqrt(252)

print("Sharpe Ratio:", sharpe_ratio)