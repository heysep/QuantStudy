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
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(df["Close"], label="Close Price")
plt.title("Samsung Price")

plt.subplot(2, 1, 2)
plt.plot(df["return_log"], label="Log Return")
plt.plot(df["vol_20"], label="20-day Volatility")
plt.legend()
plt.title("Returns & Volatility")

plt.tight_layout()
plt.show()

#퀀트에서 로그 수익률을 쓰는 이유
# 수익률 분포가 정규분포에 더 가까워짐 (ex: -50%를 복구하려면 100% 상승을 해야함 -> 로그 수익률을 쓰면 50% 상승으로 복구 가능)