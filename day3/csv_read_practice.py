import pandas as pd

df = pd.read_csv("samsung_daily.csv")
print(df.head())
print(df.info())

df = pd.read_csv("samsung_daily.csv", parse_dates=["Date"])
df = df.set_index("Date")

print(df.index)
print(df.head())

# 2022년 이후
print(df["2022-01-01":])

# 특정 기간
print(df["2023-01-01":"2023-03-31"])

