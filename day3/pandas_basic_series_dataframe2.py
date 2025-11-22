import pandas as pd

data = {
    "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
    "open": [70000, 71000, 70500],
    "close": [71000, 70500, 71500],
    "volume": [1000000, 1200000, 1100000],
}

df = pd.DataFrame(data)
print(df)
print( "shape:", df.shape)      # (3, 4)
print( "columns:", df.columns)    # Index(['date', 'open', 'close', 'volume'])
print( "index:", df.index)      # RangeIndex(0, 3)

# 컬럼
print(df["close"])
print(df[["open", "close"]])

# 행 (iloc)
print(df.iloc[0])       # 첫 번째 행
print(df.iloc[0:2])     # 0,1 행

# 행 + 컬럼 같이
print(df.loc[0, "close"])
print(df.loc[0:1, ["date", "close"]])
