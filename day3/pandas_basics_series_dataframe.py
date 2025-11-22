"""
Day 3: pandas Series / DataFrame 기본 + CSV 로딩 연습
"""

import numpy as np
import pandas as pd

s = pd.Series([10, 20, 30, 40], name="price")
print(s)
print(type(s))
print( "values:", s.values)   # numpy array
print( "index:", s.index)    # RangeIndex
print( "name:", s.name)     # "price"

# 인덱싱
print(s[0])       # 첫 번째 값
print(s[1:3])     # 슬라이스

# 통계 함수
print("mean:", s.mean())    # 평균
print("std :", s.std())     # 표준편차
print("max :", s.max())     # 최대값
print("min :", s.min())     # 최소값
print("describe:\n", s.describe())  # 요약 통계
