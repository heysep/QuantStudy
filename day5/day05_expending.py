# Day 5: expending 연습
# expending으로 할수 있는것들 : 주간 데이터, 월간 데이터 (누적구간을 잡아 통계를 계산하는 함수)
# 퀀트에서 사용처 : 누적 평균 수익률, 누적 최대/최소로 드로다운 계산(현재 값이 과거 평균보다 높은지 낮은지지)
# rolling()은 최근 N개만 보는 고정 길이 창이고
# expending()은 최근 N개를 보는 고정 길이 창이 아니라 처음부터 끝까지 보는 창이다.

#주의할점
#NAN이 있는 경우 그래프가 끊기거나 백테스팅에서 정상적이 계산이 되지 않는다
#그래서 퀀트에서는 NaN을 전처리를 한다
#NAN을 전처리할때는 해당 row를 삭제를 하거나 특정 컬럼기준으로 NAN을 제거하거나 데이터가 중간에 비면 0으로 채워넣는다


import pandas as pd

s = pd.Series([10, 20, 30, 40, 50])

print(s.rolling(3).mean()) # 결과: NaN, NaN, 20.0, 30.0, 40.0
print(s.expanding().mean()) # 결과: 10.0, 15.0, 20.0, 25.0, 30.0
print(s.expanding().sum()) # 결과: 10, 30, 60, 100, 150
print(s.expanding().max()) # 결과: 10, 20, 30, 40, 50
print(s.expanding().min()) # 결과: 10, 20, 30, 40, 50
print(s.expanding().std()) # 결과: NaN, NaN, 8.16, 11.18, 14.14

