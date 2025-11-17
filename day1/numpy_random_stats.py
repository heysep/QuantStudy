import numpy as np

# 1) 난수 생성
data = np.random.randn(100000)

# 2) 평균
mean_value = np.mean(data)

# 3) 분산
var_value = np.var(data)

# 4) 표준편차
std_value = np.sqrt(var_value)

print("개수:", len(data))
print("평균:", mean_value)
print("분산:", var_value)
print("표준편차:", std_value)

# 평균이 0에 가까워지는 이유
# 많이 뽑을수록 전체 평균에 딱 맞게 균형이 맞춰지기 때문이다.

# 분산이 1에 가까워지는 이유
# 많이 뽑을수록 데이터의 퍼짐 정도가 원래 정규분포의 퍼짐(=1)에 맞춰지기 때문이다.

# 표준편차가 1에 가까워지는 이유
# 표준편차는 퍼짐의 크기라, 분산이 1에 맞춰지면 자연스럽게 √1 = 1이 나오기 때문이다.