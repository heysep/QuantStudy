#Day 2: 난수 생성 + 평균/분산/표준편차 + 모수/표본
import numpy as np

np.random.seed(0)  # 재현 가능하게

# 정규분포 (평균 0, 표준편차 1)
a = np.random.randn(10)

# 일반 정규분포 지정 (평균, 표준편차)
b = np.random.normal(loc=5, scale=2, size=10)

# 균등분포 (0~1)
c = np.random.rand(10)

# 균등분포 (low~high)
d = np.random.uniform(low=-1, high=1, size=10)

print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)

x1 = np.random.randn(1000)      # (1000,)
x2 = np.random.randn(1000, 3)   # (1000, 3)

print(x1.shape)  # (1000,)
print(x2.shape)  # (1000, 3)
