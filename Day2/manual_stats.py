import numpy as np
import math

np.random.seed(1)
x = np.random.randn(100)  # 일부러 100개 정도만

n = len(x)
mean = x.mean()1) 모분산(전체 분산, n으로 나누기)

# 모분산 (population variance) : n으로 나눔
pop_var = np.sum((x - mean) ** 2) / n

# 표본분산 (sample variance) : (n-1)으로 나눔
sample_var = np.sum((x - mean) ** 2) / (n - 1)

print("np.var (default, ddof=0)   :", x.var())
print("np.var (ddof=1, sample var):", x.var(ddof=1))
print("직접 pop_var               :", pop_var)
print("직접 sample_var            :", sample_var)


# 1) 모분산(전체 분산, n으로 나누기) : 전체를 다 알고 있으면 그냥 n으로 나누면 된다.
# 2) 표본분산(샘플 분산, n−1로 나누기) : 일부만 알고 있으면 n−1로 나누어야 한다.