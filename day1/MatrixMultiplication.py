import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
C = np.array([1, 2])

# 행렬 곱셈
print(A @ B)
print(np.dot(A, B))

# 전치행렬 (행과 열을 바꿔서 계산산)
print(A.T)

# 브로드캐스팅
print("브로드캐스팅 : ", A + C)
