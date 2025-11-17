import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# 원소별(Element-wise) 연산 : 두 배열의 각 원소를 대응하여 연산을 수행한다. (shape가 같아야 하거나 broadcasting이 가능해야 한다)
print(x + y)   # [5 7 9]
print(x - y)   # [-3 -3 -3]
print(x * y)   # [ 4 10 18]
print(x / y)   # [0.25 0.4  0.5 ]

# 스칼라(scalar) 연산 : 배열의 모든 원소에 대해 동일한 연산을 수행한다.
print("스칼라 연산 : ", 2 * x)      # [2 4 6]
print("스칼라 연산 : ", x + 10)     # [11 12 13]
