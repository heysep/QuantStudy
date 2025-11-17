import numpy as np

a = np.array([1, 2, 3])           # 1차원 ndarray
b = np.array([[1, 2, 3],
              [4, 5, 6]])         # 2차원 ndarray

c = np.array([[1, 2, 3],
              [4, 5, 6]])          # 2차원 ndarray

print(type(a)) 
print(c.ndim)  # 출력예시: 2 (차원 수)
print(c.shape) # 출력예시: (2, 3) : 2행 3열

print(a.shape)  # (3, )           → 원소 3개짜리 1차원은 (3, ) 이렇게 표시된다.

# Python 기본 list보다:
# 메모리 배치가 연속적이고
# 타입이 모두 같고(int, float 등)
# 벡터/행렬 연산이 매우 빠르게 동작하도록 설계됨.
# 그래서 수학적 계산이 빠르게 되야하는 퀀트 분야에서는 list 대신 ndarray를 사용하는 것이 좋음.

#몰랐던점
# 1. np.array를 ndarray 라고 불리는데 numpy 내부에서 정의된 핵심 클래스 이름이 numpy.ndarray 이기 때문이다.