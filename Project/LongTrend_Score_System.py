import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_rsi(prices, period=14):
    """
    RSI(Relative Strength Index) 계산
    RSI = 100 - (100 / (1 + RS))
    RS = 평균 상승폭 / 평균 하락폭
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_stock_data(ticker, period='2y'):
    """
    yfinance를 사용하여 주식 데이터 가져오기
    최소 1년 이상의 데이터가 필요하므로 2년치를 가져옴
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        raise ValueError(f"티커 '{ticker}'의 데이터를 가져올 수 없습니다.")
    
    return df


def calculate_ma200_score(df):
    """
    1) 200일선 점수 계산 (40% 비중)
    현재가가 200일선 위에 있으면 100점, 아니면 0점
    """
    if len(df) < 200:
        return 0, "데이터 부족 (200일 미만)"
    
    # 200일 이동평균선 계산
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # 최신 데이터
    current_price = df['Close'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    
    # 200일선 위에 있으면 100점
    if current_price > ma200:
        score = 100
        status = f"200일선 위 ({current_price:.2f} > {ma200:.2f})"
    else:
        score = 0
        status = f"200일선 아래 ({current_price:.2f} < {ma200:.2f})"
    
    return score, status


def calculate_momentum12_score(df):
    """
    2) 12개월 모멘텀 점수 계산 (30% 비중)
    12개월 전 가격 대비 현재 가격이 양수면 100점, 아니면 0점
    """
    if len(df) < 252:  # 1년 = 약 252 거래일
        return 0, "데이터 부족 (12개월 미만)"
    
    # 12개월 전 가격 (약 252 거래일 전)
    price_12m_ago = df['Close'].iloc[-252]
    current_price = df['Close'].iloc[-1]
    
    # 12개월 수익률 계산
    momentum = (current_price - price_12m_ago) / price_12m_ago * 100
    
    # 양수면 100점
    if momentum > 0:
        score = 100
        status = f"12개월 모멘텀 양수 (+{momentum:.2f}%)"
    else:
        score = 0
        status = f"12개월 모멘텀 음수 ({momentum:.2f}%)"
    
    return score, status


def calculate_oversold_score(df):
    """
    3) 단기 과매도·조정 구간 점수 계산 (20% 비중)
    RSI 30 이하 또는 MA20 대비 -2~3% 이탈이면 100점, 아니면 0점
    """
    if len(df) < 20:
        return 0, "데이터 부족 (20일 미만)"
    
    # RSI 계산
    rsi = calculate_rsi(df['Close'], period=14)
    current_rsi = rsi.iloc[-1]
    
    # MA20 계산
    df['MA20'] = df['Close'].rolling(window=20).mean()
    current_price = df['Close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    
    # MA20 대비 이탈률 계산
    deviation_from_ma20 = (current_price - ma20) / ma20 * 100
    
    # 조건 확인: RSI 30 이하 또는 MA20 대비 -2~3% 이탈
    is_oversold = (current_rsi <= 30) or (-3 <= deviation_from_ma20 <= -2)
    
    if is_oversold:
        score = 100
        if current_rsi <= 30:
            status = f"과매도 구간 (RSI: {current_rsi:.2f} ≤ 30)"
        else:
            status = f"조정 구간 (MA20 대비 {deviation_from_ma20:.2f}%)"
    else:
        score = 0
        status = f"정상 구간 (RSI: {current_rsi:.2f}, MA20 대비: {deviation_from_ma20:.2f}%)"
    
    return score, status


def calculate_volatility_score(df):
    """
    4) 변동성 안정 여부 점수 계산 (10% 비중)
    변동성이 낮을수록 높은 점수 (0~100점)
    최근 20일 변동성을 기준으로 정규화
    """
    if len(df) < 20:
        return 0, "데이터 부족 (20일 미만)"
    
    # 로그 수익률 계산
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 최근 20일 변동성 계산 (연간화)
    volatility_20d = df['log_return'].tail(20).std() * np.sqrt(252) * 100
    
    # 변동성을 점수로 변환 (0~100점)
    # 변동성 10% 이하면 100점, 50% 이상이면 0점으로 선형 보간
    if volatility_20d <= 10:
        score = 100
    elif volatility_20d >= 50:
        score = 0
    else:
        # 선형 보간: 10% = 100점, 50% = 0점
        score = 100 * (1 - (volatility_20d - 10) / 40)
    
    status = f"변동성: {volatility_20d:.2f}%"
    
    return score, status


def evaluate_stock(ticker):
    """
    주식 평가 시스템 메인 함수
    티커를 입력받아 4가지 지표로 종합 점수를 계산
    """
    print(f"\n{'='*60}")
    print(f"주식 평가 시스템: {ticker}")
    print(f"{'='*60}\n")
    
    # 데이터 가져오기
    try:
        df = get_stock_data(ticker)
        print(f"데이터 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"현재가: ${df['Close'].iloc[-1]:.2f}\n")
    except Exception as e:
        print(f"오류 발생: {e}")
        return
    
    # 각 지표 점수 계산
    scores = {}
    statuses = {}
    
    # 1) 200일선 (40%)
    scores['ma200'], statuses['ma200'] = calculate_ma200_score(df)
    
    # 2) 12개월 모멘텀 (30%)
    scores['momentum12'], statuses['momentum12'] = calculate_momentum12_score(df)
    
    # 3) 단기 과매도 (20%)
    scores['oversold'], statuses['oversold'] = calculate_oversold_score(df)
    
    # 4) 변동성 (10%)
    scores['volatility'], statuses['volatility'] = calculate_volatility_score(df)
    
    # 가중 평균으로 최종 점수 계산
    weights = {
        'ma200': 0.40,
        'momentum12': 0.30,
        'oversold': 0.20,
        'volatility': 0.10
    }
    
    final_score = (
        scores['ma200'] * weights['ma200'] +
        scores['momentum12'] * weights['momentum12'] +
        scores['oversold'] * weights['oversold'] +
        scores['volatility'] * weights['volatility']
    )
    
    # 결과 출력
    print("=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"1. 200일선 (40%): {scores['ma200']:.0f}점 - {statuses['ma200']}")
    print(f"2. 12개월 모멘텀 (30%): {scores['momentum12']:.0f}점 - {statuses['momentum12']}")
    print(f"3. 단기 과매도·조정 (20%): {scores['oversold']:.0f}점 - {statuses['oversold']}")
    print(f"4. 변동성 안정 (10%): {scores['volatility']:.0f}점 - {statuses['volatility']}")
    print("-" * 60)
    print(f"최종 점수: {final_score:.1f}/100")
    print("=" * 60)
    
    # 투자 의견
    if final_score >= 70:
        opinion = "매우 우수 - 장기 투자 적합"
    elif final_score >= 50:
        opinion = "양호 - 투자 고려 가능"
    elif final_score >= 30:
        opinion = "보통 - 신중한 판단 필요"
    else:
        opinion = "주의 - 투자 신중"
    
    print(f"투자 의견: {opinion}\n")
    
    return final_score


if __name__ == "__main__":
    # 티커 입력 받기
    ticker = input("평가할 주식 티커를 입력하세요 (예: AAPL, TSLA, MSFT): ").strip().upper()
    
    if not ticker:
        print("티커를 입력해주세요.")
    else:
        evaluate_stock(ticker)
