import sys
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

# 모듈 임포트 (같은 Project 폴더 내)
from engine.data_loader import fetch_data
from engine.backtester import Backtester
from strategies.trend_momentum import TrendMomentumStrategy
from Screening_Engine import QuantScreeningEngine # [NEW] 스크리닝 엔진 임포트

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=== 퀀트 전략 실행 (Screening + Momentum) ===")
    
    # 1. 사용자 입력
    try:
        input_str = input("\n투자하실 총 금액을 입력해주세요 (예: 100000000): ")
        initial_capital = int(input_str.replace(",", "").replace("_", ""))
        if initial_capital <= 0: raise ValueError
        print(f"-> 설정된 투자금: {initial_capital:,.0f} 원\n")
    except:
        print("-> 기본값 1억원으로 설정합니다.\n")
        initial_capital = 100_000_000

    # 2. 1차 스크리닝 실행
    print("\n[Step 1] 1차 스크리닝(Pre-Screening) 실행 중...")
    screener = QuantScreeningEngine()
    screened_tickers = screener.run()
    
    if not screened_tickers:
        print("스크리닝 결과 종목이 없습니다. 기본 ETF로 대체합니다.")
        screened_tickers = ['SPY', 'QQQ', 'IWM', 'VNQ', 'GLD', 'TLT', 'HYG', 'EEM']
    else:
        print(f"\n-> 스크리닝 통과 종목 ({len(screened_tickers)}개): {screened_tickers}")

    # 3. 데이터 수집 (백테스트용 장기 데이터)
    # 스크리닝은 최근 데이터로 했지만, 모멘텀 전략 백테스트를 위해 과거 데이터도 필요
    # 여기서는 3년치 데이터를 가져와서 테스트
    print("\n[Step 2] 전략 실행용 데이터 수집 중...")
    
    # 벤치마크(SPY) 추가
    if 'SPY' not in screened_tickers:
        screened_tickers.append('SPY')
        
    start_date = datetime.now() - timedelta(days=365*3) # 3년
    data = fetch_data(screened_tickers, start_date=start_date)

    if data is None:
        return

    # 4. 전략 및 백테스터 초기화 [수정: top_k=5]
    # 스크리닝된 종목들 중에서 다시 모멘텀/추세로 Top 5 선정
    strategy = TrendMomentumStrategy(xgb_model=None, target_vol=0.12, top_k=5)
    backtester = Backtester(strategy, data, initial_capital=initial_capital)

    # 5. 백테스트 실행
    # 데이터 시작일 + 252일(1년) 후부터 백테스트 시작 (모멘텀 지표 계산 확보)
    bt_start = data.index[0] + timedelta(days=252)
    if bt_start >= data.index[-1]:
        print("데이터 기간이 너무 짧아 백테스트를 건너뜁니다.")
    else:
        print(f"\n[Step 3] 백테스트 실행 ({bt_start.date()} ~ 현재)...")
        backtester.run(start_date=bt_start)
        backtester.analyze_performance()

    # 6. 오늘 기준 포트폴리오 제안
    print("\n" + "="*50)
    print(f" [최종] 오늘 기준 포트폴리오 배분 제안")
    print(f" 투자금: {initial_capital:,.0f} 원")
    print("="*50)
    
    current_weights = strategy.rebalance(data)
    active_weights = current_weights[current_weights > 0].sort_values(ascending=False)
    
    print(f"{'Ticker':<10} | {'비중':<10} | {'매수 금액':<15}")
    print("-" * 45)
    
    stock_sum = 0
    for ticker, w in active_weights.items():
        buy_amt = w * initial_capital
        stock_sum += buy_amt
        print(f"{ticker:<10} | {w*100:>6.2f}%   | {buy_amt:>15,.0f} 원")
    
    print("-" * 45)
    cash_w = 1.0 - active_weights.sum()
    cash_amt = initial_capital - stock_sum
    
    if cash_w >= 0.001: # 0.1% 이상일 때만 출력
        print(f"{'CASH':<10} | {cash_w*100:>6.2f}%   | {cash_amt:>15,.0f} 원")
    elif cash_w < -0.001:
        print(f"{'LEVERAGE':<10} | {active_weights.sum()*100:>6.2f}%   | (차입 필요)")
    else:
        print(f"{'FULL INV':<10} | 100.00%   | 잔액 없음")
        
    print("="*50)
    print("\n* 주의: 위 백테스트 결과는 '현재 시점의 스크리닝 종목'을 과거로 시뮬레이션한 것으로,")
    print("  생존 편향(Survivorship Bias)이 포함되어 있어 실제 과거 성과보다 좋게 나올 수 있습니다.")

if __name__ == "__main__":
    main()

