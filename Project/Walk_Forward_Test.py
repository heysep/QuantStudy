import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt

# 모듈 임포트
from engine.data_loader import fetch_data
from engine.backtester import Backtester
from strategies.trend_momentum import TrendMomentumStrategy
from Screening_Engine import QuantScreeningEngine

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_walk_forward_analysis():
    print("=== Walk-Forward Analysis (2010 ~ Present) ===")
    print("* 주의: 생존 편향(Survivorship Bias)이 존재합니다.")
    print("* 현재 S&P 500 종목을 과거로 시뮬레이션합니다.\n")

    # 1. 유니버스 확보 (현재 기준)
    screener = QuantScreeningEngine()
    screener.get_universe() # 티커만 가져옴
    tickers = screener.tickers
    
    # 벤치마크 추가
    if 'SPY' not in tickers: tickers.append('SPY')
    
    # 테스트 속도를 위해 일부만 샘플링할지? -> 아니오, 전체 다 받아야 정확함.
    # 하지만 500개 15년치는 너무 큼.
    # 전략이 'Top 5'만 뽑으므로, 사실 'Screening'을 통과할만한 애들만 있어도 됨.
    # 여기서는 일단 전체 다운로드 시도 (시간 걸림)
    print(f"\n[Data] 전체 데이터 다운로드 중 (2010-01-01 ~ 현재)...")
    data = fetch_data(tickers, start_date=datetime(2010, 1, 1))
    
    if data is None:
        return

    # 2. 구간 설정 (3년 단위)
    start_year = 2010
    end_year = datetime.now().year
    
    periods = []
    curr = start_year
    while curr < end_year:
        s_date = datetime(curr, 1, 1)
        e_date = datetime(curr + 3, 1, 1) - timedelta(days=1)
        if e_date > datetime.now():
            e_date = datetime.now()
        periods.append((s_date, e_date))
        curr += 3

    # 3. 구간별 백테스트
    results = []
    
    print(f"\n{'='*80}")
    print(f"{'Period':<20} | {'CAGR(Strat)':<12} | {'CAGR(SPY)':<12} | {'Diff':<10} | {'MDD':<10}")
    print(f"{'-'*80}")

    for s_date, e_date in periods:
        period_str = f"{s_date.year}~{e_date.year}"
        
        # 해당 구간 데이터 슬라이싱
        # 백테스트 시작일은 데이터 시작일 + 1년 (지표 계산용)
        bt_start_date = s_date + timedelta(days=365)
        
        if bt_start_date >= e_date:
            continue
            
        # 데이터 슬라이싱 (앞에 1년치 여유분 포함해서 자름)
        slice_start = s_date
        sub_data = data[(data.index >= slice_start) & (data.index <= e_date)]
        
        if sub_data.empty: continue

        # 전략 실행
        strategy = TrendMomentumStrategy(top_k=5)
        backtester = Backtester(strategy, sub_data, initial_capital=100_000_000)
        
        # 백테스트 (Quiet mode)
        # print를 억제하고 싶지만 Backtester 구조상 출력됨. 
        # 여기서는 그냥 실행.
        
        equity = backtester.run(start_date=bt_start_date)
        
        # 성과 계산
        if equity is None or len(equity) < 10:
            continue
            
        # CAGR
        days = (equity.index[-1] - equity.index[0]).days
        years = days / 365.25
        total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
        cagr = (1 + total_ret) ** (1/years) - 1
        
        # MDD
        dd = (equity - equity.cummax()) / equity.cummax()
        mdd = dd.min()
        
        # Benchmark CAGR
        spy_sub = sub_data['SPY']
        spy_sub = spy_sub[spy_sub.index >= bt_start_date]
        spy_total = (spy_sub.iloc[-1] / spy_sub.iloc[0]) - 1
        spy_cagr = (1 + spy_total) ** (1/years) - 1
        
        results.append({
            'Period': period_str,
            'Strategy': cagr,
            'Benchmark': spy_cagr,
            'MDD': mdd
        })
        
        print(f"{period_str:<20} | {cagr*100:>11.2f}% | {spy_cagr*100:>11.2f}% | {cagr-spy_cagr:>+9.2%}p | {mdd*100:>9.2f}%")

    # 4. 종합 분석
    print(f"{'='*80}")
    
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        avg_strat = df_res['Strategy'].mean()
        avg_bench = df_res['Benchmark'].mean()
        win_rate = len(df_res[df_res['Strategy'] > df_res['Benchmark']]) / len(df_res)
        
        print(f"\n[Summary]")
        print(f"Average CAGR (Strategy): {avg_strat*100:.2f}%")
        print(f"Average CAGR (Benchmark): {avg_bench*100:.2f}%")
        print(f"Win Rate (vs SPY): {win_rate*100:.0f}% ({len(df_res[df_res['Strategy'] > df_res['Benchmark']])}/{len(df_res)})")
        
        # CSV 저장
        df_res.to_csv('walk_forward_results.csv', index=False)
        print("\n결과가 'walk_forward_results.csv'에 저장되었습니다.")

if __name__ == "__main__":
    run_walk_forward_analysis()
