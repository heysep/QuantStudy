import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class StaplesValuationEngine:
    def __init__(self, tickers):
        self.tickers = tickers
        self.data = {}
        self.results = {}

    def fetch_data(self):
        """데이터 수집: 주가, EPS, 재무지표"""
        print(f"데이터 수집 중... (대상: {len(self.tickers)}개)")
        
        for ticker in self.tickers:
            try:
                print(f"  - {ticker} 데이터 가져오는 중...")
                stock = yf.Ticker(ticker)
                
                # 1. 주가 데이터 (최근 10년)
                hist = stock.history(period="10y")
                if hist.empty:
                    print(f"    경고: {ticker} 주가 데이터 없음")
                    continue
                
                # Timezone 제거 (비교 오류 방지)
                hist.index = hist.index.tz_localize(None)
                
                # 2. 재무 데이터 (분기별 EPS, EBITDA 등)
                # yfinance의 financials는 연간/분기 데이터를 제공함
                # TTM EPS 계산을 위해 분기 데이터 사용
                q_income = stock.quarterly_income_stmt
                q_bs = stock.quarterly_balance_sheet
                q_cf = stock.quarterly_cashflow
                
                # Forward EPS (Yahoo Finance Analyst Estimates)
                info = stock.info
                forward_eps = info.get('forwardEps')
                current_price = info.get('currentPrice')
                if not current_price:
                    current_price = hist['Close'].iloc[-1]
                
                # 데이터 저장 구조
                self.data[ticker] = {
                    'history': hist,
                    'financials': {
                        'q_income': q_income,
                        'q_bs': q_bs,
                        'q_cf': q_cf
                    },
                    'info': info,
                    'forward_eps': forward_eps,
                    'current_price': current_price
                }
                
            except Exception as e:
                print(f"    오류 발생 ({ticker}): {e}")

    def calculate_historical_metrics(self):
        """과거 밸류에이션 지표 계산 (Historical PER, P/B, Dividend Yield)"""
        print("\n지표 계산 중...")
        
        for ticker, data in self.data.items():
            hist = data['history']
            q_income = data['financials']['q_income']
            
            # --- 1. Historical PER 계산 ---
            # 분기별 EPS를 가져와서 TTM(Trailing Twelve Months) EPS 계산
            # 날짜별로 보간하여 Daily PER 산출
            
            if q_income is not None and not q_income.empty and 'Diluted EPS' in q_income.index:
                eps_q = q_income.loc['Diluted EPS']
                # 최신순이므로 역순 정렬
                eps_q = eps_q.sort_index()
                
                # TTM EPS = 최근 4분기 합
                eps_ttm = eps_q.rolling(window=4).sum()
                
                # Index Timezone 제거
                eps_ttm.index = pd.to_datetime(eps_ttm.index).tz_localize(None)
                
                # 일별 데이터로 확장 (ffill)
                # 재무제표 발표일 기준이 정확하지만, 간편하게 해당 분기 말일 기준으로 매핑 후 ffill
                eps_ttm_daily = eps_ttm.reindex(hist.index, method='ffill')
                
                # PER = Price / TTM EPS
                per_series = hist['Close'] / eps_ttm_daily
                
                # 이상치 제거 (적자 전환 등)
                per_series = per_series[per_series > 0]
                per_series = per_series[per_series < 100] # 100배 이상은 제외 (일시적 이익 급감 등)
                
                data['metrics'] = {
                    'per_series': per_series,
                    'eps_ttm_daily': eps_ttm_daily
                }
            else:
                # EPS 데이터 부족 시 처리
                print(f"    경고: {ticker} EPS 데이터 부족으로 Historical PER 계산 불가")
                data['metrics'] = {'per_series': pd.Series()}

            # --- 2. 배당 수익률 (Dividend Yield) ---
            # yfinance history에 'Dividends' 컬럼이 있음
            # TTM 배당금 계산
            dividends = hist['Dividends']
            div_ttm = dividends.rolling(window=365).sum() # 1년치 합계
            div_yield = (div_ttm / hist['Close']) * 100
            
            data['metrics']['div_yield_series'] = div_yield

    def calculate_valuation(self):
        """적정주가 산출 (평균 PER 회귀 모형)"""
        print("\n밸류에이션 수행 중...")
        
        for ticker, data in self.data.items():
            metrics = data.get('metrics')
            if not metrics or metrics['per_series'].empty:
                continue
            
            per_series = metrics['per_series']
            forward_eps = data['forward_eps']
            current_price = data['current_price']
            
            # 1. 평균 PER 계산 (5년, 10년)
            # 데이터가 충분하지 않으면 가능한 기간만 사용
            avg_per_5y = per_series.tail(252*5).mean()
            avg_per_10y = per_series.mean() # 전체 기간 (최대 10년)
            
            # 최근 PER
            current_per = per_series.iloc[-1] if not per_series.empty else None
            
            # 2. 적정주가 계산 (Target Price)
            # Target Price = Forward EPS * Avg PER
            if forward_eps:
                target_price_5y = forward_eps * avg_per_5y
                target_price_10y = forward_eps * avg_per_10y
                
                upside_5y = (target_price_5y / current_price - 1) * 100
                upside_10y = (target_price_10y / current_price - 1) * 100
            else:
                target_price_5y = None
                target_price_10y = None
                upside_5y = None
                upside_10y = None
            
            # 결과 저장
            self.results[ticker] = {
                'Current Price': current_price,
                'Forward EPS': forward_eps,
                'Current PER': current_per,
                'Avg PER (5y)': avg_per_5y,
                'Avg PER (10y)': avg_per_10y,
                'Target Price (5y PER)': target_price_5y,
                'Upside (5y)': upside_5y,
                'Target Price (10y PER)': target_price_10y,
                'Upside (10y)': upside_10y,
                'Div Yield': data['metrics']['div_yield_series'].iloc[-1] if not data['metrics']['div_yield_series'].empty else 0.0
            }

    def generate_report(self):
        """보고서 출력 (텍스트 중심, 리츠 분석 스타일)"""
        print("\n" + "="*80)
        print("필수소비재(Consumer Staples) 밸류에이션 리포트")
        print(" * 핵심: 이익 안정성이 높은 기업은 '평균 PER'로 회귀하는 경향이 강함")
        print(" * 방식: Forward EPS × 5년 평균 PER = 적정 주가")
        print("="*80)
        
        # --- STEP 1: 밸류에이션 (평균 PER 회귀) ---
        print("\n" + "="*80)
        print("STEP 1: 평균 PER 대비 밸류에이션 (Upside 확인)")
        print(" * Upside > 10%: 저평가 (매수 기회)")
        print(" * Upside < -10%: 고평가 (조정 가능성)")
        print("="*80)
        
        print(f"{'Ticker':<6} | {'Price':<8} | {'Fwd EPS':<8} | {'Avg PER':<8} | {'Target':<9} | {'Upside':<8} | {'Status':<12}")
        print("-" * 90)
        
        # Top Pick 선정을 위한 리스트
        top_picks = []
        
        for ticker, res in self.results.items():
            price = res['Current Price']
            f_eps = res['Forward EPS'] if res['Forward EPS'] else 0
            a_per = res['Avg PER (5y)'] if res['Avg PER (5y)'] else 0
            target = res['Target Price (5y PER)'] if res['Target Price (5y PER)'] else 0
            upside = res['Upside (5y)'] if res['Upside (5y)'] else 0
            div = res['Div Yield']
            
            # 상태 판단
            if upside >= 10:
                status = "★ Undervalued" # 저평가
                top_picks.append((ticker, upside, div, a_per))
            elif upside <= -10:
                status = "Overvalued"  # 고평가
            else:
                status = "Fair Value"  # 적정가
            
            print(f"{ticker:<6} | ${price:<7.2f} | ${f_eps:<7.2f} | {a_per:<8.2f} | ${target:<8.2f} | {upside:>7.1f}% | {status:<12}")
            
        # --- STEP 2: 배당 매력도 (안전마진) ---
        print("\n" + "="*80)
        print("STEP 2: 배당 수익률 (안전마진)")
        print(" * 필수소비재는 채권 성격이 있어 배당 수익률이 중요함")
        print("="*80)
        print(f"{'Ticker':<6} | {'Div Yield':<10} | {'Evaluation':<15}")
        print("-" * 60)
        
        for ticker, res in self.results.items():
            div = res['Div Yield']
            if div >= 3.0:
                eval_str = "Good (3% 이상)"
            elif div >= 2.0:
                eval_str = "Normal"
            else:
                eval_str = "Low"
                
            print(f"{ticker:<6} | {div:>8.2f}% | {eval_str:<15}")

        # --- 종합 Top Pick ---
        print("\n" + "="*80)
        print("종합 Top Pick 후보 (저평가 + 배당)")
        print("="*80)
        
        if top_picks:
            # Upside 순으로 정렬
            top_picks.sort(key=lambda x: x[1], reverse=True)
            
            for item in top_picks:
                t, up, d, per = item
                print(f"★ {t}")
                print(f"   - 상승여력(Upside): {up:+.1f}% (목표가 도달 시)")
                print(f"   - 배당수익률: {d:.2f}%")
                print(f"   - 적용 PER: {per:.1f}배 (과거 5년 평균)")
                print("")
        else:
            print("현재 기준 '저평가(Undervalued)' 종목이 없습니다.")

if __name__ == "__main__":
    # 분석 대상: 대표 필수소비재 기업
    staples_tickers = ['KO', 'PG', 'PEP', 'WMT', 'CL', 'COST']
    
    engine = StaplesValuationEngine(staples_tickers)
    engine.fetch_data()
    engine.calculate_historical_metrics()
    engine.calculate_valuation()
    engine.generate_report()
