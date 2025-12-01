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
        """데이터 수집: 주가, Analyst Estimates, 재무지표"""
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
                
                # 2. 재무 데이터 (분기별 EPS)
                q_income = stock.quarterly_income_stmt
                
                # 3. Analyst Estimates (Forward EPS) - 핵심 수정 사항
                # yfinance의 'earnings_estimate' 테이블 사용 (더 정확함)
                forward_eps = None
                eps_source = "N/A"
                
                try:
                    estimates = stock.earnings_estimate
                    if estimates is not None and not estimates.empty:
                        # '+1y' (다음 연도) 또는 '0y' (올해) 추정치 사용
                        # 보수적인 평가를 위해 '+1y' (Next Year) 사용 권장
                        if '+1y' in estimates.index:
                            forward_eps = estimates.loc['+1y', 'avg']
                            eps_source = "Analyst Est (+1y)"
                        elif '0y' in estimates.index:
                            forward_eps = estimates.loc['0y', 'avg']
                            eps_source = "Analyst Est (0y)"
                except Exception as e:
                    print(f"    Estimates 가져오기 실패: {e}")
                
                # Fallback: Estimates가 없으면 info 사용
                if forward_eps is None:
                    forward_eps = stock.info.get('forwardEps')
                    eps_source = "Yahoo Info (Fallback)"
                
                # 현재 주가
                current_price = hist['Close'].iloc[-1]
                
                # 데이터 저장 구조
                self.data[ticker] = {
                    'history': hist,
                    'financials': {
                        'q_income': q_income
                    },
                    'forward_eps': forward_eps,
                    'eps_source': eps_source,
                    'current_price': current_price
                }
                
            except Exception as e:
                print(f"    오류 발생 ({ticker}): {e}")

    def calculate_historical_metrics(self):
        """과거 밸류에이션 지표 계산 (Robust Logic 적용)"""
        print("\n지표 계산 중 (Robust Logic 적용)...")
        
        for ticker, data in self.data.items():
            hist = data['history']
            q_income = data['financials']['q_income']
            
            # --- 1. Historical PER 계산 (Median & Outlier Filtering) ---
            
            if q_income is not None and not q_income.empty and 'Diluted EPS' in q_income.index:
                eps_q = q_income.loc['Diluted EPS']
                eps_q = eps_q.sort_index()
                
                # TTM EPS = 최근 4분기 합
                eps_ttm = eps_q.rolling(window=4).sum()
                eps_ttm.index = pd.to_datetime(eps_ttm.index).tz_localize(None)
                
                # 일별 데이터로 확장 (ffill)
                eps_ttm_daily = eps_ttm.reindex(hist.index, method='ffill')
                
                # PER = Price / TTM EPS
                per_series = hist['Close'] / eps_ttm_daily
                
                # [핵심 수정] 이상치 제거 (Outlier Filtering)
                # Staples 섹터의 정상 PER 범위: 5배 ~ 60배 (일시적 이익 급감 제외)
                per_series = per_series[(per_series > 5) & (per_series < 60)]
                
                data['metrics'] = {
                    'per_series': per_series,
                    'eps_ttm_daily': eps_ttm_daily
                }
            else:
                print(f"    경고: {ticker} EPS 데이터 부족으로 Historical PER 계산 불가")
                data['metrics'] = {'per_series': pd.Series()}

            # --- 2. 배당 수익률 (Dividend Yield) - 정확한 TTM 계산 ---
            # [핵심 수정] 최근 4회 지급분 합산 (단순 365일 rolling 아님)
            dividends = hist['Dividends']
            # 배당이 있는 날만 필터링
            actual_dividends = dividends[dividends > 0]
            
            if len(actual_dividends) >= 4:
                div_ttm_val = actual_dividends.tail(4).sum()
            else:
                div_ttm_val = actual_dividends.sum() # 4회 미만이면 있는대로 합산
                
            div_yield = (div_ttm_val / data['current_price']) * 100
            
            data['metrics']['div_yield_val'] = div_yield

    def calculate_valuation(self):
        """적정주가 산출 (Median PER 회귀 모형)"""
        print("\n밸류에이션 수행 중...")
        
        for ticker, data in self.data.items():
            metrics = data.get('metrics')
            if not metrics or metrics['per_series'].empty:
                continue
            
            per_series = metrics['per_series']
            forward_eps = data['forward_eps']
            current_price = data['current_price']
            
            # [핵심 수정] 평균(Mean) 대신 중위값(Median) 사용
            # 이상치(Outlier)에 강건한 지표 사용
            median_per_5y = per_series.tail(252*5).median()
            median_per_10y = per_series.median()
            
            # [보수적 접근] 5년과 10년 중 더 낮은 PER 적용 (거품 제거)
            applied_per = min(median_per_5y, median_per_10y)
            per_period = "5y" if applied_per == median_per_5y else "10y"
            
            # 현재 PER (최근 30일 Median - 노이즈 제거)
            current_per = per_series.tail(30).median()
            
            # 적정주가 계산 (Target Price)
            if forward_eps:
                target_price = forward_eps * applied_per
                upside = (target_price / current_price - 1) * 100
            else:
                target_price = 0
                upside = 0
            
            # 결과 저장
            self.results[ticker] = {
                'Current Price': current_price,
                'Forward EPS': forward_eps,
                'EPS Source': data['eps_source'],
                'Current PER': current_per,
                'Median PER (5y)': median_per_5y,
                'Median PER (10y)': median_per_10y,
                'Applied PER': applied_per,
                'PER Period': per_period,
                'Target Price': target_price,
                'Upside': upside,
                'Div Yield': data['metrics']['div_yield_val']
            }

    def generate_report(self):
        """보고서 출력 (텍스트 중심, 리츠 분석 스타일)"""
        print("\n" + "="*80)
        print("필수소비재(Consumer Staples) 밸류에이션 리포트 (Research Level)")
        print(" * 핵심: Median PER(중위값)와 Analyst Estimates를 사용한 정밀 분석")
        print(" * 보수적 적용: 5년 vs 10년 Median 중 '더 낮은 값'을 적용하여 거품 제거")
        print("="*80)
        
        # --- STEP 1: 밸류에이션 (Median PER 회귀) ---
        print("\n" + "="*80)
        print("STEP 1: Median PER 대비 밸류에이션 (Upside 확인)")
        print(" * Upside > 10%: 저평가 (매수 기회)")
        print(" * Upside < -10%: 고평가 (조정 가능성)")
        print("="*80)
        
        print(f"{'Ticker':<6} | {'Price':<8} | {'Fwd EPS':<8} | {'Apply PER':<9} | {'Target':<9} | {'Upside':<8} | {'Status':<12}")
        print("-" * 95)
        
        # Top Pick 선정을 위한 리스트
        top_picks = []
        
        for ticker, res in self.results.items():
            price = res['Current Price']
            f_eps = res['Forward EPS'] if res['Forward EPS'] else 0
            a_per = res['Applied PER'] if not pd.isna(res['Applied PER']) else 0
            p_period = res['PER Period']
            target = res['Target Price']
            upside = res['Upside']
            div = res['Div Yield']
            
            # 상태 판단
            if upside >= 10:
                status = "★ Undervalued" # 저평가
                top_picks.append((ticker, upside, div, a_per, p_period))
            elif upside <= -10:
                status = "Overvalued"  # 고평가
            else:
                status = "Fair Value"  # 적정가
            
            print(f"{ticker:<6} | ${price:<7.2f} | ${f_eps:<7.2f} | {a_per:<5.1f}({p_period}) | ${target:<8.2f} | {upside:>7.1f}% | {status:<12}")
            
        # --- STEP 2: 배당 매력도 (안전마진) ---
        print("\n" + "="*80)
        print("STEP 2: 배당 수익률 (안전마진)")
        print(" * 최근 4회 실제 지급분 합산 기준 (정확도 향상)")
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
                t, up, d, per, p_period = item
                print(f"★ {t}")
                print(f"   - 상승여력(Upside): {up:+.1f}% (목표가 도달 시)")
                print(f"   - 배당수익률: {d:.2f}%")
                print(f"   - 적용 PER: {per:.1f}배 ({p_period} Median)")
                print("")
        else:
            print("현재 기준 '저평가(Undervalued)' 종목이 없습니다.")

        # --- 데이터 소스 정보 ---
        print("\n[참고: EPS 데이터 소스]")
        for ticker, res in self.results.items():
            print(f" - {ticker}: {res['EPS Source']}")

if __name__ == "__main__":
    # 분석 대상: 대표 필수소비재 기업
    staples_tickers = ['KO', 'PG', 'PEP', 'WMT', 'CL', 'COST']
    
    engine = StaplesValuationEngine(staples_tickers)
    engine.fetch_data()
    engine.calculate_historical_metrics()
    engine.calculate_valuation()
    engine.generate_report()
