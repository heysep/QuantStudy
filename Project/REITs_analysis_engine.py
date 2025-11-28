import yfinance as yf
import pandas as pd
import numpy as np
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

class REITsAnalyzer:
    """
    시가총액 상위 리츠(REITs) 분석 엔진
    1) P/AFFO (Valuation)
    2) NAV Discount (Price Level)
    3) Debt Structure (Financial Health)
    """
    
    def __init__(self):
        # 시가총액 상위 주요 리츠 리스트 (섹터별 대표 종목)
        self.tickers = {
            'PLD': 'Industrial (물류)',
            'AMT': 'Telecom (통신타워)',
            'EQIX': 'Data Center (데이터센터)',
            'CCI': 'Telecom (통신타워)',
            'PSA': 'Self-Storage (창고)',
            'O': 'Retail (리테일)',
            'SPG': 'Retail (쇼핑몰)',
            'WELL': 'Healthcare (헬스케어)',
            'DLR': 'Data Center (데이터센터)',
            'VTR': 'Healthcare (헬스케어)'
        }
        self.data = []
        
    def fetch_data(self):
        """데이터 수집 및 기본 지표 계산"""
        print(f"\n{'='*70}")
        print(f"리츠(REITs) 분석 데이터 수집 중... (대상: {len(self.tickers)}개)")
        print(f" * P/AFFO 대용으로 P/OCF 사용 (무료 데이터 한계)")
        print(f" * NAV 대용으로 Book Value 사용 (P/B Ratio)")
        print(f"{'='*70}")
        
        results = []
        
        for ticker, sector in self.tickers.items():
            try:
                print(f"analyzing {ticker} ({sector})...")
                stock = yf.Ticker(ticker)
                info = stock.info
                
                if not info:
                    print(f"  -> {ticker}: 정보 없음, 건너뜀")
                    continue

                # 필수 데이터 추출
                price = info.get('currentPrice') or info.get('previousClose')
                market_cap = info.get('marketCap')
                
                # 1. P/AFFO 대용 지표: Price / Operating Cash Flow per Share
                operating_cashflow = info.get('operatingCashflow')
                shares_outstanding = info.get('sharesOutstanding')
                
                p_affo_proxy = None
                if price and operating_cashflow and shares_outstanding:
                    ocf_per_share = operating_cashflow / shares_outstanding
                    if ocf_per_share > 0:
                        p_affo_proxy = price / ocf_per_share

                # 2. NAV 대용 지표: P/B Ratio
                price_to_book = info.get('priceToBook')
                
                # 3. 부채 구조 (Debt Structure) - info 실패 시 재무제표 확인
                total_debt = info.get('totalDebt')
                total_assets = info.get('totalAssets')
                ebitda = info.get('ebitda')
                interest_expense = info.get('interestExpense')
                
                # 재무제표에서 보완
                try:
                    bs = stock.balance_sheet
                    fin = stock.financials
                    
                    if total_debt is None and not bs.empty:
                        if 'Total Debt' in bs.index:
                            total_debt = bs.loc['Total Debt'].iloc[0]
                    
                    if total_assets is None and not bs.empty:
                        if 'Total Assets' in bs.index:
                            total_assets = bs.loc['Total Assets'].iloc[0]
                            
                    if interest_expense is None and not fin.empty:
                        if 'Interest Expense' in fin.index:
                            interest_expense = fin.loc['Interest Expense'].iloc[0]
                            
                    if ebitda is None and not fin.empty:
                        # EBITDA 근사 계산: Operating Income + Depreciation (간단 버전)
                        if 'Operating Income' in fin.index:
                            op_income = fin.loc['Operating Income'].iloc[0]
                            ebitda = op_income # 감가상각 정보 없으면 영업이익으로 대체
                except Exception as e_fin:
                    # 재무제표 접근 실패 시 무시
                    pass

                # 부채 비율
                debt_ratio = None
                if total_debt and total_assets and total_assets > 0:
                    debt_ratio = (total_debt / total_assets) * 100
                    
                # 이자보상배율
                interest_coverage = None
                if ebitda and interest_expense and interest_expense != 0:
                    interest_coverage = ebitda / abs(interest_expense)

                # 배당 수익률 처리 (단위 보정)
                raw_div = info.get('dividendYield')
                if raw_div:
                    # 0.05 -> 5.0%, 5.0 -> 5.0%
                    if raw_div < 1:
                        dividend_yield = raw_div * 100
                    else:
                        dividend_yield = raw_div
                else:
                    dividend_yield = 0
                
                results.append({
                    'Ticker': ticker,
                    'Sector': sector,
                    'Price': price,
                    'P/OCF(Proxy)': p_affo_proxy, 
                    'P/B': price_to_book,         
                    'Debt_Ratio': debt_ratio,     
                    'Interest_Cov': interest_coverage, 
                    'Div_Yield': dividend_yield
                })
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
                continue
                
        self.data = pd.DataFrame(results)
        return self.data

    def analyze_metrics(self):
        """수집된 데이터를 바탕으로 분석 및 점수 산출"""
        if self.data.empty:
            print("데이터 수집 실패: 유효한 데이터가 없습니다.")
            return

        df = self.data
        
        # P/OCF 평균 계산 (이상치 제거 없이 단순 평균)
        valid_p_ocf = df['P/OCF(Proxy)'].dropna()
        avg_p_ocf = valid_p_ocf.mean() if not valid_p_ocf.empty else 0
        
        print(f"\n{'='*80}")
        print("STEP 1: 동일 섹터(전체 평균) 대비 밸류에이션 (P/AFFO Proxy)")
        print(f" * Proxy: Price / Operating Cash Flow (P/OCF)")
        print(f"{'='*80}")
        
        if avg_p_ocf > 0:
            print(f"분석 대상 리츠 평균 P/OCF: {avg_p_ocf:.2f}배")
        else:
            print("평균 P/OCF 계산 불가")
            
        print("-" * 80)
        print(f"{'티커':<6} | {'P/OCF':<10} | {'평가 (평균 대비)':<15} | {'섹터'}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            p_ocf = row['P/OCF(Proxy)']
            if pd.isna(p_ocf):
                status = "데이터 없음"
            elif avg_p_ocf > 0:
                if p_ocf < avg_p_ocf * 0.8:
                    status = "★ 매우 저평가"
                elif p_ocf < avg_p_ocf:
                    status = "저평가"
                else:
                    status = "고평가(프리미엄)"
            else:
                status = "-"
                
            p_ocf_str = f"{p_ocf:.2f}" if not pd.isna(p_ocf) else "-"
            print(f"{row['Ticker']:<6} | {p_ocf_str:<10} | {status:<15} | {row['Sector']}")

        print(f"\n{'='*80}")
        print("STEP 2: NAV 대비 할인율 체크 (P/B Ratio 활용)")
        print(" * P/B < 1.0: 자산가치 대비 할인 (안전마진 확보)")
        print(" * 금리 인하 국면에서 NAV 할인 리츠는 반등 가능성 높음")
        print(f"{'='*80}")
        print(f"{'티커':<6} | {'P/B Ratio':<10} | {'NAV 할인 여부':<20} | {'배당수익률'}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            pb = row['P/B']
            if pd.isna(pb):
                nav_status = "데이터 없음"
            elif pb < 0:
                nav_status = "자본잠식/음수 (주의)"
            elif pb < 0.9:
                nav_status = "★ 대폭 할인 (Buy)"
            elif pb < 1.05:
                nav_status = "적정 가치 (Fair)"
            else:
                nav_status = "프리미엄 거래 중"
                
            pb_str = f"{pb:.2f}" if not pd.isna(pb) else "-"
            div_str = f"{row['Div_Yield']:.2f}%"
            print(f"{row['Ticker']:<6} | {pb_str:<10} | {nav_status:<20} | {div_str}")

        print(f"\n{'='*80}")
        print("STEP 3: 부채 구조 확인 (리츠 밸류에이션의 80%는 '부채 구조'가 좌우)")
        print(" * 부채비율 < 50%: 매우 안정적")
        print(" * 이자보상배율 > 2.5: 금리 인상/변동에도 버틸 체력")
        print(f"{'='*80}")
        print(f"{'티커':<6} | {'부채비율(%)':<12} | {'이자보상배율':<12} | {'재무 건전성 평가'}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            debt_ratio = row['Debt_Ratio']
            int_cov = row['Interest_Cov']
            
            score = 0
            risk_msg = []
            
            # 부채비율 평가
            if pd.isna(debt_ratio):
                d_str = "-"
            else:
                d_str = f"{debt_ratio:.1f}%"
                if debt_ratio < 45: score += 2
                elif debt_ratio < 60: score += 1
                else: risk_msg.append("부채비중 높음")
            
            # 이자보상배율 평가
            if pd.isna(int_cov):
                i_str = "-"
            else:
                i_str = f"{int_cov:.2f}"
                if int_cov > 3.5: score += 2
                elif int_cov > 2.0: score += 1
                else: risk_msg.append("이자부담 큼")
            
            # 종합 평가
            if score >= 3:
                health = "★ 최우수 (매우 안전)"
            elif score >= 2:
                health = "양호 (안전)"
            elif score >= 1:
                health = "보통"
            else:
                health = "주의 (" + ", ".join(risk_msg) + ")"
                
            print(f"{row['Ticker']:<6} | {d_str:<12} | {i_str:<12} | {health}")
            
        # 종합 추천
        print(f"\n{'='*80}")
        print("종합 Top Pick 후보 선정 (저평가 + 재무안정 + 고배당)")
        print(f"{'='*80}")
        
        candidates = []
        for _, row in df.iterrows():
            # 조건: 
            # 1. P/OCF가 평균보다 낮고
            # 2. 부채비율이 60% 미만이며 (리츠 특성상 부채 허용치가 좀 높음)
            # 3. 배당수익률이 3% 이상인 경우
            
            is_undervalued = False
            if not pd.isna(row['P/OCF(Proxy)']) and avg_p_ocf > 0 and row['P/OCF(Proxy)'] < avg_p_ocf:
                is_undervalued = True
                
            is_safe = False
            if not pd.isna(row['Debt_Ratio']) and row['Debt_Ratio'] < 60:
                if not pd.isna(row['Interest_Cov']) and row['Interest_Cov'] > 2.0:
                    is_safe = True
            
            if is_undervalued and is_safe and row['Div_Yield'] > 3.0:
                candidates.append(row)
        
        if candidates:
            for cand in candidates:
                print(f"★ {cand['Ticker']} ({cand['Sector']})")
                print(f"   - P/OCF: {cand['P/OCF(Proxy)']:.2f} (평균대비 저평가)")
                print(f"   - P/B: {cand['P/B']:.2f} (NAV 대비 가치)")
                print(f"   - 배당수익률: {cand['Div_Yield']:.2f}%")
                print(f"   - 재무상태: 부채비율 {cand['Debt_Ratio']:.1f}%")
                print("")
        else:
            print("엄격한 기준(저평가+안전+고배당)을 모두 만족하는 종목이 없습니다.")
            print("Step 1~3의 개별 지표를 참고하여 투자 판단을 내리세요.")

if __name__ == "__main__":
    analyzer = REITsAnalyzer()
    try:
        analyzer.fetch_data()
        analyzer.analyze_metrics()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

