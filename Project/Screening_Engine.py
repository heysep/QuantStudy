import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta
import time

class QuantScreeningEngine:
    def __init__(self):
        self.tickers = []
        self.data = None
        self.info_data = {} # To store fundamental data
        self.screened_tickers = []
        self.final_scores = pd.DataFrame()

    def get_universe(self):
        """1. 유니버스 정의: S&P 500"""
        print("=== 1. 유니버스 설정 (S&P 500) ===")
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            df = pd.read_html(StringIO(response.text))[0]
            self.tickers = df['Symbol'].tolist()
            self.tickers = [t.replace('.', '-') for t in self.tickers]
            
            # 섹터 정보도 저장
            self.sector_map = df.set_index('Symbol')['GICS Sector'].to_dict()
            # 키 포맷 통일 (BRK.B -> BRK-B)
            self.sector_map = {k.replace('.', '-'): v for k, v in self.sector_map.items()}
            
            print(f"-> S&P 500 종목 수: {len(self.tickers)}개 로드 완료")
        except Exception as e:
            print(f"Error fetching S&P 500: {e}")
            # Fallback for testing
            self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'JPM', 'V']
            self.sector_map = {t: 'Technology' for t in self.tickers} # Dummy sectors

    def fetch_data(self, period="2y"):
        """데이터 수집 (가격 + 기본적 분석 데이터)"""
        print(f"\n=== 데이터 수집 중 (대상: {len(self.tickers)}개, 기간: {period}) ===")
        
        # 1. 가격 데이터 (Batch download)
        try:
            self.data = yf.download(self.tickers, period=period, group_by='ticker', progress=True, threads=True)
            # MultiIndex 처리
            if isinstance(self.data.columns, pd.MultiIndex):
                pass 
        except Exception as e:
            print(f"가격 데이터 다운로드 실패: {e}")
            return

        # 2. 펀더멘털 데이터 (개별 호출 필요 - 느림, 필요한 것만 가져오기)
        # 실제로는 API 제한 때문에 500개를 다 가져오기 힘들 수 있음.
        # 여기서는 '가격 기반' 필터링을 먼저 하고, 살아남은 종목에 대해서만 펀더멘털을 조회하는 전략 사용
        print("-> 펀더멘털 데이터는 1차 가격 필터링 후 조회합니다.")

    def apply_filters(self):
        """2. 위험 요소 필터링 & 3. 전략 필터"""
        print("\n=== 스크리닝 시작 ===")
        import traceback
        
        passed_tickers = []
        
        # 유효한 티커만 순회
        if isinstance(self.data.columns, pd.MultiIndex):
            valid_tickers = [t for t in self.tickers if t in self.data.columns.levels[0]]
        else:
            valid_tickers = self.tickers if len(self.tickers) == 1 else []
            if not valid_tickers and not self.data.empty:
                 pass

        total = len(valid_tickers)
        print(f"-> 분석 대상: {total}개 종목")

        for i, ticker in enumerate(valid_tickers):
            try:
                df = self.data[ticker].copy()
                
                if df.empty or len(df) < 250:
                    continue
                
                # 중복 컬럼 제거
                df = df.loc[:, ~df.columns.duplicated()]
                df = df.ffill()
                
                # Helper to get scalar from potential DataFrame/Series
                def get_scalar(series_or_df):
                    if isinstance(series_or_df, pd.DataFrame):
                        series_or_df = series_or_df.iloc[:, 0]
                    return float(series_or_df.iloc[-1])

                # 값 추출
                current_price = get_scalar(df['Close'])
                
                # --- [2] 위험 요소 필터링 ---
                
                # (1) 유동성 필터
                avg_vol_20 = df['Volume'].rolling(20).mean()
                if isinstance(avg_vol_20, pd.DataFrame): avg_vol_20 = avg_vol_20.iloc[:, 0]
                avg_vol_20 = float(avg_vol_20.iloc[-1])
                
                avg_amt_20 = avg_vol_20 * current_price
                if avg_amt_20 < 20_000_000:
                    continue

                # (2) 변동성 필터
                pct_change = df['Close'].pct_change()
                if isinstance(pct_change, pd.DataFrame): pct_change = pct_change.iloc[:, 0]
                
                vol_1y = float(pct_change.std() * np.sqrt(252))
                if vol_1y > 0.6: 
                    continue
                    
                # (3) 동전주 제외
                if current_price < 5:
                    continue

                # --- [3] 전략 맞춤형 필터 ---
                
                # (1) 장기 추세
                ma200_series = df['Close'].rolling(200).mean()
                if isinstance(ma200_series, pd.DataFrame): ma200_series = ma200_series.iloc[:, 0]
                ma200 = float(ma200_series.iloc[-1])
                
                if current_price < ma200:
                    continue
                    
                # (2) 조정률 필터
                high_3m_series = df['Close'].rolling(63).max()
                if isinstance(high_3m_series, pd.DataFrame): high_3m_series = high_3m_series.iloc[:, 0]
                high_3m = float(high_3m_series.iloc[-1])
                
                if high_3m == 0: continue
                
                dd = (current_price / high_3m) - 1
                
                if not (-0.30 <= dd <= -0.00): 
                    continue

                passed_tickers.append(ticker)
                
            except Exception as e:
                # print(f"Error processing {ticker}: {e}")
                # traceback.print_exc()
                continue
        
        print(f"-> 1차(가격/거래량/추세) 필터 통과: {len(passed_tickers)} / {total} 개")
        self.screened_tickers = passed_tickers

    def get_fundamental_data_and_score(self):
        """펀더멘털 데이터 조회 및 최종 점수화"""
        print("\n=== 펀더멘털 조회 및 최종 점수 산출 ===")
        
        results = []
        
        # Helper to get scalar
        def get_scalar(series_or_df):
            if isinstance(series_or_df, pd.DataFrame):
                series_or_df = series_or_df.iloc[:, 0]
            return float(series_or_df.iloc[-1])

        # SPY 데이터 가져오기 (벤치마크)
        try:
            spy = yf.download('SPY', period='1y', progress=False)
            if isinstance(spy.columns, pd.MultiIndex):
                spy = spy.xs('SPY', axis=1, level=0) if 'SPY' in spy.columns.levels[0] else spy
            
            spy_close = spy['Close']
            if isinstance(spy_close, pd.DataFrame): spy_close = spy_close.iloc[:, 0]
            
            spy_ret_6m = float(spy_close.pct_change(126).iloc[-1])
        except:
            spy_ret_6m = 0.05 # Fallback
        
        for ticker in self.screened_tickers:
            try:
                df = self.data[ticker].copy()
                # 중복 컬럼 제거
                df = df.loc[:, ~df.columns.duplicated()]
                df = df.ffill()
                
                # Close Series 추출
                close_s = df['Close']
                if isinstance(close_s, pd.DataFrame): close_s = close_s.iloc[:, 0]
                
                # Volume Series 추출
                vol_s = df['Volume']
                if isinstance(vol_s, pd.DataFrame): vol_s = vol_s.iloc[:, 0]

                # --- 점수 요소 계산 ---
                
                # 1. 모멘텀 (12개월, 6개월)
                ret_12m = float(close_s.pct_change(252).iloc[-1])
                ret_6m = float(close_s.pct_change(126).iloc[-1])
                ret_3m = float(close_s.pct_change(63).iloc[-1])
                
                # 2. 변동성 (낮을수록 좋음)
                vol = float(close_s.pct_change().std() * np.sqrt(252))
                if vol == 0: vol = 0.001
                
                # 3. 상대강도 (vs SPY)
                rs_spy = ret_6m - spy_ret_6m
                
                # 4. 거래량 강도 (최근 5일 vs 20일)
                vol_5 = float(vol_s.rolling(5).mean().iloc[-1])
                vol_20 = float(vol_s.rolling(20).mean().iloc[-1])
                vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0
                
                # 5. 조정폭 (눌림목 점수)
                high_3m = float(close_s.rolling(63).max().iloc[-1])
                cur_price = float(close_s.iloc[-1])
                dd = (cur_price / high_3m) - 1 
                
                # 종합 점수 (가중치 적용)
                score = (
                    (ret_12m * 0.25) + 
                    (ret_6m * 0.15) + 
                    (rs_spy * 0.30) + 
                    (vol_ratio * 0.1) + 
                    ((1/vol) * 0.05)
                )
                
                results.append({
                    'Ticker': ticker,
                    'Sector': self.sector_map.get(ticker, 'Unknown'),
                    'Score': score,
                    'Ret_12m': ret_12m,
                    'Ret_6m': ret_6m,
                    'RS_SPY': rs_spy,
                    'Vol_Ratio': vol_ratio,
                    'Drawdown': dd,
                    'Price': cur_price
                })
                
            except Exception as e:
                continue
        
        # DataFrame 변환
        res_df = pd.DataFrame(results)
        if res_df.empty:
            print("추출된 종목이 없습니다.")
            return pd.DataFrame()
            
        # 랭킹 산정
        res_df['Rank'] = res_df['Score'].rank(ascending=False)
        res_df = res_df.sort_values('Rank')
        
        self.final_scores = res_df
        print(f"\n-> 최종 분석 완료: {len(res_df)}개 종목")
        return res_df

    def run(self):
        self.get_universe()
        self.fetch_data()
        self.apply_filters()
        df = self.get_fundamental_data_and_score()
        
        # CSV 저장
        if not df.empty:
            df.to_csv('screening_results.csv', index=False)
            print("결과 저장 완료: screening_results.csv")
            
            # 상위 10개 출력
            print("\n[Top 10 Screened Stocks]")
            print(df.head(10)[['Ticker', 'Sector', 'Score', 'Ret_6m', 'Drawdown', 'Vol_Ratio']])
            
            return df['Ticker'].head(30).tolist() # 상위 30개 리턴
        else:
            return []

if __name__ == "__main__":
    engine = QuantScreeningEngine()
    engine.run()
