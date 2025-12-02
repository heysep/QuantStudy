import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

class MomentumEngine:
    def __init__(self, top_n=50):
        self.top_n = top_n
        self.tickers = []
        self.data = None
        self.features = pd.DataFrame()
        self.model = None
        self.scores = None

    def get_sp500_tickers(self):
        """S&P 500 티커 가져오기 (위키피디아)"""
        print("S&P 500 티커 가져오는 중...")
        try:
            import requests
            from io import StringIO
            
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            table = pd.read_html(StringIO(response.text), match='Symbol')
            df = table[0]
            self.tickers = df['Symbol'].tolist()
            # yfinance 호환을 위해 점(.)을 대시(-)로 변경 (예: BRK.B -> BRK-B)
            self.tickers = [t.replace('.', '-') for t in self.tickers]
            print(f"총 {len(self.tickers)}개 티커 발견.")
        except Exception as e:
            print(f"티커 가져오기 실패: {e}")
            # 위키피디아 실패 시 기본 티커 리스트 사용
            self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'V', 'JNJ'] 
            print(f"기본 리스트 {len(self.tickers)}개 사용.")

    def fetch_data(self, period="3y"):
        """전체 종목 데이터 다운로드 (yfinance)"""
        print(f"{len(self.tickers)}개 종목 데이터 다운로드 중 (기간: {period})...")
        try:
            self.data = yf.download(self.tickers, period=period, group_by='ticker', progress=True, threads=True)
            # MultiIndex 컬럼 처리 (필요 시)
            if isinstance(self.data.columns, pd.MultiIndex):
                pass
        except Exception as e:
            print(f"데이터 다운로드 오류: {e}")

    def calculate_features(self):
        """모멘텀 지표 계산 (핵심 로직)"""
        print("지표 계산 중...")
        
        feature_list = []
        
        # 데이터가 있는 티커만 필터링
        valid_tickers = [t for t in self.tickers if t in self.data.columns.levels[0]]
        
        for ticker in valid_tickers:
            try:
                df = self.data[ticker].copy()
                if df.empty:
                    continue
                
                # 종가(Close) 컬럼 확인
                if 'Close' not in df.columns:
                    continue

                # 결측치 채우기 (ffill)
                df = df.ffill()

                # 1. 수익률 (Returns)
                # 12개월(252일), 6개월(126일), 3개월(63일), 1개월(21일)
                df['Ret_12m'] = df['Close'].pct_change(252)
                df['Ret_6m'] = df['Close'].pct_change(126)
                df['Ret_3m'] = df['Close'].pct_change(63)
                df['Ret_1m'] = df['Close'].pct_change(21)
                
                # 2. 모멘텀 퀄리티 (12개월 - 1개월)
                # 최근 급등을 제외한 장기 추세의 건전성
                df['Mom_Quality'] = df['Ret_12m'] - df['Ret_1m']
                
                # 3. 변동성 (Volatility)
                # 126일(6개월) 기준 일간 수익률 표준편차 * 연율화(sqrt(252))
                df['Vol_126d'] = df['Close'].pct_change().rolling(126).std() * np.sqrt(252)
                
                # 4. RSTR (Trend Stability)
                # 모멘텀 / 변동성 (안정적인 상승인지 평가)
                df['RSTR'] = df['Ret_12m'] / df['Vol_126d']
                
                # 5. 목표 변수 (Target): 향후 1개월 수익률
                # 학습을 위해 미래 데이터를 현재 시점으로 당겨옴 (Shift -21)
                df['Target_Next_1m'] = df['Close'].shift(-21) / df['Close'] - 1
                
                # 티커 컬럼 추가
                df['Ticker'] = ticker
                
                # 필요한 컬럼만 선택
                cols = ['Ticker', 'Ret_12m', 'Ret_6m', 'Ret_3m', 'Ret_1m', 'Mom_Quality', 'Vol_126d', 'RSTR', 'Target_Next_1m', 'Close']
                df_features = df[cols]
                
                # [중요] 지표(Features)가 없는 행은 제거하되, 타겟(Target)이 없는 최신 데이터는 유지
                # 최신 데이터는 예측(Prediction)에 사용됨
                feature_cols = ['Ret_12m', 'Ret_6m', 'Ret_3m', 'Ret_1m', 'Mom_Quality', 'Vol_126d', 'RSTR']
                df_features = df_features.dropna(subset=feature_cols)
                
                feature_list.append(df_features)
                
            except Exception as e:
                continue
        
        if not feature_list:
            print("계산된 지표가 없습니다.")
            return

        self.features = pd.concat(feature_list)
        
        # 6. 상대 강도 (RS Rank)
        # 동일 시점 내에서 12개월 수익률 순위 (0~100점)
        print("RS Rank 계산 중...")
        self.features['RS_Rank'] = self.features.groupby(level=0)['Ret_12m'].rank(pct=True) * 100
        
        # RS Rank 계산 불가한 행 제거
        self.features = self.features.dropna(subset=['RS_Rank'])
        
        print(f"총 데이터 포인트: {len(self.features)}")

    def train_model(self):
        """XGBoost 모델 학습 (다음 달 수익률 예측)"""
        print("XGBoost 모델 학습 중...")
        
        # 학습용 데이터 준비
        # 타겟(미래 수익률)이 존재하는 과거 데이터만 사용
        train_data = self.features.dropna(subset=['Target_Next_1m'])
        
        X = train_data[['Ret_12m', 'Ret_6m', 'Ret_3m', 'Mom_Quality', 'RSTR', 'RS_Rank']]
        y = train_data['Target_Next_1m']
        
        # 학습/테스트 데이터 분리 (시계열 고려)
        # 과거 80% 학습, 최근 20% 테스트
        dates = train_data.index.unique().sort_values()
        split_date = dates[int(len(dates) * 0.8)]
        
        X_train = X[X.index < split_date]
        y_train = y[X.index < split_date]
        X_test = X[X.index >= split_date]
        y_test = y[X.index >= split_date]
        
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # 모델 평가
        score = self.model.score(X_test, y_test)
        print(f"테스트 세트 R^2 점수: {score:.4f}")
        
        # 특성 중요도 (Feature Importance) 출력
        print("특성 중요도:")
        for name, imp in zip(X.columns, self.model.feature_importances_):
            print(f"  - {name}: {imp:.4f}")

    def evaluate_current(self):
        """현재 시점 평가 및 Top 50 선정"""
        print("현재 시장 평가 중...")
        
        # 각 티커별 가장 최신 데이터 가져오기
        latest_data = self.features.groupby('Ticker').last().reset_index()
        
        # 예측용 입력 데이터
        X_latest = latest_data[['Ret_12m', 'Ret_6m', 'Ret_3m', 'Mom_Quality', 'RSTR', 'RS_Rank']]
        
        # XGBoost 점수 예측 (기대 수익률)
        latest_data['XGB_Score'] = self.model.predict(X_latest)
        
        # 기관 스타일 점수 (참고용)
        # 공식: 0.6*(12m-1m) + 0.2*(6m) + 0.1*(3m) + 0.1*(RSTR)
        latest_data['Inst_Score'] = (
            0.6 * latest_data['Mom_Quality'] +
            0.2 * latest_data['Ret_6m'] +
            0.1 * latest_data['Ret_3m'] +
            0.1 * latest_data['RSTR']
        )
        
        # 모멘텀 상태 판단 (Strong/Weak)
        # 기준: RS Rank > 60, 모든 모멘텀 지표 양수
        latest_data['Momentum_Status'] = 'Weak'
        cond_strong = (
            (latest_data['RS_Rank'] > 60) & 
            (latest_data['Mom_Quality'] > 0) & 
            (latest_data['RSTR'] > 0) & 
            (latest_data['Ret_3m'] > 0)
        )
        latest_data.loc[cond_strong, 'Momentum_Status'] = 'Strong'
        
        # XGBoost 점수 기준 정렬
        self.scores = latest_data.sort_values(by='XGB_Score', ascending=False)
        
        # 상위 N개 선정
        top_picks = self.scores.head(self.top_n)
        
        print(f"\n[XGBoost 모델 기반 Top {self.top_n} 추천]")
        print(f"{'Ticker':<6} | {'XGB Score':<9} | {'Inst Score':<10} | {'RS Rank':<7} | {'Status':<8} | {'Price':<8}")
        print("-" * 70)
        
        for _, row in top_picks.iterrows():
            print(f"{row['Ticker']:<6} | {row['XGB_Score']:>9.4f} | {row['Inst_Score']:>10.4f} | {row['RS_Rank']:>7.1f} | {row['Momentum_Status']:<8} | ${row['Close']:<7.2f}")
            
        # 결과 저장
        self.scores.to_csv('momentum_scores.csv', index=False)
        print("\n전체 결과가 'momentum_scores.csv'에 저장되었습니다.")

if __name__ == "__main__":
    engine = MomentumEngine(top_n=50)
    engine.get_sp500_tickers()
    engine.fetch_data()
    engine.calculate_features()
    engine.train_model()
    engine.evaluate_current()
