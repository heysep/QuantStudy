import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import zscore

# XGBoost 라이브러리 임포트 (설치되어 있지 않은 경우 예외 처리)
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("Warning: xgboost가 설치되지 않았습니다. 더미 예측을 사용합니다.")

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 1. Data & Feature Layer: MomentumEngine (XGBoost)
# -----------------------------------------------------------------------------
class MomentumEngine:
    """
    Data Layer & Feature Layer & Model Layer
    - 기술적 지표(모멘텀, 변동성)를 계산하고
    - XGBoost 모델을 통해 초과수익(Alpha)을 예측하는 엔진
    """
    def __init__(self, xgb_model=None):
        """
        Args:
            xgb_model: 사전에 학습된 XGBoost 모델 객체 (또는 sklearn wrapper)
        """
        self.model = xgb_model

    def build_features(self, prices_df: pd.DataFrame, benchmark_series: pd.Series = None) -> pd.DataFrame:
        """
        가격 데이터를 받아 ML 모델 및 전략에 사용할 피처를 생성합니다.
        
        Args:
            prices_df: 종목별 수정주가 (Columns: Tickers, Index: Date)
            benchmark_series: 벤치마크 지수 (Optional)
            
        Returns:
            Feature DataFrame (MultiIndex 또는 Wide Format, 여기서는 최신 시점 기준 DataFrame 반환)
        """
        # 0 또는 음수 데이터 처리 및 로그 변환
        clean_prices = prices_df.copy()
        clean_prices[clean_prices <= 0] = np.nan
        log_prices = np.log(clean_prices.ffill())
        
        # 기준 시점 (마지막 날짜)
        current_idx = prices_df.index[-1]
        
        # -----------------------------------------------------
        # (1) 절대 모멘텀 팩터 계산
        # -----------------------------------------------------
        # 12M-1M: 최근 1개월을 제외한 지난 11개월 수익률 (Reversal 방지)
        # 252일 거래일 기준
        mom_12m = log_prices.diff(252)
        mom_1m = log_prices.diff(21)
        mom_12m_1m = mom_12m - mom_1m # 로그 수익률이므로 차감 = 수익률 나누기 효과
        
        # 6M, 3M 모멘텀
        mom_6m = log_prices.diff(126)
        mom_3m = log_prices.diff(63)
        
        # -----------------------------------------------------
        # (2) 변동성 및 퀄리티 팩터
        # -----------------------------------------------------
        # 20일, 60일 변동성 (연율화)
        daily_ret = prices_df.pct_change()
        vol_20d = daily_ret.rolling(20).std() * np.sqrt(252)
        vol_60d = daily_ret.rolling(60).std() * np.sqrt(252)
        vol_12m = daily_ret.rolling(252).std() * np.sqrt(252)
        
        # Momentum Quality (RSTR): 수익률 / 변동성
        # 여기서는 12M 수익률 대비 12M 변동성 비율 사용
        mom_quality = mom_12m / vol_12m.replace(0, np.nan)
        
        # -----------------------------------------------------
        # 데이터 취합 (현재 시점)
        # -----------------------------------------------------
        # 실제 ML 파이프라인에서는 전체 기간 데이터를 stack해서 사용하지만,
        # 여기서는 리밸런싱 시점(t)의 횡단면 데이터(Cross-section)를 생성합니다.
        feat_df = pd.DataFrame({
            'mom_12m_1m': mom_12m_1m.loc[current_idx],
            'mom_6m': mom_6m.loc[current_idx],
            'mom_3m': mom_3m.loc[current_idx],
            'vol_20d': vol_20d.loc[current_idx],
            'vol_60d': vol_60d.loc[current_idx],
            'mom_quality': mom_quality.loc[current_idx]
        })
        
        # (3) 상대 모멘텀 (Cross-Sectional Rank)
        # 섹터 내 순위가 중요하나, 여기서는 전체 유니버스 내 순위(0.0 ~ 1.0) 사용
        feat_df['rs_rank'] = feat_df['mom_12m_1m'].rank(pct=True)
        
        # 결측치 처리 (상장 기간 부족 등) -> 0 또는 평균으로 대체
        feat_df.fillna(0, inplace=True)
        
        return feat_df

    def predict_score(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        Feature DataFrame을 입력받아 XGBoost 모델로 기대 초과수익(Score)을 예측합니다.
        """
        # 모델이 없으면 룰 기반 스코어로 대체 (Fallback)
        if self.model is None:
            # 간단한 모멘텀 + 퀄리티 결합 점수
            score = (
                0.4 * feature_df['mom_12m_1m'] + 
                0.3 * feature_df['mom_6m'] + 
                0.3 * feature_df['mom_quality']
            )
            return score.rename("ml_score")
        
        try:
            # feature_df의 컬럼 순서가 학습시와 동일해야 함을 주의
            if isinstance(self.model, xgb.Booster):
                dtest = xgb.DMatrix(feature_df)
                preds = self.model.predict(dtest)
            else:
                # sklearn API (XGBRegressor 등)
                preds = self.model.predict(feature_df)
                
            return pd.Series(preds, index=feature_df.index, name='ml_score')
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            return pd.Series(0, index=feature_df.index, name='ml_score')


# -----------------------------------------------------------------------------
# 2. Trend & Vol Layer: TrendEngine
# -----------------------------------------------------------------------------
class TrendEngine:
    """
    - Trend Score: 장단기 EMA를 이용한 추세 점수 산출
    - Volatility: Realized + EWMA Vol 및 공분산 행렬 추정
    """
    def build_trend_features(self, prices_df: pd.DataFrame) -> pd.Series:
        """
        Dual-Horizon Trend Score (Short + Long)
        """
        # EMA 계산
        ema_20 = prices_df.ewm(span=20, adjust=False).mean()
        ema_120 = prices_df.ewm(span=120, adjust=False).mean()
        
        last_price = prices_df.iloc[-1]
        last_ema_20 = ema_20.iloc[-1]
        last_ema_120 = ema_120.iloc[-1]
        
        # 추세 강도 (Distance from MA)
        # (Price - MA) / MA
        trend_short = (last_price - last_ema_20) / last_ema_20
        trend_long = (last_price - last_ema_120) / last_ema_120
        
        # Z-score 정규화 (횡단면)
        def safe_zscore(s):
            if s.std() == 0: return s * 0
            return (s - s.mean()) / s.std()

        z_short = safe_zscore(trend_short.fillna(0))
        z_long = safe_zscore(trend_long.fillna(0))
        
        # 최종 Trend Score 합성
        trend_score_raw = 0.5 * z_short + 0.5 * z_long
        
        # Cap 적용 (이상치 제어, -2.0 ~ 2.0)
        trend_score = trend_score_raw.clip(-2.0, 2.0)
        
        return trend_score.rename("trend_score")

    def estimate_vol(self, prices_df: pd.DataFrame, span_vol=60):
        """
        변동성 및 공분산 행렬 추정
        
        Returns:
            sigma_series: 종목별 변동성 (연율화)
            cov_matrix: 종목간 공분산 행렬 (연율화)
        """
        returns = prices_df.pct_change().fillna(0)
        
        # 1. Realized Vol (20일 단순 이동 표준편차)
        realized_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # 2. EWMA Vol (지수 가중 이동 표준편차, span=60)
        ewma_vol = returns.ewm(span=span_vol).std().iloc[-1] * np.sqrt(252)
        
        # 최종 변동성: 둘의 평균 사용 (반응성 + 안정성)
        final_sigma = 0.5 * realized_vol + 0.5 * ewma_vol
        
        # 공분산 행렬 (EWMA Covariance)
        # 가장 최근 시점의 공분산 행렬 추출
        cov_matrix = returns.ewm(span=span_vol).cov().iloc[-len(prices_df.columns):]
        # MultiIndex(Date, Ticker) -> Ticker Index로 정리
        cov_matrix = cov_matrix.droplevel(0)
        
        # 연율화 (Daily Cov -> Annual Cov)
        cov_matrix_annual = cov_matrix * 252
        
        return final_sigma, cov_matrix_annual


# -----------------------------------------------------------------------------
# 3. Portfolio Layer: PortfolioEngine
# -----------------------------------------------------------------------------
class PortfolioEngine:
    """
    Signal Layer + Portfolio Construction
    - Signal 합성 (ML + Mom + Trend)
    - 자산 선택 (Selection)
    - 비중 산출 (Risk Parity / Vol Targeting)
    """
    def __init__(self, target_vol=0.10, max_leverage=2.0, top_k=5):
        self.target_vol = target_vol     # 목표 변동성 (예: 연 10%)
        self.max_leverage = max_leverage # 최대 레버리지 제한 (예: 2배)
        self.top_k = top_k               # 포트폴리오 편입 종목 수

    def build_signal(self, ml_score, rule_mom, trend_score, weights=None):
        """
        여러 소스의 스코어를 결합하여 최종 시그널 생성
        """
        if weights is None:
            # 기본 가중치: ML 50%, 룰기반 20%, 추세 30%
            weights = {"ml": 0.5, "mom": 0.2, "trend": 0.3}
            
        def safe_zscore(s):
            if s.std() == 0 or s.isna().all(): return s.fillna(0) * 0
            return (s - s.mean()) / s.std()

        # 각 스코어 정규화 (Z-score)
        z_ml = safe_zscore(ml_score.fillna(0))
        z_mom = safe_zscore(rule_mom.fillna(0))
        z_trend = safe_zscore(trend_score.fillna(0))
        
        # 가중 합산
        final_signal = (
            weights["ml"] * z_ml + 
            weights["mom"] * z_mom + 
            weights["trend"] * z_trend
        )
        
        # Long-Only 전략: 음수 시그널은 0으로 처리
        final_signal = final_signal.clip(lower=0)
        
        return final_signal

    def make_weights(self, signal_series, sigma_series, cov_matrix):
        """
        최종 포트폴리오 비중 산출 프로세스
        Selection -> Signal-based Weighting -> Volatility Targeting
        """
        # 1. 자산 선택 (Top K)
        # 시그널이 양수인 종목 중에서 상위 K개
        valid_signal = signal_series[signal_series > 0]
        selected_tickers = valid_signal.nlargest(self.top_k).index
        
        if len(selected_tickers) == 0:
            return pd.Series(0.0, index=signal_series.index)
            
        # 선택된 자산들의 데이터 추출
        sel_signal = valid_signal.loc[selected_tickers]
        sel_sigma = sigma_series.loc[selected_tickers]
        sel_cov = cov_matrix.loc[selected_tickers, selected_tickers]
        
        # 2. Risk-Adjusted Weighting (Risk Parity style)
        # Weight ~ Signal^p / Volatility
        # 강한 시그널에 더 비중을 두되, 변동성이 크면 패널티
        p = 1.0 # Signal Power
        raw_weights = (sel_signal ** p) / sel_sigma.replace(0, np.inf)
        
        # 1차 정규화 (합계 = 1.0)
        if raw_weights.sum() == 0:
            return pd.Series(0.0, index=signal_series.index)
        w_pre = raw_weights / raw_weights.sum()
        
        # 3. Volatility Targeting (레버리지 조절)
        # 포트폴리오 예상 변동성 계산: sqrt(w^T * Cov * w)
        w_vec = w_pre.values
        cov_mat = sel_cov.values
        
        port_var = w_vec.T @ cov_mat @ w_vec
        port_vol = np.sqrt(port_var) # 연간 예상 변동성
        
        # 목표 변동성 대비 레버리지 비율 산출
        # 예: 목표 10%인데 현재 포트폴리오가 5%라면 2배 레버리지 사용
        if port_vol == 0:
            leverage = 0
        else:
            leverage = self.target_vol / port_vol
            
        # 최대 레버리지 제한 적용
        final_leverage = min(leverage, self.max_leverage)
        
        # 최종 비중 계산
        w_final = w_pre * final_leverage
        
        # 전체 유니버스 인덱스에 매핑 (선택 안된 종목은 0)
        full_weights = pd.Series(0.0, index=signal_series.index)
        full_weights.loc[selected_tickers] = w_final
        
        # 디버깅 정보 출력 (선택적으로 사용)
        # print(f"  [Allocation] TargetVol: {self.target_vol:.1%}, Est.Vol: {port_vol:.1%}, Leverage: {final_leverage:.2f}x")
        
        return full_weights


# -----------------------------------------------------------------------------
# 4. Master Strategy: Orchestrator
# -----------------------------------------------------------------------------
class MasterStrategy:
    """
    전체 레이어를 조립하여 최종 전략을 실행하는 클래스
    """
    def __init__(self, xgb_model=None, target_vol=0.10, top_k=5):
        self.mom_engine = MomentumEngine(xgb_model)
        self.trend_engine = TrendEngine()
        self.port_engine = PortfolioEngine(target_vol=target_vol, max_leverage=2.0, top_k=top_k)
        
    def rebalance(self, prices_df: pd.DataFrame, date=None):
        """
        특정 시점의 가격 데이터를 받아 리밸런싱 비중을 산출
        """
        # 1. Feature & Model Layer (Momentum)
        features = self.mom_engine.build_features(prices_df)
        ml_score = self.mom_engine.predict_score(features)
        
        # 룰 기반 모멘텀 (보조 지표로 사용)
        rule_mom = features['mom_12m_1m']
        
        # 2. Trend & Vol Layer
        trend_score = self.trend_engine.build_trend_features(prices_df)
        sigma, cov = self.trend_engine.estimate_vol(prices_df)
        
        # 3. Signal Layer
        # 가중치: ML 40%, 룰모멘텀 20%, 추세 40%
        signal = self.port_engine.build_signal(
            ml_score=ml_score,
            rule_mom=rule_mom,
            trend_score=trend_score,
            weights={"ml": 0.4, "mom": 0.2, "trend": 0.4}
        )
        
        # 4. Portfolio Layer (Vol Targeting)
        final_weights = self.port_engine.make_weights(signal, sigma, cov)
        
        return final_weights


# -----------------------------------------------------------------------------
# 실행 테스트 (Main)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== 퀀트 전략 엔진 테스트 시작 ===")
    
    # 1. 데이터 가져오기 (yfinance 활용)
    # 글로벌 주요 자산군 ETF (주식, 채권, 부동산, 원자재 등)
    tickers = ['SPY', 'QQQ', 'IWM', 'VNQ', 'GLD', 'TLT', 'HYG', 'EEM']
    print(f"대상 자산: {tickers}")
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*3) # 최근 3년 데이터
    
    print("데이터 다운로드 중...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        data = data.ffill() # 결측치 보정
        
        if data.empty:
            print("데이터 다운로드 실패. 인터넷 연결을 확인하세요.")
        else:
            print(f"데이터 준비 완료. (Rows: {len(data)})")
            
            # 2. 전략 엔진 초기화
            # 실제로는 학습된 XGB 모델을 로드해서 넣어야 함. 여기선 None (Rule-based Fallback 사용)
            strategy = MasterStrategy(xgb_model=None, target_vol=0.12, top_k=4)
            
            # 3. 리밸런싱 실행 (오늘 날짜 기준)
            print("\n[오늘 기준 포트폴리오 리밸런싱]")
            weights = strategy.rebalance(data)
            
            # 4. 결과 출력
            print("-" * 40)
            print(f"{'Ticker':<10} | {'Weight':<10} | {'Value (1억 투자시)'}")
            print("-" * 40)
            
            invest_amount = 100_000_000 # 1억원
            
            # 비중이 있는 종목만 출력
            active_weights = weights[weights > 0].sort_values(ascending=False)
            
            for ticker, w in active_weights.items():
                val = w * invest_amount
                print(f"{ticker:<10} | {w:>9.2%} | {val:>15,.0f} 원")
                
            print("-" * 40)
            total_lev = weights.sum()
            print(f"Total Leverage: {total_lev:.2%} (Cash: {1-total_lev:.2%})")
            
            # 5. 간단한 백테스트 시뮬레이션 (최근 6개월, 월간 리밸런싱)
            print("\n[최근 12개월 월간 리밸런싱 시뮬레이션]")
            
            # 월말 날짜 추출
            monthly_dates = data.resample('M').last().index
            history = []
            
            # 최소 데이터 기간(1년) 확보 후 시작
            sim_dates = [d for d in monthly_dates if d > data.index[0] + timedelta(days=252)]
            sim_dates = sim_dates[-12:] # 최근 1년치만
            
            for d in sim_dates:
                # 해당 날짜까지의 데이터 슬라이싱
                sub_data = data.loc[:d]
                
                # 리밸런싱
                w = strategy.rebalance(sub_data)
                
                # 기록
                top_pick = w.idxmax() if w.sum() > 0 else "None"
                leverage = w.sum()
                history.append({
                    "Date": d.date(),
                    "Top_Pick": top_pick,
                    "Leverage": f"{leverage:.2f}x",
                    "Num_Assets": len(w[w>0])
                })
                
            hist_df = pd.DataFrame(history)
            print(hist_df)

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

