import pandas as pd
import numpy as np
from scipy.stats import zscore

# XGBoost 임포트
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("Warning: xgboost가 설치되지 않았습니다.")

# -----------------------------------------------------------------------------
# 1. MomentumEngine (Data & Feature & Model)
# -----------------------------------------------------------------------------
class MomentumEngine:
    def __init__(self, xgb_model=None):
        self.model = xgb_model

    def build_features(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        clean_prices = prices_df.copy()
        clean_prices[clean_prices <= 0] = np.nan
        log_prices = np.log(clean_prices.ffill())
        
        current_idx = prices_df.index[-1]
        
        mom_12m = log_prices.diff(252)
        mom_1m = log_prices.diff(21)
        mom_12m_1m = mom_12m - mom_1m
        
        mom_6m = log_prices.diff(126)
        mom_3m = log_prices.diff(63)
        
        daily_ret = prices_df.pct_change()
        vol_20d = daily_ret.rolling(20).std() * np.sqrt(252)
        vol_60d = daily_ret.rolling(60).std() * np.sqrt(252)
        vol_12m = daily_ret.rolling(252).std() * np.sqrt(252)
        
        mom_quality = mom_12m / vol_12m.replace(0, np.nan)
        
        feat_df = pd.DataFrame({
            'mom_12m_1m': mom_12m_1m.loc[current_idx],
            'mom_6m': mom_6m.loc[current_idx],
            'mom_3m': mom_3m.loc[current_idx],
            'vol_20d': vol_20d.loc[current_idx],
            'vol_60d': vol_60d.loc[current_idx],
            'mom_quality': mom_quality.loc[current_idx]
        })
        
        feat_df['rs_rank'] = feat_df['mom_12m_1m'].rank(pct=True)
        feat_df.fillna(0, inplace=True)
        
        return feat_df

    def predict_score(self, feature_df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            # [수정] 월가 모멘텀 정석: 12M-1M Cross-sectional Rank만 사용
            # 복잡한 가중 평균은 오히려 섹터 편향 발생
            score = feature_df['mom_12m_1m']
            return score.rename("ml_score")
        
        try:
            if isinstance(self.model, xgb.Booster):
                dtest = xgb.DMatrix(feature_df)
                preds = self.model.predict(dtest)
            else:
                preds = self.model.predict(feature_df)
            return pd.Series(preds, index=feature_df.index, name='ml_score')
        except Exception as e:
            print(f"Prediction Error: {e}")
            return pd.Series(0, index=feature_df.index, name='ml_score')

# -----------------------------------------------------------------------------
# 2. TrendEngine (Trend & Vol)
# -----------------------------------------------------------------------------
class TrendEngine:
    def build_trend_features(self, prices_df: pd.DataFrame) -> pd.Series:
        """
        [수정] 월가 CTA 방식: Price > EMA 단순 크로스오버
        - Price > EMA20 이면 +1
        - Price > EMA120 이면 +1
        - 최대 +2 (강한 상승 추세)
        """
        ema_20 = prices_df.ewm(span=20, adjust=False).mean()
        ema_120 = prices_df.ewm(span=120, adjust=False).mean()
        
        last_price = prices_df.iloc[-1]
        last_ema_20 = ema_20.iloc[-1]
        last_ema_120 = ema_120.iloc[-1]
        
        # 단순 크로스오버 시그널
        signal_short = (last_price > last_ema_20).astype(float)  # 0 or 1
        signal_long = (last_price > last_ema_120).astype(float)  # 0 or 1
        
        # 합산 (0~2)
        trend_score = signal_short + signal_long
        
        return trend_score.rename("trend_score")

    def estimate_vol(self, prices_df: pd.DataFrame, span_vol=60):
        returns = prices_df.pct_change().fillna(0)
        realized_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        ewma_vol = returns.ewm(span=span_vol).std().iloc[-1] * np.sqrt(252)
        final_sigma = 0.5 * realized_vol + 0.5 * ewma_vol
        
        cov_matrix = returns.ewm(span=span_vol).cov().iloc[-len(prices_df.columns):]
        cov_matrix = cov_matrix.droplevel(0)
        cov_matrix_annual = cov_matrix * 252
        
        return final_sigma, cov_matrix_annual

# -----------------------------------------------------------------------------
# 3. PortfolioEngine (Allocation)
# -----------------------------------------------------------------------------
class PortfolioEngine:
    def __init__(self, target_vol=0.10, max_leverage=2.0, top_k=5):
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.top_k = top_k

    def build_signal(self, ml_score, rule_mom, trend_score, weights=None):
        if weights is None:
            weights = {"ml": 0.5, "mom": 0.2, "trend": 0.3}
            
        def safe_zscore(s):
            if s.std() == 0 or s.isna().all(): return s.fillna(0) * 0
            return (s - s.mean()) / s.std()

        z_ml = safe_zscore(ml_score.fillna(0))
        z_mom = safe_zscore(rule_mom.fillna(0))
        z_trend = safe_zscore(trend_score.fillna(0))
        
        final_signal = (
            weights["ml"] * z_ml + 
            weights["mom"] * z_mom + 
            weights["trend"] * z_trend
        )
        return final_signal.clip(lower=0)

    def make_weights(self, signal_series, sigma_series, cov_matrix):
        """
        [수정] Vol Targeting 버그 픽스
        - 선택된 자산의 비중을 1로 정규화한 후 공분산 계산
        - 이후 Target Vol 대비 레버리지 산출
        """
        valid_signal = signal_series[signal_series > 0]
        selected_tickers = valid_signal.nlargest(self.top_k).index
        
        if len(selected_tickers) == 0:
            return pd.Series(0.0, index=signal_series.index)
            
        sel_signal = valid_signal.loc[selected_tickers]
        sel_sigma = sigma_series.loc[selected_tickers]
        sel_cov = cov_matrix.loc[selected_tickers, selected_tickers]
        
        # Risk-Adjusted Allocation (비중 합 = 1로 정규화)
        raw_weights = (sel_signal ** 1.0) / sel_sigma.replace(0, np.inf)
        if raw_weights.sum() == 0: return pd.Series(0.0, index=signal_series.index)
        
        # [핵심] 비중 합 = 1.0으로 정규화 (단위 포트폴리오)
        w_unit = raw_weights / raw_weights.sum()
        
        # [핵심] 정규화된 비중으로 포트폴리오 Vol 계산
        w_vec = w_unit.values
        cov_mat = sel_cov.values
        port_var = w_vec.T @ cov_mat @ w_vec
        port_vol = np.sqrt(max(port_var, 1e-8))  # 0 방지
        
        # Target Vol 대비 레버리지 계산
        if port_vol == 0: leverage = 0
        else: leverage = self.target_vol / port_vol
            
        final_leverage = min(leverage, self.max_leverage)
        
        # 최종 비중 = 단위 비중 * 레버리지
        w_final = w_unit * final_leverage
        
        full_weights = pd.Series(0.0, index=signal_series.index)
        full_weights.loc[selected_tickers] = w_final
        
        return full_weights

# -----------------------------------------------------------------------------
# 4. Master Strategy (Main Entry for Strategy)
# -----------------------------------------------------------------------------
class TrendMomentumStrategy:
    def __init__(self, xgb_model=None, target_vol=0.12, top_k=5):
        """
        [수정] top_k 기본값 3 → 5로 확대 (과최적화 방지)
        """
        self.mom_engine = MomentumEngine(xgb_model)
        self.trend_engine = TrendEngine()
        self.port_engine = PortfolioEngine(target_vol=target_vol, max_leverage=2.0, top_k=top_k)
        
    def rebalance(self, prices_df: pd.DataFrame):
        # 1. Feature
        features = self.mom_engine.build_features(prices_df)
        ml_score = self.mom_engine.predict_score(features)
        rule_mom = features['mom_12m_1m']
        
        # 2. Trend & Vol
        trend_score = self.trend_engine.build_trend_features(prices_df)
        sigma, cov = self.trend_engine.estimate_vol(prices_df)
        
        # 3. Signal
        signal = self.port_engine.build_signal(
            ml_score=ml_score,
            rule_mom=rule_mom,
            trend_score=trend_score
        )
        
        # 4. Weight
        final_weights = self.port_engine.make_weights(signal, sigma, cov)
        return final_weights

