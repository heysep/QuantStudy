import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

class Backtester:
    def __init__(self, strategy, price_data, benchmark_ticker='SPY', initial_capital=100_000_000):
        self.strategy = strategy
        self.data = price_data
        self.benchmark_ticker = benchmark_ticker
        self.initial_capital = initial_capital
        self.equity_curve = None
        self.results_df = None

    def run(self, start_date=None):
        """월간 리밸런싱 백테스트 실행"""
        if start_date:
            self.data = self.data[self.data.index >= start_date]
            
        # 결과 저장용
        portfolio_value = [self.initial_capital]
        dates = [self.data.index[0]]
        
        current_cash = self.initial_capital
        current_holdings = pd.Series(dtype=float)
        
        print(f"[Backtest] 시작: {self.data.index[0].date()} ~ {self.data.index[-1].date()}")
        print("진행 중...", end="", flush=True)
        
        for i, date in enumerate(self.data.index):
            if i == 0: continue
            
            # [수정] 월말 리밸런싱으로 변경 (월초 → 월말)
            # 다음 날이 다음 달이면 오늘이 월말
            next_date_idx = i + 1
            if next_date_idx < len(self.data.index):
                is_rebalance_day = (self.data.index[next_date_idx].month != date.month)
            else:
                is_rebalance_day = False  # 마지막 날
            
            # 자산 가치 평가
            daily_prices = self.data.loc[date]
            prev_prices = self.data.loc[self.data.index[i-1]]
            
            if not current_holdings.empty:
                asset_returns = daily_prices / prev_prices - 1
                for ticker, val in current_holdings.items():
                    if ticker in asset_returns:
                        current_holdings[ticker] *= (1 + asset_returns[ticker])
            
            total_value = current_cash + current_holdings.sum()
            
            # 리밸런싱
            if is_rebalance_day:
                print(".", end="", flush=True)
                # [Optimization] 전체 역사를 다 넘기면 점점 느려짐.
                # 전략이 필요한 최소 기간(예: 1년=252일) + 여유분(예: 150일)만 슬라이싱해서 전달
                lookback_window = 400 
                if i > lookback_window:
                    sub_data = self.data.iloc[i-lookback_window:i]
                else:
                    sub_data = self.data.iloc[:i]
                
                # 전략 객체에서 비중 받아오기
                target_weights = self.strategy.rebalance(sub_data)
                
                current_holdings = total_value * target_weights
                current_cash = total_value - current_holdings.sum()
            
            portfolio_value.append(total_value)
            dates.append(date)
            
        print(" 완료!")
        self.equity_curve = pd.Series(portfolio_value, index=dates)
        return self.equity_curve

    def analyze_performance(self):
        """성과 지표 계산 및 리포트 출력"""
        if self.equity_curve is None:
            print("백테스트를 먼저 실행해주세요.")
            return

        # 벤치마크 수익률
        spy_data = self.data[self.benchmark_ticker]
        spy_curve = (spy_data / spy_data.iloc[0]) * self.initial_capital
        spy_curve = spy_curve.reindex(self.equity_curve.index, method='ffill')
        
        # 일별 수익률
        strat_ret = self.equity_curve.pct_change().fillna(0)
        bench_ret = spy_curve.pct_change().fillna(0)
        
        # 지표 계산
        strat_total = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        bench_total = (spy_curve.iloc[-1] / spy_curve.iloc[0]) - 1
        
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = days / 365.25
        strat_cagr = (1 + strat_total) ** (1/years) - 1
        bench_cagr = (1 + bench_total) ** (1/years) - 1
        
        # MDD
        cum_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cum_max) / cum_max
        strat_mdd = drawdown.min()
        
        cum_max_b = spy_curve.cummax()
        drawdown_b = (spy_curve - cum_max_b) / cum_max_b
        bench_mdd = drawdown_b.min()
        
        # Sharpe & Vol
        strat_sharpe = (strat_ret.mean() * 252) / (strat_ret.std() * np.sqrt(252))
        bench_sharpe = (bench_ret.mean() * 252) / (bench_ret.std() * np.sqrt(252))
        strat_vol = strat_ret.std() * np.sqrt(252)
        bench_vol = bench_ret.std() * np.sqrt(252)
        
        print(f"\n{'='*60}")
        print(f" 성과 분석 리포트 ({days}일간)")
        print(f"{'='*60}")
        print(f"{'Metric':<15} | {'Strategy':<12} | {'Benchmark':<12} | {'Diff':<10}")
        print(f"{'-'*60}")
        print(f"{'CAGR':<15} | {strat_cagr*100:>11.2f}% | {bench_cagr*100:>11.2f}% | {strat_cagr-bench_cagr:>+9.2%}p")
        print(f"{'MDD':<15} | {strat_mdd*100:>11.2f}% | {bench_mdd*100:>11.2f}% | {strat_mdd-bench_mdd:>+9.2%}p")
        print(f"{'Sharpe':<15} | {strat_sharpe:>11.2f}  | {bench_sharpe:>11.2f}  | {strat_sharpe-bench_sharpe:>+9.2f}")
        print(f"{'Total Return':<15} | {strat_total*100:>11.2f}% | {bench_total*100:>11.2f}% | {strat_total-bench_total:>+9.2%}p")
        print(f"{'='*60}")
        
        # 시각화
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.equity_curve, label='Strategy', color='red', linewidth=2)
        plt.plot(spy_curve, label='Benchmark', color='gray', linestyle='--')
        plt.title('Portfolio Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(drawdown * 100, label='Strategy DD', color='blue', linewidth=1)
        plt.plot(drawdown_b * 100, label='Benchmark DD', color='gray', linestyle=':', alpha=0.5)
        plt.fill_between(drawdown.index, drawdown * 100, 0, color='blue', alpha=0.1)
        plt.title('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

