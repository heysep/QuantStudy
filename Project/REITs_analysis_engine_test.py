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

from REITs_analysis_engine import REITsAnalyzer

class REITsBacktester:
    def __init__(self):
        self.analyzer = REITsAnalyzer()
        self.tickers = list(self.analyzer.tickers.keys())
        self.benchmark_ticker = 'SPY'
        self.start_date = (datetime.now() - timedelta(days=365*4)).strftime('%Y-%m-%d') # 최근 4년
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.monthly_investment = 1000.0 # 매월 투자금 ($1,000)
        
    def fetch_data(self):
        """데이터 수집"""
        print(f"데이터 수집 중... ({self.start_date} ~ {self.end_date})")
        
        # 1. 주가 데이터 (REITs + SPY)
        all_tickers = self.tickers + [self.benchmark_ticker]
        self.price_data = yf.download(all_tickers, start=self.start_date, end=self.end_date, progress=False)['Close']
        self.price_data = self.price_data.ffill() # 결측치 보정
        
        # 2. 재무 데이터 (Operating Cash Flow)
        self.fundamental_data = pd.DataFrame()
        
        print("재무 데이터 수집 및 처리 중...")
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                
                # 분기별 데이터
                q_cf = stock.quarterly_cashflow
                q_bs = stock.quarterly_balance_sheet
                
                if q_cf.empty or q_bs.empty:
                    continue
                    
                # OCF & Shares
                if 'Operating Cash Flow' in q_cf.index:
                    ocf = q_cf.loc['Operating Cash Flow']
                elif 'Total Cash From Operating Activities' in q_cf.index:
                    ocf = q_cf.loc['Total Cash From Operating Activities']
                else:
                    continue
                    
                if 'Ordinary Shares Number' in q_bs.index:
                    shares = q_bs.loc['Ordinary Shares Number']
                elif 'Share Issued' in q_bs.index:
                    shares = q_bs.loc['Share Issued']
                else:
                    shares = pd.Series(stock.info.get('sharesOutstanding'), index=ocf.index)
                
                # DataFrame 생성
                df_fund = pd.DataFrame({'OCF': ocf, 'Shares': shares})
                df_fund.index = pd.to_datetime(df_fund.index).tz_localize(None)
                df_fund = df_fund.sort_index()
                
                # OCF per Share
                df_fund['OCF_per_Share'] = df_fund['OCF'] / df_fund['Shares']
                
                # Daily Forward Fill
                full_idx = self.price_data.index
                df_daily_fund = df_fund['OCF_per_Share'].reindex(full_idx, method='ffill')
                
                self.fundamental_data[ticker] = df_daily_fund
                
            except Exception as e:
                print(f"  -> {ticker} 오류: {e}")
                
        print("데이터 준비 완료.")
        
    def calculate_metrics(self):
        """지표 계산: 개별 리츠의 P/OCF 및 괴리율"""
        self.valuation_ratios = pd.DataFrame(index=self.price_data.index)
        
        for ticker in self.tickers:
            if ticker in self.fundamental_data.columns:
                price = self.price_data[ticker]
                ocf_ps = self.fundamental_data[ticker]
                
                # P/OCF
                p_ocf = price / ocf_ps
                p_ocf[ocf_ps <= 0] = np.nan
                
                # 6개월 이동평균 (126일)
                ma_126 = p_ocf.rolling(window=126).mean()
                
                # 괴리율 (Current / MA) - 낮을수록 저평가
                ratio = p_ocf / ma_126
                self.valuation_ratios[ticker] = ratio
                
    def run_backtest(self):
        """월 적립식 백테스트 (Event-Driven)"""
        print("\n백테스트 실행 중 (Monthly DCA)...")
        
        # 1. 전략 포트폴리오 (SPY + REITs)
        self.portfolio = {
            'Cash': 0.0,
            'SPY': 0.0,
            'Holdings': {t: 0.0 for t in self.tickers}
        }
        
        # 2. 벤치마크 포트폴리오 (Only SPY)
        self.benchmark_portfolio = {
            'Cash': 0.0,
            'SPY': 0.0
        }
        
        # 3. 리츠 Only 포트폴리오 (Only REITs)
        self.reits_only_portfolio = {
            'Cash': 0.0,
            'Holdings': {t: 0.0 for t in self.tickers}
        }
        
        # 기록용
        self.history = []
        
        # 월초 날짜 식별 (각 월의 첫 거래일)
        monthly_trading_days = self.price_data.groupby(self.price_data.index.to_period('M')).apply(lambda x: x.index[0])
        monthly_trading_days = set(monthly_trading_days)
        
        # 일별 루프 (평가액 계산을 위해)
        dates = self.price_data.index
        
        # 투자 내역 기록
        self.invest_log = []
        
        # 총 투자금 추적 (실제 투자 발생 시 증가)
        current_total_invested = 0.0
        
        for date in dates:
            # 1. 월초 투자 실행 여부 확인
            if date in monthly_trading_days:
                current_total_invested += self.monthly_investment
                
                # 벤치마크 투자 ($1000 -> SPY)
                spy_price = self.price_data.loc[date, 'SPY']
                if not np.isnan(spy_price):
                    shares_to_buy = self.monthly_investment / spy_price
                    self.benchmark_portfolio['SPY'] += shares_to_buy
                
                # 전략 투자 & 리츠 Only 투자
                # 가장 저평가된 리츠 선정
                if date in self.valuation_ratios.index:
                    ratios = self.valuation_ratios.loc[date]
                    valid_ratios = ratios[ratios < 1.0].sort_values()
                    
                    target_reit = None
                    if not valid_ratios.empty:
                        target_reit = valid_ratios.index[0] # Top 1
                        val_ratio = valid_ratios.iloc[0]
                    
                    if target_reit:
                        # 전략: $500 SPY + $500 REIT
                        amt_each = self.monthly_investment / 2
                        
                        # SPY 매수
                        self.portfolio['SPY'] += amt_each / spy_price
                        
                        # REIT 매수 (전략)
                        reit_price = self.price_data.loc[date, target_reit]
                        if not np.isnan(reit_price):
                            self.portfolio['Holdings'][target_reit] += amt_each / reit_price
                            
                            # REIT Only 매수 ($1000 전액)
                            self.reits_only_portfolio['Holdings'][target_reit] += self.monthly_investment / reit_price
                            
                            self.invest_log.append({
                                'Date': date,
                                'Action': 'Buy',
                                'REIT': target_reit,
                                'Ratio': f"{val_ratio:.2f}",
                                'Price': f"{reit_price:.2f}"
                            })
                    else:
                        # 저평가 종목 없음 -> 현금 보유
                        self.portfolio['Cash'] += self.monthly_investment
                        self.reits_only_portfolio['Cash'] += self.monthly_investment
                        
                        self.invest_log.append({
                            'Date': date,
                            'Action': 'Save Cash',
                            'REIT': '-',
                            'Ratio': '-',
                            'Price': '-'
                        })
            
            # 2. 일별 평가액 계산
            current_spy_price = self.price_data.loc[date, 'SPY']
            
            # 전략 포트폴리오 가치
            strat_val = self.portfolio['Cash']
            strat_val += self.portfolio['SPY'] * current_spy_price
            for t, shares in self.portfolio['Holdings'].items():
                if shares > 0:
                    p = self.price_data.loc[date, t]
                    if not np.isnan(p):
                        strat_val += shares * p
            
            # 벤치마크 포트폴리오 가치
            bench_val = self.benchmark_portfolio['Cash']
            bench_val += self.benchmark_portfolio['SPY'] * current_spy_price
            
            # 리츠 Only 포트폴리오 가치
            reits_val = self.reits_only_portfolio['Cash']
            for t, shares in self.reits_only_portfolio['Holdings'].items():
                if shares > 0:
                    p = self.price_data.loc[date, t]
                    if not np.isnan(p):
                        reits_val += shares * p
            
            self.history.append({
                'Date': date,
                'Strategy_Value': strat_val,
                'Benchmark_Value': bench_val,
                'REITs_Only_Value': reits_val,
                'Total_Invested': current_total_invested
            })
            
        self.results_df = pd.DataFrame(self.history).set_index('Date')
        
    def report(self):
        """결과 보고"""
        print(f"\n{'='*60}")
        print("월 적립식(DCA) 페어 트레이딩 백테스트 결과")
        print(f"{'='*60}")
        
        final_row = self.results_df.iloc[-1]
        total_invested = final_row['Total_Invested']
        
        strat_final = final_row['Strategy_Value']
        bench_final = final_row['Benchmark_Value']
        reits_final = final_row['REITs_Only_Value']
        
        strat_return = (strat_final / total_invested - 1) * 100
        bench_return = (bench_final / total_invested - 1) * 100
        reits_return = (reits_final / total_invested - 1) * 100
        
        print(f"총 투자 원금: ${total_invested:,.0f}")
        print("-" * 60)
        print(f"{'Portfolio':<20} | {'Final Value':<15} | {'Return':<10}")
        print("-" * 60)
        print(f"{'Strategy (Mix)':<20} | ${strat_final:,.0f} | {strat_return:>6.2f}%")
        print(f"{'Benchmark (SPY)':<20} | ${bench_final:,.0f} | {bench_return:>6.2f}%")
        print(f"{'REITs Only':<20} | ${reits_final:,.0f} | {reits_return:>6.2f}%")
        print("-" * 60)
        
        # --- 수익률 정밀 분석 (Inflow 제외) ---
        # 1. 일별 입금액 계산
        inflows = self.results_df['Total_Invested'].diff().fillna(self.results_df['Total_Invested'].iloc[0])
        
        # 2. 일별 순수익 (Total Change - Inflow)
        strat_pnl = self.results_df['Strategy_Value'].diff().fillna(0) - inflows
        bench_pnl = self.results_df['Benchmark_Value'].diff().fillna(0) - inflows
        
        # 3. 일별 수익률 (PnL / Previous Value)
        prev_strat_val = self.results_df['Strategy_Value'].shift(1).fillna(0)
        prev_bench_val = self.results_df['Benchmark_Value'].shift(1).fillna(0)
        
        # 분모가 0이거나 아주 작을 때 0 처리
        strat_daily_ret = np.where(prev_strat_val > 1, strat_pnl / prev_strat_val, 0.0)
        bench_daily_ret = np.where(prev_bench_val > 1, bench_pnl / prev_bench_val, 0.0)
        
        strat_daily_ret = pd.Series(strat_daily_ret, index=self.results_df.index)
        bench_daily_ret = pd.Series(bench_daily_ret, index=self.results_df.index)
        
        # 4. Rolling 1-Year Return (252 days)
        strat_roll_1y = (1 + strat_daily_ret).rolling(window=252).apply(np.prod, raw=True) - 1
        bench_roll_1y = (1 + bench_daily_ret).rolling(window=252).apply(np.prod, raw=True) - 1
        
        # 5. Active Return (Strategy - Benchmark)
        active_return_1y = strat_roll_1y - bench_roll_1y
        
        # 연도별 수익률 비교 (Time-Weighted Return 기준)
        print("\n[연도별 성과 (Time-Weighted Return)]")
        print(f"{'Year':<6} | {'Strategy':<10} | {'Benchmark':<10} | {'Active':<10}")
        print("-" * 45)
        
        yearly_res = pd.DataFrame({'S': strat_daily_ret, 'B': bench_daily_ret})
        yearly_groups = yearly_res.groupby(yearly_res.index.year)
        
        for year, data in yearly_groups:
            s_yr = (1 + data['S']).prod() - 1
            b_yr = (1 + data['B']).prod() - 1
            diff = s_yr - b_yr
            print(f"{year:<6} | {s_yr*100:>9.2f}% | {b_yr*100:>9.2f}% | {diff*100:>9.2f}%p")

        # 최근 매수 로그 (마지막 5개)
        print(f"\n{'='*60}")
        print("최근 매수 활동:")
        for log in self.invest_log[-5:]:
            print(f" {log['Date'].date()} | {log['Action']:<10} | {log['REIT']:<5} | Ratio: {log['Ratio']}")
            
        # 시각화 (Subplots)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # 1. Portfolio Value
        ax1.plot(self.results_df['Strategy_Value'], label='Strategy (SPY + REITs)', linewidth=2)
        ax1.plot(self.results_df['Benchmark_Value'], label='Benchmark (SPY Only)', linestyle='--')
        ax1.plot(self.results_df['REITs_Only_Value'], label='REITs Only (Timing)', linestyle='-.', color='green', alpha=0.7)
        ax1.plot(self.results_df['Total_Invested'], label='Total Invested', linestyle=':', color='gray')
        
        ax1.set_title('Monthly DCA Portfolio Value')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling 1-Year Active Return
        ax2.plot(active_return_1y * 100, color='purple', label='Rolling 1Y Active Return (Strategy - SPY)')
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.fill_between(active_return_1y.index, active_return_1y * 100, 0, where=(active_return_1y > 0), color='red', alpha=0.2, label='Outperform')
        ax2.fill_between(active_return_1y.index, active_return_1y * 100, 0, where=(active_return_1y < 0), color='blue', alpha=0.2, label='Underperform')
        
        ax2.set_title('Rolling 1-Year Active Return (Strategy vs SPY)')
        ax2.set_ylabel('Excess Return (%p)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Monthly Excess Return (Bar Chart)
        monthly_excess = yearly_res.resample('M').apply(lambda x: (1+x).prod() - 1)
        monthly_diff = (monthly_excess['S'] - monthly_excess['B']) * 100
        
        colors = ['red' if v > 0 else 'blue' for v in monthly_diff]
        ax3.bar(monthly_diff.index, monthly_diff, color=colors, width=20, alpha=0.6)
        ax3.axhline(0, color='black', linewidth=1)
        ax3.set_title('Monthly Excess Return (Strategy - SPY)')
        ax3.set_ylabel('Diff (%p)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    backtester = REITsBacktester()
    backtester.fetch_data()
    backtester.calculate_metrics()
    backtester.run_backtest()
    backtester.report()
