import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class MomentumBacktester:
    def __init__(self, start_date='2023-01-01', top_n=30):
        self.start_date = pd.to_datetime(start_date)
        self.top_n = top_n
        self.tickers = []
        self.data = None
        self.features = pd.DataFrame()
        self.results = []
        
    def get_tickers(self):
        # Reuse logic or hardcode top liquid names for speed if needed
        # For proof, using full S&P 500 is better
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            self.tickers = table[0]['Symbol'].tolist()
            self.tickers = [t.replace('.', '-') for t in self.tickers]
            # Add SPY for benchmark
            if 'SPY' not in self.tickers:
                self.tickers.append('SPY')
        except:
            self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'V', 'JNJ', 'SPY']

    def fetch_data(self):
        print("데이터 다운로드 중 (3년)...")
        # Download 3 years to have enough history for features before start_date
        self.data = yf.download(self.tickers, period="3y", group_by='ticker', progress=True, threads=True)
        
    def calculate_features(self, df):
        # Helper to calc features for a single ticker dataframe
        df = df.copy()
        df = df.ffill()
        
        # Features
        df['Ret_12m'] = df['Close'].pct_change(252)
        df['Ret_6m'] = df['Close'].pct_change(126)
        df['Ret_3m'] = df['Close'].pct_change(63)
        df['Ret_1m'] = df['Close'].pct_change(21)
        df['Mom_Quality'] = df['Ret_12m'] - df['Ret_1m']
        df['Vol_126d'] = df['Close'].pct_change().rolling(126).std() * np.sqrt(252)
        df['RSTR'] = df['Ret_12m'] / df['Vol_126d']
        
        # Target (Next Month Return)
        df['Target_Next_1m'] = df['Close'].shift(-21) / df['Close'] - 1
        
        return df

    def prepare_all_features(self):
        print("전체 지표 계산 중...")
        feature_list = []
        valid_tickers = [t for t in self.tickers if t in self.data.columns.levels[0]]
        
        for ticker in valid_tickers:
            try:
                df = self.data[ticker].copy()
                if df.empty or 'Close' not in df.columns:
                    continue
                    
                df_feat = self.calculate_features(df)
                df_feat['Ticker'] = ticker
                
                # Keep necessary columns
                cols = ['Ticker', 'Ret_12m', 'Ret_6m', 'Ret_3m', 'Ret_1m', 'Mom_Quality', 'Vol_126d', 'RSTR', 'Target_Next_1m', 'Close']
                feature_list.append(df_feat[cols])
            except:
                continue
                
        self.features = pd.concat(feature_list)
        
        # RS Rank (Cross-sectional)
        print("RS Rank 계산 중...")
        self.features['RS_Rank'] = self.features.groupby(level=0)['Ret_12m'].rank(pct=True) * 100
        
    def run_backtest(self):
        print(f"\n백테스트 시작 (Start Date: {self.start_date.date()})...")
        
        # Get unique dates from features, filtered by start_date
        # We rebalance every 21 trading days (approx monthly)
        all_dates = self.features.index.unique().sort_values()
        rebalance_dates = [d for d in all_dates if d >= self.start_date]
        # Resample to monthly (approx every 21 days)
        rebalance_dates = rebalance_dates[::21] 
        
        portfolio_value = 10000.0
        spy_value = 10000.0
        
        history = []
        
        for i, date in enumerate(rebalance_dates[:-1]):
            # 1. Train Model (Data up to 'date')
            # We need targets to be known, so we use data up to (date - 1 month) for training labels
            # But for features, we use data up to 'date'
            
            # Training Data: All history up to 'date' where Target is known (non-NaN)
            train_mask = (self.features.index < date) & (self.features['Target_Next_1m'].notna())
            train_data = self.features[train_mask]
            
            if len(train_data) < 1000:
                print(f"Skipping {date.date()}: Not enough training data")
                continue
                
            X_train = train_data[['Ret_12m', 'Ret_6m', 'Ret_3m', 'Mom_Quality', 'RSTR', 'RS_Rank']]
            y_train = train_data['Target_Next_1m']
            
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # 2. Select Stocks (At 'date')
            current_mask = (self.features.index == date)
            current_data = self.features[current_mask].copy()
            
            if current_data.empty:
                continue
                
            X_curr = current_data[['Ret_12m', 'Ret_6m', 'Ret_3m', 'Mom_Quality', 'RSTR', 'RS_Rank']]
            current_data['Pred_Score'] = model.predict(X_curr)
            
            # Pick Top N
            top_picks = current_data.sort_values('Pred_Score', ascending=False).head(self.top_n)
            selected_tickers = top_picks['Ticker'].tolist()
            
            # 3. Calculate Return (Next Period)
            # Return is 'Target_Next_1m' which is actual return from date to date+21
            # We assume equal weight
            avg_return = top_picks['Target_Next_1m'].mean()
            
            # Benchmark Return (SPY)
            spy_row = self.features[(self.features['Ticker'] == 'SPY') & (self.features.index == date)]
            if not spy_row.empty:
                spy_return = spy_row['Target_Next_1m'].values[0]
            else:
                spy_return = 0.0
            
            # Update Value
            portfolio_value *= (1 + avg_return)
            spy_value *= (1 + spy_return)
            
            print(f"[{date.date()}] Port: {portfolio_value:.0f} (+{avg_return*100:.1f}%) vs SPY: {spy_value:.0f} (+{spy_return*100:.1f}%)")
            
            history.append({
                'Date': date,
                'Portfolio': portfolio_value,
                'SPY': spy_value,
                'Return': avg_return,
                'SPY_Return': spy_return
            })
            
        self.results = pd.DataFrame(history)
        
    def generate_report(self):
        if self.results.empty:
            print("결과가 없습니다.")
            return
            
        total_return = (self.results['Portfolio'].iloc[-1] / 10000 - 1) * 100
        spy_total_return = (self.results['SPY'].iloc[-1] / 10000 - 1) * 100
        
        print("\n" + "="*50)
        print("백테스트 최종 결과")
        print("="*50)
        print(f"전략 수익률: {total_return:.2f}%")
        print(f"SPY 수익률 : {spy_total_return:.2f}%")
        print(f"초과 수익률: {total_return - spy_total_return:.2f}%p")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.results['Date'], self.results['Portfolio'], label='Momentum Strategy', linewidth=2)
        plt.plot(self.results['Date'], self.results['SPY'], label='S&P 500 (SPY)', linestyle='--', color='gray')
        plt.title('Momentum Strategy vs S&P 500 (Backtest)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($10k Start)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('backtest_result.png')
        print("결과 그래프 저장됨: backtest_result.png")

if __name__ == "__main__":
    # Run backtest from 2023
    bt = MomentumBacktester(start_date='2023-01-01', top_n=30)
    bt.get_tickers()
    bt.fetch_data()
    bt.prepare_all_features()
    bt.run_backtest()
    bt.generate_report()
