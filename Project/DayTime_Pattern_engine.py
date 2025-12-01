import yfinance as yf
import pandas as pd
import numpy as np

# Define Sector ETFs
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Financials': 'XLF',
    'Health Care': 'XLV',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Industrials': 'XLI',
    'Utilities': 'XLU',
    'Energy': 'XLE',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC'
}

def analyze_sector_seasonality():
    print("Fetching data for Sector ETFs...")
    
    results = {}
    
    for sector, ticker in SECTOR_ETFS.items():
        print(f"Processing {sector} ({ticker})...")
        try:
            # Fetch max history
            data = yf.download(ticker, period="max", progress=False)
            
            if data.empty:
                print(f"No data found for {ticker}")
                continue
            
            # Debug: Print columns
            # print(f"Columns for {ticker}: {data.columns}")

            # Handle MultiIndex columns (Ticker, Price Type) or (Price Type, Ticker)
            # Recent yfinance might return columns like ('Adj Close', 'XLK')
            if isinstance(data.columns, pd.MultiIndex):
                # Try to find 'Adj Close' or 'Close' at level 0
                if 'Adj Close' in data.columns.get_level_values(0):
                    prices = data['Adj Close']
                    # If it's still a DataFrame with tickers as columns, select the ticker
                    if isinstance(prices, pd.DataFrame) and ticker in prices.columns:
                        prices = prices[ticker]
                    elif isinstance(prices, pd.DataFrame) and len(prices.columns) == 1:
                         prices = prices.iloc[:, 0]
                elif 'Close' in data.columns.get_level_values(0):
                    prices = data['Close']
                    if isinstance(prices, pd.DataFrame) and ticker in prices.columns:
                        prices = prices[ticker]
                    elif isinstance(prices, pd.DataFrame) and len(prices.columns) == 1:
                         prices = prices.iloc[:, 0]
                else:
                    print(f"No price data found in MultiIndex for {ticker}")
                    continue
            else:
                # Standard Index
                if 'Adj Close' in data.columns:
                    prices = data['Adj Close']
                elif 'Close' in data.columns:
                    prices = data['Close']
                else:
                    print(f"No price data for {ticker}")
                    continue
            
            # Ensure prices is a Series
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]

            # Resample to Monthly Start and get the first price of the month? 
            # Or Resample to Monthly End?
            # Standard practice: Monthly Returns based on Month End prices.
            monthly_prices = prices.resample('ME').last()
            
            # Calculate Monthly Returns
            monthly_returns = monthly_prices.pct_change().dropna()
            
            # Add Month column (1-12)
            # monthly_returns is a Series, we need to convert to DataFrame to add columns if we want, 
            # or just use index.
            
            # Create a DataFrame for analysis
            df_analysis = pd.DataFrame({'Return': monthly_returns})
            df_analysis['Month'] = df_analysis.index.month
            
            # Calculate Win Rate and Avg Return by Month
            monthly_stats = df_analysis.groupby('Month')['Return'].agg(
                Win_Rate=lambda x: (x > 0).mean() * 100,
                Avg_Return=lambda x: x.mean() * 100
            )
            
            results[sector] = monthly_stats['Win_Rate']
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*50)
    print("Monthly Win Rate (%) by Sector (Probability of Rise)")
    print("="*50)
    
    # Combine results into a single DataFrame
    if results:
        final_df = pd.DataFrame(results)
        
        # Format settings
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.1f}'.format)
        
        output_str = ""
        output_str += "="*50 + "\n"
        output_str += "Monthly Win Rate (%) by Sector (Probability of Rise)\n"
        output_str += "="*50 + "\n"
        output_str += str(final_df) + "\n\n"
        
        output_str += "="*50 + "\n"
        output_str += "Best Month for each Sector:\n"
        output_str += "="*50 + "\n"
        for sector in final_df.columns:
            best_month = final_df[sector].idxmax()
            best_win_rate = final_df[sector].max()
            output_str += f"{sector}: Month {best_month} ({best_win_rate:.1f}%)\n"

        output_str += "\n" + "="*50 + "\n"
        output_str += "Best Sector for each Month:\n"
        output_str += "="*50 + "\n"
        for month in final_df.index:
            best_sector = final_df.loc[month].idxmax()
            best_win_rate = final_df.loc[month].max()
            output_str += f"Month {month}: {best_sector} ({best_win_rate:.1f}%)\n"
            
        print(output_str)
        
        with open("seasonality_results.txt", "w", encoding="utf-8") as f:
            f.write(output_str)
        print("Results saved to seasonality_results.txt")

if __name__ == "__main__":
    analyze_sector_seasonality()
