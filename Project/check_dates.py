import yfinance as yf
import pandas as pd

def check_data():
    tickers = ['PLTR', 'FISV', 'GOOGL']
    print(f"Downloading data for {tickers}...")
    data = yf.download(tickers, period="1mo", group_by='ticker')
    
    for t in tickers:
        print(f"\n--- {t} ---")
        df = data[t]
        print(df.tail())

if __name__ == "__main__":
    check_data()
