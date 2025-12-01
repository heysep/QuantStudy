import yfinance as yf
import pandas as pd

ticker = "PEP"
stock = yf.Ticker(ticker)

print(f"--- {ticker} Info ---")
print(f"Forward EPS (Info): {stock.info.get('forwardEps')}")
print(f"Trailing EPS (Info): {stock.info.get('trailingEps')}")

print("\n--- Estimates ---")
try:
    # Check for earnings estimates
    # Note: yfinance structure changes often, checking common attributes
    if hasattr(stock, 'earnings_estimate'):
        print("stock.earnings_estimate:")
        print(stock.earnings_estimate)
    
    if hasattr(stock, 'analysis'):
        print("stock.analysis:")
        print(stock.analysis)
        
    if hasattr(stock, 'calendar'):
        print("stock.calendar:")
        print(stock.calendar)
        
except Exception as e:
    print(f"Error checking estimates: {e}")
