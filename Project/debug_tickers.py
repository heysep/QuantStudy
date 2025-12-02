import pandas as pd
import requests
from io import StringIO

def get_tickers():
    print("Fetching tickers...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print(f"Response status: {response.status_code}")
        
        table = pd.read_html(StringIO(response.text))
        df = table[0]
        print(f"Columns: {df.columns}")
        print(df.head())
        tickers = df['Symbol'].tolist()
        print(f"Found {len(tickers)} tickers.")
        print(tickers[:5])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_tickers()
