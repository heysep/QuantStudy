import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_data(tickers, start_date=None, end_date=None):
    """
    yfinance를 통해 데이터를 수집하고 전처리합니다.
    """
    if end_date is None:
        end_date = datetime.today()
    
    if start_date is None:
        start_date = end_date - timedelta(days=365*3)
        
    print(f"[Data] 다운로드 시작 ({start_date.date()} ~ {end_date.date()})...")
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        # 결측치 보정 (Forward Fill)
        data = data.ffill()
        
        if data.empty:
            print("[Data] 다운로드 실패: 데이터가 비어있습니다.")
            return None
            
        print(f"[Data] 준비 완료. (Rows: {len(data)})")
        return data
        
    except Exception as e:
        print(f"[Data] 오류 발생: {e}")
        return None

