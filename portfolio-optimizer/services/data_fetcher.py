import yfinance as yf

def fetch_data(ticker: str, period="5y"):
    data = yf.download(ticker, period=period)
    if data.empty:
        raise ValueError("Nie znaleziono danych dla podanego tickera.")
    return data[['Close']]