import yfinance as yf

def fetch_data(ticker: str, period="5y"):
    data = yf.download(ticker, period=period)
    if data.empty:
        raise ValueError("Nie znaleziono danych dla podanego tickera.")
    return data[['Close']]


def fetch_company_info(ticker: str):
    info = yf.Ticker(ticker).info
    if info is None or not info:
        raise ValueError("Nie znaleziono danych informacyjnych dla podanego tickera.")
    return info