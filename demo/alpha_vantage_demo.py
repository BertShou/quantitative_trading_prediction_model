import pandas as pd
import requests


def get_alpha_vantage_data(api_key, symbol, function='TIME_SERIES_DAILY'):
    """
    Fetch stock data from Alpha Vantage API.

    Args:
        api_key (str): Your Alpha Vantage API key.
        symbol (str): Stock ticker symbol (e.g., 'AAPL' for Apple).
        function (str): API function (e.g., 'TIME_SERIES_DAILY', 'TIME_SERIES_WEEKLY').

    Returns:
        pd.DataFrame: Stock data DataFrame or None if an error occurs.
    """
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Check if the API returned an error
        if 'Error Message' in data:
            print(f"API Error: {data['Error Message']}")
            return None

        # Extract time series data
        time_series_key = next(key for key in data if 'Time Series' in key)
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None


# Example usage
if __name__ == "__main__":
    api_key = 'HBL5OGPKCT1WTRUV'
    symbol = 'AAPL'  # Example stock symbol
    df = get_alpha_vantage_data(api_key, symbol)
    if df is not None:
        print(df.head())
    else:
        print("Failed to retrieve data.")
