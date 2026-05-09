import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def fetch_data(ticker, start_date="2025-01-01", end_date="2025-12-31"):
    data = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def calculate_avwap(data, period=20):
    """
    Calculate the Anchored Volume Weighted Average Price (AVWAP) for a given period.
    """
    # Calculate the typical price with min_periods=1 to avoid NaN values at the start
    typical_price = (data['Close'] * data['Volume']).rolling(window=period, min_periods=1).sum() / data['Volume'].rolling(window=period, min_periods=1).sum()
    
    data['AVWAP'] = typical_price
    return data['AVWAP']

def log_returns(data):
    """
    Calculate the logarithmic returns of the stock.
    """
    log_ret = np.log(data['Close'] / data['Close'].shift(1))
    data['Log_Returns'] = log_ret
    return data['Log_Returns']

#strategy implementation

def avwap_strategy(data, avwap_period=20):
    """
    Implement a simple trading strategy based on AVWAP.
    """
    data['AVWAP'] = calculate_avwap(data, avwap_period)
    data['Log_Returns'] = log_returns(data)
    
    # Generate trading signals
    data['Signal'] = 0
    data.loc[data['Close'] > data['AVWAP'], 'Signal'] = 1  # Buy signal
    data.loc[data['Close'] < data['AVWAP'], 'Signal'] = -1  # Sell signal
    
    return data


#signal x log returns
def signal_log_returns(data):
    """
    Calculate the product of trading signals and logarithmic returns.
    """
    data['Signal_Log_Returns'] = data['Signal'] * data['Log_Returns']
    return data

def calculate_performance(data):
    """
    Calculate the cumulative returns of the strategy.
    """
    data['Cumulative_Returns'] = (1 + data['Signal_Log_Returns']).cumprod() - 1
    # winrate
    data['Win_Rate'] = (data['Signal_Log_Returns'] > 0).mean()
    return data

def test_strategy(ticker):
    data = fetch_data(ticker)
    data = avwap_strategy(data)
    data = signal_log_returns(data)
    data = calculate_performance(data)
    print(summary := data[['Close', 'AVWAP', 'Signal', 'Log_Returns', 'Signal_Log_Returns', 'Cumulative_Returns', 'Win_Rate']].describe())
    print(data[['Close', 'AVWAP', 'Signal', 'Log_Returns', 'Signal_Log_Returns', 'Cumulative_Returns', 'Win_Rate']].tail())
    return data

def plot_results(data, ticker):
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.plot(data.index, data['AVWAP'], label='AVWAP', linestyle='--')
    plt.title(f'{ticker} Close Price and AVWAP')
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['Cumulative_Returns'], label='Cumulative Returns', color='green')
    plt.title(f'{ticker} Cumulative Returns from AVWAP Strategy')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


#test the strategy on a portfolio of stocks

def test_portfolio(portfolio):
    results = {}
    for ticker in portfolio:
        print(f"Testing strategy for {ticker}...")
        data = test_strategy(ticker)
        results[ticker] = data
    cum_return_of_portfolio = pd.DataFrame({ticker: data['Cumulative_Returns'] for ticker, data in results.items()})
    cum_return_of_portfolio['Average_Cumulative_Returns'] = cum_return_of_portfolio.mean(axis=1)
    plt.figure(figsize=(14, 7))
    for ticker in portfolio:
        plt.plot(cum_return_of_portfolio.index, cum_return_of_portfolio[ticker], label=f'{ticker} Cumulative Returns')
    plt.plot(cum_return_of_portfolio.index, cum_return_of_portfolio['Average_Cumulative_Returns'], label='Average Cumulative Returns', color='black', linestyle='--')
    plt.title('Cumulative Returns of AVWAP Strategy for Portfolio')
    plt.xlabel('Date')
    plt.legend()
    plt.show()
    return results

BIST50 = [
    "AKBNK.IS","ARCLK.IS","ASELS.IS","BIMAS.IS","EKGYO.IS",
    "EREGL.IS","FROTO.IS","GARAN.IS","HALKB.IS","ISCTR.IS",
    "KCHOL.IS","KRDMD.IS","MGROS.IS","PETKM.IS",
    "PGSUS.IS","SAHOL.IS","SISE.IS","TCELL.IS","THYAO.IS",
    "TUPRS.IS","VAKBN.IS","VESTL.IS","YKBNK.IS","AEFES.IS",
    "AKSEN.IS","ALARK.IS","ENKAI.IS","GUBRF.IS","TTKOM.IS",
] 

portfolio = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
portdf = test_portfolio(BIST50)
