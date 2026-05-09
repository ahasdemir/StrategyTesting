import pandas as pd
import numpy as np
import yfinance as yf
import os
from scipy.optimize import minimize

script_dir = os.path.dirname(os.path.abspath(__file__))
tickers = pd.read_csv(os.path.join(script_dir, "tickers.csv"), skiprows=[1])["Ticker"].tolist()

def get_clean_data(tickers, start_date, end_date):
    # Sadece Close fiyatlarını çek ve MultiIndex'i otomatik yönet
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    data = data.ffill().bfill()
    data = data.dropna(axis=1, how='all')
    return data

def get_momentum_top_n(data, n=10):
    """Calculates 12-1 momentum and returns the top N tickers."""
    # 252 days (1 year) vs 21 days (1 month)
    momentum_scores = (data.shift(21) / data.shift(252)) - 1
    # Get the latest available scores
    latest_scores = momentum_scores.iloc[-1].dropna()
    top_n = latest_scores.sort_values(ascending=False).head(n)
    return top_n.index.tolist()

def portfolio_stats(weights, returns):
    """Calculates annualized return, volatility, and Sharpe ratio."""
    # Annualized mean returns and covariance
    mean_ret = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    port_ret = np.sum(mean_ret * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    # Assuming Risk-Free Rate = 0.0 (Adjust for BIST if needed)
    sharpe = port_ret / port_vol
    return port_ret, port_vol, sharpe

def optimize_weights(returns):
    """Finds weights that maximize the Sharpe Ratio."""
    num_assets = len(returns.columns)
    
    # Objective function: Minimize negative Sharpe
    def objective(weights):
        return -portfolio_stats(weights, returns)[2]

    # Constraints: Sum of weights = 100%
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    # Bounds: No shorting (0% to 100% per stock)
    # Hint: You can change 1.0 to 0.3 to prevent over-concentration
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    init_guess = np.array(num_assets * [1. / num_assets])
    
    opt_result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return opt_result.x


# 1. Fetch Data
df = get_clean_data(tickers, "2023-01-01", "2026-05-04")

# 2. Momentum Filter
top_10 = get_momentum_top_n(df, n=5) # Selecting 5 for example
print(f"Top Momentum Tickers: {top_10}")

# 3. MPT Optimization on the Top N
# Calculate daily log returns for selected stocks
log_returns = np.log(df[top_10] / df[top_10].shift(1)).dropna()
opt_weights = optimize_weights(log_returns)

# 4. Results
results = pd.DataFrame({'Ticker': top_10, 'Weight': opt_weights})
results['Weight'] = results['Weight'].map(lambda x: f"{x:.2%}")
print("\nOptimal Portfolio Allocation:")
print(results)

# Final Portfolio Stats
p_ret, p_vol, p_sharpe = portfolio_stats(opt_weights, log_returns)
print(f"\nExpected Annual Return: {p_ret:.2%}")
print(f"Expected Volatility: {p_vol:.2%}")
print(f"Sharpe Ratio: {p_sharpe:.2f}")

