import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import cvxpy as cp


def fetch_data(ticker, start_date="2025-01-01", end_date="2025-12-31"):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.dropna()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def run_wmamr(tickers, start_date, end_date, w=5, epsilon=0.9):
    """
    Implements the WMAMR strategy from image_57ef3b.png.
    
    Parameters:
    - w: Window size for the moving average.
    - epsilon: The mean reversion threshold (control parameter).
    """
    # 1. Fetch Adjusted Close data
    df = yf.download(tickers, start=start_date, end=end_date)['Close']
    df = df.dropna()
    
    n_assets = len(tickers)
    n_days = len(df)
    
    # Initialize weights (Uniform/Equal Weight)
    b = np.ones(n_assets) / n_assets
    portfolio_value = [1.0]
    
    # 2. Daily Portfolio Management
    for t in range(w, n_days - 1):
        # Current and historical prices
        current_prices = df.iloc[t].values
        window = df.iloc[t-w+1 : t+1].values
        
        # Step A: Prediction (Formula #11 in image_57ef3b.png)
        # x_hat = (1/w) * (1 + 1/x_t + ... + 1/product(x))
        # This effectively translates to: Moving Average of past prices / Current Price
        x_hat = np.mean(window, axis=0) / current_prices
        
        # Step B: Update Rule (Optimization from image_57ef3b.png)
        # Goal: minimize 0.5 * ||b_next - b||^2 
        # Subject to: b_next @ x_hat <= epsilon, sum(b_next) = 1, b_next >= 0
        b_next = cp.Variable(n_assets)
        objective = cp.Minimize(0.5 * cp.sum_squares(b_next - b))
        constraints = [
            cp.sum(b_next) == 1,
            b_next >= 0,
            b_next @ x_hat <= epsilon
        ]
        
        prob = cp.Problem(objective, constraints)
        try:
            # Solving the quadratic program
            prob.solve(solver=cp.OSQP, silent=True)
            if b_next.value is not None:
                b = b_next.value
        except Exception:
            # If optimization fails (e.g., epsilon is too restrictive), keep current weights
            pass
            
        # 3. Performance Tracking
        next_prices = df.iloc[t+1].values
        relative_return = next_prices / current_prices
        daily_perf = np.dot(b, relative_return)
        portfolio_value.append(portfolio_value[-1] * daily_perf)
        
    return pd.Series(portfolio_value, index=df.index[w:], name="WMAMR_Wealth")

BIST50 = [
    "AKBNK.IS","ARCLK.IS","ASELS.IS","BIMAS.IS","EKGYO.IS",
    "EREGL.IS","FROTO.IS","GARAN.IS","HALKB.IS","ISCTR.IS",
    "KCHOL.IS","KRDMD.IS","MGROS.IS","PETKM.IS",
    "PGSUS.IS","SAHOL.IS","SISE.IS","TCELL.IS","THYAO.IS",
    "TUPRS.IS","VAKBN.IS","VESTL.IS","YKBNK.IS","AEFES.IS",
    "AKSEN.IS","ALARK.IS","ENKAI.IS","GUBRF.IS","TTKOM.IS",
] 

start_date, end_date = "2026-01-01", "2026-04-30"
# --- Example Usage ---
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
results = run_wmamr(BIST50, start_date, end_date, w=5, epsilon=0.95)
xu100 = yf.download("XU100.IS", start=start_date, end=end_date)['Close']
xu100 = xu100 / xu100.iloc[0]  # Normalize to start at 1

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(results, label="WMAMR Strategy", color='blue')
plt.plot(xu100, label="XU100", linestyle='--')
plt.title("WMAMR Strategy Performance")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()

