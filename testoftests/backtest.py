import pandas as pd
import numpy as np
import yfinance as yf
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
tickers = pd.read_csv(os.path.join(script_dir, "tickers.csv"), skiprows=[1])["Ticker"].tolist()

def run_backtest(tickers, start_date, end_date, rebalance_freq=21):
    df = yf.download(tickers, start=start_date, end=end_date)["Close"]
    df = df.ffill().bfill()
    
    # Günlük basit getiriler (Gelecek getirileri hesaplamak için)
    returns_df = df.pct_change().dropna()
    
    # Backtest sonuçlarını tutacak listeler
    portfolio_returns = []
    dates = []
    
    # Backtest Başlangıcı: En az 1 yıllık veri (252 gün) biriktikten sonra başlar
    for i in range(252, len(df) - rebalance_freq, rebalance_freq):
        # Analiz penceresi (Geçmiş 1 yıl)
        lookback_data = df.iloc[i-252:i]
        
        # --- 1. Momentum Filtresi ---
        # 12-1 Ay Momentum (Son 1 ayı hariç tutarak son 1 yıla bakış)
        mom_scores = (lookback_data.iloc[-21] / lookback_data.iloc[0]) - 1
        top_tickers = mom_scores.sort_values(ascending=False).head(10).index.tolist()
        
        # --- 2. MPT Optimizasyonu (Seçilen 10 hisse için) ---
        selected_returns = lookback_data[top_tickers].pct_change().dropna()
        if selected_returns.empty: continue
            
        opt_weights = get_optimal_weights(selected_returns)
        
        # --- 3. Gelecek Getiriyi Hesapla ---
        # Bir sonraki rebalance dönemine kadar olan getiriler
        future_returns = returns_df[top_tickers].iloc[i : i + rebalance_freq]
        
        # Portföy getirisi = Ağırlıklar * Varlık Getirileri
        step_returns = (future_returns * opt_weights).sum(axis=1)
        
        portfolio_returns.extend(step_returns.tolist())
        dates.extend(future_returns.index.tolist())

    return pd.Series(portfolio_returns, index=dates)

def get_optimal_weights(returns):
    num_assets = len(returns.columns)
    mean_ret = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    def objective(weights):
        p_ret = np.sum(mean_ret * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        # Sharpe Oranı Maksimizasyonu (Negatifini minimize ediyoruz)
        return - (p_ret / p_vol) if p_vol != 0 else 0

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, 0.25) for _ in range(num_assets)) # Maks %25 ağırlık sınırı
    
    init_guess = np.array(num_assets * [1. / num_assets])
    res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x if res.success else init_guess


strat_returns = run_backtest(tickers, "2022-01-01", "2026-05-04")

# Performans Metrikleri
cum_returns = (1 + strat_returns).cumprod()
total_return = cum_returns.iloc[-1] - 1
annualized_return = (1 + total_return)**(252/len(strat_returns)) - 1
max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()

print(f"Toplam Getiri: %{total_return*100:.2f}")
print(f"Yıllık Bileşik Getiri (CAGR): %{annualized_return*100:.2f}")
print(f"Maksimum Kayıp (Drawdown): %{max_drawdown*100:.2f}")

# Grafik
plt.figure(figsize=(12, 6))
cum_returns.plot(title="Momentum + MPT Stratejisi Kümülatif Getiri")
plt.grid(True)
plt.show()