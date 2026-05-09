import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_smash_day_b(ticker, start_date, end_date, trend_index=20, time_index=5, atr_length=20, atr_stop=6):
    # 1. Veri Çekme
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Gösterge Hesaplamaları
    # ATR (Stop Loss için)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=atr_length).mean()

    # 3. Backtest Değişkenleri
    trades = []
    in_position = 0 # 0: Nakit, 1: Long, -1: Short
    entry_price = 0
    stop_loss = 0
    hold_days = 0
    entry_bar_idx = 0

    # Döngü (Trend_Index kadar geriye bakabilmek için oradan başlıyoruz)
    for i in range(max(trend_index, 2), len(df)):
        
        # POZİSYONDA DEĞİLSEK (Giriş Ara)
        if in_position == 0:
            # --- LONG SETUP & FILTER ---
            # Setup: Dünkü kapanış, önceki günün düşüğünden küçük (Smash Day)
            setup_long = df['Close'].iloc[i-1] < df['Low'].iloc[i-2]
            # Filter: Dünkü kapanış, n gün önceki kapanıştan büyük (Yükselen Trend)
            filter_long = df['Close'].iloc[i-1] > df['Close'].iloc[i-trend_index]
            
            # Entry: Bugünün yükseği, dünkü yükseği geçerse (Buy Stop)
            if setup_long and filter_long:
                buy_stop_level = df['High'].iloc[i-1]
                if df['High'].iloc[i] >= buy_stop_level:
                    in_position = 1
                    entry_price = buy_stop_level # Stop tetiklendiği yer
                    entry_bar_idx = i
                    hold_days = 0
                    
                    # Stop Loss: ATR bazlı veya Quick Exit (Giriş barı ve bir önceki barın en düşüğü)
                    atr_sl = entry_price - (df['ATR'].iloc[i] * atr_stop)
                    quick_sl = min(df['Low'].iloc[i], df['Low'].iloc[i-1])
                    stop_loss = max(atr_sl, quick_sl)

            # --- SHORT SETUP & FILTER ---
            setup_short = df['Close'].iloc[i-1] > df['High'].iloc[i-2]
            filter_short = df['Close'].iloc[i-1] < df['Close'].iloc[i-trend_index]
            
            if setup_short and filter_short:
                sell_stop_level = df['Low'].iloc[i-1]
                if df['Low'].iloc[i] <= sell_stop_level:
                    in_position = -1
                    entry_price = sell_stop_level
                    entry_bar_idx = i
                    hold_days = 0
                    
                    # Stop Loss: ATR veya Quick Exit (Giriş barı ve bir önceki barın en yükseği)
                    atr_sl = entry_price + (df['ATR'].iloc[i] * atr_stop)
                    quick_sl = max(df['High'].iloc[i], df['High'].iloc[i-1])
                    stop_loss = min(atr_sl, quick_sl)

        # POZİSYONDAYSAK (Çıkış Kontrolü)
        elif in_position == 1:
            hold_days += 1
            # Stop Loss Kontrol
            if df['Low'].iloc[i] <= stop_loss:
                trades.append((stop_loss - entry_price) / entry_price)
                in_position = 0
            # Time Exit Kontrol
            elif hold_days >= time_index:
                trades.append((df['Close'].iloc[i] - entry_price) / entry_price)
                in_position = 0

        elif in_position == -1:
            hold_days += 1
            if df['High'].iloc[i] >= stop_loss:
                trades.append((entry_price - stop_loss) / entry_price)
                in_position = 0
            elif hold_days >= time_index:
                trades.append((entry_price - df['Close'].iloc[i]) / entry_price)
                in_position = 0

    # 4. Performans Analizi
    if not trades: return "İşlem açılmadı."
    
    trades = np.array(trades)
    wr = np.sum(trades > 0) / len(trades)
    cum_ret = np.prod(1 + trades) - 1
    
    print(f"--- {ticker} Smash Day Type B Sonuçları ---")
    print(f"Trend Bakış (Trend_Index): {trend_index} gün")
    print(f"Tutma Süresi (Time_Index): {time_index} gün")
    print(f"Toplam İşlem: {len(trades)}")
    print(f"Kazanma Oranı (WR): %{wr*100:.2f}")
    print(f"Kümülatif Getiri: %{cum_ret*100:.2f}")

    # Grafik
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumprod(1 + trades))
    plt.title(f"{ticker} Strateji Gelişimi (Equity Curve)")
    plt.grid(True)
    plt.show()

# Örnek: BIST 100 Endeksi (XU100.IS) üzerinden test
backtest_smash_day_b("XU100.IS", "2025-01-01", "2026-01-01", trend_index=10, time_index=5)