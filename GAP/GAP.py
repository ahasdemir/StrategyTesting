import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_gap_strategy(ticker, start_date, end_date, filter_lookback=20, time_index=10, atr_length=20, atr_stop=3):

    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Gösterge Hesaplamaları
    # ATR Hesaplama
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=atr_length).mean()

    # Trend Filtresi (Donchian Channels)
    df['UpperChannel'] = df['High'].shift(1).rolling(window=filter_lookback).max()
    df['LowerChannel'] = df['Low'].shift(1).rolling(window=filter_lookback).min()

    # 3. Sinyal ve Backtest Değişkenleri
    df['Returns'] = 0.0
    trades = []
    
    # Backtest Döngüsü
    in_position = 0 # 0: Yok, 1: Long, -1: Short
    entry_price = 0
    stop_loss = 0
    hold_days = 0

    for i in range(1, len(df)):
        current_date = df.index[i]
        
        # Pozisyonda Değilsek Giriş Ara
        if in_position == 0:
            # LONG SETUP: Gap Up + Trend Filter
            if df['Low'].iloc[i] > df['High'].iloc[i-1]: # Gap Up
                if df['High'].iloc[i] > df['UpperChannel'].iloc[i]: # Trend Long
                    in_position = 1
                    entry_price = df['Open'].iloc[i]
                    # Stop: Giriş - (ATR * ATR_Stop)
                    stop_loss = entry_price - (df['ATR'].iloc[i] * atr_stop)
                    # Pattern Exit: Boşluğun bir tık altı
                    pattern_stop = df['High'].iloc[i-1]
                    stop_loss = max(stop_loss, pattern_stop)
                    hold_days = 0
                    
            # SHORT SETUP: Gap Down + Trend Filter
            elif df['High'].iloc[i] < df['Low'].iloc[i-1]: # Gap Down
                if df['Low'].iloc[i] < df['LowerChannel'].iloc[i]: # Trend Short
                    in_position = -1
                    entry_price = df['Open'].iloc[i]
                    # Stop: Giriş + (ATR * ATR_Stop)
                    stop_loss = entry_price + (df['ATR'].iloc[i] * atr_stop)
                    # Pattern Exit: Boşluğun bir tık üstü
                    pattern_stop = df['Low'].iloc[i-1]
                    stop_loss = min(stop_loss, pattern_stop)
                    hold_days = 0

        # Pozisyondaysak Çıkış Kontrolü
        elif in_position == 1: # Long Pozisyon
            hold_days += 1
            # Stop Loss Kontrol (Düşük fiyat stopun altına indi mi?)
            if df['Low'].iloc[i] <= stop_loss:
                pnl = (stop_loss - entry_price) / entry_price
                trades.append(pnl)
                in_position = 0
            # Time Exit Kontrol
            elif hold_days >= time_index:
                pnl = (df['Close'].iloc[i] - entry_price) / entry_price
                trades.append(pnl)
                in_position = 0
                
        elif in_position == -1: # Short Pozisyon
            hold_days += 1
            if df['High'].iloc[i] >= stop_loss:
                pnl = (entry_price - stop_loss) / entry_price
                trades.append(pnl)
                in_position = 0
            elif hold_days >= time_index:
                pnl = (entry_price - df['Close'].iloc[i]) / entry_price
                trades.append(pnl)
                in_position = 0

    # 4. Performans Metrikleri
    if not trades:
        return "İşlem gerçekleşmedi."
    
    trades = np.array(trades)
    win_rate = np.sum(trades > 0) / len(trades)
    cum_return = np.prod(1 + trades) - 1
    avg_trade = np.mean(trades)
    max_drawdown = np.min(trades) # Basit işlem bazlı drawdown

    print(f"--- {ticker} Strateji Sonuçları ---")
    print(f"Toplam İşlem: {len(trades)}")
    print(f"Kazanma Oranı (WR): %{win_rate*100:.2f}")
    print(f"Kümülatif Getiri: %{cum_return*100:.2f}")
    print(f"İşlem Başı Ortalama Getiri: %{avg_trade*100:.2f}")
    
    # Görselleştirme
    plt.figure(figsize=(10,5))
    plt.plot(np.cumprod(1 + trades))
    plt.title(f"{ticker} Strateji Özsermaye Eğrisi")
    plt.xlabel("İşlem Sayısı")
    plt.ylabel("Büyüme Katsayısı")
    plt.grid(True)
    plt.show()

# Örnek Kullanım: Türk Hava Yolları üzerinde test
backtest_gap_strategy("ASELS.IS", "2025-01-01", "2026-01-01")