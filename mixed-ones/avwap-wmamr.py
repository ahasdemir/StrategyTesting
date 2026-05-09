import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def calculate_avwap(df, anchor_idx=0):
    """Belirli bir indeksten (anchor) itibaren AVWAP hesaplar."""
    pv = (df['Close'] * df['Volume']).iloc[anchor_idx:].cumsum()
    v = df['Volume'].iloc[anchor_idx:].cumsum()
    avwap = pv / v
    # Eksik günleri (anchor öncesi) doldur
    return avwap.reindex(df.index).ffill()

def run_av_wmamr(tickers, start_date, end_date, epsilon=0.95):
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Çoklu sembollerde yfinance yapısı (Close ve Volume ayrımı)
    adj_close = data['Close']
    volume = data['Volume']
    
    n_assets = len(tickers)
    b = np.ones(n_assets) / n_assets  # Başlangıç ağırlıkları
    wealth = [1.0]

    # Her varlık için AVWAP hesapla (Örneğin verinin başından itibaren)
    avwaps = pd.DataFrame()
    for tkr in tickers:
        temp_df = pd.DataFrame({'Close': adj_close[tkr], 'Volume': volume[tkr]})
        avwaps[tkr] = calculate_avwap(temp_df, anchor_idx=0)

    # Simülasyon (Strateji 10. günden başlasın)
    for t in range(10, len(adj_close) - 1):
        p_t = adj_close.iloc[t].values
        avwap_t = avwaps.iloc[t].values
        
        # Predictor: Fiyatın AVWAP'a göre oranı
        x_hat = avwap_t / p_t
        
        # WMAMR Optimizasyonu (image_57ef3b.png'deki kural)
        b_next = cp.Variable(n_assets)
        obj = cp.Minimize(0.5 * cp.sum_squares(b_next - b))
        constraints = [
            cp.sum(b_next) == 1,
            b_next >= 0,
            b_next @ x_hat <= epsilon
        ]
        
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.OSQP, silent=True)
            if b_next.value is not None:
                b = b_next.value
        except:
            pass # Çözüm bulunamazsa mevcut ağırlıkla devam et

        # Getiri hesaplama
        r_t = adj_close.iloc[t+1].values / p_t
        wealth.append(wealth[-1] * np.dot(b, r_t))

    return pd.Series(wealth, index=adj_close.index[10:])

BIST50 = [
    "AKBNK.IS","ARCLK.IS","ASELS.IS","BIMAS.IS","EKGYO.IS",
    "EREGL.IS","FROTO.IS","GARAN.IS","HALKB.IS","ISCTR.IS",
    "KCHOL.IS","KRDMD.IS","MGROS.IS","PETKM.IS",
    "PGSUS.IS","SAHOL.IS","SISE.IS","TCELL.IS","THYAO.IS",
    "TUPRS.IS","VAKBN.IS","VESTL.IS","YKBNK.IS","AEFES.IS",
    "AKSEN.IS","ALARK.IS","ENKAI.IS","GUBRF.IS","TTKOM.IS",
] 

# Örnek Kullanım: BIST 100 lokomotif hisseler
hisseler = ["THYAO.IS", "EREGL.IS", "ASELS.IS", "KCHOL.IS"]

UNIVERSE = {
    # Bankacılık
    "GARAN.IS":  "Garanti BBVA",
    "AKBNK.IS":  "Akbank",
    "ISCTR.IS":  "İş Bankası",
    "YKBNK.IS":  "Yapı Kredi",
    # Holding
    "KCHOL.IS":  "Koç Holding",
    "SAHOL.IS":  "Sabancı Holding",
    # Sanayi / Otomotiv
    "FROTO.IS":  "Ford Otosan",
    "TOASO.IS":  "Tofaş",
    "EREGL.IS":  "Ereğli Demir Çelik",
    # Enerji / Petrokimya
    "TUPRS.IS":  "Tüpraş",
    "PETKM.IS":  "Petkim",
    # Havacılık / Ulaşım
    "THYAO.IS":  "Türk Hava Yolları",
    "PGSUS.IS":  "Pegasus",
    # Perakende / Gıda
    "BIMAS.IS":  "BİM Mağazalar",
    "MGROS.IS":  "Migros",
    # Telekom / Teknoloji
    "TCELL.IS":  "Turkcell",
    "ASELS.IS":  "Aselsan",
    # Cam / Kimya
    "SISE.IS":   "Şişe Cam",
    # İnşaat / GYO
    "ENKAI.IS":  "Enka İnşaat",
    "EKGYO.IS":  "Emlak Konut GYO",
}

sonuc = run_av_wmamr(list(UNIVERSE.keys())[:5], "2025-01-01", "2026-01-01")

plt.figure(figsize=(12, 6))
plt.plot(sonuc.index, sonuc.values, label='AVWAP-WMAMR Stratejisi')
plt.title('AVWAP-WMAMR Stratejisi Getirisi')
plt.xlabel('Tarih')
plt.ylabel('Toplam Getiri')
plt.legend()
plt.grid()
plt.show()