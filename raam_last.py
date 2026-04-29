import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import warnings
warnings.filterwarnings("ignore")
from bist_symbols import SECTORS

# ─────────────────────────────────────────
# KONFİGÜRASYON — buradan ayarlayın
# ─────────────────────────────────────────
UNIVERSE = {
    "GARAN.IS": "Garanti BBVA",
    "AKBNK.IS": "Akbank",
    "ISCTR.IS": "İş Bankası",
    "YKBNK.IS": "Yapı Kredi",
    "KCHOL.IS": "Koç Holding",
    "SAHOL.IS": "Sabancı Holding",
    "FROTO.IS": "Ford Otosan",
    "TOASO.IS": "Tofaş",
    "EREGL.IS": "Ereğli Demir Çelik",
    "TUPRS.IS": "Tüpraş",
    "PETKM.IS": "Petkim",
    "THYAO.IS": "Türk Hava Yolları",
    "PGSUS.IS": "Pegasus",
    "BIMAS.IS": "BİM Mağazalar",
    "MGROS.IS": "Migros",
    "TCELL.IS": "Turkcell",
    "ASELS.IS": "Aselsan",
    "SISE.IS":  "Şişe Cam",
    "ENKAI.IS": "Enka İnşaat",
    "EKGYO.IS": "Emlak Konut GYO",
}
for sector in SECTORS:
    for sym in SECTORS[sector]:
        UNIVERSE[sym] = sym  # Sembolün kendisi ad olarak eklenir (böylece sektör dışı hisseler de gösterilir)

CASH = "XU030.IS"

CFG = {
    "top_n":        5,       # kaç hisse seç
    "wM":           0.5,     # momentum ağırlığı
    "wV":           0.3,     # volatilite ağırlığı
    "wC":           0.2,     # korelasyon ağırlığı
    "lambda_ewma":  0.94,    # RiskMetrics lambda
    "mom_days":     84,      # ~4 ay momentum penceresi
    "corr_days":    84,      # ~4 ay korelasyon penceresi
    "atr_period":   42,      # ATR periyodu (21=hızlı, 42=orta, 63=yavaş)
    # Trend modu:
    #   "multiplier" → Long=+bonus, Short=-malus (eski)
    #   "filter"     → Short olan hisseler doğrudan nakit'e çevrilir
    #   "score_only" → Trend tamamen görmezden gelinir (sadece M+V+C)
    "trend_mode":   "filter",
    "trend_bonus":  0.15,    # multiplier modunda ±bonus oranı
}

# ─────────────────────────────────────────
# VERİ
# ─────────────────────────────────────────
def download_data():
    tickers = list(UNIVERSE.keys()) + [CASH]
    raw   = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
    close = raw["Close"].ffill()
    high  = raw["High"].ffill()
    low   = raw["Low"].ffill()
    valid = close.columns[close.isna().mean() < 0.15]
    return close[valid], high[valid], low[valid]

# ─────────────────────────────────────────
# FAKTÖRLER
# ─────────────────────────────────────────
def compute_momentum(close):
    return close.pct_change(CFG["mom_days"]).iloc[-1]

def compute_volatility(close):
    ret  = close.pct_change().fillna(0)
    span = 2 / (1 - CFG["lambda_ewma"]) - 1
    var  = (ret ** 2).ewm(span=span, adjust=False).mean()
    return np.sqrt(var * 252).ewm(span=10).mean().iloc[-1]

def compute_avg_correlation(close):
    ret    = close.pct_change().dropna().tail(CFG["corr_days"])
    corr_m = ret.corr()
    n      = len(corr_m)
    return (corr_m.sum() - 1) / (n - 1)

def compute_atr_trend(close, high, low):
    p   = CFG["atr_period"]
    signals, bands = {}, {}
    for col in close.columns:
        h, l, c = high[col], low[col], close[col]
        pc  = c.shift(1)
        tr  = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        atr = tr.rolling(p).mean()

        upper = h.rolling(p).max() + atr
        lower = l.rolling(p).min() + atr

        sig = pd.Series(np.nan, index=c.index)
        sig[h > upper.shift(1)] =  1.0   # Long
        sig[l < lower.shift(1)] = -1.0   # Short
        sig = sig.ffill().fillna(1.0)

        signals[col] = sig.iloc[-1]
        bands[col] = {
            "upper":    round(upper.iloc[-1], 2),
            "lower":    round(lower.iloc[-1], 2),
            "price":    round(c.iloc[-1], 2),
            "signal":   "🟢 Long" if sig.iloc[-1] == 1.0 else "🔴 Short",
        }
    return pd.Series(signals), bands

# ─────────────────────────────────────────
# SIRALAMA + PORTFÖY
# ─────────────────────────────────────────
def generate_signal(close, high, low):
    assets = [c for c in UNIVERSE if c in close.columns]

    mom          = compute_momentum(close[assets])
    vol          = compute_volatility(close[assets])
    corr         = compute_avg_correlation(close[assets])
    trend, bands = compute_atr_trend(close[assets], high[assets], low[assets])

    common = mom.index.intersection(vol.index).intersection(
             corr.index).intersection(trend.index)
    mom, vol, corr, trend = (
        mom[common], vol[common], corr[common], trend[common]
    )
    n = len(common)

    rank_m = mom.rank(ascending=True)
    rank_v = vol.rank(ascending=False)
    rank_c = corr.rank(ascending=False)

    wM, wV, wC = CFG["wM"], CFG["wV"], CFG["wC"]
    base = (
        wM * (n + 1 - rank_m) +
        wV * (n + 1 - rank_v) +
        wC * (n + 1 - rank_c)
    )

    mode = CFG["trend_mode"]
    if mode == "multiplier":
        bonus = CFG["trend_bonus"]
        adj   = trend.map({1.0: -bonus, -1.0: bonus}).fillna(0)
        score = base * (1 + adj)
    elif mode == "filter":
        score = base.copy()   # sıralama saf M+V+C, trend filtre olarak sonra uygulanır
    else:  # score_only
        score = base.copy()

    ranked = score.sort_values()
    top5   = ranked.head(CFG["top_n"]).index.tolist()

    allocation = {}
    per_w  = 1.0 / CFG["top_n"]
    cash_w = 0.0

    for ticker in top5:
        neg_mom   = mom[ticker] <= 0
        short_sig = (mode == "filter") and (trend[ticker] == -1.0)

        if neg_mom or short_sig:
            cash_w += per_w
        else:
            allocation[ticker] = per_w

    if cash_w > 0:
        allocation[CASH] = allocation.get(CASH, 0) + cash_w

    return allocation, score, base, mom, vol, corr, trend, ranked, bands

# ─────────────────────────────────────────
# RAPOR
# ─────────────────────────────────────────
def print_report(allocation, score, base, mom, vol, corr, trend, ranked, bands, close):
    today     = date.today().strftime("%d %B %Y")
    last_data = close.index[-1].strftime("%d %B %Y")
    mode_tr   = {"multiplier": "Çarpan", "filter": "Filtre", "score_only": "Yok"}
    mode_str  = mode_tr.get(CFG["trend_mode"], CFG["trend_mode"])

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║            RAAM — BIST AYLIK SİNYAL RAPORU                  ║
║  Rapor Tarihi  : {today:<43}║
║  Son Veri      : {last_data:<43}║
║  ATR Periyodu  : {CFG["atr_period"]:<43}║
║  Trend Modu    : {mode_str:<43}║
╚══════════════════════════════════════════════════════════════╝
""")

    # Portföy
    print("📊 PORTFÖY DAĞILIMI (Bu Ay):\n")
    print(f"  {'Ticker':<12} {'Şirket':<26} {'Ağırlık':>8}  Durum")
    print(f"  {'-'*65}")
    total_w = 0
    for ticker, w in sorted(allocation.items(), key=lambda x: -x[1]):
        name   = UNIVERSE.get(ticker, "Nakit (XU030)")
        if ticker == CASH:
            status = "💵 Nakit"
        else:
            t_str  = bands.get(ticker, {}).get("signal", "")
            status = f"✅ Mom(+) {t_str}"
        print(f"  {ticker:<12} {name:<26} {w:>7.1%}  {status}")
        total_w += w
    print(f"  {'─'*65}")
    print(f"  {'TOPLAM':<38} {total_w:>7.1%}")

    # Skor tablosu
    long_n  = int((trend == 1.0).sum())
    short_n = int((trend == -1.0).sum())
    print(f"\n📈 TÜM VARLIKLAR — Skor Tablosu  "
          f"[ATR({CFG['atr_period']}): {long_n}🟢 Long | {short_n}🔴 Short]\n")
    print(f"  {'#':<3} {'Ticker':<12} {'Şirket':<22} {'Skor':>6} {'Baz':>6} "
          f"{'Mom':>7} {'Vol':>7} {'Corr':>7}  {'Trend':<14} {'Fiyat':>10} Seçildi")
    print(f"  {'-'*115}")

    for i, (ticker, sc) in enumerate(ranked.items(), 1):
        name  = UNIVERSE.get(ticker, ticker)[:21]
        b_v   = base.get(ticker, np.nan)
        m_v   = mom.get(ticker, np.nan)
        v_v   = vol.get(ticker, np.nan)
        c_v   = corr.get(ticker, np.nan)
        t_str = bands.get(ticker, {}).get("signal", "N/A")
        price = bands.get(ticker, {}).get("price", np.nan)

        in_port  = ticker in allocation and ticker != CASH
        to_cash  = (i <= CFG["top_n"]) and (ticker not in allocation or ticker == CASH)
        neg_flag = " ⚠️neg.mom" if m_v <= 0 and i <= CFG["top_n"] else ""
        srt_flag = " 🔴→Nakit"  if trend.get(ticker, 1) == -1.0 and i <= CFG["top_n"] \
                                    and CFG["trend_mode"] == "filter" else ""
        mark = "⭐" if in_port else ("→💵" if to_cash else "")

        print(f"  {i:<3} {ticker:<12} {name:<22} {sc:>6.2f} {b_v:>6.2f} "
              f"{m_v:>7.1%} {v_v:>7.1%} {c_v:>7.3f}  {t_str:<14} {price:>10,.2f}"
              f"  {mark}{neg_flag}{srt_flag}")

    # ATR bant detayı (seçilen hisseler için)
    print(f"\n🔍 ATR({CFG['atr_period']}) Bant Detayı (Top {CFG['top_n']}):\n")
    top_tickers = ranked.head(CFG["top_n"]).index
    print(f"  {'Ticker':<12} {'Fiyat':>10} {'Upper Band':>12} {'Lower Band':>12} {'Sinyal'}")
    print(f"  {'-'*60}")
    for tk in top_tickers:
        b = bands.get(tk, {})
        print(f"  {tk:<12} {b.get('price',0):>10,.2f} {b.get('upper',0):>12,.2f} "
              f"{b.get('lower',0):>12,.2f} {b.get('signal','')}")

    print(f"""
📌 UYGULAMA:
  • Sinyal: Ayın son işlem günü hesapla
  • Uygula: Ertesi ayın ilk işlem günü açılışında
  • Trend modu '{CFG["trend_mode"]}': Short sinyalli hisseler → Nakit'e çevrildi
  • İşlem maliyeti modele dahil değil (~%0.1-0.2)

⚠️  Yatırım tavsiyesi değildir.
""")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Veri indiriliyor (son 2 yıl)...")
    close, high, low = download_data()

    print("Faktörler hesaplanıyor...\n")
    allocation, score, base, mom, vol, corr, trend, ranked, bands = \
        generate_signal(close, high, low)

    print_report(allocation, score, base, mom, vol, corr, trend, ranked, bands, close)