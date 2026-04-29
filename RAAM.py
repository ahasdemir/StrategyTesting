import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 1. UNIVERSE — BIST 30 Blue Chips (sektör çeşitlendirmesi)
#    Tüm tickerlar Yahoo Finance'de .IS uzantısıyla çalışır
# ─────────────────────────────────────────
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

# Nakit karşılığı: XU030.IS (BIST 30 endeksi)
# İdeal: kısa vadeli devlet tahvili endeksi
CASH       = "XU030.IS"
TOP_N      = 5
WEIGHTS    = {"wM": 0.5, "wV": 0.3, "wC": 0.2}
LAMBDA     = 0.94
ATR_PERIOD = 42
MOM_MONTHS = 4
NA_THRESH  = 0.20   # %20'den fazla eksik veri olan tickerları çıkar

# ─────────────────────────────────────────
# 2. VERİ İNDİR
# ─────────────────────────────────────────
def download_data(start="2012-01-01"):
    tickers = list(UNIVERSE.keys()) + [CASH]
    print(f"Yahoo Finance'den {len(tickers)} ticker indiriliyor...")
    raw   = yf.download(tickers, start=start, auto_adjust=True, progress=True)
    close = raw["Close"].ffill()
    high  = raw["High"].ffill()
    low   = raw["Low"].ffill()

    valid = close.columns[close.isna().mean() < NA_THRESH]
    dropped = set(tickers) - set(valid)
    if dropped:
        print(f"  ⚠ Yetersiz veri nedeniyle çıkarılan: {dropped}")
    print(f"  ✓ Kullanılan ticker sayısı: {len(valid)} → {list(valid)}")
    return close[valid], high[valid], low[valid]

# ─────────────────────────────────────────
# 3. MOMENTUM (M)
# ─────────────────────────────────────────
def compute_momentum(close, months=MOM_MONTHS):
    return close.pct_change(months * 21)

# ─────────────────────────────────────────
# 4. VOLATİLİTE (V) — RiskMetrics EWMA λ=0.94
# ─────────────────────────────────────────
def compute_volatility(close, lam=LAMBDA, smooth=10):
    ret = close.pct_change().fillna(0)
    var = ret ** 2
    # Vectorized EWMA — pandas ewm ile λ'ya karşılık gelen span
    # λ = 1 - 2/(span+1) → span = 2/(1-λ) - 1
    span = 2 / (1 - lam) - 1   # λ=0.94 → span≈32.3
    var_ewm = var.ewm(span=span, adjust=False).mean()
    vol = np.sqrt(var_ewm * 252)
    return vol.ewm(span=smooth).mean()

# ─────────────────────────────────────────
# 5. ORTALAMA KORELASYON (C) — 4 aylık rolling
#    Vektörize versiyon (hızlı)
# ─────────────────────────────────────────
def compute_avg_correlation(close, months=MOM_MONTHS):
    period  = months * 21
    ret     = close.pct_change()
    cols    = close.columns.tolist()
    n       = len(cols)
    avg_corr = pd.DataFrame(np.nan, index=close.index, columns=cols)

    for i in range(period, len(close), 1):
        window = ret.iloc[i - period : i]
        if window.isna().all().any():
            continue
        corr_m = window.corr()
        # Diyagonal hariç ortalama: (toplam - n) / (n*(n-1))
        row_sum = corr_m.sum() - 1.0          # her asset için kendi korelasyonu çıkar
        avg_corr.iloc[i] = row_sum / (n - 1)

    return avg_corr.astype(float)

# ─────────────────────────────────────────
# 6. ATR TREND/BREAKOUT (T) — Vektörize
# ─────────────────────────────────────────
def compute_atr_trend(close, high, low, period=ATR_PERIOD):
    pc  = close.shift(1)
    tr  = pd.concat([
        (high - low),
        (high - pc).abs(),
        (low  - pc).abs()
    ], axis=1, keys=["hl","hc","lc"])

    # Multi-level columns → per-asset max
    atr_dict = {}
    sig_dict = {}
    for col in close.columns:
        h, l, c = high[col], low[col], close[col]
        pc_col  = c.shift(1)
        tr_col  = pd.concat([h-l, (h-pc_col).abs(), (l-pc_col).abs()], axis=1).max(axis=1)
        atr     = tr_col.rolling(period).mean()

        upper = h.rolling(period).max() + atr   # RAAM: her iki banda da ekle
        lower = l.rolling(period).min() + atr

        sig = pd.Series(2.0, index=c.index)
        long_mask  = h.shift(1) > upper.shift(1)
        short_mask = l.shift(1) < lower.shift(1)
        sig[short_mask] = -2.0
        sig[long_mask]  =  2.0
        sig = sig.ffill()

        atr_dict[col] = atr
        sig_dict[col] = sig

    return pd.DataFrame(sig_dict)

# ─────────────────────────────────────────
# 7. SIRALAMA MODELİ — Aylık
# ─────────────────────────────────────────
def rank_assets(mom, vol, corr, trend, close):
    asset_cols = [c for c in UNIVERSE.keys() if c in close.columns]
    cash_col   = CASH if CASH in close.columns else None
    all_cols   = asset_cols + ([cash_col] if cash_col else [])
    month_ends = close.resample("ME").last().index

    wM, wV, wC = WEIGHTS["wM"], WEIGHTS["wV"], WEIGHTS["wC"]
    records = []

    for date in month_ends:
        if date not in mom.index:
            continue

        m = mom.loc[date,  asset_cols].dropna()
        v = vol.loc[date,  asset_cols].dropna()
        c = corr.loc[date, asset_cols].dropna()
        t = trend.loc[date, asset_cols].fillna(2.0)

        common = m.index.intersection(v.index).intersection(c.index)
        if len(common) < TOP_N + 1:
            continue

        m, v, c, t = m[common], v[common], c[common], t[common]
        n = len(common)

        rank_m = m.rank(ascending=True,  na_option="bottom")
        rank_v = v.rank(ascending=False, na_option="bottom")
        rank_c = c.rank(ascending=False, na_option="bottom")

        score = (
            wM * (n + 1 - rank_m) +
            wV * (n + 1 - rank_v) +
            wC * (n + 1 - rank_c)
        ) * t

        selected = score.nsmallest(TOP_N).index.tolist()

        row = {col: 0.0 for col in all_cols}
        cash_w = 0.0
        per_w  = 1.0 / TOP_N

        for ticker in selected:
            if m.get(ticker, 0) > 0:
                row[ticker] = per_w
            else:
                cash_w += per_w

        if cash_col:
            row[cash_col] = row.get(cash_col, 0.0) + cash_w
        row["date"] = date
        records.append(row)

    df = pd.DataFrame(records).set_index("date").fillna(0.0)
    return df

# ─────────────────────────────────────────
# 8. BACKTEST
# ─────────────────────────────────────────
def backtest(close, alloc_df):
    daily_ret = close.pct_change()
    cols      = [c for c in alloc_df.columns if c in daily_ret.columns]
    dates     = alloc_df.index.tolist()
    series    = []

    for i in range(len(dates) - 1):
        mask   = (daily_ret.index > dates[i]) & (daily_ret.index <= dates[i+1])
        period = daily_ret.loc[mask, cols]
        w      = alloc_df.loc[dates[i], cols]
        series.append(period.dot(w))

    return pd.concat(series).sort_index()

# ─────────────────────────────────────────
# 9. PERFORMANS
# ─────────────────────────────────────────
def performance_summary(returns, name):
    r  = returns.dropna()
    cum = (1 + r).cumprod()
    ny  = len(r) / 252
    tot = cum.iloc[-1] - 1
    cagr    = (1 + tot) ** (1 / ny) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = cagr / ann_vol if ann_vol else np.nan
    dd      = cum / cum.cummax() - 1
    max_dd  = dd.min()
    calmar  = cagr / abs(max_dd) if max_dd else np.nan
    win_rate = (r > 0).mean()

    print(f"\n{'━'*50}")
    print(f"  {name}")
    print(f"{'━'*50}")
    print(f"  Toplam Getiri (TRY nominal) : {tot:>10.1%}")
    print(f"  CAGR                        : {cagr:>10.1%}")
    print(f"  Yıllık Volatilite           : {ann_vol:>10.1%}")
    print(f"  Sharpe Oranı                : {sharpe:>10.2f}")
    print(f"  Maks. Drawdown              : {max_dd:>10.1%}")
    print(f"  Calmar Oranı                : {calmar:>10.2f}")
    print(f"  Kazanma Oranı (günlük)      : {win_rate:>10.1%}")
    return cum

# ─────────────────────────────────────────
# 10. PLOT
# ─────────────────────────────────────────
def plot_results(raam_cum, bench_cum, alloc_df, close):
    asset_cols = [c for c in UNIVERSE.keys() if c in alloc_df.columns]
    cash_col   = CASH if CASH in alloc_df.columns else None
    plot_cols  = asset_cols + ([cash_col] if cash_col else [])

    fig, axes = plt.subplots(3, 1, figsize=(15, 13))
    fig.suptitle("RAAM — Borsa İstanbul (TRY Nominal)", fontsize=14, fontweight="bold")

    # 1) Equity curves
    ax = axes[0]
    raam_cum.plot(ax=ax, label="RAAM-BIST", color="#1f77b4", lw=2.5)
    if bench_cum is not None:
        bench_cum.plot(ax=ax, label="XU100 Benchmark", color="#d62728", lw=1.8, ls="--")
    ax.set_title("Birikim Eğrisi (1 = Başlangıç Değeri)")
    ax.legend(); ax.set_ylabel("Büyüme Katsayısı"); ax.grid(alpha=0.3)

    # 2) Drawdown
    ax2 = axes[1]
    dd = raam_cum / raam_cum.cummax() - 1
    ax2.fill_between(dd.index, dd, 0, color="#1f77b4", alpha=0.4)
    dd.plot(ax=ax2, color="#1f77b4", lw=1.5)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Düşüş %"); ax2.grid(alpha=0.3)

    # 3) Allocation stacked area
    ax3 = axes[2]
    labels = {k: UNIVERSE.get(k, "Nakit") for k in plot_cols}
    alloc_plot = alloc_df[[c for c in plot_cols if c in alloc_df.columns]]
    alloc_plot = alloc_plot.rename(columns=labels)
    alloc_plot.plot.area(ax=ax3, stacked=True, linewidth=0, cmap="tab20")
    ax3.set_title("Aylık Portföy Dağılımı")
    ax3.set_ylabel("Ağırlık"); ax3.grid(alpha=0.3)
    ax3.legend(loc="upper left", fontsize=7, ncol=4)

    plt.tight_layout()
    plt.savefig("raam_bist_results.png", dpi=150, bbox_inches="tight")
    print("\n  📈 Grafik 'raam_bist_results.png' olarak kaydedildi.")
    plt.show()

# ─────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    close, high, low = download_data(start="2012-01-01")

    asset_cols = [c for c in UNIVERSE if c in close.columns]
    print(f"\nEvrende {len(asset_cols)} hisse + nakit proxy")

    print("Momentum hesaplanıyor...")
    mom = compute_momentum(close)

    print("Volatilite hesaplanıyor (EWMA)...")
    vol = compute_volatility(close)

    print("Korelasyon hesaplanıyor (bu birkaç dakika sürebilir)...")
    corr = compute_avg_correlation(close)

    print("ATR Trend/Breakout hesaplanıyor...")
    trend = compute_atr_trend(close, high, low)

    print("Sıralama modeli çalışıyor...")
    alloc_df = rank_assets(mom, vol, corr, trend, close)

    print("Backtest yapılıyor...")
    port_ret = backtest(close, alloc_df)

    # --- Performans ---
    raam_cum  = performance_summary(port_ret, "RAAM-BIST (TRY Nominal)")
    bench_cum = None
    if "XU100.IS" in close.columns:
        xu100_ret = close["XU100.IS"].pct_change().dropna()
        xu100_ret = xu100_ret[xu100_ret.index.isin(port_ret.index)]
        bench_cum = performance_summary(xu100_ret, "XU100 Benchmark (TRY)")
    else:
        # XU100 ayrıca indir (evren dışında)
        xu100 = yf.download("XU100.IS", start="2012-01-01",
                            auto_adjust=True, progress=False)["Close"].ffill()
        xu100_ret = xu100.squeeze().pct_change().dropna()
        xu100_ret = xu100_ret[xu100_ret.index.isin(port_ret.index)]
        bench_cum = (1 + xu100_ret).cumprod()
        bench_cum = bench_cum / bench_cum.iloc[0]
        performance_summary(xu100_ret, "XU100 Benchmark (TRY)")

    # --- Son ay dağılımı ---
    print("\n📊 Son Ay Portföy Dağılımı:")
    latest = alloc_df.iloc[-1].sort_values(ascending=False)
    for ticker, w in latest[latest > 0.005].items():
        name = UNIVERSE.get(ticker, "Nakit (XU030)")
        print(f"  {ticker:12s}  {name:25s}  {w:.1%}")

    plot_results(raam_cum, bench_cum, alloc_df, close)