import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from bist_symbols import SECTORS
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# KONFİGÜRASYON
# ───────────────────────────────────E──────
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

CASH = "XU030.IS"

CFG = {
    "top_n":       5,
    "wM":          0.5,
    "wV":          0.3,
    "wC":          0.2,
    "lambda_ewma": 0.94,
    "mom_days":    84,
    "corr_days":   84,
    "atr_period":  42,
    "trend_mode":  "filter",   # "filter" | "multiplier" | "score_only"
    "trend_bonus": 0.15,
    "start_date":  "2014-01-01",
    "na_thresh":   0.15,
}

# ─────────────────────────────────────────
# VERİ
# ─────────────────────────────────────────
def download_data(start=None):
    tickers = list(UNIVERSE.keys()) + [CASH, "XU100.IS"]
    s = start or CFG["start_date"]
    print(f"  Veri indiriliyor: {s} → bugün")
    raw   = yf.download(tickers, start=s, auto_adjust=True, progress=False)
    close = raw["Close"].ffill()
    high  = raw["High"].ffill()
    low   = raw["Low"].ffill()
    valid = close.columns[close.isna().mean() < CFG["na_thresh"]]
    dropped = set(tickers) - set(valid)
    if dropped:
        print(f"  ⚠ Çıkarılan (eksik veri): {dropped}")
    print(f"  ✓ Kullanılan: {len(valid)} ticker")
    return close[valid], high[valid], low[valid]

# ─────────────────────────────────────────
# FAKTÖRLER (tek tarih için)
# ─────────────────────────────────────────
def _momentum(close_window):
    return close_window.pct_change(CFG["mom_days"]).iloc[-1]

def _volatility(close_window):
    ret  = close_window.pct_change().fillna(0)
    span = 2 / (1 - CFG["lambda_ewma"]) - 1
    var  = (ret ** 2).ewm(span=span, adjust=False).mean()
    return np.sqrt(var * 252).ewm(span=10).mean().iloc[-1]

def _avg_corr(close_window):
    ret    = close_window.pct_change().dropna().tail(CFG["corr_days"])
    corr_m = ret.corr()
    n      = len(corr_m)
    return (corr_m.sum() - 1) / (n - 1)

def _atr_trend(close_w, high_w, low_w):
    p = CFG["atr_period"]
    signals = {}
    for col in close_w.columns:
        h, l, c = high_w[col], low_w[col], close_w[col]
        pc  = c.shift(1)
        tr  = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        atr = tr.rolling(p).mean()
        upper = h.rolling(p).max() + atr
        lower = l.rolling(p).min() + atr
        sig   = pd.Series(np.nan, index=c.index)
        sig[h > upper.shift(1)] =  1.0
        sig[l < lower.shift(1)] = -1.0
        sig = sig.ffill().fillna(1.0)
        signals[col] = sig.iloc[-1]
    return pd.Series(signals)

def _score_and_alloc(mom, vol, corr, trend):
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
    else:
        score = base.copy()

    ranked   = score.sort_values()
    top5     = ranked.head(CFG["top_n"]).index.tolist()

    alloc  = {}
    per_w  = 1.0 / CFG["top_n"]
    cash_w = 0.0
    for ticker in top5:
        neg_mom   = mom[ticker] <= 0
        short_sig = (mode == "filter") and (trend[ticker] == -1.0)
        if neg_mom or short_sig:
            cash_w += per_w
        else:
            alloc[ticker] = per_w
    if cash_w > 0:
        alloc[CASH] = alloc.get(CASH, 0) + cash_w
    return alloc, score, base, mom, vol, corr, trend, ranked

# ─────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────
def run_backtest(close, high, low):
    assets      = [c for c in UNIVERSE if c in close.columns]
    all_cols    = assets + ([CASH] if CASH in close.columns else [])
    daily_ret   = close.pct_change()

    # Warm-up: momentum(84) + ATR(42) + biraz pay → 130 gün
    warmup      = CFG["mom_days"] + CFG["atr_period"] + 10
    month_ends  = close.resample("ME").last().index
    month_ends  = month_ends[month_ends > close.index[warmup]]

    records     = []   # aylık portföy ağırlıkları
    port_rets   = []   # günlük portföy getirileri

    print(f"  Backtest: {month_ends[0].date()} → {month_ends[-1].date()}  "
          f"({len(month_ends)} ay)")

    prev_alloc = None
    for i, rebal_date in enumerate(month_ends[:-1]):
        next_date = month_ends[i + 1]

        # Sadece rebal_date'e kadar olan veriyle hesapla (no look-ahead)
        idx = close.index.get_loc(rebal_date) if rebal_date in close.index \
              else close.index.searchsorted(rebal_date) - 1

        c_w = close.iloc[:idx+1][assets]
        h_w = high.iloc[:idx+1][assets]
        l_w = low.iloc[:idx+1][assets]

        if len(c_w) < warmup:
            continue

        mom   = _momentum(c_w)
        vol   = _volatility(c_w)
        corr  = _avg_corr(c_w)
        trend = _atr_trend(c_w, h_w, l_w)

        alloc, *_ = _score_and_alloc(mom, vol, corr, trend)
        prev_alloc = alloc

        rec = {"date": rebal_date, **{c: 0.0 for c in all_cols}}
        for k, v in alloc.items():
            if k in rec:
                rec[k] = v
        records.append(rec)

        # Dönem getirileri: (rebal_date, next_date]
        mask   = (daily_ret.index > rebal_date) & (daily_ret.index <= next_date)
        period = daily_ret.loc[mask, [c for c in all_cols if c in daily_ret.columns]]
        w_ser  = pd.Series({c: alloc.get(c, 0.0) for c in period.columns})
        port_rets.append(period.dot(w_ser))

    alloc_df = pd.DataFrame(records).set_index("date").fillna(0.0)
    port_ret = pd.concat(port_rets).sort_index()
    return alloc_df, port_ret

# ─────────────────────────────────────────
# PERFORMANS
# ─────────────────────────────────────────
def performance_stats(returns, name):
    r   = returns.dropna()
    cum = (1 + r).cumprod()
    ny  = len(r) / 252
    tot = cum.iloc[-1] - 1
    cagr    = (1 + tot) ** (1 / ny) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = cagr / ann_vol if ann_vol else np.nan
    dd      = cum / cum.cummax() - 1
    max_dd  = dd.min()
    calmar  = cagr / abs(max_dd) if max_dd else np.nan

    # Aylık kazanma oranı
    monthly = (1 + r).resample("ME").prod() - 1
    win_r   = (monthly > 0).mean()

    # En iyi / en kötü ay
    best_m  = monthly.max()
    worst_m = monthly.min()

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │  {name:<47}│
  ├─────────────────────────────────────────────────┤
  │  Toplam Getiri (TRY nominal) : {tot:>10.1%}          │
  │  CAGR                        : {cagr:>10.1%}          │
  │  Yıllık Volatilite           : {ann_vol:>10.1%}          │
  │  Sharpe Oranı                : {sharpe:>10.2f}          │
  │  Maks. Drawdown              : {max_dd:>10.1%}          │
  │  Calmar Oranı                : {calmar:>10.2f}          │
  │  Aylık Kazanma Oranı         : {win_r:>10.1%}          │
  │  En İyi Ay                   : {best_m:>10.1%}          │
  │  En Kötü Ay                  : {worst_m:>10.1%}          │
  └─────────────────────────────────────────────────┘""")
    return cum, dd

# ─────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────
def plot_results(raam_cum, raam_dd, bench_cum, bench_dd,
                 alloc_df, port_ret, close):
    assets     = [c for c in UNIVERSE if c in alloc_df.columns]
    cash_label = "Nakit (XU030)"

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"RAAM — Borsa İstanbul  |  ATR({CFG['atr_period']})  "
        f"Trend: {CFG['trend_mode']}  |  TRY Nominal",
        fontsize=14, fontweight="bold"
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    RAAM_COLOR  = "#1f77b4"
    BENCH_COLOR = "#d62728"

    # 1) Birikim eğrisi
    ax1 = fig.add_subplot(gs[0, :])
    raam_cum.plot(ax=ax1, color=RAAM_COLOR, lw=2.5, label="RAAM-BIST")
    if bench_cum is not None:
        bench_cum.plot(ax=ax1, color=BENCH_COLOR, lw=1.8,
                       ls="--", label="XU100 Benchmark")
    ax1.set_title("Birikim Eğrisi (1 = Başlangıç)")
    ax1.legend(fontsize=10); ax1.grid(alpha=0.3)
    ax1.set_ylabel("Büyüme Katsayısı")

    # 2) Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(raam_dd.index, raam_dd * 100, 0,
                     color=RAAM_COLOR, alpha=0.4, label="RAAM")
    if bench_dd is not None:
        ax2.fill_between(bench_dd.index, bench_dd * 100, 0,
                         color=BENCH_COLOR, alpha=0.25, label="XU100")
    ax2.set_title("Drawdown (%)"); ax2.grid(alpha=0.3)
    ax2.set_ylabel("%"); ax2.legend(fontsize=9)

    # 3) Aylık getiriler (heatmap)
    ax3 = fig.add_subplot(gs[2, :])
    monthly = (1 + port_ret).resample("ME").prod() - 1
    colors  = [RAAM_COLOR if x > 0 else BENCH_COLOR for x in monthly]
    ax3.bar(monthly.index, monthly * 100, color=colors,
            width=20, alpha=0.8)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_title("Aylık Getiriler (%)"); ax3.grid(alpha=0.3)
    ax3.set_ylabel("%")

    # 4) Portföy dağılımı (stacked area)
    ax4 = fig.add_subplot(gs[3, :])
    plot_cols = assets + ([CASH] if CASH in alloc_df.columns else [])
    labels    = {k: UNIVERSE.get(k, cash_label) for k in plot_cols}
    adf       = alloc_df[[c for c in plot_cols if c in alloc_df.columns]].rename(columns=labels)
    adf.plot.area(ax=ax4, stacked=True, linewidth=0, cmap="tab20")
    ax4.set_title("Aylık Portföy Dağılımı")
    ax4.set_ylabel("Ağırlık"); ax4.grid(alpha=0.3)
    ax4.legend(loc="upper left", fontsize=7, ncol=5)

    plt.savefig("raam_bist_backtest.png", dpi=150, bbox_inches="tight")
    print("\n  📈 Grafik: raam_bist_backtest.png")
    plt.show()

# ─────────────────────────────────────────
# CANLI SİNYAL (backtest ile aynı mantık)
# ─────────────────────────────────────────
def live_signal(close, high, low):
    assets = [c for c in UNIVERSE if c in close.columns]
    mom    = _momentum(close[assets])
    vol    = _volatility(close[assets])
    corr   = _avg_corr(close[assets])
    trend  = _atr_trend(close[assets], high[assets], low[assets])
    alloc, score, base, mom, vol, corr, trend, ranked = \
        _score_and_alloc(mom, vol, corr, trend)
    return alloc, score, base, mom, vol, corr, trend, ranked

def print_signal(alloc, score, base, mom, vol, corr, trend, ranked, close):
    today     = date.today().strftime("%d %B %Y")
    last_data = close.index[-1].strftime("%d %B %Y")
    long_n    = int((trend == 1.0).sum())
    short_n   = int((trend == -1.0).sum())
    mode_tr   = {"filter": "Filtre", "multiplier": "Çarpan", "score_only": "Yok"}

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           RAAM — BIST AYLIK SİNYAL RAPORU                   ║
║  Rapor  : {today:<50}║
║  Veri   : {last_data:<50}║
║  Mod    : ATR({CFG["atr_period"]}) — Trend: {mode_tr.get(CFG["trend_mode"],""):<38}║
╚══════════════════════════════════════════════════════════════╝
""")
    print("📊 PORTFÖY DAĞILIMI:\n")
    print(f"  {'Ticker':<12} {'Şirket':<26} {'Ağırlık':>8}  Durum")
    print(f"  {'-'*65}")
    for ticker, w in sorted(alloc.items(), key=lambda x: -x[1]):
        name   = UNIVERSE.get(ticker, "Nakit (XU030)")
        status = "💵 Nakit" if ticker == CASH else \
                 f"✅ Mom:{mom.get(ticker,0):+.1%} Trend:{'🟢' if trend.get(ticker,1)==1 else '🔴'}"
        print(f"  {ticker:<12} {name:<26} {w:>7.1%}  {status}")

    print(f"\n📈 SKOR TABLOSU  [{long_n}🟢 Long | {short_n}🔴 Short]\n")
    print(f"  {'#':<3} {'Ticker':<12} {'Şirket':<22} {'Skor':>6} "
          f"{'Mom':>7} {'Vol':>7} {'Corr':>7}  {'Trend':<8} Seçildi")
    print(f"  {'-'*90}")
    for i, (ticker, sc) in enumerate(ranked.items(), 1):
        name  = UNIVERSE.get(ticker, ticker)[:21]
        t_str = "🟢 Long" if trend.get(ticker, 1) == 1.0 else "🔴 Short"
        mark  = "⭐" if (ticker in alloc and ticker != CASH) else \
                ("→💵" if i <= CFG["top_n"] else "")
        neg   = " ⚠️" if mom.get(ticker, 0) <= 0 and i <= CFG["top_n"] else ""
        print(f"  {i:<3} {ticker:<12} {name:<22} {sc:>6.2f} "
              f"{mom.get(ticker,0):>7.1%} {vol.get(ticker,0):>7.1%} "
              f"{corr.get(ticker,0):>7.3f}  {t_str:<12} {mark}{neg}")

    print(f"""
📌 Sinyal ayın son işlem gününde hesaplanır,
   ertesi ayın ilk işlem gününde uygulanır.
⚠️  Yatırım tavsiyesi değildir.
""")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  RAAM — BIST  |  Backtest + Canlı Sinyal")
    print("═"*55 + "\n")

    # ── Veri ──
    close, high, low = download_data()

    # ── BACKTEST ──
    print("\n📊 BACKTEST ÇALIŞIYOR...\n")
    alloc_df, port_ret = run_backtest(close, high, low)

    # Benchmark: XU100
    bench_ret  = None
    bench_cum  = None
    bench_dd   = None
    if "XU100.IS" in close.columns:
        br = close["XU100.IS"].pct_change().dropna()
        br = br[br.index.isin(port_ret.index)]
        bench_ret = br

    print("\n📈 PERFORMANS KARŞILAŞTIRMASI\n")
    raam_cum, raam_dd = performance_stats(port_ret, "RAAM-BIST (TRY Nominal)")
    if bench_ret is not None:
        bench_cum, bench_dd = performance_stats(bench_ret, "XU100 Benchmark (TRY)")

    # Yıllık getiri tablosu
    print("\n📅 YILLIK GETİRİLER:\n")
    raam_yr  = (1 + port_ret).resample("YE").prod() - 1
    print(f"  {'Yıl':<6} {'RAAM':>8}", end="")
    if bench_ret is not None:
        bench_yr = (1 + bench_ret).resample("YE").prod() - 1
        print(f"  {'XU100':>8}  {'Fark':>8}", end="")
    print()
    print(f"  {'-'*40}")
    for yr in raam_yr.index:
        r = raam_yr.get(yr, np.nan)
        print(f"  {yr.year:<6} {r:>8.1%}", end="")
        if bench_ret is not None:
            b   = bench_yr.get(yr, np.nan)
            dif = r - b
            clr = "+" if dif >= 0 else ""
            print(f"  {b:>8.1%}  {clr}{dif:>7.1%}", end="")
        print()

    # ── CANLI SİNYAL ──
    print("\n" + "═"*55)
    print("  CANLI SİNYAL (Bugünkü Veri)")
    print("═"*55)
    alloc, score, base, mom, vol, corr, trend, ranked = \
        live_signal(close, high, low)
    print_signal(alloc, score, base, mom, vol, corr, trend, ranked, close)

    # ── PLOT ──
    plot_results(raam_cum, raam_dd, bench_cum, bench_dd,
                 alloc_df, port_ret, close)