#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMPS-SMO Matplotlib Grafik Koleksiyonu
Referans: arXiv:2505.03659 | Borsa İstanbul BIST Uygulaması
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ── Tema ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0a0e1a",
    "axes.facecolor":    "#111827",
    "axes.edgecolor":    "#374151",
    "axes.labelcolor":   "#d1d5db",
    "axes.grid":         True,
    "grid.color":        "#1f2937",
    "grid.linewidth":    0.6,
    "text.color":        "#e5e7eb",
    "xtick.color":       "#9ca3af",
    "ytick.color":       "#9ca3af",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "legend.fontsize":   8,
    "legend.framealpha": 0.3,
    "legend.edgecolor":  "#374151",
    "legend.facecolor":  "#111827",
    "lines.linewidth":   1.5,
    "font.family":       "DejaVu Sans",
    "savefig.facecolor": "#0a0e1a",
    "savefig.dpi":       150,
})

ACCENT   = "#00d2ff"
GOLD     = "#fbbf24"
GREEN    = "#22c55e"
RED      = "#ef4444"
PURPLE   = "#a78bfa"
ORANGE   = "#fb923c"

PALETTE = [
    "#00d2ff","#ff6b6b","#ffd700","#7cfc00","#ff69b4","#00ced1",
    "#ff8c00","#9370db","#32cd32","#ff4500","#1e90ff","#adff2f","#dc143c",
]

# ── Veri yükle ────────────────────────────────────────────────────────────────
RESULT_PATH = Path(__file__).parent / "results.json"
with open(RESULT_PATH, encoding="utf-8") as f:
    R = json.load(f)

dates      = R["tarihler"]["test"]
lmps_w     = np.array(R["zenginlik_egrileri"]["LMPS-SMO"])
base_w     = {k: np.array(v) for k, v in R["zenginlik_egrileri"]["baz_stratejiler"].items()}
mix_wt     = np.array(R["karisim_agirliklari"])       # (T, M)
train_loss = np.array(R["egitim_kaybi"])
metrics    = R["metrikler"]
cand_names = R["aday_stratejiler"]
ds         = R["veri_seti"]

T     = len(dates)
x_idx = np.arange(T)
# İnce tarih etiketi (her ~60 günde bir)
tick_step  = max(1, T // 8)
tick_pos   = x_idx[::tick_step]
tick_lbl   = [dates[i][:7] for i in tick_pos]  # YYYY-MM

# Günlük log-getiri (LMPS)
lmps_rets    = np.diff(np.log(np.maximum(lmps_w, 1e-9)))
base_rets    = {k: np.diff(np.log(np.maximum(v, 1e-9))) for k, v in base_w.items()}
all_names    = ["LMPS-SMO"] + list(base_w.keys())
all_wealths  = [lmps_w] + list(base_w.values())
all_colors   = PALETTE[:len(all_names)]

# Çizim kayıt klasörü
OUT = Path(__file__).parent / "plots"
OUT.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Şekil 1 — Ana 3×3 Grafik Koleksiyonu
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 15))
fig.suptitle(
    "LMPS-SMO · BIST Portföy Analizi  |  arXiv:2505.03659",
    fontsize=14, fontweight="bold", color=ACCENT, y=0.98
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)


# ── 1-A  Birikimli Zenginlik (tüm stratejiler) ───────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])   # üst satır geniş
for i, (name, w) in enumerate(zip(all_names, all_wealths)):
    lw   = 2.5 if name == "LMPS-SMO" else 0.9
    ls   = "-"  if name == "LMPS-SMO" else "--"
    zord = 10   if name == "LMPS-SMO" else 2
    ax1.plot(x_idx, w, color=all_colors[i], lw=lw, ls=ls, label=name, zorder=zord)
ax1.set_title("Birikimli Zenginlik Eğrileri (Test Dönemi)")
ax1.set_ylabel("Birikimli Servet")
ax1.set_xticks(tick_pos); ax1.set_xticklabels(tick_lbl, rotation=30, ha="right")
ax1.legend(ncol=3, loc="upper left")
ax1.axhline(1.0, color="#4b5563", lw=0.8, ls=":")


# ── 1-B  Meta-Eğitim Kayıp Eğrisi ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
epochs = np.arange(1, len(train_loss) + 1)
ax2.plot(epochs, train_loss, color=GOLD, lw=2, marker="o", markersize=5)
ax2.fill_between(epochs, train_loss, train_loss.min(), alpha=0.2, color=GOLD)
ax2.set_title("Meta-Eğitim Kayıp Eğrisi")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Kayıp")
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


# ── 2-A  Karışım Ağırlığı Evrimi (istiflenmiş alan) ──────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
cand_colors = [ACCENT, RED, GREEN, PURPLE]
bottom = np.zeros(T)
for m, (cname, col) in enumerate(zip(cand_names, cand_colors)):
    wm = mix_wt[:, m]
    ax3.fill_between(x_idx, bottom, bottom + wm, alpha=0.75, color=col, label=cname)
    bottom += wm
ax3.set_ylim(0, 1)
ax3.set_title("Karışım Ağırlıklarının Zaman İçindeki Değişimi")
ax3.set_ylabel("Ağırlık")
ax3.set_xticks(tick_pos); ax3.set_xticklabels(tick_lbl, rotation=30, ha="right")
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax3.legend(loc="upper right", ncol=2)


# ── 2-B  Sharpe Oranı Çubuk Grafiği ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
sharpes = [m["sharpe_orani"] for m in metrics]
names_m = [m["isim"] for m in metrics]
cols_bar = [ACCENT if n == "LMPS-SMO" else
            (GREEN if s > 0 else RED) for n, s in zip(names_m, sharpes)]
bars = ax4.barh(names_m[::-1], sharpes[::-1], color=cols_bar[::-1], height=0.65, edgecolor="#0a0e1a")
ax4.axvline(0, color="#6b7280", lw=0.8)
ax4.set_title("Sharpe Oranı Karşılaştırması")
ax4.set_xlabel("Sharpe")
for bar, val in zip(bars, sharpes[::-1]):
    ax4.text(val + 0.02 * (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", ha="left" if val >= 0 else "right",
             fontsize=7, color="#e5e7eb")


# ── 3-A  Düşüş Eğrileri (Drawdown) ───────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
def drawdown(w):
    cum = np.maximum(w, 1e-9)
    rm  = np.maximum.accumulate(cum)
    return (rm - cum) / (rm + 1e-9)

ax5.plot(x_idx, drawdown(lmps_w) * 100, color=ACCENT, lw=1.8, label="LMPS-SMO", zorder=5)
for name, w in list(base_w.items())[:5]:
    ax5.plot(x_idx, drawdown(w) * 100, lw=0.7, alpha=0.55, label=name)
ax5.set_title("Maksimum Düşüş Eğrileri")
ax5.set_ylabel("Düşüş (%)")
ax5.set_xticks(tick_pos); ax5.set_xticklabels(tick_lbl, rotation=30, ha="right")
ax5.invert_yaxis()
ax5.legend(fontsize=7)


# ── 3-B  Yıllık Getiri & Volatilite Scatter ──────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
for i, m in enumerate(metrics):
    col  = ACCENT if m["isim"] == "LMPS-SMO" else all_colors[i]
    size = 90     if m["isim"] == "LMPS-SMO" else 45
    zord = 10     if m["isim"] == "LMPS-SMO" else 3
    ax6.scatter(m["yillik_volatilite"], m["yillik_getiri"],
                color=col, s=size, zorder=zord, edgecolors="#0a0e1a", lw=0.5)
    ax6.annotate(m["isim"], (m["yillik_volatilite"], m["yillik_getiri"]),
                 fontsize=6.5, color=col,
                 xytext=(4, 2), textcoords="offset points")
ax6.axhline(0, color="#4b5563", lw=0.7, ls=":")
ax6.set_title("Risk-Getiri Uzayı")
ax6.set_xlabel("Yıllık Volatilite (%)"); ax6.set_ylabel("Yıllık Getiri (%)")


# ── 3-C  Log-Getiri Dağılım Histogramı ───────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
ax7.hist(lmps_rets * 100, bins=40, color=ACCENT, alpha=0.7, density=True, label="LMPS-SMO")
for name, lr in list(base_rets.items())[:3]:
    ax7.hist(lr * 100, bins=40, alpha=0.3, density=True, label=name, histtype="step", lw=1.2)
ax7.axvline(0, color="#6b7280", lw=0.8)
ax7.set_title("Günlük Log-Getiri Dağılımı")
ax7.set_xlabel("Log-Getiri (%)"); ax7.set_ylabel("Yoğunluk")
ax7.legend(fontsize=7)

plt.savefig(OUT / "01_ana_panel.png", bbox_inches="tight")
plt.close()
print("✓ 01_ana_panel.png")


# ══════════════════════════════════════════════════════════════════════════════
# Şekil 2 — Karışım Ağırlıkları Detay (4 alt grafik)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
fig2.suptitle("Aday Strateji Karışım Ağırlıkları — Detaylı Görünüm", fontsize=13,
              fontweight="bold", color=ACCENT)

for m, (cname, col, ax) in enumerate(zip(cand_names, cand_colors, axes2.ravel())):
    wm = mix_wt[:, m]
    ax.plot(x_idx, wm * 100, color=col, lw=1.5)
    ax.fill_between(x_idx, 0, wm * 100, alpha=0.25, color=col)
    ax.set_title(f"Aday: {cname}", color=col)
    ax.set_ylabel("Ağırlık (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lbl, rotation=30, ha="right")
    # İstatistikler
    ax.axhline(wm.mean() * 100, color=col, lw=1.0, ls="--", alpha=0.7,
               label=f"Ort: {wm.mean()*100:.1f}%")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "02_karisim_agirliklari.png", bbox_inches="tight")
plt.close()
print("✓ 02_karisim_agirliklari.png")


# ══════════════════════════════════════════════════════════════════════════════
# Şekil 3 — Performans Metrikleri Isı Haritası
# ══════════════════════════════════════════════════════════════════════════════
metric_keys = ["birikimli_servet", "yillik_getiri", "yillik_volatilite",
               "sharpe_orani", "maks_dusus"]
metric_labels = ["Birikimli\nServet", "Yıllık\nGetiri %", "Yıllık\nVolatilite %",
                 "Sharpe\nOranı", "Maks.\nDüşüş %"]

strat_names  = [m["isim"] for m in metrics]
data_matrix  = np.array([[m[k] for k in metric_keys] for m in metrics])

# Z-score normalize (sütun bazlı, görsel karşılaştırma için)
col_mean = data_matrix.mean(axis=0)
col_std  = data_matrix.std(axis=0) + 1e-9
data_z   = (data_matrix - col_mean) / col_std

# Düşüş ve volatilite için yönü tersine çevir (küçük = iyi)
data_z[:, 2] = -data_z[:, 2]   # volatilite
data_z[:, 4] = -data_z[:, 4]   # maks düşüş

cmap = LinearSegmentedColormap.from_list(
    "bist", ["#ef4444", "#111827", "#22c55e"], N=256
)

fig3, ax3h = plt.subplots(figsize=(10, 7))
fig3.patch.set_facecolor("#0a0e1a")
im = ax3h.imshow(data_z, cmap=cmap, aspect="auto", vmin=-2.5, vmax=2.5)

ax3h.set_xticks(range(len(metric_keys)))
ax3h.set_xticklabels(metric_labels, fontsize=9)
ax3h.set_yticks(range(len(strat_names)))
ax3h.set_yticklabels(strat_names, fontsize=9)

# Hücre değerleri
for r in range(len(strat_names)):
    for c in range(len(metric_keys)):
        raw = data_matrix[r, c]
        txt = f"{raw:.2f}" if abs(raw) < 100 else f"{raw:.0f}"
        col_txt = "#ffffff" if abs(data_z[r, c]) < 1.2 else "#000000"
        ax3h.text(c, r, txt, ha="center", va="center", fontsize=7.5,
                  color=col_txt, fontweight="bold" if strat_names[r] == "LMPS-SMO" else "normal")

# LMPS-SMO satırını vurgula
lmps_row = strat_names.index("LMPS-SMO")
ax3h.add_patch(plt.Rectangle((-0.5, lmps_row - 0.5), len(metric_keys), 1,
                              fill=False, edgecolor=ACCENT, lw=2.5))

cb = fig3.colorbar(im, ax=ax3h, fraction=0.03, pad=0.02)
cb.set_label("Z-skor (Normalleştirilmiş)", fontsize=8)
ax3h.set_title("Performans Metrikleri Isı Haritası  (yeşil=iyi, kırmızı=kötü)",
               fontsize=11, fontweight="bold", color=ACCENT)

plt.tight_layout()
plt.savefig(OUT / "03_metrik_isi_haritasi.png", bbox_inches="tight")
plt.close()
print("✓ 03_metrik_isi_haritasi.png")


# ══════════════════════════════════════════════════════════════════════════════
# Şekil 4 — Kayan Sharpe Oranı (60 günlük pencere)
# ══════════════════════════════════════════════════════════════════════════════
WIN = 60

def rolling_sharpe(log_rets, win=WIN):
    sr = np.full(len(log_rets), np.nan)
    for t in range(win, len(log_rets)):
        s = log_rets[t-win:t]
        mu, sig = s.mean(), s.std()
        sr[t] = (mu / (sig + 1e-9)) * np.sqrt(252)
    return sr

fig4, ax4r = plt.subplots(figsize=(14, 5))
ax4r.axhline(0, color="#4b5563", lw=0.8)

rs_lmps = rolling_sharpe(lmps_rets)
ax4r.plot(x_idx[1:], rs_lmps, color=ACCENT, lw=2.2, label="LMPS-SMO", zorder=5)
ax4r.fill_between(x_idx[1:], 0, rs_lmps, where=rs_lmps > 0,
                  alpha=0.15, color=GREEN, interpolate=True)
ax4r.fill_between(x_idx[1:], 0, rs_lmps, where=rs_lmps < 0,
                  alpha=0.15, color=RED, interpolate=True)

for name, lr in list(base_rets.items())[:4]:
    rs = rolling_sharpe(lr)
    ax4r.plot(x_idx[1:], rs, lw=0.9, alpha=0.5, ls="--", label=name)

ax4r.set_title(f"Kayan Sharpe Oranı ({WIN} Günlük Pencere)", fontsize=12,
               fontweight="bold", color=ACCENT)
ax4r.set_ylabel("Sharpe Oranı (yıllıklandırılmış)")
ax4r.set_xticks(tick_pos); ax4r.set_xticklabels(tick_lbl, rotation=30, ha="right")
ax4r.legend(ncol=3)

plt.tight_layout()
plt.savefig(OUT / "04_kayan_sharpe.png", bbox_inches="tight")
plt.close()
print("✓ 04_kayan_sharpe.png")


# ══════════════════════════════════════════════════════════════════════════════
# Şekil 5 — Düşüş Dönemleri Analizi (tüm stratejiler ısı haritası)
# ══════════════════════════════════════════════════════════════════════════════
def dd_series(w):
    cum = np.maximum(w, 1e-9)
    rm  = np.maximum.accumulate(cum)
    return (rm - cum) / (rm + 1e-9) * 100

fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1])
fig5.suptitle("Düşüş Analizi", fontsize=13, fontweight="bold", color=ACCENT)

# Üst: Tüm stratejiler ısı haritası
dd_matrix = np.array([dd_series(w) for w in all_wealths])   # (13, T)
im5 = ax5a.imshow(dd_matrix, aspect="auto", cmap="YlOrRd",
                  extent=[0, T, -0.5, len(all_names) - 0.5])
ax5a.set_yticks(range(len(all_names)))
ax5a.set_yticklabels(all_names, fontsize=8)
ax5a.set_xticks(tick_pos); ax5a.set_xticklabels(tick_lbl, rotation=30, ha="right")
ax5a.set_title("Tüm Stratejilerin Düşüş Haritası (%)", fontsize=10)
cb5 = fig5.colorbar(im5, ax=ax5a, fraction=0.02, pad=0.01)
cb5.set_label("Düşüş (%)", fontsize=8)

# Alt: LMPS-SMO vs en iyi baz (en yüksek Sharpe)
best_base = max(base_rets, key=lambda n: next(
    m["sharpe_orani"] for m in metrics if m["isim"] == n))
ax5b.fill_between(x_idx, dd_series(lmps_w),  alpha=0.5, color=ACCENT, label="LMPS-SMO")
ax5b.fill_between(x_idx, dd_series(base_w[best_base]), alpha=0.35, color=GOLD,
                  label=f"{best_base} (en iyi baz)")
ax5b.set_title(f"LMPS-SMO vs {best_base} — Düşüş Karşılaştırması")
ax5b.set_ylabel("Düşüş (%)"); ax5b.invert_yaxis()
ax5b.set_xticks(tick_pos); ax5b.set_xticklabels(tick_lbl, rotation=30, ha="right")
ax5b.legend()

plt.tight_layout()
plt.savefig(OUT / "05_dusus_analizi.png", bbox_inches="tight")
plt.close()
print("✓ 05_dusus_analizi.png")


# ══════════════════════════════════════════════════════════════════════════════
# Şekil 6 — Aylık Getiri Mozaiği (Takvim ısı haritası)
# ══════════════════════════════════════════════════════════════════════════════
from collections import defaultdict

# Günlük getiriyi ay-yıl bazında grupla
monthly_ret = defaultdict(list)
for t in range(T - 1):
    ym = dates[t][:7]  # YYYY-MM
    dr = float(np.log(max(lmps_w[t+1], 1e-9)) - np.log(max(lmps_w[t], 1e-9)))
    monthly_ret[ym].append(dr)

sorted_months = sorted(monthly_ret.keys())
years  = sorted(set(m[:4] for m in sorted_months))
months = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"]
month_nums = {f"{int(m):02d}": i for i, m in enumerate(range(1, 13))}

grid = np.full((len(years), 12), np.nan)
for ym, rets in monthly_ret.items():
    y, m = ym[:4], ym[5:7]
    ri = years.index(y)
    ci = month_nums[m]
    grid[ri, ci] = sum(rets) * 100   # aylık log-getiri %

fig6, ax6c = plt.subplots(figsize=(14, max(3, len(years) * 0.8 + 1.5)))
vmax = max(abs(np.nanmax(grid)), abs(np.nanmin(grid)))
cmap6 = LinearSegmentedColormap.from_list("rg", ["#ef4444","#111827","#22c55e"], N=256)
im6 = ax6c.imshow(grid, cmap=cmap6, aspect="auto",
                  vmin=-vmax, vmax=vmax,
                  extent=[-0.5, 11.5, len(years) - 0.5, -0.5])

for ri, yr in enumerate(years):
    for ci in range(12):
        v = grid[ri, ci]
        if not np.isnan(v):
            txt_col = "#000" if abs(v) > vmax * 0.6 else "#fff"
            ax6c.text(ci, ri, f"{v:.1f}%", ha="center", va="center",
                      fontsize=7.5, color=txt_col, fontweight="bold")

ax6c.set_xticks(range(12)); ax6c.set_xticklabels(months, fontsize=9)
ax6c.set_yticks(range(len(years))); ax6c.set_yticklabels(years, fontsize=9)
ax6c.set_title("LMPS-SMO Aylık Log-Getiri Mozaiği (%)",
               fontsize=12, fontweight="bold", color=ACCENT)
cb6 = fig6.colorbar(im6, ax=ax6c, fraction=0.02, pad=0.01, orientation="vertical")
cb6.set_label("Aylık Log-Getiri (%)", fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "06_aylik_mozaik.png", bbox_inches="tight")
plt.close()
print("✓ 06_aylik_mozaik.png")


# ══════════════════════════════════════════════════════════════════════════════
# Şekil 7 — Metrik Çubuk/Radar Karşılaştırması
# ══════════════════════════════════════════════════════════════════════════════
fig7, axes7 = plt.subplots(1, 4, figsize=(16, 5))
fig7.suptitle("Strateji Performans Karşılaştırması", fontsize=13,
              fontweight="bold", color=ACCENT)

metric_cfg = [
    ("birikimli_servet",  "Birikimli Servet",   False),
    ("yillik_getiri",     "Yıllık Getiri (%)",   False),
    ("sharpe_orani",      "Sharpe Oranı",        False),
    ("maks_dusus",        "Maks. Düşüş (%)",     True),   # invert (küçük iyi)
]

for ax, (key, label, inv) in zip(axes7, metric_cfg):
    vals = [m[key] for m in metrics]
    cols = [ACCENT if m["isim"] == "LMPS-SMO" else
            (GREEN if v > 0 else RED) for m, v in zip(metrics, vals)]
    strats = [m["isim"] for m in metrics]
    order = np.argsort(vals)
    v_ord = [vals[i] for i in order]
    c_ord = [cols[i] for i in order]
    n_ord = [strats[i] for i in order]
    bars = ax.barh(n_ord, v_ord, color=c_ord, height=0.6, edgecolor="#0a0e1a")
    ax.axvline(0, color="#6b7280", lw=0.7)
    ax.set_title(label, fontsize=9)
    ax.tick_params(axis="y", labelsize=7.5)

plt.tight_layout()
plt.savefig(OUT / "07_metrik_cubuk.png", bbox_inches="tight")
plt.close()
print("✓ 07_metrik_cubuk.png")


# ══════════════════════════════════════════════════════════════════════════════
# Şekil 8 — Meta-Eğitim Detay Paneli
# ══════════════════════════════════════════════════════════════════════════════
fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(12, 5))
fig8.suptitle("MAML Meta-Eğitim Analizi", fontsize=12, fontweight="bold", color=ACCENT)

# Sol: Kayıp
epochs = np.arange(1, len(train_loss) + 1)
ax8a.plot(epochs, train_loss, color=GOLD, lw=2, marker="o", ms=6, zorder=3)
ax8a.fill_between(epochs, train_loss, train_loss.min(), alpha=0.2, color=GOLD)
# Trend çizgisi
z = np.polyfit(epochs, train_loss, 1)
trend = np.poly1d(z)(epochs)
ax8a.plot(epochs, trend, color=RED, lw=1.2, ls="--", alpha=0.7, label=f"Eğilim: {z[0]:.5f}/epoch")
ax8a.set_title("Meta-Eğitim Kayıp Eğrisi"); ax8a.set_xlabel("Epoch"); ax8a.set_ylabel("Kayıp")
ax8a.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax8a.legend()

# Sağ: Hiper-parametreler bilgi kutusu + karışım ağırlığı özeti
ax8b.axis("off")
rows = [
    ("Hiperparametre", "Değer"),
    ("─────────────", "─────────"),
    ("Aday strateji (M)", "4"),
    ("Destek kümesi (K)", "10"),
    ("Sorgu kümesi (Q)", "1"),
    ("Görev sayısı (H)", "8"),
    ("Epoch sayısı", "20"),
    ("İç döngü LR (α)", "0.001"),
    ("Meta LR (β)", "0.0005"),
    ("İşlem maliyeti (δ)", "0.001"),
    ("LSTM gizli boyutu", "32"),
    ("Dikkat boyutu (d)", "8"),
    ("─────────────", "─────────"),
    ("Eğitim gün", str(ds["egitim_gun"])),
    ("Test gün", str(ds["test_gun"])),
    ("Hisse sayısı", str(ds["hisse_sayisi"])),
]
table = ax8b.table(cellText=rows, loc="center", cellLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.3)
for key, cell in table.get_celld().items():
    cell.set_facecolor("#111827")
    cell.set_edgecolor("#1f2937")
    cell.set_text_props(color="#e5e7eb")
    if key[0] == 0:
        cell.set_text_props(color=ACCENT, fontweight="bold")
ax8b.set_title("Model & Eğitim Parametreleri", fontsize=9, pad=10)

plt.tight_layout()
plt.savefig(OUT / "08_egitim_detay.png", bbox_inches="tight")
plt.close()
print("✓ 08_egitim_detay.png")


print(f"\nTüm grafikler → {OUT}/")
print(f"Toplam 8 grafik dosyası oluşturuldu.")
