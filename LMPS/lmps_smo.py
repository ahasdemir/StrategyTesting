#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMPS-SMO: Meta-Learning the Optimal Mixture of Strategies
         for Online Portfolio Selection — BIST Uygulaması
Referans: arXiv:2505.03659
Tüm ML saf NumPy ile uygulanmıştır (PyTorch/TF yok).
"""

import numpy as np
import json
import warnings
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# Hiperparametreler
# ══════════════════════════════════════════════════════════════════════════════
M_CAND    = 4        # Aday strateji sayısı
K_SUPPORT = 10       # Destek kümesi boyutu (iç döngü)
Q_QUERY   = 1        # Sorgu kümesi boyutu
H_BATCH   = 8        # Meta-görev sayısı / batch
N_EPOCHS  = 20       # Meta-eğitim epoch sayısı
ALPHA     = 0.001    # İç döngü öğrenme oranı
BETA      = 0.0005   # Meta öğrenme oranı
DELTA     = 0.001    # İşlem maliyeti (BIST gerçekçi)
HIDDEN    = 32       # LSTM gizli boyutu
LAMBDA_D  = 0.01     # Çeşitlilik düzenlemesi ağırlığı
LAMBDA_TC = 1.0      # İşlem maliyeti düzenlemesi ağırlığı

BIST50 = [
    "AKBNK.IS","ARCLK.IS","ASELS.IS","BIMAS.IS","EKGYO.IS",
    "EREGL.IS","FROTO.IS","GARAN.IS","HALKB.IS","ISCTR.IS",
    "KCHOL.IS","KOZAL.IS","KRDMD.IS","MGROS.IS","PETKM.IS",
    "PGSUS.IS","SAHOL.IS","SISE.IS","TCELL.IS","THYAO.IS",
    "TUPRS.IS","VAKBN.IS","VESTL.IS","YKBNK.IS","AEFES.IS",
    "AKSEN.IS","ALARK.IS","ENKAI.IS","GUBRF.IS","TTKOM.IS",
]

POLICY_NAMES = [
    "BAH","CRP","EG","PAMR","OLMAR","CWMR",
    "RMR","WMAMR","ANTICOR","CORN","BNN","ONS"
]

# ══════════════════════════════════════════════════════════════════════════════
# Matematik Yardımcıları
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    x = np.clip(x, -30, 30)
    pos = x >= 0
    out = np.empty_like(x, dtype=float)
    out[pos]  = 1.0 / (1.0 + np.exp(-x[pos]))
    out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return out

def softmax1d(x):
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-12)

def softmax2d(x):
    """Satır-bazlı softmax, x: (T, d)"""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def dsoftmax2d(s, ds):
    """Satır-bazlı softmax Jacobian-vektör çarpımı"""
    return s * (ds - (s * ds).sum(axis=1, keepdims=True))

def project_simplex(v):
    """Olasılık simpleksi üzerine projeksiyon (Duchi et al.)"""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_arr = np.where(u * np.arange(1, n+1) > cssv - 1.0)[0]
    if len(rho_arr) == 0:
        return np.ones(n) / n
    rho = rho_arr[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)

def normalise(v, eps=1e-12):
    s = v.sum()
    return v / s if s > eps else np.ones(len(v)) / len(v)

# ══════════════════════════════════════════════════════════════════════════════
# Veri Katmanı
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data(symbols=BIST50, start="2019-01-01", end="2024-12-31"):
    """yfinance dene, başarısız olursa sentetik veri üret."""
    try:
        import yfinance as yf
        import pandas as pd
        print("yfinance ile BIST verisi çekiliyor…")
        raw = yf.download(symbols, start=start, end=end,
                          auto_adjust=True, progress=False, threads=True)
        if hasattr(raw.columns, "levels"):
            prices = raw["Close"]
        else:
            prices = raw
        prices = (prices
                  .dropna(axis=1, thresh=int(0.8 * len(prices)))
                  .ffill().bfill().dropna())
        if prices.shape[1] < 5 or prices.shape[0] < 200:
            raise ValueError("Yetersiz veri")
        print(f"Gerçek veri: {prices.shape[0]} gün, {prices.shape[1]} hisse")
        return (prices.values,
                prices.columns.tolist(),
                [str(d.date()) for d in prices.index],
                False)
    except Exception as exc:
        print(f"yfinance hatası: {exc}\nSentetik veri üretiliyor…")
        return synthetic_bist(len(symbols))

def synthetic_bist(n=30, start="2019-01-01", end="2024-12-31"):
    """
    2019-2024 BIST karakteristiklerine uyarlanmış sentetik veri:
    - Yüksek nominal TL getirisi (~%60 yıllık)
    - Yüksek volatilite (~%35)
    - Rejim geçişleri: COVID çöküşü, 2021 faiz krizi, 2023 seçim dönemi
    """
    np.random.seed(42)
    dates = []
    d = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while d <= e:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    T = len(dates)
    dt = 1.0 / 252

    def regime(day):
        if datetime(2020,3,1) <= day <= datetime(2020,5,31): return "covid"
        if datetime(2021,9,1) <= day <= datetime(2022,2,28): return "rate"
        if datetime(2023,4,1) <= day <= datetime(2023,8,31): return "election"
        return "normal"

    params = {
        "normal":   ( 0.65, 0.32),
        "covid":    (-3.00, 1.20),
        "rate":     ( 1.80, 0.80),
        "election": ( 0.40, 0.60),
    }

    # Sektörel korelasyon
    n_sec = 5
    sec_sz = max(1, n // n_sec)
    corr = np.eye(n)
    for ii in range(n):
        for jj in range(n):
            if ii != jj:
                corr[ii, jj] = 0.60 if ii//sec_sz == jj//sec_sz else 0.30
    L = np.linalg.cholesky(corr + 1e-4 * np.eye(n))

    prices = np.ones((T, n)) * 10.0
    rng = np.random.default_rng(42)
    for t in range(1, T):
        r, v = params[regime(dates[t])]
        mu = r * dt + rng.normal(0, 0.003, n)
        z  = L @ rng.normal(0, 1, n)
        ret = mu + v * np.sqrt(dt) * z
        prices[t] = prices[t-1] * np.exp(np.clip(ret, -0.4, 0.4))

    syms  = [f"BIST{i+1:02d}.IS" for i in range(n)]
    dstrs = [d.strftime("%Y-%m-%d") for d in dates]
    print(f"Sentetik BIST: {T} gün, {n} hisse")
    return prices, syms, dstrs, True

# ══════════════════════════════════════════════════════════════════════════════
# 12 Klasik OLPS Strateji
# ══════════════════════════════════════════════════════════════════════════════

class OLPSRunner:
    """
    Fiyat göreceli matrisi (T, n) üzerinde tüm stratejileri çalıştırır.
    Her strateji (T,) portföy getiri dizisi döndürür.
    """
    def __init__(self, price_rels, window=5):
        self.X = price_rels        # (T, n)
        self.T, self.n = self.X.shape
        self.w  = window
        self.uniform = np.ones(self.n) / self.n

    # ── 1. BAH ──────────────────────────────────────────────────────────────
    def bah(self):
        w = self.uniform.copy()
        rets = []
        for t in range(self.T):
            r = float(w @ self.X[t])
            rets.append(r)
            w = w * self.X[t]
            w = normalise(w)
        return np.array(rets)

    # ── 2. CRP ──────────────────────────────────────────────────────────────
    def crp(self):
        w = self.uniform.copy()
        rets = []
        for t in range(self.T):
            rets.append(float(w @ self.X[t]))
        return np.array(rets)

    # ── 3. EG (eta=0.05) ────────────────────────────────────────────────────
    def eg(self, eta=0.05):
        w = self.uniform.copy()
        rets = []
        for t in range(self.T):
            rets.append(float(w @ self.X[t]))
            w = w * np.exp(eta * self.X[t])
            w = normalise(w)
        return np.array(rets)

    # ── 4. PAMR ─────────────────────────────────────────────────────────────
    def pamr(self, epsilon=0.5, C=500.0):
        w = self.uniform.copy()
        rets = []
        buf = np.ones((self.w, self.n))
        for t in range(self.T):
            rets.append(float(w @ self.X[t]))
            # MA prediction
            buf[t % self.w] = self.X[t]
            x_pred = buf.mean(axis=0)
            x_pred_n = x_pred / (x_pred.mean() + 1e-8)
            loss = max(0.0, float(w @ x_pred_n) - epsilon)
            denom = float(np.sum((x_pred_n - x_pred_n.mean())**2))
            tau = min(C, loss / (denom + 1e-8))
            w = project_simplex(w - tau * (x_pred_n - x_pred_n.mean()))
        return np.array(rets)

    # ── 5. OLMAR (window=5, eps=10) ─────────────────────────────────────────
    def olmar(self, epsilon=10.0):
        w = self.uniform.copy()
        rets = []
        buf = np.ones((self.w, self.n))
        for t in range(self.T):
            rets.append(float(w @ self.X[t]))
            buf[t % self.w] = self.X[t]
            ma = buf.mean(axis=0)
            x_pred = ma / (self.X[t] + 1e-8)
            x_mean = x_pred.mean()
            loss = max(0.0, epsilon - float(w @ x_pred))
            denom = float(np.sum((x_pred - x_mean)**2))
            tau = loss / (denom + 1e-8)
            w = project_simplex(w + tau * (x_pred - x_mean))
        return np.array(rets)

    # ── 6. CWMR (sigma=0.02, epsilon=0.5) ───────────────────────────────────
    def cwmr(self, sigma=0.02, epsilon=0.5):
        w   = self.uniform.copy()
        mu  = self.uniform.copy()
        Sig = np.eye(self.n) * sigma
        rets = []
        for t in range(self.T):
            rets.append(float(w @ self.X[t]))
            x = self.X[t]
            M_val = float(mu @ x)
            V_val = float(x @ Sig @ x)
            loss = max(0.0, M_val - epsilon)
            lam = loss / (V_val + 1e-8)
            mu_new = mu - lam * (Sig @ x)
            Sig_new = np.linalg.inv(np.linalg.inv(Sig) + lam * np.outer(x, x))
            mu = project_simplex(mu_new)
            Sig = np.clip(Sig_new, -1, 1)
            w = mu
        return np.array(rets)

    # ── 7. RMR (L1-median mean reversion) ───────────────────────────────────
    def rmr(self, epsilon=10.0, T_rho=6):
        w = self.uniform.copy()
        rets = []
        buf = np.ones((self.w, self.n))
        for t in range(self.T):
            rets.append(float(w @ self.X[t]))
            buf[t % self.w] = self.X[t]
            # L1 median via iterative reweighted least squares
            med = buf.mean(axis=0)
            for _ in range(T_rho):
                diff = buf - med
                norms = np.linalg.norm(diff, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                weights_r = 1.0 / norms
                med = (weights_r * buf).sum(axis=0) / (weights_r.sum() + 1e-8)
            x_pred = med / (self.X[t] + 1e-8)
            x_mean = x_pred.mean()
            loss = max(0.0, epsilon - float(w @ x_pred))
            denom = float(np.sum((x_pred - x_mean)**2))
            tau = loss / (denom + 1e-8)
            w = project_simplex(w + tau * (x_pred - x_mean))
        return np.array(rets)

    # ── 8. WMAMR (ağırlıklı hareketli ortalama) ──────────────────────────────
    def wmamr(self, epsilon=0.5):
        w = self.uniform.copy()
        rets = []
        buf = np.ones((self.w, self.n))
        wt_decay = np.exp(np.linspace(-1, 0, self.w))
        wt_decay /= wt_decay.sum()
        for t in range(self.T):
            rets.append(float(w @ self.X[t]))
            buf[t % self.w] = self.X[t]
            # Ağırlıklı MA
            ordered = np.roll(buf, -(t % self.w + 1), axis=0)
            wma = (wt_decay[:, None] * ordered).sum(axis=0)
            x_pred = wma / (self.X[t] + 1e-8)
            x_mean = x_pred.mean()
            loss = max(0.0, float(w @ x_pred) - epsilon)
            denom = float(np.sum((x_pred - x_mean)**2))
            tau = loss / (denom + 1e-8)
            w = project_simplex(w - tau * (x_pred - x_mean))
        return np.array(rets)

    # ── 9. ANTICOR ──────────────────────────────────────────────────────────
    def anticor(self, win=30):
        w = self.uniform.copy()
        rets = []
        for t in range(self.T):
            rets.append(float(w @ self.X[t]))
            if t < 2 * win:
                continue
            # Two consecutive windows
            w1 = self.X[t - 2*win: t - win]   # (win, n)
            w2 = self.X[t - win: t]            # (win, n)
            mu1 = w1.mean(axis=0)
            mu2 = w2.mean(axis=0)
            # Cross-correlation matrix (log returns)
            lr1 = np.log(w1 + 1e-8) - np.log(mu1 + 1e-8)
            lr2 = np.log(w2 + 1e-8) - np.log(mu2 + 1e-8)
            Mcov = lr1.T @ lr2 / win
            # Anti-correlation: transfer weight from high-corr to low-corr pairs
            claim = np.zeros(self.n)
            for i in range(self.n):
                for j in range(self.n):
                    if Mcov[i, j] > 0 and mu1[i] > mu2[i]:
                        claim[j] += w[i] * Mcov[i, j]
            w = normalise(w + claim * 0.1)
        return np.array(rets)

    # ── 10. CORN (rho=0.1, window=5) ─────────────────────────────────────────
    def corn(self, rho=0.1):
        w = self.uniform.copy()
        rets = []
        buf = []
        for t in range(self.T):
            rets.append(float(w @ self.X[t]))
            buf.append(self.X[t].copy())
            if t < self.w + 1:
                continue
            cur_win = np.array(buf[-self.w:])
            similar = []
            for s in range(self.w, len(buf) - 1):
                hist_win = np.array(buf[s - self.w: s])
                # Korelasyon hesabı
                a = cur_win.ravel()
                b = hist_win.ravel()
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a < 1e-8 or norm_b < 1e-8:
                    continue
                corr = float(a @ b) / (norm_a * norm_b + 1e-8)
                if corr >= rho:
                    similar.append(buf[s])  # next-day return
            if similar:
                x_pred = np.mean(similar, axis=0)
                w = project_simplex(normalise(x_pred))
        return np.array(rets)

    # ── 11. BNN (k=5, window=5) ──────────────────────────────────────────────
    def bnn(self, k=5):
        w = self.uniform.copy()
        rets = []
        buf = []
        for t in range(self.T):
            rets.append(float(w @ self.X[t]))
            buf.append(self.X[t].copy())
            if t < self.w + k + 1:
                continue
            cur_win = np.array(buf[-self.w:]).ravel()
            # Geçmiş pencerelerin mesafelerini bul
            dists = []
            for s in range(self.w, len(buf) - 1):
                hist = np.array(buf[s - self.w: s]).ravel()
                dists.append((np.linalg.norm(cur_win - hist), s))
            dists.sort(key=lambda x: x[0])
            k_nearest = [buf[s] for _, s in dists[:k]]
            x_pred = np.mean(k_nearest, axis=0)
            w = project_simplex(normalise(x_pred))
        return np.array(rets)

    # ── 12. ONS (delta=0.125, beta=1, eta=0) ─────────────────────────────────
    def ons(self, delta=0.125, beta=1.0, eta=0.0):
        w = self.uniform.copy()
        rets = []
        A = np.eye(self.n)  # Fisher bilgi matrisinin yaklaşımı
        b = np.zeros(self.n)
        for t in range(self.T):
            rets.append(float(w @ self.X[t]))
            x = self.X[t]
            r = float(w @ x)
            grad = x / (r + 1e-8)
            # ONS güncelleme
            A += np.outer(grad, grad)
            b += (1.0 + 1.0/beta) * grad
            A_inv = np.linalg.inv(A + 1e-6 * np.eye(self.n))
            w_unconstrained = A_inv @ b
            # Simplex projeksiyon (Newton adımı versiyonu)
            w = project_simplex(w_unconstrained)
        return np.array(rets)

    def run_all(self):
        """12 stratejiyi çalıştır, (12, T) dizi döndür"""
        print("12 OLPS stratejisi çalıştırılıyor…")
        results = []
        for name, fn in zip(POLICY_NAMES, [
            self.bah, self.crp, self.eg, self.pamr,
            self.olmar, self.cwmr, self.rmr, self.wmamr,
            self.anticor, self.corn, self.bnn, self.ons
        ]):
            print(f"  {name}…", end=" ", flush=True)
            try:
                r = fn()
            except Exception as exc:
                print(f"HATA: {exc}, CRP kullanılıyor")
                r = np.ones(self.T)
            results.append(r)
            cum = float(np.prod(r))
            print(f"Birikimli: {cum:.4f}")
        return np.array(results)  # (12, T)

    def run_all_portfolios(self):
        """
        Her zaman adımı için portföy ağırlıklarını döndür: (12, T, n).
        Strateji birleştirmesi için gerekli.
        """
        print("Portföy ağırlıkları hesaplanıyor…")
        all_portfolios = []
        for name, fn_name in zip(POLICY_NAMES, [
            "bah_w","crp_w","eg_w","pamr_w",
            "olmar_w","cwmr_w","rmr_w","wmamr_w",
            "anticor_w","corn_w","bnn_w","ons_w"
        ]):
            w_seq = getattr(self, fn_name, None)
            if w_seq is None:
                w_seq = self._run_portfolio_tracking(name)
            all_portfolios.append(w_seq)
        return np.array(all_portfolios)  # (12, T, n)

    def _run_portfolio_tracking(self, name):
        """Portföy ağırlık dizisini döndür"""
        w = self.uniform.copy()
        ws = []
        for t in range(self.T):
            ws.append(w.copy())
        return np.array(ws)


# ══════════════════════════════════════════════════════════════════════════════
# K-means ile Aday Politika Seçimi
# ══════════════════════════════════════════════════════════════════════════════

def kmeans_policy_selection(policy_returns, n_clusters=4, n_init=10, max_iter=300):
    """
    12 politikayı K-means ile kümelere ayır.
    Her kümeden en yüksek birikimli getirili politikayı seç.
    policy_returns: (12, T)
    returns: Seçilen M=4 politikanın indeksleri
    """
    print(f"\nK-means ile {n_clusters} aday strateji seçiliyor…")
    X = policy_returns  # (12, T)
    n_policies = X.shape[0]

    best_inertia = np.inf
    best_labels  = None

    for _ in range(n_init):
        # Rastgele merkez başlatma
        idx = np.random.choice(n_policies, n_clusters, replace=False)
        centroids = X[idx].copy()

        for _ in range(max_iter):
            # Ata
            dists  = np.array([[np.linalg.norm(x - c) for c in centroids] for x in X])
            labels = dists.argmin(axis=1)
            # Güncelle
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if (labels == k).any() else centroids[k]
                for k in range(n_clusters)
            ])
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        inertia = sum(
            np.linalg.norm(X[i] - centroids[labels[i]])**2
            for i in range(n_policies)
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels  = labels.copy()

    # Her kümeden en iyi politikayı seç (en yüksek birikimli getiri)
    selected = []
    cumrets = policy_returns.prod(axis=1)  # (12,)
    for k in range(n_clusters):
        cluster_idx = np.where(best_labels == k)[0]
        if len(cluster_idx) == 0:
            cluster_idx = [0]
        best_in_cluster = cluster_idx[cumrets[cluster_idx].argmax()]
        selected.append(int(best_in_cluster))

    # Tekrar eden varsa farklılarıyla tamamla
    selected = list(dict.fromkeys(selected))
    remaining = [i for i in range(n_policies) if i not in selected]
    remaining.sort(key=lambda i: -cumrets[i])
    while len(selected) < n_clusters:
        selected.append(remaining.pop(0))

    names = [POLICY_NAMES[i] for i in selected]
    print(f"Seçilen stratejiler: {names} (indeksler: {selected})")
    return selected[:n_clusters]


# ══════════════════════════════════════════════════════════════════════════════
# LSTM + Öz-Dikkat Modeli (Saf NumPy)
# ══════════════════════════════════════════════════════════════════════════════

def lstm_step(x, h, c, W, b):
    """Tek LSTM adımı. W: (4H, input+H), b: (4H,)"""
    H = len(h)
    xh    = np.concatenate([x, h])
    gates = W @ xh + b
    f = sigmoid(gates[:H])
    i = sigmoid(gates[H:2*H])
    g = np.tanh(gates[2*H:3*H])
    o = sigmoid(gates[3*H:])
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    cache = (x.copy(), h.copy(), c.copy(), xh.copy(), f, i, g, o, c_new.copy())
    return h_new, c_new, cache


def lstm_layer_forward(x_seq, W, b, H):
    """x_seq: (T, input_size) → hs: (T, H), cache_list"""
    T = x_seq.shape[0]
    h = np.zeros(H)
    c = np.zeros(H)
    hs, caches = [], []
    for t in range(T):
        h, c, cache = lstm_step(x_seq[t], h, c, W, b)
        hs.append(h.copy())
        caches.append(cache)
    return np.array(hs), caches


def lstm_layer_backward(dhs, caches, W, b):
    """
    BPTT. dhs: (T, H)
    Döndürür: dx_seq (T, input_size), dW, db
    """
    T  = len(caches)
    H  = W.shape[0] // 4
    in_sz = caches[0][0].shape[0]
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    dx = np.zeros((T, in_sz))
    dh_next = np.zeros(H)
    dc_next = np.zeros(H)

    for t in reversed(range(T)):
        x, h_prev, c_prev, xh, f, i, g, o, c_new = caches[t]
        dh = dhs[t] + dh_next

        tanh_c = np.tanh(c_new)
        do  = dh * tanh_c
        dc  = dh * o * (1.0 - tanh_c**2) + dc_next
        df  = dc * c_prev
        di  = dc * g
        dg  = dc * i
        dc_next = dc * f

        d_f = f * (1.0 - f) * df
        d_i = i * (1.0 - i) * di
        d_g = (1.0 - g**2) * dg
        d_o = o * (1.0 - o) * do

        dgates = np.concatenate([d_f, d_i, d_g, d_o])
        dW     += np.outer(dgates, xh)
        db     += dgates
        dxh     = W.T @ dgates
        dx[t]   = dxh[:in_sz]
        dh_next = dxh[in_sz:]

    return dx, dW, db


class LMPSModel:
    """
    2 Katmanlı LSTM + Öz-Dikkat ağı, saf NumPy BPTT ile.
    Giriş: (T, M_CAND) — aday politika getirileri
    Çıkış: (T, M_CAND) — karışım ağırlıkları (softmax)
    """
    def __init__(self, M=M_CAND, H=HIDDEN):
        self.M, self.H = M, H
        self.d = max(4, H // 4)   # dikkat boyutu
        self._init_params()

    def _init_params(self):
        M, H, d = self.M, self.H, self.d
        scale = lambda fan_in, fan_out: np.sqrt(2.0 / (fan_in + fan_out))

        self.W1   = np.random.randn(4*H, M+H) * scale(M, H)
        self.b1   = np.zeros(4*H)
        self.W2   = np.random.randn(4*H, H+H) * scale(H, H)
        self.b2   = np.zeros(4*H)
        self.Wq   = np.random.randn(d, H) * 0.02
        self.Wk   = np.random.randn(d, H) * 0.02
        self.Wv   = np.random.randn(H, H) * 0.02
        self.Wout = np.random.randn(M, H) * 0.02
        self.bout = np.zeros(M)

    # ── Parametre paketleme ──────────────────────────────────────────────────
    def pack(self):
        return np.concatenate([
            self.W1.ravel(), self.b1,
            self.W2.ravel(), self.b2,
            self.Wq.ravel(), self.Wk.ravel(), self.Wv.ravel(),
            self.Wout.ravel(), self.bout
        ])

    def unpack(self, theta):
        M, H, d = self.M, self.H, self.d
        specs = [
            ("W1",   (4*H, M+H)), ("b1",   (4*H,)),
            ("W2",   (4*H, H+H)), ("b2",   (4*H,)),
            ("Wq",   (d, H)),     ("Wk",   (d, H)),
            ("Wv",   (H, H)),
            ("Wout", (M, H)),     ("bout", (M,)),
        ]
        params = {}
        idx = 0
        for name, shape in specs:
            n = int(np.prod(shape))
            params[name] = theta[idx:idx+n].reshape(shape)
            idx += n
        return params

    def set_params(self, theta):
        p = self.unpack(theta)
        for k, v in p.items():
            setattr(self, k, v)

    # ── İleri Geçiş ─────────────────────────────────────────────────────────
    def forward(self, x_seq, p=None):
        """
        x_seq: (T, M)
        p: parametre dict (None ise kendi param. kullanılır)
        Döndürür: weights (T, M), cache
        """
        if p is None:
            p = {k: getattr(self, k) for k in
                 ["W1","b1","W2","b2","Wq","Wk","Wv","Wout","bout"]}

        T, M = x_seq.shape
        H, d = self.H, self.d

        # LSTM1
        hs1, cache1 = lstm_layer_forward(x_seq, p["W1"], p["b1"], H)

        # LSTM2
        hs2, cache2 = lstm_layer_forward(hs1, p["W2"], p["b2"], H)

        # Öz-dikkat (nedensel maske)
        Q   = hs2 @ p["Wq"].T                         # (T, d)
        K_  = hs2 @ p["Wk"].T                         # (T, d)
        V   = hs2 @ p["Wv"].T                         # (T, H)
        sc  = Q @ K_.T / np.sqrt(d)                   # (T, T)
        mask = np.triu(np.full((T,T), -1e9), k=1)
        sc  += mask
        attn    = softmax2d(sc)                        # (T, T)
        context = attn @ V                             # (T, H)

        # Çıkış katmanı
        logits  = context @ p["Wout"].T + p["bout"]   # (T, M)
        weights = softmax2d(logits)                    # (T, M)

        cache = dict(x_seq=x_seq, hs1=hs1, hs2=hs2,
                     cache1=cache1, cache2=cache2,
                     Q=Q, K=K_, V=V, attn=attn, context=context,
                     logits=logits, weights=weights, p=p)
        return weights, cache

    # ── Geri Geçiş (BPTT) ────────────────────────────────────────────────────
    def backward(self, dweights, cache):
        """
        dweights: (T, M) — çıkış ağırlıklarına göre kayıp gradyanı
        Döndürür: param_grads dict
        """
        T, M = dweights.shape
        H, d = self.H, self.d
        p    = cache["p"]

        weights = cache["weights"]; context = cache["context"]
        attn    = cache["attn"];    Q = cache["Q"]
        K_      = cache["K"];       V = cache["V"]
        hs2     = cache["hs2"];     hs1 = cache["hs1"]
        cache1  = cache["cache1"]; cache2 = cache["cache2"]

        # 1) Çıkış softmax
        dlogits = dsoftmax2d(weights, dweights)            # (T, M)
        dWout   = dlogits.T @ context                       # (M, H)
        dbout   = dlogits.sum(axis=0)                       # (M,)
        dcontext = dlogits @ p["Wout"]                      # (T, H)

        # 2) Dikkat
        dV    = attn.T @ dcontext                           # (T, H)
        dattn = dcontext @ V.T                              # (T, T)
        dsc   = dsoftmax2d(attn, dattn)                     # (T, T)
        mask  = np.triu(np.ones((T,T), dtype=bool), k=1)
        dsc[mask] = 0.0
        dsc  /= np.sqrt(d)
        dQ    = dsc @ K_                                    # (T, d)
        dK_   = dsc.T @ Q                                  # (T, d)
        dWq   = dQ.T @ hs2                                  # (d, H)
        dWk   = dK_.T @ hs2                                 # (d, H)
        dWv   = dV.T @ hs2                                  # (H, H)
        dhs2_attn = dQ @ p["Wq"] + dK_ @ p["Wk"] + dV @ p["Wv"]  # (T, H)

        # 3) LSTM2 BPTT
        dhs1_from_lstm2, dW2, db2 = lstm_layer_backward(dhs2_attn, cache2, p["W2"], p["b2"])

        # 4) LSTM1 BPTT
        _, dW1, db1 = lstm_layer_backward(dhs1_from_lstm2, cache1, p["W1"], p["b1"])

        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2,
                    Wq=dWq, Wk=dWk, Wv=dWv, Wout=dWout, bout=dbout)

    def grad_pack(self, grads):
        return np.concatenate([
            grads["W1"].ravel(), grads["b1"].ravel(),
            grads["W2"].ravel(), grads["b2"].ravel(),
            grads["Wq"].ravel(), grads["Wk"].ravel(), grads["Wv"].ravel(),
            grads["Wout"].ravel(), grads["bout"].ravel()
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Kayıp Fonksiyonu (Denklem 3)
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_loss(weights, policy_rets, prev_weights=None):
    """
    L = -log(w·r)  +  λ_D * (-H(w))  +  λ_TC * δ * ||w - w_prev||₁
    weights      : (M,) — öngörülen karışım ağırlıkları
    policy_rets  : (M,) — aday politika getirileri
    prev_weights : (M,) veya None
    Döndürür: (scalar kayıp, dweights (M,))
    """
    eps = 1e-8
    r   = float(np.dot(weights, policy_rets))
    r   = max(r, eps)

    # MSE terimi (negatif log-zenginlik)
    loss_mse  = -np.log(r)
    dloss_mse = -policy_rets / r                          # (M,) w'ye göre gradyan

    # Çeşitlilik terimi (entropi maksimizasyonu → minimize edilecek negatif entropi)
    entropy   = -float(np.sum(weights * np.log(weights + eps)))
    loss_div  = -entropy
    dloss_div = np.log(weights + eps) + 1.0               # (M,) w'ye göre

    # İşlem maliyeti terimi
    if prev_weights is not None:
        tc_diff   = np.abs(weights - prev_weights)
        loss_tc   = DELTA * tc_diff.sum()
        dloss_tc  = DELTA * np.sign(weights - prev_weights)
    else:
        loss_tc   = 0.0
        dloss_tc  = np.zeros_like(weights)

    total   = loss_mse + LAMBDA_D * loss_div + LAMBDA_TC * loss_tc
    dtotal  = dloss_mse + LAMBDA_D * dloss_div + LAMBDA_TC * dloss_tc
    return total, dtotal


def sequence_loss_and_grad(model, x_seq, policy_ret_seq, p_flat, prev_w=None):
    """
    Bir dizi üzerinde kayıp ve gradyan hesapla.
    x_seq          : (T, M) — politika getiri penceresi (model girişi)
    policy_ret_seq : (T, M) — gerçek politika getirileri
    p_flat         : düzleştirilmiş parametre vektörü
    """
    model.set_params(p_flat)
    weights_seq, cache = model.forward(x_seq)
    T = weights_seq.shape[0]

    total_loss = 0.0
    dweights = np.zeros_like(weights_seq)

    for t in range(T):
        pw = prev_w if (t == 0 and prev_w is not None) else \
             (weights_seq[t-1] if t > 0 else None)
        l, dw = portfolio_loss(weights_seq[t], policy_ret_seq[t], pw)
        total_loss += l
        dweights[t] = dw

    grads_dict = model.backward(dweights / T, cache)
    grad_vec   = model.grad_pack(grads_dict)
    return total_loss / T, grad_vec


# ══════════════════════════════════════════════════════════════════════════════
# MAML Meta-Eğitim (FOMAML)
# ══════════════════════════════════════════════════════════════════════════════

def maml_train(model, cand_returns, n_epochs=N_EPOCHS, H_batch=H_BATCH,
               K=K_SUPPORT, Q=Q_QUERY, alpha=ALPHA, beta=BETA):
    """
    FOMAML meta-eğitimi.
    cand_returns: (T_train, M_CAND) — eğitim dönemi aday politika getirileri
    """
    print("\nMAML meta-eğitimi başlıyor…")
    T, M = cand_returns.shape
    theta = model.pack()
    losses = []
    window = K + Q

    for epoch in range(n_epochs):
        epoch_loss  = 0.0
        n_batches   = 0
        meta_grad   = np.zeros_like(theta)

        # Geçerli pencere sayısı
        valid_starts = list(range(K, T - Q))
        if len(valid_starts) < H_batch:
            valid_starts = valid_starts * (H_batch // max(1, len(valid_starts)) + 1)
        np.random.shuffle(valid_starts)

        for b_start in range(0, min(len(valid_starts) - H_batch, 200 * H_batch), H_batch):
            task_grads = []
            task_losses = []

            for h in range(H_batch):
                t0 = valid_starts[b_start + h]
                # Destek kümesi: t0-K .. t0-1
                sup_x   = cand_returns[t0-K:t0]          # (K, M)
                sup_y   = cand_returns[t0-K:t0]          # aynı (getiri tahmini)
                # Sorgu: t0 .. t0+Q
                qry_x   = cand_returns[t0:t0+Q]          # (Q, M)
                qry_y   = cand_returns[t0:t0+Q]

                if len(sup_x) < K or len(qry_x) < Q:
                    continue

                # İç döngü: destek kümesi üzerinde uyarla
                theta_i = theta.copy()
                for _ in range(1):   # tek iç adım (FOMAML)
                    _, g_sup = sequence_loss_and_grad(model, sup_x, sup_y, theta_i)
                    theta_i  = theta_i - alpha * g_sup

                # Sorgu kümesi üzerinde gradyan hesapla (uyarlanmış param. ile)
                q_loss, g_qry = sequence_loss_and_grad(model, qry_x, qry_y, theta_i)
                task_grads.append(g_qry)
                task_losses.append(q_loss)

            if not task_losses:
                continue

            # Meta-güncelleme
            avg_grad   = np.mean(task_grads, axis=0)
            avg_loss   = float(np.mean(task_losses))
            theta      = theta - beta * avg_grad
            meta_grad += avg_grad
            epoch_loss += avg_loss
            n_batches  += 1

        avg_epoch_loss = epoch_loss / max(1, n_batches)
        losses.append(avg_epoch_loss)
        print(f"  Epoch {epoch+1:2d}/{n_epochs}  Kayıp: {avg_epoch_loss:.6f}")

    # En iyi parametreleri modele yükle
    model.set_params(theta)
    print("Meta-eğitim tamamlandı.")
    return losses


# ══════════════════════════════════════════════════════════════════════════════
# Meta-Test: Çevrimiçi Uyarlamalı Çıkarım
# ══════════════════════════════════════════════════════════════════════════════

def meta_test(model, cand_returns_train, cand_returns_test,
              cand_portfolios_test, all_policy_rets_test,
              K=K_SUPPORT, alpha=ALPHA):
    """
    Her test günü:
    1. Son K günü destek kümesi olarak al
    2. İç adımla theta'yı uyarla
    3. Karışım ağırlıklarını tahmin et
    4. Portföy oluştur ve getiriyi hesapla
    """
    print("\nMeta-test (çevrimiçi uyarlamalı çıkarım)…")
    T_test, M = cand_returns_test.shape
    theta_star = model.pack()

    lmps_rets    = []
    weight_hist  = []
    prev_w_lmps  = None
    prev_portfolio = None

    # Tüm veriyi birleştir (destek için geçmişe erişim)
    all_cand = np.vstack([cand_returns_train[-K:], cand_returns_test])

    for t in range(T_test):
        # Destek: son K adım
        sup_x   = all_cand[t:t+K]            # (K, M)
        sup_y   = all_cand[t:t+K]

        # İç uyarlama
        theta_i = theta_star.copy()
        if len(sup_x) == K:
            _, g_sup = sequence_loss_and_grad(model, sup_x, sup_y, theta_i)
            theta_i  = theta_i - alpha * g_sup

        # Tahmin (bugünkü adım için)
        model.set_params(theta_i)
        query_x = all_cand[t+K:t+K+1]        # (1, M)
        if len(query_x) == 0:
            query_x = all_cand[t+K-1:t+K]

        w_pred, _ = model.forward(query_x)
        w_mix     = w_pred[-1]                 # (M,) karışım ağırlıkları
        weight_hist.append(w_mix.copy())

        # Gerçek getiri: w_mix · politika_getirileri_t
        policy_r = cand_returns_test[t]        # (M,)
        ret      = float(np.dot(w_mix, policy_r))

        # İşlem maliyeti
        if prev_w_lmps is not None:
            tc = DELTA * float(np.sum(np.abs(w_mix - prev_w_lmps)))
            ret = ret - tc

        lmps_rets.append(max(ret, 1e-4))
        prev_w_lmps = w_mix.copy()

    # Meta parametrelerini geri yükle
    model.set_params(theta_star)

    return np.array(lmps_rets), np.array(weight_hist)


# ══════════════════════════════════════════════════════════════════════════════
# Performans Metrikleri
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(rets, name=""):
    """
    rets: günlük portföy getiri dizisi (çarpımsal, 1.01 = %1 artış)
    """
    rets = np.array(rets)
    rets = np.maximum(rets, 1e-8)
    log_rets   = np.log(rets)
    T          = len(rets)
    ann_factor = 252

    cum_wealth  = float(np.prod(rets))
    ann_return  = float(np.exp(log_rets.mean() * ann_factor) - 1.0)
    ann_vol     = float(log_rets.std() * np.sqrt(ann_factor))
    sharpe      = (ann_return / ann_vol) if ann_vol > 1e-8 else 0.0

    # Maksimum düşüş
    cum_curve  = np.cumprod(rets)
    running_max = np.maximum.accumulate(cum_curve)
    drawdowns   = (running_max - cum_curve) / (running_max + 1e-8)
    max_dd      = float(drawdowns.max())

    return {
        "isim":              name,
        "birikimli_servet":  round(cum_wealth, 4),
        "yillik_getiri":     round(ann_return * 100, 2),
        "yillik_volatilite": round(ann_vol * 100, 2),
        "sharpe_orani":      round(sharpe, 4),
        "maks_dusus":        round(max_dd * 100, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HTML Gösterge Paneli (Chart.js)
# ══════════════════════════════════════════════════════════════════════════════

def build_dashboard(results):
    """Tek dosya etkileşimli HTML gösterge paneli oluştur."""

    dates_test  = results["tarihler"]["test"]
    lmps_wealth = results["zenginlik_egrileri"]["LMPS-SMO"]
    baselines   = results["zenginlik_egrileri"]["baz_stratejiler"]
    weights_ev  = results["karisim_agirliklari"]
    train_loss  = results["egitim_kaybi"]
    metrics     = results["metrikler"]
    cand_names  = results["aday_stratejiler"]
    ds_info     = results["veri_seti"]

    # ── Renk paleti ─────────────────────────────────────────────────────────
    colors = [
        "#00d2ff","#ff6b6b","#ffd700","#7cfc00","#ff69b4","#00ced1",
        "#ff8c00","#9370db","#32cd32","#ff4500","#1e90ff","#adff2f",
        "#dc143c"
    ]

    def js_arr(lst):
        return json.dumps([round(v, 6) for v in lst])

    def js_str_arr(lst):
        return json.dumps(lst)

    # ── Baz strateji veri setleri ────────────────────────────────────────────
    baseline_datasets = ""
    for ci, (bname, bw) in enumerate(baselines.items()):
        col = colors[(ci + 1) % len(colors)]
        baseline_datasets += f"""
        {{
            label: '{bname}',
            data: {js_arr(bw)},
            borderColor: '{col}',
            backgroundColor: '{col}22',
            borderWidth: 1.2,
            pointRadius: 0,
            borderDash: [4,3],
            tension: 0.1
        }},"""

    # ── Karışım ağırlığı veri setleri ────────────────────────────────────────
    weight_datasets = ""
    w_arr = np.array(weights_ev)   # (T_test, M)
    for m, cname in enumerate(cand_names):
        col = colors[m % len(colors)]
        wm  = w_arr[:, m].tolist()
        weight_datasets += f"""
        {{
            label: '{cname}',
            data: {js_arr(wm)},
            backgroundColor: '{col}88',
            borderColor: '{col}',
            borderWidth: 1,
            fill: true,
            tension: 0.3,
            pointRadius: 0
        }},"""

    # ── Metrik tablosu ────────────────────────────────────────────────────────
    metric_rows = ""
    for m in metrics:
        is_lmps = m["isim"] == "LMPS-SMO"
        row_cls = "lmps-row" if is_lmps else ""
        metric_rows += f"""
        <tr class="{row_cls}">
            <td>{m['isim']}</td>
            <td>{m['birikimli_servet']:.4f}</td>
            <td>{m['yillik_getiri']:.2f}%</td>
            <td>{m['yillik_volatilite']:.2f}%</td>
            <td>{m['sharpe_orani']:.4f}</td>
            <td>{m['maks_dusus']:.2f}%</td>
        </tr>"""

    epochs = list(range(1, len(train_loss) + 1))

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LMPS-SMO · BIST Portföy Analizi</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0a0e1a; --card: #111827; --border: #1f2937;
    --text: #e5e7eb; --muted: #9ca3af; --accent: #00d2ff;
    --green: #22c55e; --red: #ef4444; --gold: #fbbf24;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
         font-family: 'Segoe UI', system-ui, sans-serif;
         min-height: 100vh; }}
  header {{ background: linear-gradient(135deg,#0f172a,#1e3a5f);
            padding: 24px 32px; border-bottom: 1px solid var(--border); }}
  header h1 {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); }}
  header p  {{ color: var(--muted); font-size: 0.85rem; margin-top: 4px; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 99px;
    font-size: 0.72rem; font-weight: 600; margin-left: 8px;
    background: #00d2ff22; color: var(--accent); border: 1px solid var(--accent);
  }}
  main {{ padding: 24px 32px; max-width: 1600px; margin: 0 auto; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .grid3 {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; margin-bottom: 20px; }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
  }}
  .card h2 {{ font-size: 0.95rem; font-weight: 600; color: var(--muted);
              text-transform: uppercase; letter-spacing: .05em; margin-bottom: 16px; }}
  .chart-wrap {{ position: relative; height: 280px; }}
  .chart-wrap.tall {{ height: 360px; }}
  .kpi {{ text-align: center; }}
  .kpi .val {{ font-size: 2rem; font-weight: 700; color: var(--accent); }}
  .kpi .lbl {{ font-size: 0.78rem; color: var(--muted); margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: #1f2937; color: var(--muted); padding: 10px 12px;
        text-align: right; font-weight: 600; position: sticky; top: 0; }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid #1f293799; }}
  td:first-child {{ text-align: left; font-weight: 500; }}
  tr:hover td {{ background: #ffffff08; }}
  .lmps-row td {{ color: var(--accent); font-weight: 700; background: #00d2ff0d; }}
  .info-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; }}
  .info-item {{ background: #1f2937; border-radius: 8px; padding: 14px; }}
  .info-item .il {{ font-size: 0.75rem; color: var(--muted); }}
  .info-item .iv {{ font-size: 1.05rem; font-weight: 600; margin-top: 4px; }}
  footer {{ text-align: center; padding: 24px; color: var(--muted); font-size: 0.78rem;
            border-top: 1px solid var(--border); margin-top: 20px; }}
  @media (max-width: 900px) {{
    .grid2, .grid3 {{ grid-template-columns: 1fr; }}
    .info-grid {{ grid-template-columns: 1fr 1fr; }}
  }}
</style>
</head>
<body>

<header>
  <h1>LMPS-SMO <span class="badge">arXiv:2505.03659</span></h1>
  <p>Meta-Learning ile Optimal Strateji Karışımı · Borsa İstanbul (BIST) Uygulaması</p>
</header>

<main>

<!-- Veri Seti Bilgisi -->
<div class="card" style="margin-bottom:20px">
  <h2>Veri Seti Bilgisi</h2>
  <div class="info-grid">
    <div class="info-item">
      <div class="il">Borsa</div>
      <div class="iv">{ds_info['borsa']}</div>
    </div>
    <div class="info-item">
      <div class="il">Hisse Senedi Sayısı</div>
      <div class="iv">{ds_info['hisse_sayisi']}</div>
    </div>
    <div class="info-item">
      <div class="il">Dönem</div>
      <div class="iv">{ds_info['donem']}</div>
    </div>
    <div class="info-item">
      <div class="il">Eğitim / Test</div>
      <div class="iv">{ds_info['egitim_gun']} / {ds_info['test_gun']} gün</div>
    </div>
    <div class="info-item">
      <div class="il">Aday Stratejiler</div>
      <div class="iv">{', '.join(cand_names)}</div>
    </div>
    <div class="info-item">
      <div class="il">İşlem Maliyeti</div>
      <div class="iv">δ = {DELTA}</div>
    </div>
    <div class="info-item">
      <div class="il">Veri Kaynağı</div>
      <div class="iv">{'Gerçek (yfinance)' if not ds_info.get('sentetik') else 'Sentetik (BIST kalibreli)'}</div>
    </div>
    <div class="info-item">
      <div class="il">Model</div>
      <div class="iv">2-Katman LSTM + Öz-Dikkat</div>
    </div>
  </div>
</div>

<!-- Birikimli Zenginlik -->
<div class="card" style="margin-bottom:20px">
  <h2>Birikimli Zenginlik Eğrileri (Test Dönemi)</h2>
  <div class="chart-wrap tall">
    <canvas id="wealthChart"></canvas>
  </div>
</div>

<!-- Alt satır: Karışım ağırlıkları + Eğitim kaybı -->
<div class="grid2">
  <div class="card">
    <h2>Karışım Ağırlıklarının Zaman İçindeki Değişimi</h2>
    <div class="chart-wrap">
      <canvas id="weightChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>Meta-Eğitim Kayıp Eğrisi</h2>
    <div class="chart-wrap">
      <canvas id="lossChart"></canvas>
    </div>
  </div>
</div>

<!-- Performans metrikleri tablosu -->
<div class="card" style="margin-bottom:20px; overflow-x:auto">
  <h2>Performans Metrikleri</h2>
  <table>
    <thead>
      <tr>
        <th>Strateji</th>
        <th>Birikimli Servet</th>
        <th>Yıllık Getiri</th>
        <th>Yıllık Volatilite</th>
        <th>Sharpe Oranı</th>
        <th>Maks. Düşüş</th>
      </tr>
    </thead>
    <tbody>
      {metric_rows}
    </tbody>
  </table>
</div>

</main>

<footer>LMPS-SMO · arXiv:2505.03659 · BIST Uygulaması · Saf NumPy Uygulaması</footer>

<script>
Chart.defaults.color = '#9ca3af';
Chart.defaults.borderColor = '#1f2937';

const testDates = {js_str_arr(dates_test)};

// ── Birikimli zenginlik ──────────────────────────────────────────────────────
const wealthCtx = document.getElementById('wealthChart').getContext('2d');
new Chart(wealthCtx, {{
  type: 'line',
  data: {{
    labels: testDates,
    datasets: [
      {{
        label: 'LMPS-SMO',
        data: {js_arr(lmps_wealth)},
        borderColor: '#00d2ff',
        backgroundColor: '#00d2ff18',
        borderWidth: 2.5,
        pointRadius: 0,
        tension: 0.2,
        fill: false,
        order: 0
      }},
      {baseline_datasets}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ position: 'top', labels: {{ boxWidth: 12, font: {{ size: 11 }} }} }},
      tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(4)}}` }} }}
    }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 12, maxRotation: 0 }}, grid: {{ color: '#1f293766' }} }},
      y: {{ title: {{ display: true, text: 'Birikimli Servet' }}, grid: {{ color: '#1f293766' }} }}
    }}
  }}
}});

// ── Karışım ağırlıkları ──────────────────────────────────────────────────────
const weightCtx = document.getElementById('weightChart').getContext('2d');
new Chart(weightCtx, {{
  type: 'line',
  data: {{
    labels: testDates,
    datasets: [{weight_datasets}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ position: 'top', labels: {{ boxWidth: 12, font: {{ size: 11 }} }} }},
      tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{(ctx.parsed.y*100).toFixed(1)}}%` }} }}
    }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 10, maxRotation: 0 }}, grid: {{ color: '#1f293766' }} }},
      y: {{
        stacked: true,
        min: 0, max: 1,
        title: {{ display: true, text: 'Ağırlık' }},
        grid: {{ color: '#1f293766' }},
        ticks: {{ callback: v => (v*100).toFixed(0)+'%' }}
      }}
    }}
  }}
}});

// ── Eğitim kaybı ─────────────────────────────────────────────────────────────
const lossCtx = document.getElementById('lossChart').getContext('2d');
new Chart(lossCtx, {{
  type: 'line',
  data: {{
    labels: {js_str_arr([str(e) for e in epochs])},
    datasets: [{{
      label: 'Meta-Eğitim Kaybı',
      data: {js_arr(train_loss)},
      borderColor: '#fbbf24',
      backgroundColor: '#fbbf2422',
      borderWidth: 2,
      pointRadius: 4,
      pointBackgroundColor: '#fbbf24',
      tension: 0.3,
      fill: true
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top', labels: {{ boxWidth: 12, font: {{ size: 11 }} }} }}
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Epoch' }}, grid: {{ color: '#1f293766' }} }},
      y: {{ title: {{ display: true, text: 'Kayıp' }}, grid: {{ color: '#1f293766' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
# Ana Fonksiyon
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(__file__).parent
    print("=" * 65)
    print(" LMPS-SMO: BIST Portföy Optimizasyonu")
    print("=" * 65)

    # ── 1. Veri ──────────────────────────────────────────────────────────────
    prices, symbols, dates, is_synthetic = fetch_data()
    T_total = prices.shape[0]
    n_stocks = prices.shape[1]

    # Fiyat göreli matrisi: x_t = p_t / p_{t-1}
    price_rels = prices[1:] / (prices[:-1] + 1e-8)  # (T-1, n)
    dates_pr   = dates[1:]
    T = len(price_rels)

    # Eğitim/Test bölünmesi (%75 / %25)
    split     = int(T * 0.75)
    X_train   = price_rels[:split]
    X_test    = price_rels[split:]
    dates_test = dates_pr[split:]
    T_test     = len(X_test)
    print(f"\nEğitim: {split} gün  |  Test: {T_test} gün")

    # ── 2. 12 OLPS stratejisi çalıştır ───────────────────────────────────────
    runner_train = OLPSRunner(X_train)
    policy_rets_train = runner_train.run_all()   # (12, T_train)

    runner_test  = OLPSRunner(X_test)
    policy_rets_test  = runner_test.run_all()    # (12, T_test)

    # ── 3. K-means ile aday strateji seçimi ─────────────────────────────────
    cand_idx = kmeans_policy_selection(policy_rets_train, n_clusters=M_CAND)
    cand_names = [POLICY_NAMES[i] for i in cand_idx]

    # Aday politika getirileri: (T, M)
    cand_train = policy_rets_train[cand_idx].T   # (T_train, M)
    cand_test  = policy_rets_test[cand_idx].T    # (T_test, M)

    # ── 4. Model başlat & MAML eğitimi ──────────────────────────────────────
    model = LMPSModel(M=M_CAND, H=HIDDEN)
    print(f"\nModel parametresi: {len(model.pack()):,}")

    train_losses = maml_train(
        model, cand_train,
        n_epochs=N_EPOCHS, H_batch=H_BATCH,
        K=K_SUPPORT, Q=Q_QUERY,
        alpha=ALPHA, beta=BETA
    )

    # ── 5. Meta-test ─────────────────────────────────────────────────────────
    lmps_rets, weight_hist = meta_test(
        model, cand_train, cand_test,
        cand_portfolios_test=None,
        all_policy_rets_test=policy_rets_test,
        K=K_SUPPORT, alpha=ALPHA
    )

    # ── 6. Birikimli zenginlik eğrileri ─────────────────────────────────────
    def cum_wealth_curve(rets):
        return list(np.cumprod(np.maximum(rets, 1e-8)))

    lmps_wealth = cum_wealth_curve(lmps_rets)

    baseline_wealth = {}
    for idx_p, pname in enumerate(POLICY_NAMES):
        baseline_wealth[pname] = cum_wealth_curve(policy_rets_test[idx_p])

    # ── 7. Metrikler ────────────────────────────────────────────────────────
    all_metrics = [compute_metrics(lmps_rets, "LMPS-SMO")]
    for idx_p, pname in enumerate(POLICY_NAMES):
        all_metrics.append(compute_metrics(policy_rets_test[idx_p], pname))

    # Tabloyu yazdır
    print("\n" + "─" * 75)
    print(f"{'Strateji':<12} {'Birikimli':>12} {'Yıllık G.%':>12} "
          f"{'Volatilite%':>12} {'Sharpe':>10} {'MaksDüşüş%':>12}")
    print("─" * 75)
    for m in all_metrics:
        marker = " ◄" if m["isim"] == "LMPS-SMO" else ""
        print(f"{m['isim']:<12} {m['birikimli_servet']:>12.4f} "
              f"{m['yillik_getiri']:>12.2f} {m['yillik_volatilite']:>12.2f} "
              f"{m['sharpe_orani']:>10.4f} {m['maks_dusus']:>12.2f}{marker}")
    print("─" * 75)

    # ── 8. Sonuçları kaydet ──────────────────────────────────────────────────
    results = {
        "tarihler": {
            "test":   dates_test,
        },
        "zenginlik_egrileri": {
            "LMPS-SMO":       lmps_wealth,
            "baz_stratejiler": baseline_wealth,
        },
        "karisim_agirliklari": weight_hist.tolist(),
        "egitim_kaybi":      train_losses,
        "metrikler":         all_metrics,
        "aday_stratejiler":  cand_names,
        "veri_seti": {
            "borsa":       "Borsa İstanbul (BIST)",
            "hisse_sayisi": n_stocks,
            "donem":        f"{dates[0]} – {dates[-1]}",
            "egitim_gun":   split,
            "test_gun":     T_test,
            "sentetik":     is_synthetic,
        },
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSonuçlar kaydedildi → {results_path}")

    # ── 9. Dashboard ─────────────────────────────────────────────────────────
    html = build_dashboard(results)
    dash_path = out_dir / "dashboard.html"
    with open(dash_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Gösterge paneli oluşturuldu → {dash_path}")
    print("\nTamamlandı!")
    return results
    # 9.a Basic matplotlib görselleştirme
    plt.figure(figsize=(10, 6))
    plt.plot(dates_test, lmps_wealth, label="LMPS-SMO", color="#00d2ff", linewidth=2.5)
    for idx_p, pname in enumerate(POLICY_NAMES):
        plt.plot(dates_test, baseline_wealth[pname], label=pname, linewidth=1.2, linestyle='--')
    plt.title("Birikimli Zenginlik Eğrileri (Test Dönemi)")
    plt.xlabel("Tarih")
    plt.ylabel("Birikimli Servet")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt_path = out_dir / "wealth_curves.png"
    plt.savefig(plt_path)
    print(f"Görselleştirme kaydedildi → {plt_path}")

if __name__ == "__main__":
    main()
