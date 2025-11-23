#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Martha Elias
# Author: Martha Elias  (marthaelias [at] protonmail [dot] com)
# Version: v1.0 (October 2025)
# DOI: 10.5281/zenodo.17379025
# License: Apache 2.0
#
# I'd be happy if yo like my work: https://buymeacoffee.com/marthafay
#
"""
CAUTION
Deterministic modeling is vulnerable to unnatural distortions and algorithmically triggered reactions.
Independent safety and risk management strategies are essential.

DISCLAIMER (Research Only)
This repository contains a research prototype. It is provided for educational and research purposes only. It does NOT constitute financial, investment, legal, medical, or any other professional advice. No warranty is given. Use at your own risk. Before using any outputs to inform real-world decisions, obtain advice from qualified professionals and perform independent verification.

----------------
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

Note on Terminology
-------------------
All terms, metaphors, and model names used in this repository (e.g. "Pseudoscalar Score", "Schrödinger Zone", "OddSpin", "MaxwellFlux", etc.) are original to the author. These names are not in the public domain. Any use of these names, terms, or model identifiers — especially for commercial or branding purposes — is prohibited without prior written permission from the author. This restriction applies regardless of whether the underlying code or methods are licensed under open-source terms.

xi_robotics_offline_demo.py — Robotiknahe Offline-Demo mit T7-Kopplung + Lag-/Drop-Erkennung

What it shows
- Synthetic multi-joint data: qd_j(t), qdd_j(t), tau_cmd_j(t), tau_meas_j(t)
- Windowed metrics:
- PLV_v – motion phase coherence across joints
- PLV(u, τΣ), IAI(u, τΣ) – circular coupling/orientation
- ρ_w(u, ddqΣ) – reactivity
- Odd/Chi/P – pseudoscalar orientation
- Lag(u, τΣ) – latency / desynchronization
- PLV_v (short) – rapid coherence changes
- Strategy operator Ξ with T7 coupling and adaptive weights
- Projection onto 4 decision axes (execute / guard / stop / pivot)
- Interactive plot of |Ξ|, T7, PLV (long/short), and lag

Dependencies

numpy, (optional) matplotlib

"""

import argparse, json
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

EPS = 1e-12

# ---------- Hilfsfunktionen / Fenster / Analytik ----------

def hann(n: int) -> np.ndarray:
    if n < 8:
        return np.ones(n, float)
    t = np.linspace(0, np.pi, n)
    w = 0.5 * (1 - np.cos(t)) * 2.0
    return np.clip(w, 0.05, 1.0)

def analytic(x: np.ndarray) -> np.ndarray:
    """FFT-Hilbert (Fenster-basiert, nicht kausal)"""
    x = np.asarray(x, float)
    n = x.size
    X = np.fft.fft(x)
    H = np.zeros(n)
    if n % 2 == 0:
        H[0] = H[n//2] = 1.0
        H[1:n//2] = 2.0
    else:
        H[0] = 1.0
        H[1:(n+1)//2] = 2.0
    return np.fft.ifft(X * H)

def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    wm = w.sum() + EPS
    mx = (w @ x) / wm
    my = (w @ y) / wm
    sx = np.sqrt(((w*(x-mx)**2).sum())/wm + EPS)
    sy = np.sqrt(((w*(y-my)**2).sum())/wm + EPS)
    return float(np.clip(((w*(x-mx)*(y-my)).sum())/(wm*sx*sy + EPS), -1, 1))

def plv_v(qd_win: np.ndarray, alpha: np.ndarray) -> float:
    """Bewegungs-Phasenkohärenz über Joints"""
    J, W = qd_win.shape
    z = np.empty((J, W), complex)
    for j in range(J):
        z[j] = analytic(qd_win[j])
    phi = np.angle(z)
    s_t = np.sum((alpha[:, None] * np.exp(1j * phi)), axis=0)
    return float(np.abs(np.mean(s_t)))

def plv_pair(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    zu, zv = analytic(u), analytic(v)
    dphi = np.angle(zu * np.conj(zv))
    wn = w / (w.sum() + EPS)
    return float(np.clip(np.abs(np.sum(wn * np.exp(1j * dphi))), 0, 1))

def iai_pair(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    zu, zv = analytic(u), analytic(v)
    dphi = np.angle(zu * np.conj(zv))
    wn = w / (w.sum() + EPS)
    return float(np.clip(np.sum(wn * np.sin(dphi)), -1, 1))

def odd_chiral(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float,float,float]:
    """Oddness + Chirality + Produkt P"""
    xr = x[::-1]
    x_even = 0.5*(x + xr)
    x_odd = 0.5*(x - xr)
    E_even = float((w*(x_even**2)).sum())
    E_odd  = float((w*(x_odd**2)).sum())
    Odd = float(np.clip(E_odd / max(E_even + E_odd + EPS, EPS), 0, 1))
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    num = float((w*(x*dy - y*dx)).sum())
    den = float((w*(x*x + y*y)).sum() + EPS)
    Chi = float(np.clip(num / den, -1, 1))
    return Odd, Chi, float(Odd * Chi)

def softmax(vals, T=0.6):
    v = np.asarray(vals, float) / max(T, 1e-6)
    v -= v.max()
    e = np.exp(v)
    return (e / (e.sum() + EPS)).tolist()

# ---------- Lag-/Desync-Helfer ----------

def _normxcorr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    na = np.linalg.norm(a) + EPS
    nb = np.linalg.norm(b) + EPS
    return float(np.dot(a, b) / (na * nb))

def estimate_signed_lag(a: np.ndarray, b: np.ndarray, maxlag: int = 20) -> tuple[int, float]:
    """Signierter Lag via normierter Kreuzkorrelation"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    M = int(max(1, maxlag))
    best_lag, best_c = 0, 0.0
    for lag in range(-M, M + 1):
        if lag < 0:
            aa, bb = a[:lag], b[-lag:]
        elif lag > 0:
            aa, bb = a[lag:], b[:-lag]
        else:
            aa, bb = a, b
        if aa.size < 8:
            continue
        c = _normxcorr(aa, bb)
        if abs(c) > abs(best_c):
            best_c, best_lag = c, lag
    return best_lag, best_c

# ---------- T7-Helfer ----------

def mad_scale(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if x.size == 0:
        return 1e-6
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m)) + 1e-12

def plv_from_phase(phi: np.ndarray) -> float:
    if phi.size == 0:
        return 0.0
    z = np.exp(1j * phi)
    return float(np.abs(np.mean(z)))

def t7_from_window(x: np.ndarray, phi: np.ndarray, eps: float | None = None) -> tuple[float, float]:
    """T7 = 0.4*RunLength + 0.4*Stabilität + 0.2*PLV"""
    x = np.asarray(x, float)
    if eps is None:
        eps = 1.2 * mad_scale(np.diff(x))
    dx = np.diff(x, prepend=x[0])
    run = float(np.mean(np.abs(dx) < eps))
    stab = float(np.clip(1.0 - np.tanh(0.7 * np.std(x)), 0.0, 1.0))
    plv_val = plv_from_phase(phi) if phi.size else 0.0
    t7 = float(np.clip(0.4*run + 0.4*stab + 0.2*plv_val, 0.0, 1.0))
    return t7, float(eps)

# ---------- Synthetische Robotikdaten ----------

def synth_robotic(J=6, fs=50, duration=30, seed=7):
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    f0 = 0.6
    f = f0 + rng.uniform(-0.12, 0.12, J)
    A_qd = 0.6 + 0.3 * rng.random(J)
    phi0 = rng.uniform(0, np.pi, J)

    qd = np.zeros((J, n))
    qdd = np.zeros((J, n))
    tau_cmd = np.zeros((J, n))
    tau_meas = np.zeros((J, n))
    for j in range(J):
        wj = 2 * np.pi * f[j]
        qd[j] = A_qd[j]*np.sin(wj*t + phi0[j]) + 0.10*rng.standard_normal(n)
        qdd[j] = np.gradient(qd[j], 1/fs)
        tau_cmd[j] = (0.9*A_qd[j])*np.sin(wj*t + phi0[j] + 0.25) + 0.08*rng.standard_normal(n)
        alpha = 0.92
        for k in range(1, n):
            tau_meas[j, k] = alpha*tau_meas[j, k-1] + (1-alpha)*tau_cmd[j, k] \
                             + 0.04*np.sign(qd[j, k]) + 0.04*rng.standard_normal()
    return t, qd, qdd, tau_cmd, tau_meas

# ---------- Operator mit T7 + Lag + Drop ----------

def compute_xi_timeseries(qd, qdd, tau_cmd, tau_meas,
                          fs=50, window=256, step=8,
                          alpha=None, weights=None, temp=0.6,
                          max_lag_ms=120.0, short_frac=0.25,
                          t7_hard_guard_floor=0.22):
    """
    Hauptberechnung: Ξ + T7 + Lag + MultiRes-Drop
    """
    J, N = qd.shape
    W = int(window)
    S = int(max(1, step))
    if alpha is None:
        alpha = np.ones(J)/J
    if weights is None:
        weights = dict(wC=0.45, wR=0.35, wZ=0.30, vP=0.6, vI=0.4,
                       wL=0.25, vL=0.35, wDrop=0.20)
    angles = np.deg2rad([0,90,180,270])
    labels = ["execute","guard","stop","pivot"]

    u = (alpha[:, None]*tau_cmd).sum(axis=0)
    tauS = (alpha[:, None]*tau_meas).sum(axis=0)
    ddqS = (alpha[:, None]*qdd).sum(axis=0)

    maxlag = int(round((max_lag_ms/1000.0)*fs))
    Ws = max(16, int(W*short_frac))

    out = dict(t_idx=[], absXi=[], angXi=[], probs=[], action=[],
               PLV_v=[], PLV_v_short=[], PLV_uy=[], IAI_cs=[], rho_ud=[],
               G=[], T7=[], lag_frames=[], lag_idx=[])

    for start in range(0, N - W + 1, S):
        s = slice(start, start + W)
        w = hann(W)

        PLVv_long = plv_v(qd[:, s], alpha)
        s2 = slice(start + W - Ws, start + W)
        PLVv_short = plv_v(qd[:, s2], alpha)

        rho_ud = abs(weighted_corr(u[s], ddqS[s], w))
        PLV_uy = plv_pair(u[s], tauS[s], w)
        IAI_cs = iai_pair(u[s], tauS[s], w)
        Odd, Chi, P = odd_chiral(u[s], tauS[s], w)

        u_abs = np.abs(u[s])
        phi_u = np.unwrap(np.angle(analytic(u[s])))
        T7, _ = t7_from_window(u_abs, phi_u)

        lag_samp, _ = estimate_signed_lag(u[s], tauS[s], maxlag=maxlag)
        lag_idx = np.clip(abs(lag_samp)/max(1, maxlag), 0.0, 1.0)
        lag_signed = np.sign(lag_samp) * lag_idx

        Z = (1 - PLV_uy) * (1 - rho_ud)
        drop_fast = np.clip(PLVv_long - PLVv_short, 0.0, 1.0)

        ReXi = (weights["wC"]*PLVv_long +
                weights["wR"]*rho_ud -
                weights["wZ"]*Z -
                weights["wL"]*lag_idx -
                weights["wDrop"]*drop_fast)
        ImXi = (weights["vP"]*P +
                weights["vI"]*IAI_cs +
                weights["vL"]*lag_signed)
        Xi = ReXi + 1j*ImXi

        G = (PLVv_long * PLV_uy * rho_ud) * (1.0 - 0.6*lag_idx) * T7
        T_eff = float(np.clip(temp * (1 - 0.5*(T7 - 0.5)), 0.35, 1.20))
        rot = 0.25*np.tanh(2.0*G) * np.sign(IAI_cs)
        Xi_t = Xi * np.exp(1j * rot)

        mag = float(np.abs(Xi_t))
        ang = float(np.angle(Xi_t))
        vals = [mag*np.cos(ang - th) for th in angles]
        p = softmax(vals, T=T_eff)
        if T7 < t7_hard_guard_floor:
            p = [0.06, 0.76, 0.12, 0.06]
            a = "guard"
        else:
            a = labels[int(np.argmax(p))]

        out["t_idx"].append(start + W//2)
        out["absXi"].append(mag)
        out["angXi"].append(ang)
        out["probs"].append(p)
        out["action"].append(a)
        out["PLV_v"].append(PLVv_long)
        out["PLV_v_short"].append(PLVv_short)
        out["PLV_uy"].append(PLV_uy)
        out["IAI_cs"].append(IAI_cs)
        out["rho_ud"].append(rho_ud)
        out["G"].append(G)
        out["T7"].append(T7)
        out["lag_frames"].append(lag_samp)
        out["lag_idx"].append(lag_idx)

    for k in out:
        if k == "action": continue
        out[k] = np.asarray(out[k])
    return out

# ---------- Plot ----------

def plot_results(res):
    if plt is None:
        print("Matplotlib nicht verfügbar – bitte matplotlib installieren.")
        return

    t = res["t_idx"]
    P = np.array(res["probs"])
    labels = ["execute","guard","stop","pivot"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, res["absXi"], label="|Ξ|")
    axes[0].plot(t, res.get("T7", np.zeros_like(t)), label="T7", alpha=0.7)
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_ylabel("|Ξ|, T7")

    axes[1].plot(t, res["PLV_v"], label="PLV_v (long)")
    axes[1].plot(t, res["PLV_v_short"], label="PLV_v (short)", alpha=0.8)
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylabel("PLV_v")

    axes[2].plot(t, res["lag_frames"], label="Lag (samples)")
    axes[2].plot(t, res["lag_idx"], label="LagIdx (0..1)")
    axes[2].legend(); axes[2].grid(alpha=0.3); axes[2].set_ylabel("Lag")

    for i, lab in enumerate(labels):
        axes[3].plot(P[:, i], label=lab)
    axes[3].legend(); axes[3].set_ylim(-0.02, 1.02)
    axes[3].grid(alpha=0.3); axes[3].set_ylabel("Policy"); axes[3].set_xlabel("Sample index")

    plt.tight_layout()
    plt.show()

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", type=int, default=6)
    ap.add_argument("--fs", type=int, default=50)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save", type=str, default="", help="Optional: save JSON (z.B. xi_demo_out.json)")
    ap.add_argument("--max_lag_ms", type=float, default=120.0, help="Maximaler Lag für Lag-Suche (ms)")
    ap.add_argument("--short_frac", type=float, default=0.25, help="Anteil des Hauptfensters für kurzes Zusatzfenster (0.2–0.5)")
    args = ap.parse_args()

    # --- Datensynthese ---
    t, qd, qdd, tau_cmd, tau_meas = synth_robotic(
        J=args.joints, fs=args.fs, duration=args.duration, seed=args.seed
    )

    # --- Xi-Berechnung ---
    res = compute_xi_timeseries(
        qd, qdd, tau_cmd, tau_meas,
        fs=args.fs, window=args.window, step=args.step,
        temp=args.temp, max_lag_ms=args.max_lag_ms, short_frac=args.short_frac
    )

    # --- Kurze Statistik ---
    hist = {}
    for a in res["action"]:
        hist[a] = hist.get(a, 0) + 1
    print(f"Frames: {len(res['t_idx'])} | mean|Ξ|={res['absXi'].mean():.3f} | actions={hist}")

    # --- Optional speichern ---
    if args.save:
        payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in res.items()}
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        print(f"Gespeichert: {args.save}")

    # --- Optional plotten ---
    if args.plot:
        plot_results(res)

if __name__ == "__main__":
    main()