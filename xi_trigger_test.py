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


xi_trigger_test.py — Trigger & Redirect Tests for Ξ (offline, with plots)

Purpose:
  • Simulates disturbances (Orientation Flip, Latency/Desync, Coherence Drop)
  • Evaluates reaction latency, evidence changes (|Ξ|, Gate, PLV, ρ), and recovery behavior
  • Produces visualizations and optional PNG exports

Plot Features:
  • Time axis: seconds (or samples via --xaxis samples)
  • Event visualization: shaded regions (event + pre/post windows)
  • Dual y-axes: |Ξ| (left) and Gate (right)
  • Additional panel: PLV_v, PLV(u,τΣ), ρ(u,ddqΣ)
  • Optional smoothing for readability
  • PNG export (--save plot.png)

Example:
  python xi_trigger_test.py --fs 100 --duration 40 --window 256 --step 8 --plot --save out.png

Updates:
  - Safe statistics (no "mean of empty slice" warnings)
  - Scenario windows visualized as shaded areas (event + pre/post)
  - --save <prefix> automatically saves A/B/C plots as PNG
"""

import argparse
import numpy as np
import sys
from typing import Dict, Tuple, List, Optional

# ==== Import der Demo-Funktionen ============================================
try:
    from xi_robotics_offline_demo import synth_robotic, compute_xi_timeseries
except Exception as e:
    print("Fehler: Konnte xi_robotics_offline_demo nicht importieren.\n"
          "Bitte lege dieses Testskript in denselben Ordner wie xi_robotics_offline_demo.py.\n"
          f"Import-Fehler: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ==== Hilfsfunktionen =======================================================

ACTIONS = ["execute", "guard", "stop", "pivot"]
ACT2IDX = {a: i for i, a in enumerate(ACTIONS)}

def _nearest_frame_idx(frame_centers: np.ndarray, sample_idx: int) -> int:
    """Erster Frame, dessen Fenstermitte >= sample_idx liegt."""
    idx = np.searchsorted(frame_centers, sample_idx, side="left")
    return int(np.clip(idx, 0, len(frame_centers) - 1))

def _hist(actions: List[str]) -> Dict[str, int]:
    h = {}
    for a in actions:
        h[a] = h.get(a, 0) + 1
    return h

def _policy_array(probs: List[List[float]]) -> np.ndarray:
    return np.asarray(probs, float)

def _print_header(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def _fmt(x: float) -> str:
    return "n/a" if (x is None or np.isnan(x)) else f"{x:+.3f}"

def _safe_slice(a: np.ndarray, i0: int, i1: int) -> Optional[np.ndarray]:
    i0 = max(0, int(i0)); i1 = max(0, int(i1))
    if i1 <= i0:
        return None
    return a[i0:i1]

def _safe_mean(x: Optional[np.ndarray]) -> float:
    if x is None or x.size == 0:
        return float("nan")
    return float(np.nanmean(x))

def _safe_mean_vec(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if x is None or x.size == 0:
        return None
    return np.nanmean(x, axis=0)

# ==== Störszenarien =========================================================

def scenario_orientation_flip(tau_meas: np.ndarray, start: int, end: int) -> None:
    tau_meas[:, start:end] *= -1.0

def scenario_latency_desync(tau_cmd: np.ndarray, lag_samples: int) -> None:
    if lag_samples <= 0:
        return
    J, N = tau_cmd.shape
    tau_cmd[:, lag_samples:] = tau_cmd[:, :-lag_samples]
    tau_cmd[:, :lag_samples] = 0.0

def scenario_coherence_drop(qd: np.ndarray, start: int, end: int, noise_amp: float = 0.6, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    qd[:, start:end] += noise_amp * rng.standard_normal((qd.shape[0], end - start))

# ==== Auswertung / Trigger-Detektion =======================================

def evaluate_redirect(res: Dict, event_start_smp: int, event_end_smp: int,
                      fs: int, window: int, step: int,
                      baseline_action: str, redirect_targets=("guard","pivot","stop"),
                      max_latency_frames: Optional[int] = None) -> Dict:
    """
    Robuste Pre-/Post-Auswertung um das Event-Fenster. Liefert NaN statt Warnungen,
    wenn Pre/Post nicht existieren (z.B. globales Event ab t=0).
    """
    centers = res["t_idx"]
    i_start = _nearest_frame_idx(centers, event_start_smp)
    i_end   = _nearest_frame_idx(centers, event_end_smp)

    actions = res["action"]
    probs   = _policy_array(res["probs"])
    G       = np.asarray(res["G"], float)
    Xi      = np.asarray(res["absXi"], float)
    PLVv    = np.asarray(res["PLV_v"], float)
    PLVuy   = np.asarray(res["PLV_uy"], float)
    IAI     = np.asarray(res["IAI_cs"], float)

    # Umschaltzeitpunkt
    redirect_idx = None
    for i in range(i_start, len(actions)):
        if actions[i] != baseline_action and actions[i] in redirect_targets:
            redirect_idx = i
            break

    # Recovery: Rückkehr zur Basisaktion nach Event-Ende
    recovery_idx = None
    for i in range(i_end, len(actions)):
        if actions[i] == baseline_action:
            recovery_idx = i
            break

    # Pre/Post-Fenster (robust)
    # ~letztes/erstes Viertel eines Fensterlaufs um Event
    frames_per_win = max(3, int((window / step) * 0.25))
    pre_start  = max(0, i_start - frames_per_win)
    pre_end    = i_start
    post_start = i_start
    post_end   = min(len(actions), i_start + frames_per_win)

    pre_probs = _safe_slice(probs, pre_start, pre_end)
    post_probs= _safe_slice(probs, post_start, post_end)
    pre_G     = _safe_slice(G, pre_start, pre_end)
    post_G    = _safe_slice(G, post_start, post_end)
    pre_PLVv  = _safe_slice(PLVv, pre_start, pre_end)
    post_PLVv = _safe_slice(PLVv, post_start, post_end)
    pre_PLVuy = _safe_slice(PLVuy, pre_start, pre_end)
    post_PLVuy= _safe_slice(PLVuy, post_start, post_end)
    pre_Xi    = _safe_slice(Xi, pre_start, pre_end)
    post_Xi   = _safe_slice(Xi, post_start, post_end)
    pre_IAI   = _safe_slice(IAI, pre_start, pre_end)
    post_IAI  = _safe_slice(IAI, post_start, post_end)

    # Deltas (nahe an „n/a“, wenn Pre/Post fehlt)
    delta_p = np.full(len(ACTIONS), np.nan, float)
    b = _safe_mean_vec(pre_probs); a = _safe_mean_vec(post_probs)
    if b is not None and a is not None:
        delta_p = a - b

    delta_G     = _safe_mean(post_G)    - _safe_mean(pre_G)
    delta_PLVv  = _safe_mean(post_PLVv) - _safe_mean(pre_PLVv)
    delta_PLVuy = _safe_mean(post_PLVuy)- _safe_mean(pre_PLVuy)
    delta_Xi    = _safe_mean(post_Xi)   - _safe_mean(pre_Xi)

    # Latenzen
    latency_frames = None
    latency_sec = None
    if redirect_idx is not None:
        latency_frames = int(redirect_idx - i_start)
        latency_sec = float(latency_frames * step / fs)

    recovery_frames = None
    recovery_sec = None
    if recovery_idx is not None:
        recovery_frames = int(recovery_idx - i_end)
        recovery_sec = float(recovery_frames * step / fs)

    if max_latency_frames is None:
        max_latency_frames = int(max(3, (window / step) * 0.5))  # ~halbe Fensterbreite
    pass_redirect = (redirect_idx is not None) and (latency_frames is not None) and (latency_frames <= max_latency_frames)
    pass_recovery = (recovery_idx is not None)

    return dict(
        i_start=i_start, i_end=i_end,
        pre_span=(pre_start, pre_end), post_span=(post_start, post_end),
        redirect_idx=redirect_idx, latency_frames=latency_frames, latency_sec=latency_sec,
        recovery_idx=recovery_idx, recovery_frames=recovery_frames, recovery_sec=recovery_sec,
        pass_redirect=bool(pass_redirect), pass_recovery=bool(pass_recovery),
        baseline_action=baseline_action, redirect_targets=list(redirect_targets),
        delta_p=delta_p, delta_G=delta_G, delta_PLVv=delta_PLVv, delta_PLVuy=delta_PLVuy, delta_Xi=delta_Xi,
        IAI_pre=_safe_mean(pre_IAI), IAI_post=_safe_mean(post_IAI)
    )

def pretty_print_eval(name: str, ev: Dict):
    _print_header(f"Szenario: {name}")
    print(f"Baseline-Aktion      : {ev['baseline_action']}")
    print(f"Ziel-Aktionen        : {ev['redirect_targets']}")
    if ev['redirect_idx'] is not None:
        print(f"Umschalt-Latenz      : {ev['latency_frames']} Frames  (~{ev['latency_sec']:.3f} s)")
    else:
        print(f"Umschalt-Latenz      : KEIN Wechsel erkannt")
    if ev['recovery_idx'] is not None:
        print(f"Recovery nach Ende   : {ev['recovery_frames']} Frames  (~{ev['recovery_sec']:.3f} s)")
    else:
        print(f"Recovery nach Ende   : kein Rückwechsel erkannt")

    dp = ev["delta_p"]
    labels = ACTIONS
    for i, lab in enumerate(labels):
        print(f"  Δp_{lab:<7} = {_fmt(dp[i])}")

    print(f"ΔGate                 : {_fmt(ev['delta_G'])}")
    print(f"ΔPLV_v                : {_fmt(ev['delta_PLVv'])}")
    print(f"ΔPLV(u,τΣ)            : {_fmt(ev['delta_PLVuy'])}")
    print(f"Δ|Ξ|                  : {_fmt(ev['delta_Xi'])}")
    print(f"IAI (vor→nach)        : {_fmt(ev['IAI_pre'])} → {_fmt(ev['IAI_post'])}")
    print(f"PASS Redirect?        : {ev['pass_redirect']}")
    print(f"PASS Recovery?        : {ev['pass_recovery']}")

# ==== Plot (mit Shading) ====================================================

def plot_actions(res, title: str, fs: int, step: int,
                 event_span_samples: Tuple[int,int] = None,
                 pre_post_frames: Tuple[Tuple[int,int],Tuple[int,int]] = None,
                 save_path: Optional[str] = None):
    if plt is None:
        return
    t_frames = np.asarray(res["t_idx"])
    t_sec = t_frames / fs
    P = np.array(res["probs"])
    Xi = np.array(res["absXi"])
    G = np.array(res["G"])
    a_idx = np.array([ACT2IDX[a] for a in res["action"]])
    plvv  = np.array(res["PLV_v"])
    plvuy = np.array(res["PLV_uy"])
    rho   = np.array(res["rho_ud"]) if "rho_ud" in res else np.zeros_like(plvv)

    fig, axes = plt.subplots(4, 1, figsize=(11, 8.2), sharex=True)

    # 1) Aktionen
    axes[0].plot(t_sec, a_idx, drawstyle="steps-post", color="#1f77b4")
    axes[0].set_yticks(range(len(ACTIONS))); axes[0].set_yticklabels(ACTIONS)
    axes[0].set_title(title); axes[0].grid(True, alpha=0.25)

    # 2) |Xi| & Gate (Twin)
    ax2 = axes[1]; ax2b = ax2.twinx()
    ax2.plot(t_sec, Xi, label="|Ξ|", color="#1f77b4")
    ax2b.plot(t_sec, G, label="Gate", color="#ff7f0e", alpha=0.9)
    ax2.set_ylabel("|Ξ|"); ax2b.set_ylabel("Gate")
    ax2.grid(True, alpha=0.25)
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines+lines2, labels+labels2, loc="upper left")

    # 3) PLVs & |rho|
    axes[2].plot(t_sec, plvv,  label="PLV_v (motion)", color="#2ca02c")
    axes[2].plot(t_sec, plvuy, label="PLV(u,τΣ)",       color="#d62728")
    axes[2].plot(t_sec, rho,   label="|ρ(u,ddqΣ)|",     color="#9467bd")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="upper left"); axes[2].grid(True, alpha=0.25)

    # 4) Policy
    for i, lab in enumerate(ACTIONS):
        axes[3].plot(t_sec, P[:, i], label=lab)
    axes[3].legend(ncol=4, loc="upper left"); axes[3].grid(True, alpha=0.25)
    axes[3].set_xlabel("Zeit (s)")

    # Shading
    def shade(ax):
        if event_span_samples is not None:
            es, ee = event_span_samples
            ax.axvspan(es/fs, ee/fs, color="#f39c12", alpha=0.18, lw=0)
        if pre_post_frames is not None:
            (pre_s, pre_e), (post_s, post_e) = pre_post_frames
            if pre_e > pre_s:
                ax.axvspan(t_frames[pre_s]/fs, t_frames[pre_e-1]/fs, color="#3498db", alpha=0.12, lw=0)
            if post_e > post_s:
                ax.axvspan(t_frames[post_s]/fs, t_frames[post_e-1]/fs, color="#2ecc71", alpha=0.12, lw=0)

    for ax in axes:
        shade(ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()

# ==== Main ===================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", type=int, default=6)
    ap.add_argument("--fs", type=int, default=100)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save", type=str, default="", help="Prefix für PNGs (z.B. out → out_A.png, out_B.png, out_C.png)")
    args = ap.parse_args()

    # 1) Basissignal
    t, qd, qdd, tau_cmd, tau_meas = synth_robotic(
        J=args.joints, fs=args.fs, duration=args.duration, seed=args.seed
    )

    # Baseline-Run
    base = compute_xi_timeseries(qd, qdd, tau_cmd, tau_meas,
                                 fs=args.fs, window=args.window, step=args.step, temp=args.temp)
    base_hist = _hist(base["action"])
    baseline_action = max(base_hist.items(), key=lambda kv: kv[1])[0]
    print(f"Baseline: dom. Aktion = {baseline_action}  | Verteilung = {base_hist}")

    # --- Zeitfenster für Störungen ---
    N = qd.shape[1]
    ev_start = int(0.50 * N)
    ev_dur   = int(0.15 * N)
    ev_end   = min(N-1, ev_start + ev_dur)

    # 2) Szenario A: Orientierungsflip (τ_meas *= -1 im Segment)
    tA_qd, tA_qdd = qd.copy(), qdd.copy()
    tA_tau_cmd, tA_tau_meas = tau_cmd.copy(), tau_meas.copy()
    scenario_orientation_flip(tA_tau_meas, ev_start, ev_end)
    resA = compute_xi_timeseries(tA_qd, tA_qdd, tA_tau_cmd, tA_tau_meas,
                                 fs=args.fs, window=args.window, step=args.step, temp=args.temp)
    evA = evaluate_redirect(resA, ev_start, ev_end, args.fs, args.window, args.step,
                            baseline_action, redirect_targets=("guard","pivot","stop"))
    pretty_print_eval("Orientierungs-Flip (τ_meas Segment * -1)", evA)
    if args.plot or args.save:
        pre_post = (evA["pre_span"], evA["post_span"])
        pth = f"{args.save}_A.png" if args.save else None
        plot_actions(resA, "Szenario A — Orientierungs-Flip",
                     fs=args.fs, step=args.step,
                     event_span_samples=(ev_start, ev_end),
                     pre_post_frames=pre_post, save_path=pth)

    # 3) Szenario B: Latenz/Desync (τ_cmd verzögern um LAG Samples)
    lag_samples = max(1, int(0.05 * args.fs))  # ~50ms bei fs=100
    tB_qd, tB_qdd = qd.copy(), qdd.copy()
    tB_tau_cmd, tB_tau_meas = tau_cmd.copy(), tau_meas.copy()
    scenario_latency_desync(tB_tau_cmd, lag_samples=lag_samples)
    # globales Event: [0, 1] Samples → kein Pre vorhanden (bewusst)
    resB = compute_xi_timeseries(tB_qd, tB_qdd, tB_tau_cmd, tB_tau_meas,
                                 fs=args.fs, window=args.window, step=args.step, temp=args.temp)
    evB = evaluate_redirect(resB, 0, 1, args.fs, args.window, args.step,
                            baseline_action, redirect_targets=("guard","stop"))
    pretty_print_eval(f"Latenz/Desync (τ_cmd um {lag_samples} Samples verzögert)", evB)
    if args.plot or args.save:
        pre_post = (evB["pre_span"], evB["post_span"])
        pth = f"{args.save}_B.png" if args.save else None
        plot_actions(resB, "Szenario B — Latenz/Desync",
                     fs=args.fs, step=args.step,
                     event_span_samples=(0, 1),
                     pre_post_frames=pre_post, save_path=pth)

    # 4) Szenario C: Kohärenz-Drop (qd Noise-Burst im Segment)
    tC_qd, tC_qdd = qd.copy(), qdd.copy()
    tC_tau_cmd, tC_tau_meas = tau_cmd.copy(), tau_meas.copy()
    scenario_coherence_drop(tC_qd, ev_start, ev_end, noise_amp=0.8, seed=args.seed+1)
    # qdd neu ableiten (Kohärenz konsistent halten)
    for j in range(tC_qdd.shape[0]):
        tC_qdd[j] = np.gradient(tC_qd[j], 1/args.fs)
    resC = compute_xi_timeseries(tC_qd, tC_qdd, tC_tau_cmd, tC_tau_meas,
                                 fs=args.fs, window=args.window, step=args.step, temp=args.temp)
    evC = evaluate_redirect(resC, ev_start, ev_end, args.fs, args.window, args.step,
                            baseline_action, redirect_targets=("guard","pivot"))
    pretty_print_eval("Kohärenz-Drop (qd Noise-Burst)", evC)
    if args.plot or args.save:
        pre_post = (evC["pre_span"], evC["post_span"])
        pth = f"{args.save}_C.png" if args.save else None
        plot_actions(resC, "Szenario C — Kohärenz-Drop",
                     fs=args.fs, step=args.step,
                     event_span_samples=(ev_start, ev_end),
                     pre_post_frames=pre_post, save_path=pth)

    # Zusammenfassung
    print("\n" + "-"*80)
    print("Zusammenfassung (PASS Redirect / PASS Recovery):")
    print(f"A: {evA['pass_redirect']} / {evA['pass_recovery']}")
    print(f"B: {evB['pass_redirect']} / {evB['pass_recovery']}")
    print(f"C: {evC['pass_redirect']} / {evC['pass_recovery']}")
    print("-"*80)

if __name__ == "__main__":
    main()