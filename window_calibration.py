#!/usr/bin/env python3
"""Finds a domain's own predictive-optimal window W and checks it's real
before you trust it -- this is the part that lets regime_detector.py run
on a new series without you hand-tuning W first.

Why not just use a moving average / EWMA volatility window?
-------------------------------------------------------------
A fixed or arbitrarily-chosen window (the standard moving-average
practice: "use 20 periods", "use 200 samples") implicitly assumes every
domain relaxes on the same timescale. It doesn't. A 50ms network ping
stream and a 142-year river-flow record do not share a relaxation time,
and using one borrowed from a textbook or from a different domain either
washes out real short-timescale structure (measured directly: a 200-tick
window on 50ms-cadence data averages over 10 seconds, erasing structure
that lived at ~150ms) or manufactures apparent structure from noise
where none exists. There is no portable "right" window across domains --
each domain has to be measured against its own history.

What this module does, three gates, in order:

1. `find_native_window` -- sweep candidate windows, take the |predictive
   spread| peak. Flags two known failure shapes instead of silently
   trusting the peak:
   - BOUNDARY ARTIFACT: the peak sits at the smallest W actually tested
     and decays monotonically from there -- real signal, but you never
     bracketed the true optimum, it may be smaller than anything tried.
   - UNDEFINED BELOW: the peak is only "first" because smaller windows
     produced NaN (typically heavy ties/zeros collapsing the tercile
     split), not because they were tested and found weaker. Reported
     honestly rather than mislabeled as a boundary artifact.
2. `null_check` -- shuffle the series, re-difference, recompute the
   spread at the SAME window, many times. Gives a z-score for "is this
   window's structure distinguishable from noise" without requiring a
   hand-picked reference domain.
3. Degenerate-dwell check (see `regime_detector.dwell_stats`) -- even a
   window that passes the null check can produce regimes that flicker
   every sample (median dwell <= 1), which means the "native window"
   found upstream isn't real regime structure regardless of the z-score.

Each domain is judged purely against its own shuffled self -- there is
no cross-domain classifier here, deliberately. An earlier version of
this tool tried to also output a portable cross-domain "character" label
(organically clustering vs actively-regulated vs noise) from the same
z-score. Stress-testing it (a synthetic sweep over a tunable persistence
parameter) found a real, wide, unstable decision cliff: the label could
flip from a small change in sample size or window choice near the
threshold, while the underlying z-score barely moved. That classifier
was deliberately removed rather than patched -- a portable label meant
to mean the same thing across domains contradicts the premise that a
domain is only ever witnessed against its own history. What's left here
is exactly that: the local, self-referential gates, nothing more.
"""
from collections import deque

import numpy as np

DEFAULT_WINDOWS = [2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 200, 300, 450, 650]
N_NULL = 20


def _causal_rolling_std(change, window):
    n = len(change)
    stds = np.empty(n)
    buf = deque()
    s = 0.0
    s2 = 0.0
    for i in range(n):
        v = change[i]
        buf.append(v)
        s += v
        s2 += v * v
        if len(buf) > window:
            old = buf.popleft()
            s -= old
            s2 -= old * old
        m = len(buf)
        mean = s / m
        var = max(s2 / m - mean * mean, 0.0)
        stds[i] = var ** 0.5
    return stds


def bucket_spread(x, window):
    """Causal rolling std of |change| at `window`, lagged by 1 so it
    only ever uses information available before the sample it's
    predicting, z-scored and split into terciles. Returns
    mean(next |change| | high bucket) - mean(next |change| | low
    bucket) -- the predictive spread this window's regime split buys
    you. NaN if a bucket has zero variance (e.g. heavy ties collapsing
    the split at small W on sparse/discrete data)."""
    change = np.abs(np.diff(x))
    sigma = _causal_rolling_std(change, window)
    sigma_lagged = np.empty(len(sigma))
    sigma_lagged[0] = sigma[0]
    sigma_lagged[1:] = sigma[:-1]
    warm = window
    sig = sigma_lagged[warm:]
    nxt = change[warm:]
    if sig.std() == 0 or nxt.std() == 0:
        return 0.0
    z_sig = (sig - sig.mean()) / sig.std()
    z_nxt = (nxt - nxt.mean()) / nxt.std()
    terciles = np.quantile(z_sig, [1 / 3, 2 / 3])
    bucket = np.digitize(z_sig, terciles)
    means = [z_nxt[bucket == b].mean() for b in range(3)]
    return means[2] - means[0]


def sweep(x, windows=DEFAULT_WINDOWS, max_w_frac=0.1):
    n = len(x)
    results = []
    for W in windows:
        if W >= n * max_w_frac:
            break
        results.append((W, bucket_spread(x, W)))
    return results


def find_native_window(x, windows=DEFAULT_WINDOWS):
    """Returns (W, spread, is_boundary_artifact, undefined_below)."""
    results = sweep(x, windows)
    valid = [(W, s) for W, s in results if not (isinstance(s, float) and np.isnan(s))]
    if not valid:
        return None, None, None, False
    idx_peak = max(range(len(valid)), key=lambda i: abs(valid[i][1]))
    W_peak, s_peak = valid[idx_peak]

    smallest_w_tested = results[0][0]
    peak_is_smallest_tested = idx_peak == 0 and W_peak == smallest_w_tested
    undefined_below = idx_peak == 0 and W_peak != smallest_w_tested

    is_boundary = peak_is_smallest_tested and all(
        abs(valid[i][1]) <= abs(valid[i - 1][1]) for i in range(1, min(4, len(valid)))
    )
    return W_peak, s_peak, is_boundary, undefined_below


def null_check(x, window, n_null=N_NULL, seed=42):
    """Self-generated null: shuffle x, re-difference, recompute the
    spread at the SAME window, n_null times. Returns (real_spread, z)."""
    rng = np.random.default_rng(seed)
    real_s = bucket_spread(x, window)
    null_s = np.array([bucket_spread(rng.permutation(x), window) for _ in range(n_null)])
    if null_s.std() == 0:
        return real_s, float("nan")
    z = (real_s - null_s.mean()) / null_s.std()
    return real_s, z


def acf_fft(a, maxlag):
    """Full autocorrelation via FFT, for inspecting a domain's raw
    decorrelation length directly (not required for calibrate(), useful
    for diagnostics)."""
    a = a - a.mean()
    n = len(a)
    f = np.fft.fft(a, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    return acf[: maxlag + 1]


def calibrate(x, label="", z_thresh=3.0, windows=DEFAULT_WINDOWS, verbose=True):
    """Runs all three gates and returns a dict describing whether x has
    real, structured regimes and at what window."""
    from regime_detector import detect_regimes, dwell_stats

    if verbose:
        print(f"=== calibrating: {label} (n={len(x)}) ===")
    W, spread, is_boundary, undefined_below = find_native_window(x, windows)
    if W is None:
        if verbose:
            print("  too short to sweep, aborting")
        return {"label": label, "has_structure": False, "reason": "too_short"}

    real_s, z = null_check(x, W)
    flag = ""
    if is_boundary:
        flag = "  [BOUNDARY ARTIFACT -- no real interior peak found]"
    elif undefined_below:
        flag = f"  [UNDEFINED BELOW W={W} -- smaller windows were NaN, cannot rule out a smaller peak]"
    if verbose:
        print(f"  native window W={W}  spread={spread:.3f}  null-z={z:.2f}{flag}")

    if np.isnan(z):
        if verbose:
            print("  z=nan -- null computation broke (likely heavy ties/zeros), cannot validate")
        return {"label": label, "W": W, "spread": spread, "z": z,
                "boundary_artifact": is_boundary, "undefined_below": undefined_below,
                "has_structure": False, "reason": "nan_null"}

    if abs(z) < z_thresh:
        if verbose:
            print(f"  |z|={abs(z):.2f} < {z_thresh} -- not distinguishable from noise, stopping here")
        return {"label": label, "W": W, "spread": spread, "z": z,
                "boundary_artifact": is_boundary, "undefined_below": undefined_below,
                "has_structure": False, "reason": "no_structure"}

    direction = "clustering" if spread > 0 else "anti-persistent"
    r = detect_regimes(x, window=W, warmup=W, recompute_every=max(100, W * 5))
    runs = dwell_stats(r, label=f"  {label} regimes", verbose=verbose)

    medians = {}
    for reg in range(3):
        lens = [l for reg_i, l in runs if reg_i == reg]
        medians[reg] = float(np.median(lens)) if lens else 0.0
    degenerate = all(m <= 1 for m in medians.values())
    if degenerate and verbose:
        print(f"  DEGENERATE DWELL -- median dwell <=1 in every regime ({medians}) -- "
              f"not real regime structure regardless of null-z. Do not trust W={W}.")

    return {"label": label, "W": W, "spread": spread, "z": z,
            "boundary_artifact": is_boundary, "undefined_below": undefined_below,
            "has_structure": not degenerate, "degenerate_dwell": degenerate,
            "direction": direction, "runs": runs, "dwell_medians": medians}
