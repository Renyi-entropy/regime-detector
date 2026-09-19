#!/usr/bin/env python3
"""Causal, unsupervised regime detector.

Classifies each point in a time series into low/mid/high volatility
"phase" using only the causal rolling standard deviation of |change| at
a chosen window W, with tercile boundaries computed from an EXPANDING
window (samples seen so far only). No target, no label, no lookahead:
at time t the classifier only ever sees change[0..t] and quantiles of
sigma[0..t].

This is a witness, not a predictor: it answers "given everything up to
and including now, which regime is the present moment in," not "what
happens next." See README.md for why that distinction matters and for
how to pick W with window_calibration.py.
"""
from collections import deque

import numpy as np

LABELS = ["low", "mid", "high"]


def causal_rolling_std(change, window):
    """O(n) rolling std over the last `window` samples, causal (no
    lookahead): stds[i] only depends on change[max(0, i-window+1)..i]."""
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


def detect_regimes(x, window, warmup=None, recompute_every=None):
    """Classify each sample of x into regime 0/1/2 (low/mid/high),
    strictly causal throughout.

    - `warmup` samples at the start are reported as None (not enough
      history yet to trust a tercile split). Defaults to `window`.
    - Tercile boundaries are recomputed every `recompute_every` samples
      from an EXPANDING window (all samples seen so far, not the whole
      series) rather than at every single t: a full expanding-quantile
      recompute at every t is O(n^2) and impractical past a few
      thousand samples. Between refreshes the most recently computed
      (already-causal) boundary is reused. Defaults to max(200, W*5).

    Returns a list of length len(x)-1 (one entry per diff), each either
    None (warmup) or 0/1/2.
    """
    warmup = warmup or window
    recompute_every = recompute_every or max(200, window * 5)
    change = np.abs(np.diff(x))
    sigma = causal_rolling_std(change, window)
    n = len(sigma)
    regimes = [None] * n
    lo = hi = None
    for t in range(n):
        if t < warmup:
            continue
        if lo is None or t % recompute_every == 0:
            seen = sigma[: t + 1]  # expanding, causal -- no future leakage
            lo, hi = np.quantile(seen, [1 / 3, 2 / 3])
        if sigma[t] <= lo:
            regimes[t] = 0
        elif sigma[t] <= hi:
            regimes[t] = 1
        else:
            regimes[t] = 2
    return regimes


def dwell_stats(regimes, label="", verbose=True):
    """Collapses a regime-label sequence into runs and reports how long
    each regime persists. Real regime structure should show multi-sample
    dwell times, not point-to-point flicker (median dwell <= 1 means the
    "regime" found isn't real structure -- see README's degenerate-dwell
    gate)."""
    valid = [r for r in regimes if r is not None]
    if not valid:
        return []
    runs = []
    cur = valid[0]
    length = 1
    for r in valid[1:]:
        if r == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = r
            length = 1
    runs.append((cur, length))

    if verbose:
        print(f"{label}: {len(runs)} regime runs over {len(valid)} classified samples "
              f"(mean run length = {len(valid)/len(runs):.1f} samples)")
        for reg in range(3):
            lens = [l for r, l in runs if r == reg]
            if lens:
                print(f"  {LABELS[reg]:>4}: {len(lens):>5} runs, mean dwell={np.mean(lens):.1f}, "
                      f"median={np.median(lens):.0f}, max={max(lens)}")
    return runs
