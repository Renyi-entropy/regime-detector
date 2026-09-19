"""Main phase-window and regime-switching engine.

`detect_regimes` is the actual online classifier: strictly causal, no
lookahead, no target/label. `calibrate` is the orchestrator most callers
want -- it runs window discovery (validators.find_native_window), the
shuffle-null gate (null_control.null_check), and the degenerate-dwell
gate (validators), and only then calls detect_regimes.

This is a witness, not a predictor: it answers "given everything up to
and including now, which regime is the present moment in," not "what
happens next" or "when will it change." See README.md.
"""
from collections import deque

import numpy as np

from .null_control import null_check
from .validators import dwell_medians, find_native_window, is_degenerate_dwell

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
    dwell times, not point-to-point flicker (see validators.is_degenerate_dwell)."""
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


def calibrate(x, label="", z_thresh=3.0, windows=None, verbose=True):
    """Runs window discovery + null gate + degenerate-dwell gate and
    returns a dict describing whether x has real, structured regimes
    and at what window. This is the entry point most callers want."""
    from .null_control import DEFAULT_WINDOWS

    windows = windows or DEFAULT_WINDOWS
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

    medians = dwell_medians(runs)
    degenerate = is_degenerate_dwell(medians)
    if degenerate and verbose:
        print(f"  DEGENERATE DWELL -- median dwell <=1 in every regime ({medians}) -- "
              f"not real regime structure regardless of null-z. Do not trust W={W}.")

    return {"label": label, "W": W, "spread": spread, "z": z,
            "boundary_artifact": is_boundary, "undefined_below": undefined_below,
            "has_structure": not degenerate, "degenerate_dwell": degenerate,
            "direction": direction, "runs": runs, "dwell_medians": medians}
