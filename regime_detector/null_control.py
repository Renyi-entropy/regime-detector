"""Shuffle-null verification logic.

The core statistic throughout this package is `bucket_spread`: does the
current regime bucket (low/mid/high, from the causal rolling std of
|change| at window W, lagged by one step so nothing here looks ahead)
line up with a different SCALE of change in the very next step? Split
into terciles and compare the mean next-|change| in the high bucket
against the low bucket -- that gap is the "spread." This is a
retrospective association check used only to pick and validate a
window, not a forecast of any specific future value -- see core.py's
docstring on the witness/predictor distinction.

A raw spread number means nothing on its own: some of it is always
structure you'd see even in noise, purely from how many samples land in
each bucket and how heavy the tails are. `null_check` answers "is this
spread bigger than chance" without requiring a hand-picked reference
domain to compare against -- it builds the null FROM THE SAME SERIES, by
shuffling it.

Important: the null is shuffle-then-difference, not difference-then-
shuffle. Shuffling the already-differenced series (or overlapping blocks
of it) can leak structure back in and inflate every z-score -- this was
a real bug found and fixed during development (shuffling must happen on
the raw values, then re-differenced, so temporal structure is genuinely
destroyed before the statistic is recomputed).
"""
from collections import deque

import numpy as np

DEFAULT_WINDOWS = [2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 200, 300, 450, 650]
N_NULL = 100  # measured 2026-09-20: at N_NULL=20, z's own sampling noise
              # (std over reruns with different shuffle seeds) is wide
              # enough to flip a borderline series' verdict against
              # z_thresh=3.0 on 47% of seeds -- not a tail risk, a coin
              # flip. N_NULL=100 drops that to 7% (z std 0.67 -> 0.23) at
              # ~5x the cost (~0.8s vs ~0.16s for a 6000-sample series in
              # one null_check call) -- calibrate() only calls this once
              # per series, not in a hot loop, so the cost is worth
              # paying. N_NULL=200 tightens further (flip-rate near 0)
              # for anyone calibrating a series they expect to sit near
              # the threshold and want a firmer answer.


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
    """Returns mean(next |change| | high bucket) - mean(next |change| |
    low bucket) for the given window -- a retrospective association
    strength, not a forecast.

    There is exactly ONE way this function says "this result isn't
    trustworthy": NaN. Every caller (find_native_window's NaN filter,
    calibrate()'s nan_null check) only has to know that one signal.
    Found live 2026-09-19: this used to return a bare 0.0 for the
    zero-variance case below (e.g. a constant-slope series, where
    |diff(x)| never changes and the tercile split is meaningless) --
    a SECOND, silent way of meaning "undefined" that looked exactly
    like a real, valid "no predictive spread" measurement and slipped
    straight past every downstream gate, none of which were looking
    for it. Same root cause as this package's own README example of
    the single-source-of-invariant pattern: two different code paths
    each deciding independently what counts as "not a valid result"
    will eventually disagree. Fixed by making zero-variance return the
    same NaN the tie-collapse case already does, one invariant, one
    place it's defined."""
    change = np.abs(np.diff(x))
    sigma = _causal_rolling_std(change, window)
    sigma_lagged = np.empty(len(sigma))
    sigma_lagged[0] = sigma[0]
    sigma_lagged[1:] = sigma[:-1]
    warm = window
    sig = sigma_lagged[warm:]
    nxt = change[warm:]
    if sig.std() == 0 or nxt.std() == 0:
        return float("nan")
    z_sig = (sig - sig.mean()) / sig.std()
    z_nxt = (nxt - nxt.mean()) / nxt.std()
    terciles = np.quantile(z_sig, [1 / 3, 2 / 3])
    bucket = np.digitize(z_sig, terciles)
    means = [z_nxt[bucket == b].mean() for b in range(3)]
    return means[2] - means[0]


def sweep(x, windows=DEFAULT_WINDOWS, max_w_frac=0.1):
    """bucket_spread at each candidate window, skipping any window too
    large a fraction of the series to trust. Does NOT filter NaN
    results -- a (W, nan) tuple can and does appear in the returned
    list. That's deliberate: filtering is the CALLER's job (see
    validators.find_native_window's NaN filter and its
    boundary-artifact-vs-undefined-below distinction), because only the
    caller knows whether "some windows were undefined" changes how the
    rest of the candidates should be interpreted."""
    n = len(x)
    results = []
    for W in windows:
        if W >= n * max_w_frac:
            break
        results.append((W, bucket_spread(x, W)))
    return results


def null_check(x, window, n_null=N_NULL, seed=42):
    """Shuffle x, re-difference, recompute bucket_spread at the SAME
    window, n_null times. Returns (real_spread, z) where z is the real
    spread's distance from the null distribution in null-std units.

    z can come back NaN for two DIFFERENT reasons that callers currently
    can't tell apart from z alone (calibrate() reports both under one
    "nan_null" reason): real_s itself is NaN (bucket_spread's own
    zero-variance/tie-collapse case, propagates straight through the
    arithmetic above), or real_s is a normal number but null_s.std()==0
    (the null distribution degenerated -- every shuffled trial produced
    the same spread, most often because window is close to len(x) and
    there's almost no room left to shuffle into a different bucket
    split). `np.isnan(real_s)` distinguishes the first case for a caller
    that needs to know which."""
    rng = np.random.default_rng(seed)
    real_s = bucket_spread(x, window)
    null_s = np.array([bucket_spread(rng.permutation(x), window) for _ in range(n_null)])
    if null_s.std() == 0:
        return real_s, float("nan")
    z = (real_s - null_s.mean()) / null_s.std()
    return real_s, z


def acf_fft(a, maxlag):
    """Full autocorrelation via FFT -- diagnostic only, not required by
    calibrate(), useful for inspecting a domain's raw decorrelation
    length directly."""
    a = a - a.mean()
    n = len(a)
    f = np.fft.fft(a, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    return acf[: maxlag + 1]
