"""Sanity gates: boundary-artifact, undefined-window, and degenerate-
dwell checks. This is what stands between a raw window sweep and a
number you'd actually trust.

Every gate here exists because it caught a real failure on real data
during development, not a hypothetical:

- NaN handling: a domain with heavy ties (e.g. precipitation data,
  ~60% exact-zero readings) collapses the tercile split at small
  windows, producing NaN. NaN silently "wins" Python's max() (NaN
  comparisons are always False, so it never loses a comparison) and
  silently passes `abs(nan) < threshold` (also always False) -- both
  would let a broken window through undetected without explicit
  isnan() checks.
- Boundary-artifact vs undefined-below: a peak landing first in a
  NaN-filtered candidate list does not always mean "no real interior
  peak exists." It can mean the peak was genuinely tested and found
  strongest (a real boundary artifact -- the optimum may be smaller
  than anything tried), or it can mean smaller windows were simply
  undefined (NaN) and never meaningfully compared at all. Conflating
  these produced a real false positive during development and is
  fixed here by tracking which case actually happened.
- Degenerate dwell: even a window that clears the null-check gate can
  produce a regime label that flickers every sample (median dwell <=1
  in every bucket) -- not a real phase, regardless of the z-score.
"""
import numpy as np

from .null_control import DEFAULT_WINDOWS, sweep


def find_native_window(x, windows=DEFAULT_WINDOWS):
    """Sweeps candidate windows and returns the one maximizing
    |regime-split spread| (see null_control.bucket_spread), flagged for
    known failure shapes.

    Returns (W, spread, is_boundary_artifact, undefined_below):
    - is_boundary_artifact: the peak IS the smallest window actually
      tested, and spread decays monotonically from there -- real
      signal, but the true optimum may lie below anything tried.
    - undefined_below: the peak is only first in the list because
      smaller windows were NaN (untested, not tested-and-weaker).
    """
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


def dwell_medians(runs, n_regimes=3):
    """Median dwell length per regime label, from the (label, length)
    runs produced by regime_detector.core.dwell_stats."""
    medians = {}
    for reg in range(n_regimes):
        lens = [l for reg_i, l in runs if reg_i == reg]
        medians[reg] = float(np.median(lens)) if lens else 0.0
    return medians


def is_degenerate_dwell(medians):
    """True if every regime's median dwell is <=1 sample -- flickering
    labels, not a real phase, regardless of how strong the null-z was."""
    return all(m <= 1 for m in medians.values())
