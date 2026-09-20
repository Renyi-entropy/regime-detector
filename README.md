# regime-detector

A causal, unsupervised tool that answers one question about a time
series: **"which regime is the present moment in?"** — not "what happens
next." No labels, no training set, no forecast. It witnesses tension in
a substrate against its own history, nothing more.

## Why not a moving average / EWMA volatility band?

Standard practice picks a window — 20 periods, 200 ticks, whatever the
library default is — and applies it everywhere. That silently assumes
every domain relaxes on the same timescale. It doesn't:

- A 50ms-cadence network ping stream and a 142-year river-flow record
  have nothing in common on the calendar.
- Measured directly: running a 200-sample window on 50ms-cadence RTT
  data averages over 10 seconds of history — the domain's own native
  clustering scale turned out to be ~3 samples (~150ms). The borrowed
  window washed out real structure into near-flatness.
- Run the same fixed window on a domain that's actively regulated
  (a control loop suppressing its own variance) rather than organically
  clustering, and you get a monotonic-but-drained or even *reversed*
  relationship — the moving-average band doesn't just get the scale
  wrong, it can get the direction wrong.

There is no portable "right" window across domains. `regime_detector.core.calibrate`
finds each domain's own natural window empirically -- the one where the
current regime bucket lines up most strongly with a different scale of
change right afterward -- and checks the result before trusting it,
instead of assuming one. This is a retrospective association check used
to pick and validate a window, not a forecast of any future value.

## Install

```bash
pip install -e .
```

## What's actually in here

```
regime_detector/
├── core.py             # main phase-window + regime-switching engine (calibrate, detect_regimes, dwell_stats)
├── null_control.py      # shuffle-null verification logic (bucket_spread, null_check)
└── validators.py         # sanity gates (boundary-artifact, undefined-window, degenerate-dwell checks)
examples/
└── run_tunnel_test.py    # example run against network RTT/jitter-style telemetry
```

- **`null_control.py`** — computes `bucket_spread`: does the current
  regime bucket (from the causal rolling std of |change| at window W)
  line up with a different scale of change right afterward? A
  retrospective association check, not a forecast. Then `null_check`
  shuffles the series, re-differences, and recomputes the same
  statistic many times, giving a z-score for "is this distinguishable
  from noise" without needing a hand-picked reference domain to compare
  against.
- **`validators.py`** — two gates on top of the raw sweep:
  1. **Boundary-artifact / undefined-below check** (`find_native_window`)
     — is the peak sitting at the smallest window actually tested and
     decaying monotonically from there (real signal, true optimum may be
     smaller than anything tried), or is it only "first" because smaller
     windows were undefined/NaN (heavy ties collapsing the split, never
     really compared at all)? Conflating these produced a real false
     positive during development; they're reported separately.
  2. **Degenerate-dwell check** (`is_degenerate_dwell`) — even a window
     that clears the null-check gate can produce a regime label that
     flickers every single sample. A median dwell of ≤1 is not a phase,
     so it's flagged rather than trusted.
- **`core.py`** — `detect_regimes` is the actual online classifier,
  strictly causal: the regime at time *t* is decided using only samples
  up to and including *t*, with tercile boundaries computed from an
  *expanding* window (everything seen so far), never the whole series.
  `calibrate` orchestrates all of the above into one call.
- **`examples/run_tunnel_test.py`** — synthetic RTT/jitter trace with a
  real regime shift (quiet baseline → noisy contention → quiet again),
  no external capture file required. Swap the synthetic generator for a
  real column of RTT samples to run against your own data.

```bash
python3 examples/run_tunnel_test.py
```

```
=== calibrating: tunnel RTT/jitter (n=6000) ===
  native window W=8  spread=1.458  null-z=32.18
  tunnel RTT/jitter regimes: 625 regime runs over 5991 classified samples (mean run length = 9.6 samples)
   low:   199 runs, mean dwell=8.9, median=6, max=54
   mid:   296 runs, mean dwell=5.4, median=4, max=31
  high:   130 runs, mean dwell=20.2, median=8, max=818
```

Note the `high` bucket's dwell: mean=20.2 but max=818 -- that's the
detector correctly camping in "high" for the entire noisy middle
segment, exactly the regime shift the synthetic trace was built with.

## Usage on your own data

```python
import numpy as np
from regime_detector import calibrate, detect_regimes, dwell_stats

x = np.asarray(your_series)
result = calibrate(x, label="my series")

if result["has_structure"]:
    regimes = detect_regimes(x, window=result["W"])
    dwell_stats(regimes, label="my series")
```

## What the gates actually establish

Worth being precise, because the names flatter them:

- The **null gate** scores the *window-discovery* statistic
  (`bucket_spread` at W) against a shuffle null. Passing means "there's
  a real, non-chance association at W between the recent-volatility
  bucket and the next |change|." It does **not** validate the emitted
  regime labels — `bucket_spread` splits on global terciles of the whole
  series, while `detect_regimes` splits on causal *expanding* terciles
  refreshed periodically. Related quantities, different splits, and the
  z-score was only ever computed on the first.
- The **dwell gate** is the only check that looks at the classifier's
  actual output, and it's a **floor, not evidence**: it rejects pure
  point-to-point flicker (median dwell ≤1 in every regime) and nothing
  else. Measured: a pure random walk clears it on 5/5 seeds with median
  dwells of 2–6. Clearing it means "not flicker," never "structure is
  real."

So `has_structure: True` means a window was found, it isn't chance, and
the labels don't flicker — three necessary conditions, not a proof that
the three regimes are meaningful. `calibrate()` returns
`passed_null_gate` and `passed_dwell_gate` separately so you can see
which one carried the verdict, and `reason` on every False path.

## What this deliberately does NOT do

An earlier version of this tool also tried to output a portable
cross-domain "character" label — organically clustering vs
actively-regulated vs pure noise — from a normalized version of the
same z-score. Stress-testing it against a synthetic series with a
tunable persistence parameter found a real, wide, unstable decision
cliff near the threshold: the label could flip from a small change in
sample size or window choice while the underlying z-score barely moved.

That classifier was removed, not patched. A label meant to mean the
same thing across domains contradicts the premise this tool is built
on: a domain is only ever witnessed against its own history, and there
was never supposed to be a shared vocabulary to sort every domain into.
What's here is exactly the part that survived scrutiny — window
discovery, a noise gate, a degenerate-dwell gate — nothing dressed up as
more than it is.

This is also not a predictor and isn't wired to any actuator. It reports
which regime the present is in; what a system does with that signal is
a separate decision made by whoever consumes it.

## License

MIT
