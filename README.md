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

There is no portable "right" window across domains. `window_calibration.py`
finds each domain's own predictive-optimal window empirically, and
checks the result before trusting it, instead of assuming one.

## What's actually in here

Two small modules, one demo:

- **`window_calibration.py`** — sweeps candidate windows, picks the one
  that maximizes predictive spread (does knowing "high/mid/low recent
  volatility" actually predict the next |change|?), then runs it through
  three gates before calling it real:
  1. **Boundary-artifact check** — is the peak sitting at the smallest
     window you tried, decaying monotonically from there? That means
     you never bracketed the true optimum, not that none exists.
  2. **Self-generated null check** — shuffle the series, re-difference,
     recompute the same statistic, many times. Gives a z-score for "is
     this distinguishable from noise" without needing a hand-picked
     reference domain to compare against.
  3. **Degenerate-dwell check** — even a window that clears the null
     check can produce a regime label that flickers every single
     sample. A median dwell of ≤1 is not a phase, so it's flagged
     rather than trusted.
- **`regime_detector.py`** — the actual online classifier. Strictly
  causal: the regime at time *t* is decided using only samples up to
  and including *t*, with tercile boundaries computed from an
  *expanding* window (everything seen so far), never the whole series.
  Nothing here peeks ahead.
- **`example.py`** — synthetic demo, no external data needed. Shows the
  calibrator correctly finding real regime structure in a volatility-
  clustering series, and correctly rejecting a pure random walk as
  noise.

```bash
python3 example.py
```

```
=== calibrating: synthetic volatility clustering (phi=0.9) (n=6000) ===
  native window W=5  spread=0.571  null-z=13.74
  synthetic volatility clustering (phi=0.9) regimes: 1163 regime runs over 5994 classified samples (mean run length = 5.2 samples)
   low:   346 runs, mean dwell=5.6, median=3, max=51
   mid:   512 runs, mean dwell=3.7, median=3, max=23
  high:   305 runs, mean dwell=7.1, median=5, max=58

=== calibrating: pure random walk (no regime structure) (n=6000) ===
  native window W=300  spread=-0.066  null-z=-2.65
  |z|=2.65 < 3.0 -- not distinguishable from noise, stopping here
```

## Usage on your own data

```python
import numpy as np
from window_calibration import calibrate
from regime_detector import detect_regimes, dwell_stats

x = np.asarray(your_series)
result = calibrate(x, label="my series")

if result["has_structure"]:
    regimes = detect_regimes(x, window=result["W"])
    dwell_stats(regimes, label="my series")
```

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
