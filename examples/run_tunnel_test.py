#!/usr/bin/env python3
"""Example: running regime_detector against network-tunnel-style
telemetry (RTT/jitter). Self-contained -- generates a synthetic jitter
trace with a real regime shift partway through (a quiet baseline that
suddenly gets noisy, like a link picking up contention) so there's no
external capture file required to try this out.

Swap `simulate_tunnel_jitter()` for a real column of RTT/jitter samples
(milliseconds, one per probe) to run this against your own capture.
"""
import numpy as np

from regime_detector import calibrate, detect_regimes, dwell_stats


def simulate_tunnel_jitter(n=6000, seed=0):
    """A quiet baseline segment, then a noisier segment, then back to
    quiet -- the kind of shift a real tunnel shows when a competing flow
    starts and stops sharing the same link."""
    rng = np.random.default_rng(seed)
    quiet = rng.normal(20.0, 0.5, n // 3)
    noisy = rng.normal(20.0, 4.0, n // 3)
    quiet2 = rng.normal(20.0, 0.5, n - 2 * (n // 3))
    return np.concatenate([quiet, noisy, quiet2])


if __name__ == "__main__":
    rtt_ms = simulate_tunnel_jitter()

    result = calibrate(rtt_ms, label="tunnel RTT/jitter")

    if result["has_structure"]:
        W = result["W"]
        regimes = detect_regimes(rtt_ms, window=W)
        print()
        print(f"regime label at last 20 samples: "
              f"{[regimes[i] for i in range(len(regimes) - 20, len(regimes))]}")
        dwell_stats(regimes, label="tunnel RTT/jitter (recheck)")
    else:
        print(f"\nno structure found ({result.get('reason')}) -- nothing to detect regimes on")
