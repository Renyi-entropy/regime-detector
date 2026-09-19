#!/usr/bin/env python3
"""Minimal end-to-end demo, no external data required.

Builds two synthetic series -- one with real regime structure (volatility
clustering, an AR(1) on log-volatility) and one pure white noise -- and
shows window_calibration.calibrate() correctly finding structure in the
first and correctly rejecting the second as noise.
"""
import numpy as np

from window_calibration import calibrate


def mean_reverting_vol_clustering(n, phi=0.9, seed=0):
    rng = np.random.default_rng(seed)
    log_sigma = np.zeros(n)
    for t in range(1, n):
        log_sigma[t] = phi * log_sigma[t - 1] + rng.normal(0, 0.3)
    sigma = np.exp(log_sigma)
    innov = rng.normal(0, 1, n) * sigma
    return np.cumsum(innov)


if __name__ == "__main__":
    clustered = mean_reverting_vol_clustering(6000, phi=0.9, seed=1)
    noise = np.cumsum(np.random.default_rng(2).normal(0, 1, 6000))

    calibrate(clustered, label="synthetic volatility clustering (phi=0.9)")
    print()
    calibrate(noise, label="pure random walk (no regime structure)")
