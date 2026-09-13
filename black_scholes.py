"""
Shared Black-Scholes pricing utilities.

# THEORY
All Greeks below expect an ANNUALIZED volatility and a maturity T in years.
Volatilities computed from daily log-returns (see data_loader.py) must be
annualized via annualize() before being passed here -- forgetting this
silently mispricess every option by a factor of ~1/sqrt(252) (~16x too low).
"""

import numpy as np
from scipy.stats import norm

from config import JOURS_PAR_AN


def annualize(sigma_daily):
    """Converts a daily log-return standard deviation into annualized volatility."""
    return sigma_daily * np.sqrt(JOURS_PAR_AN)


def bs_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta_put(S, K, T, r, sigma):
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def bs_delta_call(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def bs_gamma(S, K, T, r, sigma):
    """Identical for puts and calls (put-call parity)."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma):
    """Identical for puts and calls. Returned per 1.0 (100%) change in sigma."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def bs_theta_put(S, K, T, r, sigma):
    """Annualized theta (time decay). Divide by JOURS_PAR_AN for a daily figure."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return term1 + term2


def bs_theta_call(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    return term1 + term2
