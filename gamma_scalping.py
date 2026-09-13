"""
MODULE E2 -- Gamma Scalping (daily delta-hedged ATM straddle)

# THEORY
A Long Straddle (Call + Put, same strike, same maturity) is long Gamma and
long Vega, but carries directional (Delta) risk. By REHEDGING the Delta to
zero every day (buying/selling the underlying), the directional risk is
removed and what remains is pure exposure to REALIZED volatility, in
BOTH directions -- unlike the Backspread (E1), which only pays off on a
large DOWNSIDE move.

The classic result: daily P&L of a delta-hedged long option position is
approximately
    P&L_t ~= 0.5 * Gamma_t * (dS_t)^2 - Theta_t * dt
It gains when the underlying moves MORE than what implied volatility
priced in (realized > implied), and bleeds Theta when it moves less.
This module simulates the EXACT hedge (not the approximation) day by day.

# LIMITATION
No transaction costs are modeled for the daily rehedge trades. In practice,
frequent rehedging on a volatile, potentially illiquid contract erodes some
of the gamma-scalping P&L shown here.
"""

import numpy as np
import matplotlib.pyplot as plt

from config import (
    STRADDLE_BUDGET_PCT, RISK_FREE_RATE, JOURS_PAR_AN, TAILLE_CONTRAT_WTI,
    N_SIM_BLACK_SWAN, N_JOURS_SIMULATION, JOUR_CHOC, POSITION_USD,
    DRIFT_HO_CRASH, DRIFT_WTI_CRASH, RHO_CRASH,
)
from black_scholes import (
    bs_put, bs_call, bs_delta_put, bs_delta_call, annualize,
)


def setup_straddle(S0, sigma_WTI_daily, T0_jours=N_JOURS_SIMULATION):
    """ATM straddle (K = S0), sized from a fixed premium budget."""
    K = round(S0, 2)
    T0 = T0_jours / JOURS_PAR_AN
    sigma_annual = annualize(sigma_WTI_daily)

    call0 = bs_call(S0, K, T0, RISK_FREE_RATE, sigma_annual)
    put0 = bs_put(S0, K, T0, RISK_FREE_RATE, sigma_annual)
    straddle_price = call0 + put0

    budget = POSITION_USD * STRADDLE_BUDGET_PCT
    n_straddles = budget / (straddle_price * TAILLE_CONTRAT_WTI)

    print(f"\n{'=' * 60}")
    print("  MODULE E2 -- GAMMA SCALPING (DELTA-HEDGED STRADDLE)")
    print(f"{'=' * 60}")
    print(f"  Strike K (ATM)         : {K:>8.2f}$")
    print(f"  Call premium           : {call0:>8.4f}$")
    print(f"  Put premium            : {put0:>8.4f}$")
    print(f"  Straddle premium       : {straddle_price:>8.4f}$")
    print(f"  Budget                 : {budget:>10,.0f}$")
    print(f"  Straddle contracts     : {n_straddles:>8.2f}")
    print(f"{'=' * 60}")

    return {"K": K, "n_straddles": n_straddles, "straddle_price": straddle_price}


def simulate_gamma_scalping(S0, params, mu_HO, mu_WTI, sigma_HO, sigma_WTI, rho_pre,
                             sigma_HO_crash, sigma_WTI_crash,
                             n_sim=N_SIM_BLACK_SWAN, n_jours=N_JOURS_SIMULATION,
                             jour_choc=JOUR_CHOC):
    """
    Simulates a daily delta-hedged long straddle through the Black Swan.

    Returns
    -------
    pnl_scalping : total P&L per simulation (option value change + hedge P&L)
    delta_paths  : net PRE-hedge delta of the straddle each day (needed by Module F)
    """
    K, n = params["K"], params["n_straddles"]

    pnl_scalping = np.zeros(n_sim)
    delta_paths = np.zeros((n_sim, n_jours))

    for sim in range(n_sim):
        S_wti = S0
        sig_prev = annualize(sigma_WTI)

        T0_rem = n_jours / JOURS_PAR_AN
        V_prev = n * (bs_call(S_wti, K, T0_rem, RISK_FREE_RATE, sig_prev)
                      + bs_put(S_wti, K, T0_rem, RISK_FREE_RATE, sig_prev))
        delta_prev = n * (bs_delta_call(S_wti, K, T0_rem, RISK_FREE_RATE, sig_prev)
                          + bs_delta_put(S_wti, K, T0_rem, RISK_FREE_RATE, sig_prev))
        hedge_position = -delta_prev * TAILLE_CONTRAT_WTI  # barrels of WTI held short/long

        pnl = 0.0

        for jour in range(n_jours):
            T_rem = max((n_jours - jour - 1) / JOURS_PAR_AN, 1 / JOURS_PAR_AN)

            if jour < jour_choc:
                cov_t = np.array([[sigma_HO ** 2, rho_pre * sigma_HO * sigma_WTI],
                                   [rho_pre * sigma_HO * sigma_WTI, sigma_WTI ** 2]])
                mu_wti_t, sig_daily = mu_WTI, sigma_WTI
            else:
                cov_t = np.array([[sigma_HO_crash ** 2, RHO_CRASH * sigma_HO_crash * sigma_WTI_crash],
                                   [RHO_CRASH * sigma_HO_crash * sigma_WTI_crash, sigma_WTI_crash ** 2]])
                mu_wti_t, sig_daily = DRIFT_WTI_CRASH, sigma_WTI_crash

            sig_t = annualize(sig_daily)
            L_t = np.linalg.cholesky(cov_t)
            shocks = L_t @ np.random.standard_normal(2)
            r_wti = mu_wti_t + shocks[1]

            S_wti_new = S_wti * np.exp(r_wti)

            # P&L from holding yesterday's hedge as price moves today
            hedge_pnl = hedge_position * (S_wti_new - S_wti)

            V_new = n * (bs_call(S_wti_new, K, T_rem, RISK_FREE_RATE, sig_t)
                         + bs_put(S_wti_new, K, T_rem, RISK_FREE_RATE, sig_t))
            option_pnl = (V_new - V_prev) * TAILLE_CONTRAT_WTI

            pnl += hedge_pnl + option_pnl

            # Rebalance to the new delta for the next period
            delta_new = n * (bs_delta_call(S_wti_new, K, T_rem, RISK_FREE_RATE, sig_t)
                              + bs_delta_put(S_wti_new, K, T_rem, RISK_FREE_RATE, sig_t))
            hedge_position = -delta_new * TAILLE_CONTRAT_WTI

            delta_paths[sim, jour] = delta_new

            S_wti = S_wti_new
            V_prev = V_new
            sig_prev = sig_t

        pnl_scalping[sim] = pnl

    return pnl_scalping, delta_paths


def plot_module_e2(pnl_scalping, params, save_path="module_E2_gamma_scalping.png"):
    """Distribution of the gamma-scalping P&L, through the Black Swan."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pnl_scalping, bins=80, color="mediumpurple", alpha=0.75, edgecolor="none")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(np.median(pnl_scalping), color="indigo", linestyle="--",
               linewidth=2, label=f"Median: {np.median(pnl_scalping):,.0f}$")
    ax.set_title(f"Module E2 -- Gamma Scalping P&L Distribution\n"
                 f"ATM Straddle K={params['K']}$, daily delta-hedged")
    ax.set_xlabel("P&L ($) -- gamma-scalping overlay only")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: {save_path}")
