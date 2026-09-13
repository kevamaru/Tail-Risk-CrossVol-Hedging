"""
MODULE F -- Combined Book: Delta Netting Across E1 and E2

# THEORY
E1 (backspread) is a STATIC position: it is never rehedged, so its delta
DRIFTS as the underlying moves and time passes. E2 (straddle) is ACTIVELY
self-hedged to zero delta every day, in isolation.

The blind spot this creates: if a risk report only checks "is E2 hedged?"
the answer is always yes (by construction) -- yet the OPTIONS BOOK AS A
WHOLE still carries E1's drifting, unmanaged delta. Checking each leg in
isolation hides this.

This module re-simulates E1 and E2 TOGETHER, on the SAME price path per
scenario (this matters: netting two independently-simulated Monte Carlo
arrays would combine unrelated random draws and mean nothing), and
compares two risk regimes:

  ISOLATED : E2 hedges only its own delta to zero every day.
             Net options-book delta = Delta_E1(t) + 0 (E1 left to drift).
  UNIFIED  : ONE hedging trade neutralizes (Delta_E1 + Delta_E2) together
             every day -- the desk manages the combined book, not each
             leg separately.

# WHAT THIS DOES **NOT** CLAIM
This does not claim the unified approach mechanically reduces trading
volume (E1 wasn't being traded at all in the isolated case, so there is
no double-trading to net away here). The genuine finding is about RISK
VISIBILITY: the isolated view understates the book's true WTI exposure.
"""

import numpy as np
import matplotlib.pyplot as plt

from config import (
    RISK_FREE_RATE, JOURS_PAR_AN, TAILLE_CONTRAT_WTI,
    N_SIM_BLACK_SWAN, N_JOURS_SIMULATION, JOUR_CHOC,
    DRIFT_HO_CRASH, DRIFT_WTI_CRASH, RHO_CRASH,
)
from black_scholes import (
    bs_put, bs_call, bs_delta_put, bs_delta_call, annualize,
)


def simulate_combined_book(S0, backspread_params, straddle_params, nb_contrats_frozen,
                            mu_HO, mu_WTI, sigma_HO, sigma_WTI, rho_pre,
                            sigma_HO_crash, sigma_WTI_crash,
                            n_sim=N_SIM_BLACK_SWAN, n_jours=N_JOURS_SIMULATION,
                            jour_choc=JOUR_CHOC):
    """
    Simulates E1 + E2 jointly on shared price paths, under both hedging
    regimes (isolated vs unified), and tracks the net options-book delta.

    Returns a dict of arrays (per simulation): pnl totals, delta paths for
    one illustrative path (index 0) for plotting.
    """
    K1, K2 = backspread_params["K1"], backspread_params["K2"]
    short_c, long_c = backspread_params["short_contracts"], backspread_params["long_contracts"]
    K_straddle, n_straddles = straddle_params["K"], straddle_params["n_straddles"]

    pnl_isolated = np.zeros(n_sim)   # E1 (unhedged) + E2 (self-hedged)
    pnl_unified = np.zeros(n_sim)    # E1 + E2 hedged together as one book

    # Kept only for simulation 0, for the illustrative plot
    illustrative = {"delta_isolated": [], "S_path": []}

    for sim in range(n_sim):
        S_wti = S0
        sig_prev = annualize(sigma_WTI)

        val_e1_prev = (long_c * bs_put(S_wti, K2, n_jours / JOURS_PAR_AN, RISK_FREE_RATE, sig_prev)
                       - short_c * bs_put(S_wti, K1, n_jours / JOURS_PAR_AN, RISK_FREE_RATE, sig_prev))
        val_e2_prev = n_straddles * (bs_call(S_wti, K_straddle, n_jours / JOURS_PAR_AN, RISK_FREE_RATE, sig_prev)
                                      + bs_put(S_wti, K_straddle, n_jours / JOURS_PAR_AN, RISK_FREE_RATE, sig_prev))

        # ISOLATED regime: only E2 is hedged, to its own delta
        d_e2_0 = n_straddles * (bs_delta_call(S_wti, K_straddle, n_jours / JOURS_PAR_AN, RISK_FREE_RATE, sig_prev)
                                 + bs_delta_put(S_wti, K_straddle, n_jours / JOURS_PAR_AN, RISK_FREE_RATE, sig_prev))
        hedge_isolated = -d_e2_0 * TAILLE_CONTRAT_WTI

        # UNIFIED regime: one hedge neutralizes E1 + E2 together
        d_e1_0 = (long_c * bs_delta_put(S_wti, K2, n_jours / JOURS_PAR_AN, RISK_FREE_RATE, sig_prev)
                  - short_c * bs_delta_put(S_wti, K1, n_jours / JOURS_PAR_AN, RISK_FREE_RATE, sig_prev))
        hedge_unified = -(d_e1_0 + d_e2_0) * TAILLE_CONTRAT_WTI

        pnl_iso, pnl_uni = 0.0, 0.0

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

            S_new = S_wti * np.exp(r_wti)

            # --- Mark both option positions to market (same for both regimes) ---
            val_e1_new = (long_c * bs_put(S_new, K2, T_rem, RISK_FREE_RATE, sig_t)
                          - short_c * bs_put(S_new, K1, T_rem, RISK_FREE_RATE, sig_t))
            val_e2_new = n_straddles * (bs_call(S_new, K_straddle, T_rem, RISK_FREE_RATE, sig_t)
                                         + bs_put(S_new, K_straddle, T_rem, RISK_FREE_RATE, sig_t))

            options_pnl_e1 = (val_e1_new - val_e1_prev) * TAILLE_CONTRAT_WTI
            options_pnl_e2 = (val_e2_new - val_e2_prev) * TAILLE_CONTRAT_WTI

            # --- ISOLATED: E1 unhedged (its P&L is what it is),
            #     E2 covered by yesterday's self-hedge only ---
            hedge_pnl_isolated = hedge_isolated * (S_new - S_wti)
            pnl_iso += options_pnl_e1 + options_pnl_e2 + hedge_pnl_isolated

            # --- UNIFIED: one hedge covering E1 + E2 combined ---
            hedge_pnl_unified = hedge_unified * (S_new - S_wti)
            pnl_uni += options_pnl_e1 + options_pnl_e2 + hedge_pnl_unified

            # --- Rebalance both regimes' hedges for the next day ---
            d_e1_new = (long_c * bs_delta_put(S_new, K2, T_rem, RISK_FREE_RATE, sig_t)
                        - short_c * bs_delta_put(S_new, K1, T_rem, RISK_FREE_RATE, sig_t))
            d_e2_new = n_straddles * (bs_delta_call(S_new, K_straddle, T_rem, RISK_FREE_RATE, sig_t)
                                       + bs_delta_put(S_new, K_straddle, T_rem, RISK_FREE_RATE, sig_t))

            hedge_isolated = -d_e2_new * TAILLE_CONTRAT_WTI               # E2 alone
            hedge_unified = -(d_e1_new + d_e2_new) * TAILLE_CONTRAT_WTI   # E1 + E2 together

            if sim == 0:
                illustrative["delta_isolated"].append(d_e1_new)  # E1's UNMANAGED drift under "isolated"
                illustrative["S_path"].append(S_new)
                # Note: under "unified", net delta is 0 by construction every day
                # (that is the entire point of the unified hedge) -- not tracked
                # here since it carries no information beyond "always zero".

            S_wti = S_new
            val_e1_prev, val_e2_prev = val_e1_new, val_e2_new
            sig_prev = sig_t

        pnl_isolated[sim] = pnl_iso
        pnl_unified[sim] = pnl_uni

    return pnl_isolated, pnl_unified, illustrative


def print_combined_book_stats(pnl_isolated, pnl_unified):
    def _stats(pnl, label):
        print(f"\n  {label}")
        print(f"  {'-' * 42}")
        print(f"  Mean             : {np.mean(pnl):>14,.0f} $")
        print(f"  Volatility       : {np.std(pnl):>14,.0f} $")
        print(f"  VaR 99%          : {np.percentile(pnl, 1):>14,.0f} $")
        print(f"  Worst case       : {np.min(pnl):>14,.0f} $")

    print(f"\n{'=' * 60}")
    print("  MODULE F -- COMBINED BOOK: ISOLATED vs UNIFIED HEDGING")
    print(f"{'=' * 60}")
    _stats(pnl_isolated, "ISOLATED (E1 unhedged, E2 self-hedged alone)")
    _stats(pnl_unified, "UNIFIED (E1 + E2 hedged together)")
    print(f"\n  Note: differences reflect the OPPORTUNITY COST/BENEFIT of")
    print(f"  E1's unmanaged delta drift under the isolated regime -- not")
    print(f"  a transaction-cost saving (see module docstring).")
    print(f"{'=' * 60}")


def plot_module_f(illustrative, pnl_isolated, pnl_unified, save_path="module_F_combined_book.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax1 = axes[0]
    ax1.plot(illustrative["delta_isolated"], color="crimson", linewidth=1.5,
              label="Isolated: E1's unmanaged delta drift")
    ax1.axhline(0, color="black", linewidth=1, linestyle="--", label="Unified: kept at 0")
    ax1.set_title("Net Options-Book Delta\n(one illustrative path)")
    ax1.set_xlabel("Day"); ax1.set_ylabel("Net Delta (WTI contracts)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(pnl_isolated, bins=60, color="crimson", alpha=0.5, density=True, label="Isolated")
    ax2.hist(pnl_unified, bins=60, color="teal", alpha=0.5, density=True, label="Unified")
    ax2.set_title("Combined P&L Distribution (E1 + E2)")
    ax2.set_xlabel("P&L ($)"); ax2.set_ylabel("Density")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: {save_path}")
