"""
MODULE E1 -- Zero-Cost Put Ratio Backspread (tail-risk convexity overlay)

# THEORY
A linear hedge (futures) has a constant, non-adaptive sensitivity. A single
Long Put improves this, but is still a directional bet with a fixed cost.
A Put Ratio Backspread goes further: SELL 1 put closer to the money (K1) to
finance the purchase of N puts deeper out-of-the-money (K2). If sized so
that Premium(K1) == N * Premium(K2), the structure costs ~0 upfront while
remaining massively convex on a large downside move -- pure Spitznagel-style
tail-risk protection: it is designed to bleed a little in calm markets and
pay off disproportionately in a genuine crash.

# DESIGN NOTE -- why the ratio is DERIVED, not IMPOSED
An earlier version of this project tried to force both a fixed strike pair
AND an exact "1x3" ratio simultaneously. With flat (non-skewed) volatility
these two constraints are generally incompatible: for a given (K1, K2, T,
sigma), Black-Scholes determines ONE specific zero-cost ratio -- it cannot
be dictated in advance without either changing the strikes or introducing a
volatility skew. Forcing a round number here would mean fabricating the
result rather than deriving it.
Empirically, deep OTM strikes (e.g. K2 = 65% of spot) are priced near zero
with only 63 days to maturity under flat vol, producing an unusable ratio
(>150x). K1=95% / K2=85% keeps both legs meaningfully priced and happens to
produce a ratio close to 3x -- consistent with the classic "1x3" backspread,
but obtained from the pricing model rather than assumed.

# LIMITATION -- flat volatility
Real markets price deep OTM puts with a volatility SKEW (higher implied vol
than at-the-money), making them more expensive than a flat-vol model
suggests. This model likely UNDERSTATES the true cost of the long leg, and
therefore OVERSTATES how cheap real tail protection would be. See README.
"""

import numpy as np
import matplotlib.pyplot as plt

from config import (
    BACKSPREAD_K1_PCT, BACKSPREAD_K2_PCT, BACKSPREAD_SHORT_CONTRACTS,
    RISK_FREE_RATE, JOURS_PAR_AN, TAILLE_CONTRAT_WTI,
    N_SIM_BLACK_SWAN, N_JOURS_SIMULATION, JOUR_CHOC,
    DRIFT_HO_CRASH, DRIFT_WTI_CRASH, RHO_CRASH,
)
from black_scholes import (
    bs_put, bs_delta_put, bs_gamma, bs_vega, bs_theta_put, annualize,
)


def setup_backspread(S0, sigma_WTI_daily, T0_jours=N_JOURS_SIMULATION):
    """
    Prices both legs, derives the natural zero-cost ratio, and reports the
    net Greeks of the resulting position at inception.

    Returns
    -------
    dict with K1, K2, sigma_annual, short_contracts, long_contracts,
    net_cost (should be close to 0), and the net Greeks at t=0.
    """
    K1 = round(S0 * BACKSPREAD_K1_PCT, 2)
    K2 = round(S0 * BACKSPREAD_K2_PCT, 2)
    T0 = T0_jours / JOURS_PAR_AN
    sigma_annual = annualize(sigma_WTI_daily)

    p1 = bs_put(S0, K1, T0, RISK_FREE_RATE, sigma_annual)
    p2 = bs_put(S0, K2, T0, RISK_FREE_RATE, sigma_annual)

    ratio = p1 / p2   # natural zero-cost ratio, derived -- not assumed
    short_contracts = BACKSPREAD_SHORT_CONTRACTS
    long_contracts = short_contracts * ratio

    net_cost = (short_contracts * p1 - long_contracts * p2) * TAILLE_CONTRAT_WTI

    d1_p = bs_delta_put(S0, K1, T0, RISK_FREE_RATE, sigma_annual)
    d2_p = bs_delta_put(S0, K2, T0, RISK_FREE_RATE, sigma_annual)
    g1 = bs_gamma(S0, K1, T0, RISK_FREE_RATE, sigma_annual)
    g2 = bs_gamma(S0, K2, T0, RISK_FREE_RATE, sigma_annual)
    v1 = bs_vega(S0, K1, T0, RISK_FREE_RATE, sigma_annual)
    v2 = bs_vega(S0, K2, T0, RISK_FREE_RATE, sigma_annual)
    t1 = bs_theta_put(S0, K1, T0, RISK_FREE_RATE, sigma_annual)
    t2 = bs_theta_put(S0, K2, T0, RISK_FREE_RATE, sigma_annual)

    # Net = long leg (N contracts) minus short leg (1 contract), in position units
    delta_net = long_contracts * d2_p - short_contracts * d1_p
    gamma_net = long_contracts * g2 - short_contracts * g1
    vega_net = long_contracts * v2 - short_contracts * v1
    theta_net = long_contracts * t2 - short_contracts * t1

    print(f"\n{'=' * 60}")
    print("  MODULE E1 -- ZERO-COST PUT RATIO BACKSPREAD")
    print(f"{'=' * 60}")
    print(f"  K1 (short, {BACKSPREAD_K1_PCT:.0%} of spot) : {K1:>8.2f}$   premium {p1:.4f}$")
    print(f"  K2 (long,  {BACKSPREAD_K2_PCT:.0%} of spot) : {K2:>8.2f}$   premium {p2:.4f}$")
    print(f"  Natural zero-cost ratio (P1/P2)          : {ratio:>8.2f}")
    print(f"  Short contracts (K1)                     : {short_contracts:>8.1f}")
    print(f"  Long contracts  (K2)                     : {long_contracts:>8.1f}")
    print(f"  Net cost at inception                    : {net_cost:>8,.0f}$  (target: ~0)")
    print(f"{'=' * 60}")
    print("  NET GREEKS AT INCEPTION")
    print(f"  Delta net : {delta_net:>10.3f}")
    print(f"  Gamma net : {gamma_net:>10.4f}")
    print(f"  Vega  net : {vega_net:>10.3f}")
    print(f"  Theta net : {theta_net:>10.3f}  (per year; not necessarily 0 -- see README)")
    print(f"{'=' * 60}")

    return {
        "K1": K1, "K2": K2, "sigma_annual": sigma_annual,
        "short_contracts": short_contracts, "long_contracts": long_contracts,
        "net_cost": net_cost,
    }


def simulate_backspread(S0, params, mu_HO, mu_WTI, sigma_HO, sigma_WTI, rho_pre,
                         sigma_HO_crash, sigma_WTI_crash,
                         n_sim=N_SIM_BLACK_SWAN, n_jours=N_JOURS_SIMULATION,
                         jour_choc=JOUR_CHOC):
    """
    Marks the backspread to market daily through the Black Swan.
    Returns pnl_backspread : P&L of the OPTIONS OVERLAY ONLY (not the
    physical position or the linear hedge -- those are combined in main.py).
    Also returns the WTI price path array (needed by combined_book.py).
    """
    K1, K2 = params["K1"], params["K2"]
    short_c, long_c = params["short_contracts"], params["long_contracts"]

    pnl_backspread = np.zeros(n_sim)
    delta_paths = np.zeros((n_sim, n_jours))  # net option delta, for Module F

    for sim in range(n_sim):
        pnl = 0.0
        S_wti = S0
        sig_prev = annualize(sigma_WTI)

        for jour in range(n_jours):
            T_rem = max((n_jours - jour) / JOURS_PAR_AN, 1 / JOURS_PAR_AN)

            if jour < jour_choc:
                cov_t = np.array([[sigma_HO ** 2, rho_pre * sigma_HO * sigma_WTI],
                                   [rho_pre * sigma_HO * sigma_WTI, sigma_WTI ** 2]])
                mu_ho_t, mu_wti_t, sig_daily = mu_HO, mu_WTI, sigma_WTI
            else:
                cov_t = np.array([[sigma_HO_crash ** 2, RHO_CRASH * sigma_HO_crash * sigma_WTI_crash],
                                   [RHO_CRASH * sigma_HO_crash * sigma_WTI_crash, sigma_WTI_crash ** 2]])
                mu_ho_t, mu_wti_t, sig_daily = DRIFT_HO_CRASH, DRIFT_WTI_CRASH, sigma_WTI_crash

            sig_t = annualize(sig_daily)
            L_t = np.linalg.cholesky(cov_t)
            shocks = L_t @ np.random.standard_normal(2)
            r_wti = mu_wti_t + shocks[1]

            S_wti_new = S_wti * np.exp(r_wti)

            val_hier = (long_c * bs_put(S_wti, K2, T_rem + 1 / JOURS_PAR_AN, RISK_FREE_RATE, sig_prev)
                        - short_c * bs_put(S_wti, K1, T_rem + 1 / JOURS_PAR_AN, RISK_FREE_RATE, sig_prev))
            val_auj = (long_c * bs_put(S_wti_new, K2, T_rem, RISK_FREE_RATE, sig_t)
                       - short_c * bs_put(S_wti_new, K1, T_rem, RISK_FREE_RATE, sig_t))

            pnl += (val_auj - val_hier) * TAILLE_CONTRAT_WTI

            delta_paths[sim, jour] = (long_c * bs_delta_put(S_wti_new, K2, T_rem, RISK_FREE_RATE, sig_t)
                                       - short_c * bs_delta_put(S_wti_new, K1, T_rem, RISK_FREE_RATE, sig_t))

            S_wti = S_wti_new
            sig_prev = sig_t

        pnl_backspread[sim] = pnl

    return pnl_backspread, delta_paths


def plot_module_e1(pnl_backspread, params, save_path="module_E1_backspread.png"):
    """Distribution of the backspread P&L alone, through the Black Swan."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pnl_backspread, bins=80, color="gold", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(np.median(pnl_backspread), color="darkorange", linestyle="--",
               linewidth=2, label=f"Median: {np.median(pnl_backspread):,.0f}$")
    ax.set_title(f"Module E1 -- Zero-Cost Backspread P&L Distribution\n"
                 f"K1={params['K1']}$ (short) / K2={params['K2']}$ (long), through the Black Swan")
    ax.set_xlabel("P&L ($) -- options overlay only")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: {save_path}")
