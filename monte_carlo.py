"""
MODULE C -- Monte Carlo Simulation (normal regime, pre-Black Swan)

# THEORY
mu, sigma and rho are calibrated ONLY on pre-Covid data, as a desk would
in real time without knowledge of the future. 10,000 correlated paths are
then simulated via Cholesky decomposition:
    Cov = L L^T,  W = Z @ L^T  =>  Cov(W) = L L^T = Cov
This demonstrates two things: (1) the hedge does reduce variance under
normal conditions, and (2) the real Covid P&L is an out-of-distribution
event -- statistically invisible to a model calibrated on the past.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from config import N_SIMULATIONS, N_JOURS_SIMULATION, POSITION_USD, ENTRY_DATE


def calibrate_pre_covid(returns, cutoff=ENTRY_DATE):
    """Calibrates mu, sigma, rho strictly on data prior to cutoff."""
    pre_covid = returns.loc[:cutoff]

    mu_HO = pre_covid["HeatingOil"].mean()
    mu_WTI = pre_covid["WTI"].mean()
    sigma_HO = pre_covid["HeatingOil"].std()
    sigma_WTI = pre_covid["WTI"].std()
    rho_pre = pre_covid["HeatingOil"].corr(pre_covid["WTI"])

    print(f"\n{'=' * 55}")
    print("  PRE-COVID CALIBRATION")
    print(f"{'=' * 55}")
    print(f"  mu Heating Oil    : {mu_HO:.6f}")
    print(f"  mu WTI            : {mu_WTI:.6f}")
    print(f"  sigma Heating Oil : {sigma_HO:.4f}")
    print(f"  sigma WTI         : {sigma_WTI:.4f}")
    print(f"  Correlation rho   : {rho_pre:.4f}")
    print(f"{'=' * 55}")

    return mu_HO, mu_WTI, sigma_HO, sigma_WTI, rho_pre


def run_monte_carlo(mu_HO, mu_WTI, sigma_HO, sigma_WTI, rho_pre, h_star,
                     n_sim=N_SIMULATIONS, n_jours=N_JOURS_SIMULATION):
    """
    Simulates n_sim correlated (HO, WTI) paths over n_jours, in the normal regime.

    Returns
    -------
    pnl_unhedged_mc, pnl_hedged_mc : arrays of final P&L per simulation
    L : Cholesky matrix (reused elsewhere for consistency)
    """
    cov_matrix = np.array([
        [sigma_HO ** 2, rho_pre * sigma_HO * sigma_WTI],
        [rho_pre * sigma_HO * sigma_WTI, sigma_WTI ** 2],
    ])
    L = np.linalg.cholesky(cov_matrix)

    print(f"\nSimulating {n_sim:,} paths over {n_jours} days...")

    pnl_unhedged_mc = np.zeros(n_sim)
    pnl_hedged_mc = np.zeros(n_sim)

    for sim in range(n_sim):
        Z = np.random.standard_normal((2, n_jours))
        correlated_shocks = L @ Z

        returns_HO_sim = mu_HO + correlated_shocks[0]
        returns_WTI_sim = mu_WTI + correlated_shocks[1]

        pnl_ho = POSITION_USD * np.sum(returns_HO_sim)
        pnl_wti = -h_star * POSITION_USD * np.sum(returns_WTI_sim)

        pnl_unhedged_mc[sim] = pnl_ho
        pnl_hedged_mc[sim] = pnl_ho + pnl_wti

    return pnl_unhedged_mc, pnl_hedged_mc, L


def _print_stats(pnl, label):
    print(f"\n  {label}")
    print(f"  {'-' * 40}")
    print(f"  Mean             : {np.mean(pnl):>12,.0f} $")
    print(f"  Volatility       : {np.std(pnl):>12,.0f} $")
    print(f"  VaR 95%          : {np.percentile(pnl, 5):>12,.0f} $")
    print(f"  VaR 99%          : {np.percentile(pnl, 1):>12,.0f} $")
    print(f"  Worst case       : {np.min(pnl):>12,.0f} $")
    print(f"  Best case        : {np.max(pnl):>12,.0f} $")
    print(f"  % scenarios > 0  : {(pnl > 0).mean() * 100:>11.1f}%")


def print_monte_carlo_stats(pnl_unhedged_mc, pnl_hedged_mc, pnl_unhedged_covid, n_sim=N_SIMULATIONS):
    """Prints simulation stats and locates the real Covid P&L within the distribution."""
    print(f"\n{'=' * 55}")
    print(f"  MONTE CARLO RESULTS -- {n_sim:,} SIMULATIONS")
    print(f"{'=' * 55}")
    _print_stats(pnl_unhedged_mc, "UNHEDGED")
    _print_stats(pnl_hedged_mc, "HEDGED")

    vol_reduction = (1 - np.std(pnl_hedged_mc) / np.std(pnl_unhedged_mc)) * 100
    print(f"\n  Volatility reduction : {vol_reduction:>11.1f}%")

    print(f"\n{'=' * 55}")
    print("  WHERE COVID FALLS IN THE DISTRIBUTION")
    print(f"{'=' * 55}")
    pct_covid = stats.percentileofscore(pnl_unhedged_mc, pnl_unhedged_covid)
    print(f"  Real Covid P&L     : {pnl_unhedged_covid:>12,.0f} $")
    print(f"  Percentile         : {pct_covid:>11.1f}%")
    print(f"  -> Covid was a {100 - pct_covid:.1f}% unforeseeable event")
    print(f"{'=' * 55}")


def plot_module_c(pnl_unhedged_mc, pnl_hedged_mc, pnl_unhedged_covid, pnl_hedged_covid,
                   period, L, mu_HO, mu_WTI, n_jours=N_JOURS_SIMULATION,
                   save_path="module_C_monte_carlo.png"):
    """4 charts: distributions, comparison, simulated paths vs reality."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "MODULE C -- Monte Carlo (10,000 simulations)\nP&L Distribution, Normal Regime vs Real Black Swan",
        fontsize=13, fontweight="bold",
    )

    ax1 = axes[0][0]
    ax1.hist(pnl_unhedged_mc, bins=80, color="royalblue", alpha=0.7, edgecolor="none", label="MC distribution")
    ax1.axvline(np.percentile(pnl_unhedged_mc, 5), color="orange", linestyle="--", linewidth=2, label="VaR 95%")
    ax1.axvline(np.percentile(pnl_unhedged_mc, 1), color="red", linestyle="--", linewidth=2, label="VaR 99%")
    ax1.axvline(pnl_unhedged_covid, color="black", linewidth=2.5, label=f"Real Covid: {pnl_unhedged_covid:,.0f}$")
    ax1.set_title("Unhedged P&L -- Monte Carlo Distribution")
    ax1.set_xlabel("P&L ($)"); ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = axes[0][1]
    ax2.hist(pnl_hedged_mc, bins=80, color="green", alpha=0.7, edgecolor="none", label="Hedged MC distribution")
    ax2.axvline(np.percentile(pnl_hedged_mc, 5), color="orange", linestyle="--", linewidth=2, label="VaR 95%")
    ax2.axvline(np.percentile(pnl_hedged_mc, 1), color="red", linestyle="--", linewidth=2, label="VaR 99%")
    ax2.axvline(pnl_hedged_covid, color="black", linewidth=2.5, label=f"Real Covid: {pnl_hedged_covid:,.0f}$")
    ax2.set_title("Hedged P&L -- Monte Carlo Distribution")
    ax2.set_xlabel("P&L ($)"); ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    ax3 = axes[1][0]
    ax3.hist(pnl_unhedged_mc, bins=80, color="royalblue", alpha=0.5, label="Unhedged", density=True)
    ax3.hist(pnl_hedged_mc, bins=80, color="green", alpha=0.5, label="Hedged", density=True)
    ax3.axvline(pnl_unhedged_covid, color="blue", linestyle="--", linewidth=2, label=f"Covid Unhedged: {pnl_unhedged_covid:,.0f}$")
    ax3.axvline(pnl_hedged_covid, color="darkgreen", linestyle="--", linewidth=2, label=f"Covid Hedged: {pnl_hedged_covid:,.0f}$")
    ax3.set_title("Comparison -- The Hedge Reduces Variance in the Normal Regime")
    ax3.set_xlabel("P&L ($)"); ax3.set_ylabel("Density")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    ax4 = axes[1][1]
    h_star = 0.71
    for _ in range(50):
        Z = np.random.standard_normal((2, n_jours))
        shocks = L @ Z
        r_HO = mu_HO + shocks[0]
        r_WTI = mu_WTI + shocks[1]
        pnl_path = POSITION_USD * np.cumsum(r_HO) + (-h_star * POSITION_USD * np.cumsum(r_WTI))
        ax4.plot(pnl_path, color="green", alpha=0.15, linewidth=0.8)

    pnl_real_path = period["PnL_hedged"].iloc[:n_jours].values
    ax4.plot(pnl_real_path, color="red", linewidth=2.5, label="Real Covid path", zorder=5)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_title("50 MC Paths vs Covid Reality (Red)")
    ax4.set_xlabel("Days"); ax4.set_ylabel("Cumulative P&L ($)")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: {save_path}")
