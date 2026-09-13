"""
MODULE A -- Rolling Optimal Hedge Ratio

# THEORY
The optimal hedge ratio h* minimizes the variance of the hedged portfolio:
    Pi = Delta_S - h * Delta_F
Differentiating Var(Pi) with respect to h and setting it to zero:
    h* = Cov(S, F) / Var(F) = rho * (sigma_S / sigma_F)
This is a simple linear regression of the position (spot, here Jet Fuel)
on the hedging instrument (WTI futures). h* is recomputed on a 30-day
rolling window to adapt to changing market conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    ROLLING_WINDOW, ENTRY_DATE, COVID_START, COVID_DATE,
    POSITION_EUR, POSITION_USD, TAILLE_CONTRAT_WTI,
)


def compute_rolling_hedge_ratio(raw, returns, window=ROLLING_WINDOW):
    """
    Computes h*, correlation and volatilities on a rolling window.

    Returns
    -------
    results : DataFrame
        Columns ['HedgeRatio', 'Correlation', 'Vol_HO', 'Vol_WTI', 'NbContrats']
    """
    hedge_ratios, correlations, vol_HO, vol_WTI, dates = [], [], [], [], []

    for i in range(window, len(returns)):
        w = returns.iloc[i - window:i]

        cov_matrix = np.cov(w["HeatingOil"], w["WTI"])
        cov, var = cov_matrix[0][1], cov_matrix[1][1]

        sigma_HO = np.std(w["HeatingOil"])
        sigma_WTI = np.std(w["WTI"])

        rho = cov / (sigma_HO * sigma_WTI) if (sigma_HO * sigma_WTI) != 0 else 0
        h_star = cov / var if var != 0 else 0

        hedge_ratios.append(h_star)
        correlations.append(rho)
        vol_HO.append(sigma_HO)
        vol_WTI.append(sigma_WTI)
        dates.append(returns.index[i])

    results = pd.DataFrame(
        {
            "HedgeRatio": hedge_ratios,
            "Correlation": correlations,
            "Vol_HO": vol_HO,
            "Vol_WTI": vol_WTI,
        },
        index=dates,
    )

    # Number of WTI contracts to short, updated every day
    results["NbContrats"] = (results["HedgeRatio"] * POSITION_USD) / (
        raw["WTI"].reindex(results.index) * TAILLE_CONTRAT_WTI
    )

    return results


def print_entry_decision(raw, results, entry_date=ENTRY_DATE):
    """Prints the hedging decision at the entry date."""
    prix_wti = raw.loc[:entry_date]["WTI"].iloc[-1]
    h_star_entry = results.loc[:entry_date]["HedgeRatio"].iloc[-1]
    rho_entry = results.loc[:entry_date]["Correlation"].iloc[-1]
    nb_contrats = (h_star_entry * POSITION_USD) / (prix_wti * TAILLE_CONTRAT_WTI)

    print(f"\n{'=' * 55}")
    print("  DECISION ON JANUARY 15, 2020")
    print(f"{'=' * 55}")
    print(f"  Long Jet Fuel Position    : {POSITION_EUR:>12,.0f} EUR")
    print(f"  Hedge Ratio h*            : {h_star_entry:>12.4f}")
    print(f"  Correlation rho           : {rho_entry:>12.4f}")
    print(f"  WTI Price                 : {prix_wti:>12.2f} $")
    print(f"  WTI Contracts to SHORT    : {nb_contrats:>12.1f}")
    print(f"  Hedge Value               : {nb_contrats * prix_wti * TAILLE_CONTRAT_WTI:>12,.0f} $")
    print(f"{'=' * 55}")

    return h_star_entry, rho_entry, nb_contrats


def plot_module_a(results, h_star_entry, save_path="module_A_hedge_ratio.png"):
    """3 charts: evolution of h*, of rho, and of the number of contracts."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        "MODULE A -- Rolling 30-day Hedge Ratio\nWTI Futures vs Jet Fuel (Heating Oil)",
        fontsize=13, fontweight="bold",
    )

    entry = pd.to_datetime(ENTRY_DATE)
    covid_start = pd.to_datetime(COVID_START)
    covid_crash = pd.to_datetime(COVID_DATE)

    ax1 = axes[0]
    ax1.plot(results.index, results["HedgeRatio"], color="royalblue", linewidth=1.5, label="h* rolling 30d")
    ax1.axhline(h_star_entry, color="green", linestyle=":", alpha=0.7, label=f"h* at entry = {h_star_entry:.3f}")
    ax1.axvline(entry, color="green", linestyle="--", linewidth=1.5, label="Entry Jan 15")
    ax1.axvline(covid_crash, color="red", linestyle="--", linewidth=1.5, label="Black Swan")
    ax1.axvspan(covid_start, covid_crash, alpha=0.1, color="red", label="Danger zone")
    ax1.set_ylabel("Hedge Ratio h*")
    ax1.set_title("Evolution of the Optimal Hedge Ratio")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(results.index, results["Correlation"], color="darkorange", linewidth=1.5, label="Correlation rho rolling 30d")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axvline(entry, color="green", linestyle="--", linewidth=1.5)
    ax2.axvline(covid_crash, color="red", linestyle="--", linewidth=1.5, label="Black Swan")
    ax2.axvspan(covid_start, covid_crash, alpha=0.1, color="red")
    ax2.set_ylabel("Correlation rho")
    ax2.set_title("Correlation Breakdown During the Black Swan")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.2, 1.1)

    ax3 = axes[2]
    ax3.plot(results.index, results["NbContrats"], color="purple", linewidth=1.5, label="WTI contracts to short")
    ax3.axvline(entry, color="green", linestyle="--", linewidth=1.5, label="Entry")
    ax3.axvline(covid_crash, color="red", linestyle="--", linewidth=1.5, label="Black Swan")
    ax3.axvspan(covid_start, covid_crash, alpha=0.1, color="red")
    ax3.set_ylabel("Number of contracts")
    ax3.set_title("Optimal Number of Contracts -- Daily Rebalancing")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: {save_path}")
