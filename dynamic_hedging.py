"""
MODULE B -- Day-by-Day Dynamic Hedging

# THEORY
The historical data is replayed exactly as it happened: each day, the P&L
of the physical position (Jet Fuel, via the Heating Oil proxy) is compared
to the P&L of the hedge (short WTI futures, rebalanced daily according to
the previous day's h*). This measures, on REAL Covid-era data, whether the
hedge genuinely reduced variance -- and what happens once the correlation
underlying h* collapses.
"""

import pandas as pd
import matplotlib.pyplot as plt

from config import POSITION_USD, TAILLE_CONTRAT_WTI, ENTRY_DATE, COVID_START, COVID_DATE


def compute_daily_pnl(raw, results, entry_date=ENTRY_DATE):
    """
    Simulates daily P&L, with and without the hedge, on real market data.

    Returns
    -------
    period : DataFrame -- day-by-day P&L starting at entry_date
    nb_units_HO : float -- units of physical Jet Fuel held
    """
    entry_date = pd.to_datetime(entry_date)
    prix_HO_entry = raw.loc[entry_date, "HeatingOil"]

    nb_units_HO = POSITION_USD / prix_HO_entry
    print(f"HO units bought on {entry_date.date()}: {nb_units_HO:,.0f}")

    period = raw[raw.index >= entry_date].copy()

    # Physical position P&L (long Jet Fuel)
    period["PnL_HO_daily"] = period["HeatingOil"].diff() * nb_units_HO
    period["PnL_unhedged"] = period["PnL_HO_daily"].cumsum()

    # Rolling hedge ratio aligned to the period (daily rebalancing)
    h_star_series = results["HedgeRatio"].reindex(period.index, method="ffill")
    nb_contrats_series = results["NbContrats"].reindex(period.index, method="ffill")
    period["NbContrats"] = nb_contrats_series

    # Hedge P&L: short WTI -> gains when WTI falls
    period["WTI_diff"] = period["WTI"].diff()
    period["PnL_WTI_daily"] = -nb_contrats_series * period["WTI_diff"] * TAILLE_CONTRAT_WTI
    period["PnL_hedge_cum"] = period["PnL_WTI_daily"].cumsum()

    period["PnL_hedged"] = period["PnL_unhedged"] + period["PnL_hedge_cum"]
    period = period.dropna()

    return period, nb_units_HO


def print_black_swan_impact(period, covid_date=COVID_DATE):
    """Compares P&L and volatility on the day of the Black Swan."""
    covid_date = pd.to_datetime(covid_date)
    pnl_unhedged_covid = period.loc[covid_date, "PnL_unhedged"]
    pnl_hedged_covid = period.loc[covid_date, "PnL_hedged"]

    print(f"\n{'=' * 55}")
    print(f"  BLACK SWAN IMPACT -- {covid_date.date()}")
    print(f"{'=' * 55}")
    print(f"  P&L Unhedged      : {pnl_unhedged_covid:>12,.0f} $")
    print(f"  P&L Hedged        : {pnl_hedged_covid:>12,.0f} $")
    print(f"  Protection        : {pnl_hedged_covid - pnl_unhedged_covid:>12,.0f} $")
    print(f"{'=' * 55}")

    vol_unhedged = period["PnL_HO_daily"].std()
    vol_hedged = (period["PnL_HO_daily"] + period["PnL_WTI_daily"]).std()
    reduction_vol = (1 - vol_hedged / vol_unhedged) * 100

    print(f"\n  Vol P&L Unhedged   : {vol_unhedged:>10,.0f} $/day")
    print(f"  Vol P&L Hedged     : {vol_hedged:>10,.0f} $/day")
    print(f"  Vol reduction      : {reduction_vol:>10.1f}%")
    print(f"{'=' * 55}")

    return pnl_unhedged_covid, pnl_hedged_covid


def plot_module_b(period, pnl_unhedged_covid, pnl_hedged_covid,
                   covid_date=COVID_DATE, covid_start=COVID_START,
                   save_path="module_B_hedging_dynamique.png"):
    """3 charts: cumulative P&L, spot vs futures prices, rebalanced contracts."""
    covid_date = pd.to_datetime(covid_date)
    covid_start = pd.to_datetime(covid_start)

    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.suptitle(
        "MODULE B -- Daily Dynamic Hedging\nLong EUR 500K Jet Fuel + Short WTI Futures",
        fontsize=13, fontweight="bold",
    )

    ax1 = axes[0]
    ax1.plot(period.index, period["PnL_unhedged"], color="royalblue", linewidth=1.5, label="P&L Unhedged")
    ax1.plot(period.index, period["PnL_hedged"], color="green", linewidth=1.5, label="P&L Hedged")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axvline(covid_date, color="red", linestyle="--", linewidth=2, label="Black Swan Mar 18")
    ax1.axvspan(covid_start, covid_date, alpha=0.1, color="red", label="Danger zone")
    ax1.annotate(f"{pnl_unhedged_covid:,.0f}$", xy=(covid_date, pnl_unhedged_covid),
                 xytext=(30, -20), textcoords="offset points", color="royalblue", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="royalblue"))
    ax1.annotate(f"{pnl_hedged_covid:,.0f}$", xy=(covid_date, pnl_hedged_covid),
                 xytext=(30, 20), textcoords="offset points", color="green", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="green"))
    ax1.set_ylabel("Cumulative P&L ($)")
    ax1.set_title("P&L Unhedged (Blue) vs Hedged (Green)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.plot(period.index, period["HeatingOil"], color="orange", linewidth=1.5, label="Heating Oil (Jet Fuel proxy)")
    ax2_twin.plot(period.index, period["WTI"], color="gray", linewidth=1.5, linestyle="--", label="WTI (Hedge)")
    ax2.axvline(covid_date, color="red", linestyle="--", linewidth=2)
    ax2.axvspan(covid_start, covid_date, alpha=0.1, color="red")
    ax2.set_ylabel("Heating Oil ($)", color="orange")
    ax2_twin.set_ylabel("WTI ($)", color="gray")
    ax2.set_title("Jet Fuel Spot vs WTI Futures Prices")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.fill_between(period.index, period["NbContrats"], color="purple", alpha=0.4, label="WTI contracts shorted")
    ax3.plot(period.index, period["NbContrats"], color="purple", linewidth=1)
    ax3.axvline(covid_date, color="red", linestyle="--", linewidth=2, label="Black Swan")
    ax3.axvspan(covid_start, covid_date, alpha=0.1, color="red")
    ax3.set_ylabel("Number of contracts")
    ax3.set_title("Daily Rebalancing -- WTI Contract Count")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: {save_path}")
