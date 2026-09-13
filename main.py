"""
Cross-Hedge Risk Dashboard -- full orchestration (Modules A to F)

Situation: on 15 January 2020, a trader holds EUR 500,000 of physical Jet
Fuel (no liquid futures exist for it -> Heating Oil is used as a proxy).
The position is hedged with WTI futures. This is stress-tested against the
March 2020 Covid Black Swan, then progressively reinforced with:
  E1 -- a zero-cost Put Ratio Backspread (directional tail convexity)
  E2 -- a daily delta-hedged straddle (pure realized-volatility exposure)
  F  -- combined-book delta netting across E1 and E2

Usage: python main.py
"""

from config import ENTRY_DATE, COVID_DATE
import data_loader
import hedge_ratio
import dynamic_hedging
import monte_carlo
import black_swan
import backspread
import gamma_scalping
import combined_book
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
np.random.seed(42)
def main():
    # ---------------- MODULE A ----------------
    raw, returns = data_loader.load_market_data()
    results = hedge_ratio.compute_rolling_hedge_ratio(raw, returns)
    h_star_entry, rho_entry, nb_contrats = hedge_ratio.print_entry_decision(raw, results)
    hedge_ratio.plot_module_a(results, h_star_entry)
    print("\nModule A complete -- Module B next")

    # ---------------- MODULE B ----------------
    period, nb_units_HO = dynamic_hedging.compute_daily_pnl(raw, results)
    pnl_unhedged_covid, pnl_hedged_covid = dynamic_hedging.print_black_swan_impact(period)
    dynamic_hedging.plot_module_b(period, pnl_unhedged_covid, pnl_hedged_covid)
    print("\nModule B complete -- Module C next")

    # ---------------- MODULE C ----------------
    mu_HO, mu_WTI, sigma_HO, sigma_WTI, rho_pre = monte_carlo.calibrate_pre_covid(returns)
    h_star = h_star_entry  # frozen at entry, never recalibrated mid-crisis
    pnl_unhedged_mc, pnl_hedged_mc, L = monte_carlo.run_monte_carlo(
        mu_HO, mu_WTI, sigma_HO, sigma_WTI, rho_pre, h_star
    )
    monte_carlo.print_monte_carlo_stats(pnl_unhedged_mc, pnl_hedged_mc, pnl_unhedged_covid)
    monte_carlo.plot_module_c(pnl_unhedged_mc, pnl_hedged_mc, pnl_unhedged_covid,
                               pnl_hedged_covid, period, L, mu_HO, mu_WTI)
    print("\nModule C complete -- Module D next")

    # ---------------- MODULE D ----------------
    pnl_bs_unhedged, pnl_bs_hedged, h_star_mean, sigma_HO_crash, sigma_WTI_crash = \
        black_swan.inject_black_swan(mu_HO, mu_WTI, sigma_HO, sigma_WTI, rho_pre, h_star)
    black_swan.print_black_swan_stats(pnl_bs_unhedged, pnl_bs_hedged, pnl_hedged_mc)
    black_swan.plot_module_d(h_star_mean, h_star, pnl_unhedged_mc, pnl_hedged_mc,
                              pnl_bs_unhedged, pnl_bs_hedged, mu_HO, mu_WTI,
                              sigma_HO, sigma_WTI, rho_pre, sigma_HO_crash, sigma_WTI_crash)
    print("\nModule D complete -- Module E1 next")

    S0 = raw.loc[:ENTRY_DATE]["WTI"].iloc[-1]

    # ---------------- MODULE E1 -- BACKSPREAD ----------------
    bs_params = backspread.setup_backspread(S0, sigma_WTI)
    pnl_backspread, _ = backspread.simulate_backspread(
        S0, bs_params, mu_HO, mu_WTI, sigma_HO, sigma_WTI, rho_pre,
        sigma_HO_crash, sigma_WTI_crash,
    )
    backspread.plot_module_e1(pnl_backspread, bs_params)
    print("\nModule E1 complete -- Module E2 next")

    # ---------------- MODULE E2 -- GAMMA SCALPING ----------------
    st_params = gamma_scalping.setup_straddle(S0, sigma_WTI)
    pnl_scalping, _ = gamma_scalping.simulate_gamma_scalping(
        S0, st_params, mu_HO, mu_WTI, sigma_HO, sigma_WTI, rho_pre,
        sigma_HO_crash, sigma_WTI_crash,
    )
    gamma_scalping.plot_module_e2(pnl_scalping, st_params)
    print("\nModule E2 complete -- Module F next")

    # ---------------- MODULE F -- COMBINED BOOK ----------------
    pnl_isolated, pnl_unified, illustrative = combined_book.simulate_combined_book(
        S0, bs_params, st_params, nb_contrats,
        mu_HO, mu_WTI, sigma_HO, sigma_WTI, rho_pre,
        sigma_HO_crash, sigma_WTI_crash,
    )
    combined_book.print_combined_book_stats(pnl_isolated, pnl_unified)
    combined_book.plot_module_f(illustrative, pnl_isolated, pnl_unified)

    print("\n" + "=" * 60)
    print("  PROJECT COMPLETE")
    print("  Modules A, B, C, D, E1, E2, F done")
    print("=" * 60)


if __name__ == "__main__":
    main()

