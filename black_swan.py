"""
MODULE D -- Injecting the Black Swan

# THEORY
At day JOUR_CHOC, what the real Covid crash produced is forced into the
simulation: correlation collapses (0.88 -> 0.20) and volatilities spike
(x4 on Heating Oil, x5 on WTI). The hedge ratio h* stays FROZEN at its
pre-crisis value (exactly as in real life -- a desk does not recalibrate
its model mid-crash). This exposes the minimum residual variance of a
linear hedge:
    Var_min(Pi) = Var(S) * (1 - rho^2)
As rho -> 0.20, this residual variance explodes: the hedge becomes
statistically useless, or worse.
"""

import numpy as np
import matplotlib.pyplot as plt

from config import (
    RHO_CRASH, VOL_MULT_HO, VOL_MULT_WTI, DRIFT_HO_CRASH, DRIFT_WTI_CRASH,
    N_SIM_BLACK_SWAN, N_JOURS_SIMULATION, JOUR_CHOC, POSITION_USD,
)


def inject_black_swan(mu_HO, mu_WTI, sigma_HO, sigma_WTI, rho_pre, h_star,
                       n_sim=N_SIM_BLACK_SWAN, n_jours=N_JOURS_SIMULATION,
                       jour_choc=JOUR_CHOC):
    """
    Simulates n_sim paths with a Black Swan injected at day jour_choc.
    h_star stays frozen at its entry value throughout (no recalibration).

    Returns
    -------
    pnl_bs_unhedged, pnl_bs_hedged : arrays of final P&L
    h_star_mean : average evolution of the "true" h* that would have been needed
    sigma_HO_crash, sigma_WTI_crash : crisis-regime volatilities
    """
    sigma_HO_crash = sigma_HO * VOL_MULT_HO
    sigma_WTI_crash = sigma_WTI * VOL_MULT_WTI

    print(f"\n{'=' * 55}")
    print("  MODULE D -- BLACK SWAN INJECTION")
    print(f"{'=' * 55}")
    print(f"  rho before shock   : {rho_pre:.4f}")
    print(f"  rho after shock    : {RHO_CRASH:.4f}")
    print(f"  sigma HO before    : {sigma_HO:.4f}")
    print(f"  sigma HO after     : {sigma_HO_crash:.4f}  (x{VOL_MULT_HO})")
    print(f"  sigma WTI before   : {sigma_WTI:.4f}")
    print(f"  sigma WTI after    : {sigma_WTI_crash:.4f}  (x{VOL_MULT_WTI})")
    print(f"{'=' * 55}")

    pnl_bs_unhedged = np.zeros(n_sim)
    pnl_bs_hedged = np.zeros(n_sim)
    h_star_paths = np.zeros((n_sim, n_jours))

    for sim in range(n_sim):
        pnl_ho_cum, pnl_wti_cum = 0.0, 0.0
        h_current = h_star

        for jour in range(n_jours):
            if jour < jour_choc:
                cov_t = np.array([
                    [sigma_HO ** 2, rho_pre * sigma_HO * sigma_WTI],
                    [rho_pre * sigma_HO * sigma_WTI, sigma_WTI ** 2],
                ])
                mu_ho_t, mu_wti_t = mu_HO, mu_WTI
            else:
                cov_t = np.array([
                    [sigma_HO_crash ** 2, RHO_CRASH * sigma_HO_crash * sigma_WTI_crash],
                    [RHO_CRASH * sigma_HO_crash * sigma_WTI_crash, sigma_WTI_crash ** 2],
                ])
                mu_ho_t, mu_wti_t = DRIFT_HO_CRASH, DRIFT_WTI_CRASH
                # The "true" h* that would have been needed (hindsight information)
                h_current = (RHO_CRASH * sigma_HO_crash) / sigma_WTI_crash

            L_t = np.linalg.cholesky(cov_t)
            shocks = L_t @ np.random.standard_normal(2)
            r_HO, r_WTI = mu_ho_t + shocks[0], mu_wti_t + shocks[1]

            pnl_ho_cum += POSITION_USD * r_HO
            # h_star stays FROZEN at its entry value -- the core of the demonstration
            pnl_wti_cum += -h_star * POSITION_USD * r_WTI

            h_star_paths[sim, jour] = h_current

        pnl_bs_unhedged[sim] = pnl_ho_cum
        pnl_bs_hedged[sim] = pnl_ho_cum + pnl_wti_cum

    h_star_mean = h_star_paths.mean(axis=0)

    print(f"\n  h* on Day 1  (normal regime) : {h_star_mean[0]:.4f}")
    print(f"  h* on Day {jour_choc - 1} (day before shock) : {h_star_mean[jour_choc - 1]:.4f}")
    print(f"  h* on Day {jour_choc + 1} (day after shock)  : {h_star_mean[jour_choc + 1]:.4f}")
    print(f"  h* on Day {n_jours - 1} (end)              : {h_star_mean[-1]:.4f}")

    return pnl_bs_unhedged, pnl_bs_hedged, h_star_mean, sigma_HO_crash, sigma_WTI_crash


def print_black_swan_stats(pnl_bs_unhedged, pnl_bs_hedged, pnl_hedged_mc):
    """Comparative stats, normal regime vs Black Swan."""
    def _stats(pnl, label):
        print(f"\n  {label}")
        print(f"  {'-' * 40}")
        print(f"  Mean             : {np.mean(pnl):>12,.0f} $")
        print(f"  Volatility       : {np.std(pnl):>12,.0f} $")
        print(f"  VaR 95%          : {np.percentile(pnl, 5):>12,.0f} $")
        print(f"  VaR 99%          : {np.percentile(pnl, 1):>12,.0f} $")
        print(f"  Worst case       : {np.min(pnl):>12,.0f} $")

    print(f"\n{'=' * 55}")
    print("  BLACK SWAN IMPACT -- 5,000 SIMULATIONS")
    print(f"{'=' * 55}")
    _stats(pnl_bs_unhedged, "UNHEDGED -- WITH BLACK SWAN")
    _stats(pnl_bs_hedged, "HEDGED -- WITH BLACK SWAN")

    vol_normale = np.std(pnl_hedged_mc)
    vol_bs = np.std(pnl_bs_hedged)
    print(f"\n  Hedged vol, normal regime : {vol_normale:>10,.0f} $")
    print(f"  Hedged vol, Black Swan    : {vol_bs:>10,.0f} $")
    print(f"  Volatility amplification  : x{vol_bs / vol_normale:.1f}")
    print(f"{'=' * 55}")


def plot_module_d(h_star_mean, h_star, pnl_unhedged_mc, pnl_hedged_mc,
                   pnl_bs_unhedged, pnl_bs_hedged, mu_HO, mu_WTI, sigma_HO, sigma_WTI,
                   rho_pre, sigma_HO_crash, sigma_WTI_crash,
                   n_jours=N_JOURS_SIMULATION, jour_choc=JOUR_CHOC,
                   save_path="module_D_black_swan.png"):
    """4 charts: collapsing h*, distributions, simulated paths, compared volatility."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"MODULE D -- Black Swan Injection\nForced Decorrelation at Day {jour_choc} -- rho: {rho_pre:.2f} -> {RHO_CRASH:.2f}",
        fontsize=13, fontweight="bold",
    )

    ax1 = axes[0][0]
    ax1.plot(range(n_jours), h_star_mean, color="purple", linewidth=2, label="Mean h*")
    ax1.axvline(jour_choc, color="red", linestyle="--", linewidth=2, label="Black Swan injection")
    ax1.axhline(h_star, color="green", linestyle=":", linewidth=1.5, label=f"Initial h* = {h_star:.3f}")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.fill_between(range(jour_choc, n_jours), h_star_mean[jour_choc:], h_star, alpha=0.2, color="red", label="Danger zone")
    ax1.set_title("h* Becomes Useless After the Shock")
    ax1.set_xlabel("Days"); ax1.set_ylabel("Hedge Ratio h*")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = axes[0][1]
    ax2.hist(pnl_hedged_mc, bins=60, color="green", alpha=0.5, density=True, label="Hedged -- Normal Regime")
    ax2.hist(pnl_bs_hedged, bins=60, color="red", alpha=0.5, density=True, label="Hedged -- Black Swan")
    ax2.axvline(np.percentile(pnl_bs_hedged, 1), color="darkred", linestyle="--", linewidth=2, label="VaR 99% Black Swan")
    ax2.set_title("Hedged Distribution -- Normal vs Black Swan")
    ax2.set_xlabel("P&L ($)"); ax2.set_ylabel("Density")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    ax3 = axes[1][0]
    for _ in range(100):
        pnl_cum, pnl_path = 0.0, []
        for jour in range(n_jours):
            if jour < jour_choc:
                cov_t = np.array([[sigma_HO ** 2, rho_pre * sigma_HO * sigma_WTI],
                                   [rho_pre * sigma_HO * sigma_WTI, sigma_WTI ** 2]])
                mh, mw = mu_HO, mu_WTI
            else:
                cov_t = np.array([[sigma_HO_crash ** 2, RHO_CRASH * sigma_HO_crash * sigma_WTI_crash],
                                   [RHO_CRASH * sigma_HO_crash * sigma_WTI_crash, sigma_WTI_crash ** 2]])
                mh, mw = DRIFT_HO_CRASH, DRIFT_WTI_CRASH
            L_t = np.linalg.cholesky(cov_t)
            sh = L_t @ np.random.standard_normal(2)
            pnl_cum += POSITION_USD * (mh + sh[0]) + (-h_star * POSITION_USD * (mw + sh[1]))
            pnl_path.append(pnl_cum)
        ax3.plot(pnl_path, color="red", alpha=0.1, linewidth=0.8)

    ax3.axvline(jour_choc, color="black", linestyle="--", linewidth=2, label=f"Black Swan Day {jour_choc}")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title(f"100 Hedged P&L Paths -- Black Swan at Day {jour_choc}")
    ax3.set_xlabel("Days"); ax3.set_ylabel("Cumulative P&L ($)")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    ax4 = axes[1][1]
    categories = ["Unhedged\nNormal", "Hedged\nNormal", "Unhedged\nBlack Swan", "Hedged\nBlack Swan"]
    vols = [np.std(pnl_unhedged_mc), np.std(pnl_hedged_mc), np.std(pnl_bs_unhedged), np.std(pnl_bs_hedged)]
    colors = ["royalblue", "green", "darkorange", "red"]
    bars = ax4.bar(categories, vols, color=colors, alpha=0.8, edgecolor="none")
    for bar, vol in zip(bars, vols):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vols) * 0.02,
                  f"{vol:,.0f}$", ha="center", va="bottom", fontsize=9)
    ax4.set_title("P&L Volatility -- Normal vs Black Swan")
    ax4.set_ylabel("Volatility ($)")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: {save_path}")
