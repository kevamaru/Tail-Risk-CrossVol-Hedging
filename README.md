# Cross-Hedge Risk Dashboard — Distillate Trading

A self-contained Python project stress-testing a **cross-hedge** (WTI
Futures used to hedge a Jet Fuel position, via a Heating Oil proxy)
through the Covid-19 Black Swan, then progressively reinforcing it with
options overlays that capture volatility convexity.

## Situation

Imagine being a **market maker on Jet Fuel**, already holding a **€500,000
inventory position** just before closing a deal with sales on the trading
floor for a client. There is no liquid, exchange-traded Jet Fuel futures
contract, so the position is hedged with **WTI Crude futures** instead —
a classic **cross-hedge**: two related but not identical assets.

## Methodology

### Module A — Rolling Hedge Ratio (`hedge_ratio.py`)

The variance-minimizing hedge ratio:

$$h^* = \frac{Cov(S, F)}{Var(F)} = \rho \cdot \frac{\sigma_S}{\sigma_F}$$

Recomputed on a 30-day rolling window.

### Module B — Dynamic Hedging on Real Data (`dynamic_hedging.py`)

Replays the hedge day-by-day on **actual historical prices** (2018–2021).

### Module C — Monte Carlo, Normal Regime (`monte_carlo.py`)

Calibrated strictly on pre-Covid historical data using **Geometric
Brownian Motion (GBM)** with Cholesky decomposition to simulate
correlated Brownian drivers. 10,000 Monte Carlo paths confirm that the
linear cross-hedge successfully reduces portfolio variance by ~53% under
standard market conditions.

However, the realized Covid-19 P&L lands completely outside this
simulated distribution (0.0th percentile). This highlights the core
failure of standard Gaussian risk frameworks: by assuming stable
correlations and log-normal distributions, the model generates deceptive
Value-at-Risk (VaR) metrics that lure the desk into a **false sense of
security** right before a fat-tail regime shift.

### Module D — Injecting the Black Swan (`black_swan.py`)

The core vulnerability lies in the inventory structure: the physical book
holds spot Heating Oil (a middle-distillate proxy for Jet Fuel). When
governments announced flight bans to contain Covid-19, commercial
aviation was grounded almost instantly, causing physical Jet Fuel and
distillate prices to fall far faster and harder than WTI Crude futures.

This macro shock breaks the underlying cross-hedge dynamics:

- **Correlation collapse:** ρ drops from 0.8851 to 0.2000.
- **Volatility explosion:** Heating Oil daily volatility roughly
  quadruples (×4, to ~0.0628), while WTI volatility roughly quintuples
  (×5, to ~0.1029).

With the linear hedge ratio h\* **frozen** at 0.7100 (a desk does not
recalibrate mid-crash), the residual variance is:

$$Var_{min}(\Pi) = \sigma_S^2 (1 - \rho^2)$$

Because spot variance σ_S² expands while (1 − ρ²) jumps from ~0.22 to
~0.96, the short WTI futures position fails to offset the physical
inventory drop. Hedged portfolio volatility explodes roughly **8×**,
triggering the kind of variation margin calls that turn a hedge into a
second source of losses. This is why a book running only a linear hedge
needs a second layer: **long convexity, long volatility**, designed to
profit from exactly this kind of regime shift.

### Module E1 — Zero-Cost Put Ratio Backspread (`backspread.py`)

Sells 1 Put closer to the money (K1, higher premium) to finance ~3 Long
Puts deep out-of-the-money (K2, cheaper premium each). This structure
achieves a near-zero upfront cost while creating massive convexity on a
large downside move — Spitznagel/Taleb-style tail-risk protection.

> **Design note:** the ratio is *derived* from Black-Scholes pricing, not
> imposed. An earlier draft tried to force both a specific strike pair
> *and* an exact "1×3" ratio — with flat volatility these two constraints
> are generally incompatible. K1=95% / K2=85% of spot keeps both legs
> meaningfully priced at a 3-month tenor and happens to produce a ~3.3×
> ratio, close to the classic "1×3" backspread, but obtained from the
> model rather than assumed. Pushing K2 much deeper (e.g. 65% of spot)
> prices the long leg near zero at this tenor under flat vol, producing
> an unusable ratio (>150×) — a real skew or a longer tenor would be
> needed to make a deeper structure viable.

**Net Greeks at inception, and what they mean for this position:**

| Greek | Value | What it says |
|---|---|---|
| Delta | -1.13 | Near flat directional exposure at entry — this is not a bet that WTI falls tomorrow, it's a bet on a large move happening at all. |
| Gamma | +0.38 | Positive: as WTI falls, the position gets *more* short — the sensitivity itself compounds in the right direction. This is the convexity Taleb describes as antifragile. |
| Vega | +103.6 | Positive: a spike in implied volatility alone (even before price moves) adds value — exactly the mechanism that pays off when Covid-style vol arrives. |
| Theta | -66.75 / yr (~-265$/day at this size) | The cost of the insurance: this bleeds slowly in a calm market, the same way an insurance premium is paid whether or not the accident happens. |

### Module E2 — Gamma Scalping (`gamma_scalping.py`)

A delta-hedged ATM straddle (K = $57.81) on WTI futures, designed to
isolate pure realized volatility and monetize market oscillations,
independent of direction.

**Mechanics:**
- **The options (the barometer):** a long ATM straddle stays static on
  the book, providing continuous positive Gamma exposure without any
  further option execution until expiry.
- **The futures (the cash engine):** daily delta rebalancing forces
  mechanical, disciplined execution — market rallies push delta positive
  and the hedge sells futures into strength; market drops push delta
  negative and the hedge buys futures into weakness.
- **The P&L loop:** captures alpha whenever realized volatility exceeds
  the implied volatility used to price the straddle. That cash flow is
  what finances the daily Theta bleed and, in a real book, transaction
  costs.

**Position at inception:**

| | Value |
|---|---|
| Strike K (ATM) | $57.81 |
| Call premium | $3.86 |
| Put premium | $3.65 |
| Straddle premium | $7.51 |
| Budget | $11,150 |
| Straddle contracts | 1.48 |

**Validation:** with no injected crash and realized volatility matching
the volatility used for pricing, mean P&L over 3,000 paths is ~$117
against a ~$1,160 standard deviation — consistent with the theoretical
zero-mean property of a correctly delta-hedged position. Under the full
Black Swan scenario, the overlay acts as an autonomous liquidity
generator precisely when the desk needs it most.

### Module F — Combined Book (`combined_book.py`)

E1 is a **static** position — never rehedged, so its delta *drifts*. E2
is **actively** self-hedged to zero, in isolation. A risk report that only
checks "is E2 hedged?" always says yes — yet the options book as a whole
still carries E1's unmanaged delta. This module re-simulates E1 and E2
**jointly on the same price path** (netting two independently-simulated
Monte Carlo arrays would mean nothing) and compares:

- **Isolated** — E2 hedges only itself; E1's delta is left to drift.
- **Unified** — one hedge neutralizes (Delta_E1 + Delta_E2) together.

**Honest finding:** in this specific backtest, *Isolated* shows a higher
mean P&L than *Unified*. This is **not** evidence that ignoring risk is
better — it is an artifact of the simulation design: the Black Swan
always crashes WTI **downward** with certainty (Module D's methodology).
E1's naturally short delta, left unhedged, rides that guaranteed direction
for extra profit. In reality the crash direction is not known in advance;
an unhedged directional drift that happens to align with a backtest's
scripted outcome is not a repeatable edge. What *is* a genuine, direction-
agnostic result: **Unified cuts P&L volatility roughly 5×** (from
~294,000$ to ~62,000$) — a real reduction in uncertainty, independent of
which way the crash breaks.

## Limitations

- **Flat volatility.** No skew is modeled. Real markets price deep OTM
  puts with a volatility premium, making tail protection more expensive
  than shown here (see Module E1 design note).
- **No transaction costs.** Module E2's daily rehedging and Module F's
  unified execution ignore bid-ask spread and slippage.
- **Margin ≠ premium.** A "zero-cost" structure still requires margin
  and is subject to variation margin calls from the clearing house — the
  €500K position size functions as a liquidity buffer for this, not as
  the option premium itself.
- **Deterministic crash direction**, as discussed under Module F above.
- **No fixed random seed.** Monte Carlo figures shown here (Modules C,
  D, E1, E2, F) will vary slightly on each run. Set `np.random.seed(...)`
  in `main.py` for exact reproducibility if needed.

## Running

```bash
pip install -r requirements.txt
python main.py
```

Runs Modules A through F in sequence, printing full diagnostics and
saving 7 figures to the working directory.

## Project Structure

```
hedge_project/
├── config.py             # shared parameters
├── data_loader.py         # yfinance download (WTI, Heating Oil)
├── black_scholes.py         # shared Black-Scholes pricing library
├── hedge_ratio.py              # Module A
├── dynamic_hedging.py            # Module B
├── monte_carlo.py                  # Module C
├── black_swan.py                     # Module D
├── backspread.py                       # Module E1
├── gamma_scalping.py                     # Module E2
├── combined_book.py                        # Module F
├── main.py                                   # orchestration
└── requirements.txt
```

## Tech

Python — NumPy, Pandas, Matplotlib, SciPy, yfinance.
#   T a i l - R i s k - C r o s s V o l - H e d g i n g  
 