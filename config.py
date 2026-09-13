"""
Central configuration for the Cross-Hedge Risk Dashboard project.
All parameters shared across modules live here.
"""

# --- Market ---
TICKERS = ["CL=F", "HO=F"]          # WTI Futures, Heating Oil (Jet Fuel proxy)
DATA_START = "2018-01-01"
DATA_END   = "2021-12-31"

# --- Position to hedge ---
POSITION_EUR   = 500_000            # Long physical Jet Fuel
TAUX_EURUSD    = 1.115
POSITION_USD   = POSITION_EUR * TAUX_EURUSD
TAILLE_CONTRAT_WTI = 1_000          # barrels per WTI futures contract

# --- Key dates ---
ENTRY_DATE  = "2020-01-15"          # Position entry
COVID_START = "2020-02-20"          # Start of the danger zone
COVID_DATE  = "2020-03-18"          # Peak of the Black Swan

# --- Rolling Hedge Ratio ---
ROLLING_WINDOW = 30                 # days

# --- Monte Carlo ---
N_SIMULATIONS = 10_000              # Module C (normal regime)
N_SIM_BLACK_SWAN = 5_000            # Modules D, E, F
N_JOURS_SIMULATION = 63             # ~3 months of trading (Jan -> Mar 2020)
JOUR_CHOC = 30                      # The Black Swan hits on day 30

# --- Injected Black Swan parameters ---
RHO_CRASH = 0.20                    # Correlation collapse (vs ~0.88 normal)
VOL_MULT_HO  = 4                    # Heating Oil vol x4 during the crash
VOL_MULT_WTI = 5                    # WTI vol x5 during the crash
DRIFT_HO_CRASH  = -0.035            # Daily drift during the crash
DRIFT_WTI_CRASH = -0.020

# --- Module E1: Zero-Cost Put Ratio Backspread ---
BACKSPREAD_K1_PCT = 0.95            # Short leg strike (financing), 5% OTM
BACKSPREAD_K2_PCT = 0.85            # Long leg strike (convexity), 15% OTM
BACKSPREAD_SHORT_CONTRACTS = 10     # Reference size for the short leg

# --- Module E2: Gamma Scalping (delta-hedged straddle) ---
STRADDLE_BUDGET_PCT = 0.02          # 2% of position, as initial premium budget

# --- Shared option pricing ---
RISK_FREE_RATE = 0.015
JOURS_PAR_AN = 252                  # Trading days per year, used to annualize vol

