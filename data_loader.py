"""
Chargement des donnees de marche (WTI Futures, Heating Oil) via yfinance.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from config import TICKERS, DATA_START, DATA_END


def load_market_data():
    """
    Telecharge les prix de cloture WTI (CL=F) et Heating Oil (HO=F).

    Returns
    -------
    raw : DataFrame
        Prix bruts, colonnes ['WTI', 'HeatingOil'], index = dates.
    returns : DataFrame
        Rendements journaliers logarithmiques (log-returns), memes colonnes.
        Les log-returns sont preferes aux rendements simples pour les calculs
        de volatilite : ils sont additifs dans le temps et symetriques.
    """
    print("Telechargement des donnees...")
    raw = yf.download(TICKERS, start=DATA_START, end=DATA_END)["Close"]
    raw.columns = ["WTI", "HeatingOil"]
    raw = raw.dropna()

    returns = np.log(raw / raw.shift(1)).dropna()

    print(f"Donnees recuperees : {len(raw)} jours")
    print(f"Periode : {raw.index[0].date()} -> {raw.index[-1].date()}")

    return raw, returns
