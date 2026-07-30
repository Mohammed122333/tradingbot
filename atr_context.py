"""Single ATR implementation shared by live + backtest. Parity requires that
both engines get the SAME number from the SAME bars.
"""
import numpy as np
from utils import _wilders_smooth_full_njit


def true_range(highs, lows, closes):
    n = len(highs)
    tr = np.zeros(n)
    if n == 0:
        return tr
    tr[0] = highs[0] - lows[0]
    if n > 1:
        tr[1:] = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]),
                       np.abs(lows[1:] - closes[:-1]))
        )
    return tr


def atr_series(df, period=14):
    """Full Wilder ATR array aligned to df. Never returns zeros."""
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)
    tr = true_range(highs, lows, closes)
    if len(tr) == 0:
        return np.zeros(0)
    seed = float(np.mean(tr[:period])) if len(tr) >= period else float(np.mean(tr))
    if seed <= 0:
        seed = float(np.mean(highs - lows)) or 1e-9
    a = _wilders_smooth_full_njit(tr, period, seed)
    # Guard: a flat/illiquid warm-up window must not produce a 0 divisor.
    floor = max(seed * 0.01, 1e-9)
    return np.maximum(a, floor)


def atr_last(df, period=14):
    a = atr_series(df, period)
    return float(a[-1]) if len(a) else 0.0
