"""
Hierarchical multi-timeframe confluence engine.

D1 bias gates H4, H4 gates H1, H1 gates M15.  Each signal is scored
by how many aligned layers support it.  If too few layers agree the
entry is blocked — mirroring how a pro trader reads context across
timeframes before pulling the trigger.

Usage:
    from multi_tf import hierarchical_confluence

    result = hierarchical_confluence("XAUUSDm", "bullish", date_to=some_date)
    if result['gated']:
        # Too few layers agree — skip or halve size
        pass
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import HIGHER_TFS, LOWER_TFS
from utils import get_data, get_data_by_date

logger = logging.getLogger(__name__)

_htf_bias_cache = {}
import time


def _tf_bias(df: pd.DataFrame) -> str:
    """Return 'bullish', 'bearish', or 'neutral' bias for a single TF.

    Uses swing-structure-like logic:  last 3 swing highs vs swing lows.
    Falls back to a simple 50%-range test on small data.
    """
    if df is None or len(df) < 10:
        return 'neutral'

    closes = df['close'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)

    # 1) Swing-structure check on last ~20 bars
    lookback = min(20, len(df))
    sh_idx, sl_idx = [], []
    for i in range(2, lookback - 2):
        if all(highs[-i] >= highs[-i + j] for j in (-2, -1, 1, 2)):
            sh_idx.append(-i)
        if all(lows[-i] <= lows[-i + j] for j in (-2, -1, 1, 2)):
            sl_idx.append(-i)

    if len(sh_idx) >= 2 and len(sl_idx) >= 2:
        last_sh = closes[sh_idx[-1]]
        prev_sh = closes[sh_idx[-2]]
        last_sl = closes[sl_idx[-1]]
        prev_sl = closes[sl_idx[-2]]
        hh = last_sh > prev_sh
        ll = last_sl < prev_sl
        hl = last_sl > prev_sl
        lh = last_sh < prev_sh
        if hh and hl:
            return 'bullish'
        if lh and ll:
            return 'bearish'

    # 2) 50%-range fallback
    recent = df.tail(10)
    avg_range = (recent['high'] - recent['low']).mean()
    if avg_range == 0:
        return 'neutral'
    mid = (recent['high'] + recent['low']).mean() / 2
    now = closes[-1]
    dist = (now - mid) / avg_range
    if dist > 0.15:
        return 'bullish'
    if dist < -0.15:
        return 'bearish'
    return 'neutral'


def hierarchical_confluence(
    symbol: str,
    direction: str,
    date_to=None,
    min_layers: int = 2,
    use_lower: bool = True,
    use_upper: bool = True,
) -> dict:
    """Score how many timeframe layers agree with *direction*.

    Parameters
    ----------
    symbol : str
    direction : 'bullish' | 'bearish'
    date_to : datetime, optional
        Backtest-specific end date (None = live).
    min_layers : int
        Minimum aligned layers required to pass the gate.
        Recommended: 2 (H1 + M15) for scalping; 3 (D1 + H4 + H1) for swing.
    use_lower : bool
        Include M15 / H1 layers.
    use_upper : bool
        Include H4 / D1 layers.

    Returns
    -------
    dict with keys:
        score     — int (0-4, how many layers agree)
        layers    — list of (tf_name, bias)
        gated     — bool (True = too few layers agree)
        gate_msg  — str
    """
    # Use an hourly bucket cache to prevent DB spam per minute
    live = date_to is None
    if live:
        bucket = int(time.time() // 3600)
    else:
        try:
            bucket = int(pd.to_datetime(date_to).timestamp() // 3600)
        except Exception:
            bucket = str(date_to)
            
    ck = (symbol, direction, min_layers, use_lower, use_upper, bucket)
    if ck in _htf_bias_cache:
        return _htf_bias_cache[ck]

    # Determine which TFs to check (in order: D1 → H4 → H1 → M15)
    tf_list = []
    if use_upper:
        tf_list.extend([("D1", 16408), ("H4", 16388)])
    if use_lower:
        tf_list.extend([("H1", 16385), ("M15", 15)])

    # Map MT5 timeframe constants to minutes for point-in-time filtering
    _mt5_to_min = {15: 15, 16385: 60, 16388: 240, 16408: 1440}

    layers = []
    agreeing = 0

    for tf_name, tf_val in tf_list:
        try:
            bars_needed = 30 if tf_val >= 16385 else 60
            if date_to:
                df = get_data(symbol, tf_val, bars_needed, live=False, date_to=date_to)
                # Strip bars whose close time > date_to (lookahead protection)
                if df is not None and not df.empty:
                    tf_min = _mt5_to_min.get(tf_val, 60)
                    while not df.empty:
                        bar_close = df.index[-1] + pd.Timedelta(minutes=tf_min)
                        if bar_close > date_to:
                            df = df.iloc[:-1]
                        else:
                            break
            else:
                df = get_data(symbol, tf_val, bars_needed, live=True)
                if df is not None and not df.empty:
                    df = df.iloc[:-1]  # drop forming bar in live

            if df is None or len(df) < 5:
                layers.append((tf_name, 'neutral'))
                continue

            bias = _tf_bias(df)
            layers.append((tf_name, bias))
            if bias == direction:
                agreeing += 1
        except Exception as e:
            logger.debug("multi_tf: error getting bias for %s %s: %s", symbol, tf_name, e)
            layers.append((tf_name, 'error'))

    score = agreeing
    gated = score < min_layers
    gate_msg = ""
    if gated:
        gate_msg = (f"Multi-TF gate: only {score}/{len(layers)} layers agree (need ≥{min_layers}): "
                    f"{[t for t, b in layers if b != 'error']}")
        logger.debug("multi_tf: %s", gate_msg)

    result = {
        'score': agreeing,
        'layers': layers,
        'gated': gated,
        'gate_msg': f"MTF: {agreeing}/{min_layers} agreed {direction} ({[t for t,b in layers if b==direction]})" if gated else "MTF: Pass"
    }
    
    _htf_bias_cache[ck] = result
    return result
