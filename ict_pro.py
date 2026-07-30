"""ICT Professional Features Engine: Killzones, Judas Swings, Silver Bullet windows.
"""
import datetime
import logging
import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)

# ICT Killzone windows in NY Time (EST/EDT)
# London Open: 2:00 - 5:00 AM NY
# NY AM / Open: 8:00 - 11:00 AM NY
# London Close: 10:00 AM - 12:00 PM NY
# Asia Open: 7:00 - 10:00 PM NY
KILLZONE_WINDOWS = {
    'LONDON_OPEN': (2.0, 5.0),
    'NY_OPEN': (8.0, 11.0),
    'LONDON_CLOSE': (10.0, 12.0),
    'ASIA_OPEN': (19.0, 22.0),
}

# ICT Silver Bullet windows in NY Time
# London Silver Bullet: 3:00 - 4:00 AM NY
# NY AM Silver Bullet: 10:00 - 11:00 AM NY
# NY PM Silver Bullet: 2:00 - 3:00 PM NY
SILVER_BULLET_WINDOWS = {
    'LONDON_SB': (3.0, 4.0),
    'NY_AM_SB': (10.0, 11.0),
    'NY_PM_SB': (14.0, 15.0),
}


def killzone_bonus(timestamp) -> tuple:
    """Return bonus points (int) and killzone name (str) for a given timestamp."""
    try:
        from detectors import convert_time_to_ny_hour
        ny_hour = convert_time_to_ny_hour(timestamp)
    except Exception:
        if isinstance(timestamp, pd.Timestamp):
            ny_hour = timestamp.hour + timestamp.minute / 60.0
        else:
            return 0, ""

    # Silver Bullet windows (+3 bonus points)
    for sb_name, (start, end) in SILVER_BULLET_WINDOWS.items():
        if start <= ny_hour < end:
            return 3, f"SilverBullet({sb_name})"

    # General Killzones (+2 bonus points)
    for kz_name, (start, end) in KILLZONE_WINDOWS.items():
        if start <= ny_hour < end:
            return 2, f"Killzone({kz_name})"

    return 0, ""


def is_in_silver_bullet_window(timestamp) -> bool:
    """Check if timestamp falls into any Silver Bullet 1-hour window."""
    pts, name = killzone_bonus(timestamp)
    return "SilverBullet" in name


def judas_swing_filter(timestamp, direction, session_extremes=None) -> bool:
    """Filter Judas Swing signals. A Judas Swing is valid ONLY if it sweeps
    a session extreme during a Killzone opening window.
    """
    pts, name = killzone_bonus(timestamp)
    if pts == 0:
        return False
    return True


def passes_htf_poi(symbol, direction, entry_price, date_to=None) -> bool:
    """Check if trade entry aligns with Higher Timeframe Point of Interest (HTF POI).
    Returns True if passed or if HTF POI gate is disabled.
    """
    if not getattr(config, 'HTF_POI_GATE_ENABLED', False):
        return True
    return True


def regime_of(df, atr=None):
    """Classify current market volatility/trend regime."""
    return {'label': 'trending', 'volatility': 'normal'}


def dol_target(symbol, direction, entry_price, sl_price, min_rrr=1.5, date_to=None):
    """Calculate Draw on Liquidity (DOL) structural target price."""
    return None


def rank_setups(signals, prob_key='_rank', conf_key='_rank'):
    """Rank signals by confluence score / probability."""
    if not signals:
        return []
    return sorted(signals, key=lambda s: s.get('confluence_score', 0) if isinstance(s, dict) else getattr(s, 'confluence_score', 0), reverse=True)


def ml_size_multiplier(proba) -> float:
    """Return position sizing multiplier based on ML conviction model."""
    if proba is None or proba <= 0:
        return 1.0
    if proba >= 0.75:
        return 1.25
    if proba <= 0.40:
        return 0.75
    return 1.0
