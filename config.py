"""
Application configuration — constants + state-managed mutable settings.

Mutable values (risk params, feature toggles, live-trading state) are
stored in a thread-safe StateManager (state.app_state).  Backward-compat
module-level attribute access is provided via __getattr__ / __setattr__
so that existing  config.X = Y  and  config.X  code continues to work.
"""
import pandas as pd
import numpy as np
import os
import json
import logging
import threading
from queue import Queue
from typing import Any

from broker import TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, \
    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1

# ML imports (local only - no API/internet)
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.inspection import permutation_importance
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Bot Version
BOT_VERSION = "31.0.0"

OFFLINE_BACKTESTING = True  # If True, bypass MT5 history sync for backtests/grid opt

# Realistic default spread (in POINTS) used by cost_model when no real spread
# is available (offline backtest with no 'spread' column). Keys are matched as
# case-insensitive substrings of the symbol; 'DEFAULT' is the fallback.
# NOTE: cost_model NEVER lets an AUTO spread silently collapse to 0.0.
DEFAULT_SPREAD_POINTS = {
    'XAU': 20.0, 'XAG': 25.0, 'BTC': 3000.0, 'ETH': 300.0,
    'JPY': 15.0, 'DEFAULT': 12.0,
}

# ===========================================================================
# NORMALIZATION CONSTANTS
# RULE: no threshold in the strategy layer may be expressed in price units,
# points, or spreads. Everything here is (a) a fraction of ATR, (b) a fraction
# of the dealing range, or (c) dimensionless. `point` and spread belong ONLY
# to the cost/execution layer. This is what makes the bot instrument-agnostic.
# ===========================================================================
# ── Order Block / displacement (ATR-relative) ──────────────────────────────
OB_RANGE_ATR_MIN           = 0.35   # min OB-candle range as a fraction of ATR
OB_DISPLACEMENT_ATR_MULT   = 1.50   # displacement leg must clear OB by this x ATR
OB_REQUIRE_FVG             = True   # displacement must leave an FVG behind
OB_LOOKAHEAD               = 8      # bars allowed for displacement to confirm

# ── Displacement quality gate (replaces atr_mult >= 0.4, which was a no-op) ─
DISPLACEMENT_MIN_BODY_RATIO = 0.55
DISPLACEMENT_MIN_ATR_MULT   = 1.50

# ── FVG sizing ────────────────────────────────────────────────────────────
FVG_MIN_ATR_FRACTION       = 0.20   # feeds utils.fvg_size_ok(atr_fraction=...)

# ── Equal highs / lows tolerance (was threshold_points=2.0 in PRICE units!) ─
EQ_LEVEL_ATR_FRAC          = 0.15
EQ_POOL_MAX_LOOKBACK_SWINGS = 10

# ── Stops / buffers ────────────────────────────────────────────────────────
SL_BUFFER_ATR_FRAC         = 0.10   # replaces buffer = spread * 2
MIN_STOP_ATR_FRAC          = 0.05   # floor when broker trade_stops_level unknown
STRUCTURE_TP_ATR_FRAC      = 0.15   # replaces swing_high + spread * 5
TP_LIQUIDITY_EXTENSION     = 0.0    # 0 = exit BEFORE the pool (matches dol_target)

# ── Swings / sweeps: ONE definition, used everywhere ───────────────────────
SWING_LOOKBACK             = 5
SWEEP_LOOKBACK             = 5      # was 15 live vs 5 backtest -> parity break
SWING_STRICT_TIES          = True   # strict-left/loose-right fractal tie-breaking

# ── Confluence weights (were hardcoded via getattr fallbacks) ──────────────
CONFLUENCE_WEIGHTS = {
    'fvg_ob': 3, 'structure': 2, 'mss': 2, 'sweep': 2, 'ote': 1,
    'poc': 1, 'htf': 1, 'displacement': 1, 'multi_tf_aligned': 3,
    'eq_target': 2,
}

# Fractal tier -> points. You already COMPUTE these tiers and threw them away.
SWEEP_TIER_POINTS     = {1: 1, 2: 2, 3: 4}   # ST / IT / LT
STRUCTURE_TIER_POINTS = {1: 1, 2: 2, 3: 3}

# ── Killzones (NY local hours, decimal) ────────────────────────────────────
KILLZONE_ENABLED = True
KILLZONE_WINDOWS = [
    ("London",         2.0,  5.0, 2),
    ("NY AM",          8.5, 11.0, 2),
    ("Silver Bullet", 10.0, 11.0, 1),
    ("NY PM",         13.5, 16.0, 1),
]

# Macro windows (:50 -> :10 of the hour). Cheap, high-value ICT time filter.
MACRO_ENABLED = False
MACRO_BONUS   = 1

# ── Draw on Liquidity ──────────────────────────────────────────────────────
DOL_TP_ENABLED      = True
DOL_TP_BUFFER_FRAC  = 0.05
# Higher weight = more significant draw. dol_target picks the MOST significant
# pool that clears min_rr, not the nearest (which collapsed it to fixed-R).
DOL_POOL_WEIGHTS = {'PWH': 5, 'PWL': 5, 'PDH': 4, 'PDL': 4,
                    'H4_SWING': 3, 'H1_SWING': 2}

# ── HTF POI nesting ────────────────────────────────────────────────────────
HTF_POI_ENABLED      = True
HTF_POI_STRICT       = False
HTF_POI_TOL_ATR_FRAC = 0.10   # replaces HTF_POI_TOL_FRAC = 0.0005 of price

# ── OTE ────────────────────────────────────────────────────────────────────
OTE_FIB_LOW            = 0.618
OTE_FIB_HIGH           = 0.786
OTE_ANCHOR_TO_MSS_LEG  = True   # False = legacy 50-bar window (don't)

# ── Regime (regime.py is now the ONLY classifier) ──────────────────────────
REGIME_FILTER_ENABLED          = False
REGIME_LOOKBACK                = 50
REGIME_SWING_LOOKBACK          = 3
REGIME_DISPLACEMENT_MIN        = 0.35
REGIME_INSIDE_BAR_MAX          = 0.35
REGIME_RANGE_CONTRACTION_MAX   = 0.60
REGIME_RANGE_EXPANSION_MULT    = 2.00
REGIME_HIGH_VOL_SIZE_MULT      = 0.50
REGIME_CHOP_BLOCK              = True
REGIME_CHOP_CONF_BUMP          = 1

# ── ML sizing / ranking ────────────────────────────────────────────────────
ML_SIZE_MIN_MULT = 0.5
ML_SIZE_MAX_MULT = 1.5
ML_SIZE_BASELINE = 0.5
ML_RANK_ENABLED  = True

# ── Execution humanization (LIVE ONLY — never affects signals) ─────────────
HUMANIZE_EXECUTION        = False
HUMANIZE_ENTRY_DELAY_MIN  = 0.4
HUMANIZE_ENTRY_DELAY_MAX  = 2.5
HUMANIZE_SCAN_JITTER_FRAC = 0.5
HUMANIZE_COMMENT_ROTATE   = True

# ── Live loop ──────────────────────────────────────────────────────────────
LIVE_SCAN_INTERVAL_SEC     = 5.0
LIVE_MAX_HOLD_HOURS        = 72.0   # must equal SIM_MAX_HOLD_HOURS below
SIM_MAX_HOLD_HOURS         = 72.0   # was 3 days in sim vs 72h live -> parity break
PENDING_LIMIT_EXPIRY_HOURS = 2.0    # ONE value; sim no longer hardcodes 3600/7200

# ── Execution realism ──────────────────────────────────────────────────────
# 1.0 = maximum backtest/live realism. >1 makes the backtest deliberately
# pessimistic AND breaks parity, because spread feeds signal thresholds.
ESTIMATED_SPREAD_MULTIPLIER = 1.0
SLIPPAGE_MAX_POINTS         = 5
SIM_APPLY_SLIPPAGE_ON_STOP  = True   # market exits slip; limit TPs don't
SIM_APPLY_SLIPPAGE_ON_ENTRY = False  # limit entries fill at price or better

# ── Risk (institutional) ───────────────────────────────────────────────────
USE_EQUITY_DRAWDOWN             = True   # count FLOATING loss, not just realized
MAX_TOTAL_DRAWDOWN_PCT          = 10.0   # from high-water mark; 0 = off
MAX_CONSEC_LOSSES               = 6      # 0 = off
MAX_EXPOSURE_PER_SYMBOL         = 2
CORRELATION_GROUPS = {
    'USD_MAJORS': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'],
    'CRYPTO':     ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA'],
    'US_INDICES': ['US30', 'US500', 'USTEC', 'NAS', 'SPX', 'DJI'],
    'METALS':     ['XAU', 'XAG', 'GOLD', 'SILVER'],
}
MAX_TRADES_PER_CORRELATION_GROUP = 2
BROKER_REJECT_HALT_COUNT         = 5

# ── SMT ────────────────────────────────────────────────────────────────────
SMT_PARTNERS = {
    'XAU':    ['XAG', 'DXY'],   # gold's partner is silver/DXY, NOT EURUSD
    'XAG':    ['XAU', 'DXY'],
    'EURUSD': ['GBPUSD', 'DXY'],
    'GBPUSD': ['EURUSD', 'DXY'],
    'AUDUSD': ['NZDUSD', 'DXY'],
    'NZDUSD': ['AUDUSD', 'DXY'],
    'USDJPY': ['USDCHF', 'DXY'],
    'BTC':    ['ETH'],
    'ETH':    ['BTC'],
    'US30':   ['US500', 'USTEC'],
    'USTEC':  ['US500', 'US30'],
}
SMT_WARN_ON_MISSING_FEED = True   # loudly, instead of silently disabling

# ── Parity enforcement ─────────────────────────────────────────────────────
PARITY_STRICT = True   # raise instead of degrading when engines could diverge

# -------------------------
# Logger Setup (Centralized Log Queue Handler)
# -------------------------
class QueueHandler(logging.Handler):
    """Deliver log messages to a queue so the Tkinter UI can poll them asynchronously.
    Prevents background trading scanner thread from blocking on UI updates."""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

# Shared logging queue and events
log_queue = Queue()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# Console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Queue Handler (UI listener)
queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(formatter)
logger.addHandler(queue_handler)

# Persistent Live Log File (bot_live.log)
try:
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler("bot_live.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Could not initialize bot_live.log file handler: {e}")

# -------------------------
# Threading Events (immutable references)
# -------------------------
stop_event = threading.Event()
bt_stop_event = threading.Event()

# Trade memory file path (constant)
TRADE_MEMORY_FILE = "trade_memory.json"

# -------------------------
# Broker-agnostic settings
# -------------------------
BROKER_TIMEZONE = 'UTC'

# -------------------------
# Timeframes (constants)
# -------------------------
HIGHER_TFS = [("H4", TIMEFRAME_H4), ("D1", TIMEFRAME_D1)]
LOWER_TFS = [("M15", TIMEFRAME_M15), ("H1", TIMEFRAME_H1)]
ULTRA_LOW_TFS = [("M1", TIMEFRAME_M1), ("M5", TIMEFRAME_M5)]
DEFAULT_SYMBOL = "XAUUSDm"

# Account Swap Settings (Default True for Swap-Free / Islamic accounts)
SWAP_FREE_ACCOUNT = True
ENABLE_SWAP_COST = False

# -------------------------
# Strategy Method Definitions (constants)
# -------------------------
CORE_METHODS = ["FVG Return"]
EXPERIMENTAL_METHODS = ["Silver Bullet", "MMXM", "ICT Model 2022", "ICT Model 2025", "Judas Swing", "Breaker Block", "IOFR", "ICT Retest"]
ICT_METHODS = CORE_METHODS + EXPERIMENTAL_METHODS
LEGACY_OB_RETEST_METHODS = {"ICT Retest", "OB Retest", "Order Block Retest"}

# Legacy Stop Constants
TRAIL_TYPE_PARTIAL = "True Partial (50% at 1R)"
TRAIL_TYPE_ATR = "Profit Lock (50%)"
TRAIL_TYPE_PERCENT = "Percentage Trail"
TRAIL_TYPES = [TRAIL_TYPE_PARTIAL, TRAIL_TYPE_ATR, TRAIL_TYPE_PERCENT]
DEFAULT_TRAIL_PERCENT = 1.0

# FVG Stop Loss Mode Constants
FVG_SL_NORMAL = "Normal"
FVG_SL_CANDLE = "Candle Extreme"
FVG_SL_BOS = "Last BoS"
FVG_SL_MSS = "Last MSS"
FVG_SL_SWEEP = "Last Sweep"
FVG_SL_MODES = [FVG_SL_NORMAL, FVG_SL_CANDLE, FVG_SL_BOS, FVG_SL_MSS, FVG_SL_SWEEP]

# LLM sentiment (constant gate)
USE_LLM_NEWS_SENTIMENT = False

# ---------------------------------------------------------------------------
# MUTABLE STATE — synced with StateManager for thread safety
# ---------------------------------------------------------------------------
# Strategy: Eagerly load values as plain module attributes so that hot-path
# reads (backtest loops, signal scans) are native Python variable access.
# Writes proxy through the StateManager and sync the local copy.
from state import app_state

# (config_name, state_attr_name, cast_type_or_None)
_STATE_SYNC: list[tuple[str, str, type | None]] = [
    ('DAILY_LOSS_LIMIT_PCT',      'daily_loss_limit_pct', float),
    ('MAX_CONCURRENT_TRADES',     'max_concurrent_trades', int),
    ('MIN_FVG_SIZE_SPREADS',      'min_fvg_size_spreads', float),
    ('MIN_CONFLUENCE_SCORE',      'min_confluence_score', int),
    ('PENDING_LIMIT_EXPIRY_HOURS', 'pending_limit_expiry_hours', float),
    ('ESTIMATED_SPREAD_MULTIPLIER', 'estimated_spread_multiplier', float),
    ('USE_EQUITY_DRAWDOWN',        'use_equity_drawdown', bool),
    ('MAX_TOTAL_DRAWDOWN_PCT',     'max_total_drawdown_pct', float),
    ('MAX_CONSEC_LOSSES',          'max_consec_losses', int),
    ('SLIPPAGE_MAX_POINTS',       'slippage_max_points', int),
    ('COMMISSION_PER_LOT',        'commission_per_lot', float),
    ('ANTI_GAP_SL_ENABLED',       'anti_gap_sl_enabled', bool),
    ('ANTI_GAP_ATR_MULTIPLIER',   'anti_gap_atr_multiplier', float),
    ('NEWS_EMBARGO_ENABLED',      'news_embargo_enabled', bool),
    ('NEWS_FILTER_ENABLED',       'news_filter_enabled', bool),
    ('NEWS_FILTER_BUFFER_MINS',   'news_filter_buffer_mins', int),
    ('USE_SMT_DIVERGENCE',        'use_smt_divergence', bool),
    ('SMT_CORRELATED_PAIR',       'smt_correlated_pair', str),
    ('USE_VOLUME_PROFILE_CONFLUENCE', 'use_volume_profile_confluence', bool),
    ('DOL_TP_ENABLED',            'dol_tp_enabled', bool),
    ('KILLZONE_ENABLED',          'killzone_enabled', bool),
    ('HTF_POI_ENABLED',           'htf_poi_enabled', bool),
    ('CONFLUENCE_REQUIRE_MANDATORY', 'confluence_require_mandatory', bool),
    ('REGIME_FILTER_ENABLED',     'regime_filter_enabled', bool),
    ('REGIME_ADAPTIVE_PARAMS_ENABLED', 'regime_adaptive_params_enabled', bool),
    ('MULTI_TF_CONFLUENCE_ENABLED', 'multi_tf_confluence_enabled', bool),
    ('MULTI_TF_GATE_ENABLED',     'multi_tf_gate_enabled', bool),
    ('ML_SIZING_ENABLED',         'ml_sizing_enabled', bool),
    ('ML_RANK_ENABLED',           'ml_rank_enabled', bool),


    ('live_trading_running',      'live_trading_running', bool),
    ('current_activity',          'current_activity', str),
    ('slippage_recovery_tracker', 'slippage_recovery_tracker', dict),
    ('executed_ob',               'executed_ob', dict),
    ('pending_limit_orders',      'pending_limit_orders', dict),
    ('priority_scan_setups',      'priority_scan_setups', dict),
]

# Eager-load into module namespace so __getattr__ is never called for these
for _cn, _sn, _ct in _STATE_SYNC:
    _val = getattr(app_state, _sn)
    vars()[_cn] = _ct(_val) if _ct else _val

# Build lookup for write path
_STATE_WRITE: dict[str, tuple[str, type | None]] = {
    _cn: (_sn, _ct) for _cn, _sn, _ct in _STATE_SYNC
}


def __getattr__(name: str) -> Any:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


import sys as _sys


class _ConfigModule(_sys.modules[__name__].__class__):
    def __setattr__(self, name: str, value: Any) -> None:
        entry = _STATE_WRITE.get(name)
        if entry is not None:
            state_attr, cast = entry
            setattr(app_state, state_attr, cast(value) if cast else value)
            super().__setattr__(name, cast(value) if cast else value)
            return
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _ConfigModule
