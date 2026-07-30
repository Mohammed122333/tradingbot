import sys
import os
import pandas as pd
import MetaTrader5 as mt5

# Set up path to Ver2
sys.path.insert(0, r"C:\Users\adeL\Desktop\GoodBot\Ver2")

import config as cfg
from backtester import combined_backtest

# Load newtest.json to get exact params
import json
with open(r"C:\Users\adeL\Desktop\GoodBot\Ver2\newtest.json", "r") as f:
    js = json.load(f)

kwargs = {
    'symbols': ['XAUUSDm'],
    'date_from': "2020-01-02 00:00:00",
    'date_to': "2023-01-01 23:59:59",
    'initial_balance': 10000.0,
    'risk_percent': 0.25,
    'fixed_lot': 0.02,
    'risk_mode': "Risk",
    'trailing_methods': ["FVG Return"],
    'ict_params': {
        "threshold_factor": 1.4,
        "reversal_threshold": 3.0,
        "lookahead": 8,
        "fib_low": 0.618,
        "fib_high": 0.786
    },
    'ict_method': ["FVG Return"],
    'min_rrr': 1.0,
    'use_dynamic_rrr': False,
    'trade_on_all_tfs': True,
    'use_ultra_low_tf': False,
    'fvg_sl_mode': 0,
    'spread_cost': 0.0,
    'slippage_points': 0,
    'commission_per_lot': 0.0,
    'session_filter': False,
    'session_start': 13,
    'session_end': 21,
    'use_htf_filter': False,
    'use_ote_filter': False,
    'bypass_htf_conf': True,
    'trail_type': "None (Disabled)",
    'trail_params': {"trail_pct": 0.5},
    'require_bos_fvg': False,
    'enable_slippage_recovery': True,
    'anti_gap_enabled': True,
    'anti_gap_mult': 2.0,
    'fvg_sl_spread_buffer': 1.0,
    'limit_touch_fill': False,
    'fvg_displacement_only': False,
    'fvg_discount_premium_only': False,
    'fvg_recent_sweep_only': False,
    'sb_require_htf_bias': False,
    'use_symbol_profiles': False,
    'clear_cache': True,
    'ml_filter': False,
    'ml_min_confidence': 0.60,
    'use_smt_divergence': False,
    'smt_correlated_pair': "AUTO",
    'use_volume_profile': False,
    'max_concurrent_trades': 0,
    'daily_loss_limit': 1.0,
    'min_confluence_score': 2,
    'min_fvg_size': 0.0,
}

# Override config to simulate opt_worker
cfg.MIN_FVG_SIZE_SPREADS = 0.0
cfg.MIN_CONFLUENCE_SCORE = 2

# We need to manually initialize MT5 to simulate the data loading properly because we are running outside the worker cache
mt5.initialize()

print("Running backtest...")
trades, metrics = combined_backtest(**kwargs)
print(f"Trades found: {len(trades)}")

if len(trades) == 0:
    print("Zero trades found! Let's check get_data_by_date.")
    from simulation import get_data_by_date
    df = get_data_by_date('XAUUSDm', mt5.TIMEFRAME_M15, pd.Timestamp("2020-01-02"), pd.Timestamp("2023-01-01"))
    print(f"Dataframe length for M15: {len(df) if df is not None else 0}")
