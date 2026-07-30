import sys
import types
import os
import json
import pandas as pd
import datetime as dt

# Handle environments without native MetaTrader5 package (e.g. Linux / Kaggle / Colab)
try:
    import MetaTrader5 as mt5
except ImportError:
    mock_mt5 = types.ModuleType("MetaTrader5")
    mock_mt5.TIMEFRAME_M1 = 1
    mock_mt5.TIMEFRAME_M5 = 5
    mock_mt5.TIMEFRAME_M15 = 15
    mock_mt5.TIMEFRAME_H1 = 16385
    mock_mt5.TIMEFRAME_H4 = 16388
    mock_mt5.TIMEFRAME_D1 = 16408
    mock_mt5.TIMEFRAME_W1 = 32769
    mock_mt5.TIMEFRAME_MN1 = 49153
    mock_mt5.initialize = lambda *a, **k: True
    mock_mt5.shutdown = lambda *a, **k: True
    mock_mt5.last_error = lambda: (0, 'Success')
    mock_mt5.symbol_info = lambda *a, **k: None
    mock_mt5.symbol_info_tick = lambda *a, **k: None
    sys.modules["MetaTrader5"] = mock_mt5
    import MetaTrader5 as mt5

import config
config.OFFLINE_BACKTESTING = True

from backtester import combined_backtest
import utils

def run_gold_backtest():
    # Load settings from KAggle.json / OTPnoSB.json if available
    cfg_file = 'KAggle.json' if os.path.exists('KAggle.json') else 'OTPnoSB.json'
    json_cfg = {}
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, 'r', encoding='utf-8') as f:
                json_cfg = json.load(f)
            print(f"Loaded configuration from {cfg_file}")
        except Exception as e:
            print(f"Warning: Could not parse {cfg_file}: {e}")

    symbol = json_cfg.get("opt_symbol_entry", "XAUUSDm")
    date_from = json_cfg.get("opt_date_from", "2024-01-01 00:00:00")
    date_to = json_cfg.get("opt_date_to", "2025-01-01 23:59:59")
    initial_balance = float(json_cfg.get("opt_balance_entry", 10000.0))
    risk_pct = float(json_cfg.get("opt_risk_entry", 0.25))

    methods = ["FVG Return", "Silver Bullet", "MMXM", "ICT Model 2022", "ICT Model 2025", "Judas Swing", "Breaker Block", "IOFR", "ICT Retest"]

    print("==========================================================")
    print(f"        Gold (XAUUSDm) Backtest Runner")
    print(f"        Symbol:          {symbol}")
    print(f"        Date Window:     {date_from} to {date_to}")
    print(f"        Initial Balance: ${initial_balance:,.2f}")
    print(f"        Risk per Trade:  {risk_pct}%")
    print("==========================================================")

    kwargs = {
        'symbols': [symbol],
        'date_from': date_from,
        'date_to': date_to,
        'initial_balance': initial_balance,
        'risk_percent': risk_pct,
        'fixed_lot': 0.02,
        'risk_mode': "percent",
        'trailing_methods': methods,
        'mock_si': {
            'point': 0.01,
            'trade_tick_size': 0.01,
            'digits': 2,
            'spread': 20,
            'trade_stops_level': 100,
        },
        'ict_params': {
            "threshold_factor": 1.4,
            "reversal_threshold": 3.0,
            "lookahead": 8,
            "fib_low": 0.618,
            "fib_high": 0.786
        },
        'ict_method': methods,
        'min_rrr': float(json_cfg.get("opt_rrr_min", 1.5)),
        'use_dynamic_rrr': json_cfg.get("opt_use_dynamic_rrr", False),
        'trade_on_all_tfs': True,
        'use_ultra_low_tf': False,
        'fvg_sl_mode': "Normal",
        'spread_cost': float(json_cfg.get("opt_spread_cost", 0.0)),
        'slippage_points': int(json_cfg.get("opt_slippage", 0)),
        'commission_per_lot': float(json_cfg.get("opt_commission", 0.0)),
        'session_filter': json_cfg.get("opt_session_filter", False),
        'session_start': int(json_cfg.get("opt_session_start", 13)),
        'session_end': int(json_cfg.get("opt_session_end", 21)),
        'use_htf_filter': False,
        'use_ote_filter': False,
        'bypass_htf_conf': True,
        'trail_type': "True Partial (50% at 1R)",
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
        'ml_filter': json_cfg.get("opt_ml_filter", False),
        'ml_min_confidence': float(json_cfg.get("opt_ml_min_confidence", 60.0)),
        'use_smt_divergence': False,
        'smt_correlated_pair': "AUTO",
        'use_volume_profile': False,
        'max_concurrent_trades': int(json_cfg.get("opt_concurr_min", 2)),
        'daily_loss_limit': float(json_cfg.get("opt_daily_loss", 1.0)),
        'min_confluence_score': int(json_cfg.get("opt_min_conf", 2)),
        'min_fvg_size': float(json_cfg.get("opt_min_fvg", 0.0)),
    }

    print("\nExecuting Gold backtest simulation...")
    trades, metrics = combined_backtest(**kwargs)

    print("\n==========================================================")
    print("                 BACKTEST PERFORMANCE RESULTS              ")
    print("==========================================================")
    print(f" Total Executed Trades: {len(trades)}")
    print(f" Final Balance:         ${metrics.get('final_balance', initial_balance):,.2f}")
    print(f" Net Profit / Loss:     ${metrics.get('net_profit', 0.0):,.2f}")
    print(f" Total Return (ROI):    {metrics.get('roi_pct', 0.0):.2f}%")
    print(f" Max Drawdown (MtM):    {metrics.get('max_drawdown_pct', 0.0):.2f}% (${metrics.get('max_drawdown', 0.0):,.2f})")
    print(f" Calmar Ratio:          {metrics.get('calmar', 0.0):.2f}")
    print(f" Win Rate (SL/TP Exits):{metrics.get('sltp_win_rate', 0.0):.2f}%")
    print(f" Overall Win Rate:      {metrics.get('overall_win_rate', 0.0):.2f}%")
    print(f" Profit Factor:         {metrics.get('profit_factor', 0.0):.2f}")
    print("==========================================================")

    return trades, metrics

if __name__ == '__main__':
    run_gold_backtest()
