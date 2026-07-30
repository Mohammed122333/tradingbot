import os
import sys
import json
import random
import multiprocessing as mp
import itertools
from datetime import datetime, timezone
import pandas as pd
import MetaTrader5 as mt5
import datetime as dt
import pickle

import config
import utils
import backtester
from opt_worker import init_worker, run_combo

def main():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return
    
    symbol = "XAUUSDm"
    
    # BUGFIX: define the backtest window BEFORE it is used below.
    date_from = pd.Timestamp("2024-01-01", tz=config.BROKER_TIMEZONE).to_pydatetime().replace(tzinfo=None)
    date_to = pd.Timestamp("2024-12-31", tz=config.BROKER_TIMEZONE).to_pydatetime().replace(tzinfo=None)
    
    oos_split = 0.25
    full_delta = date_to - date_from
    train_date_to = date_from + full_delta * (1 - oos_split)
    oos_date_to = date_to
    
    print(f"In-Sample (Train): {date_from} to {train_date_to}")
    print(f"Out-Of-Sample (Test): {train_date_to} to {oos_date_to}")
    
    methods = ["ICT Retest", "Silver Bullet", "FVG Return", "MMXM", "ICT Model 2022", "ICT Model 2025", "Judas Swing", "Breaker Block", "IOFR"]
    
    sl_modes = [config.FVG_SL_NORMAL, config.FVG_SL_SWEEP, config.FVG_SL_CANDLE]
    trail_types = [config.TRAIL_TYPE_PARTIAL, config.TRAIL_TYPE_ATR, config.TRAIL_TYPE_PERCENT, "None"]
    use_ote_filters = [True, False]
    use_htf_filters = [True, False]
    disp_only = [True, False]
    disc_prem = [True, False]
    sweep_only = [True, False]
    vp_vals = [True, False]
    smt_vals = [True, False]
    max_concurr = [1, 2, 3]
    rrrs = [1.5, 2.0, 3.0]
    daily_losses = [0.0, 3.0, 5.0]
    sb_htf_bias = [True, False]
    
    print(f"Pre-loading data for {symbol} from {date_from} to {date_to}...")
    utils.get_data_by_date(symbol, mt5.TIMEFRAME_M1, date_from - dt.timedelta(days=1), date_to + dt.timedelta(days=30))
    utils.get_data_by_date(symbol, mt5.TIMEFRAME_M5, date_from, date_to)
    utils.get_data_by_date(symbol, mt5.TIMEFRAME_M15, date_from, date_to)
    utils.get_data_by_date(symbol, mt5.TIMEFRAME_H1, date_from - dt.timedelta(days=5), date_to)
    utils.get_data_by_date(symbol, mt5.TIMEFRAME_H4, date_from - dt.timedelta(days=20), date_to)
    utils.get_data_by_date(symbol, mt5.TIMEFRAME_D1, date_from - dt.timedelta(days=60), date_to)
    
    print(f"Pre-loading data for DXY...")
    smt_date_from = date_from - dt.timedelta(days=5)
    utils.get_data_by_date("DXY", mt5.TIMEFRAME_M15, smt_date_from, date_to)
    utils.get_data_by_date("DXY", mt5.TIMEFRAME_M5, smt_date_from, date_to)
    utils.get_data_by_date("DXY", mt5.TIMEFRAME_H1, smt_date_from, date_to)

    # H4 and D1 need more context for bias
    

    
    cache_path = os.path.join(os.getcwd(), 'opt_cache_data.pkl')
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(getattr(utils, '_full_history_cache', {}), f)
    except Exception as e:
        print(f"Failed to dump cache: {e}")
        return
        
    print("Data loaded. Generating random configurations per method...")
    import indicators
    
    ict_params = indicators.get_ict_model_parameters("Default", symbol)
    fixed_params = {
        'symbol': symbol,
        'date_from': date_from,
        'date_to': train_date_to,
        'oos_date_to': oos_date_to,
        'initial_balance': 10000.0,
        'risk_percent': 0.25,
        'fixed_lot': 0,
        'risk_mode': 'percent',
        'trailing_methods': config.LOWER_TFS,
        'ict_params': ict_params,
        'use_dynamic_rrr': False,
        'trade_all_tfs': True,
        'use_ultra_low_tf': False,
        'spread_cost': 0, # Real spread
        'slippage_pts': 2,
        'commission': 7.0,
        'session_filter': True,
        'session_start': 7,
        'session_end': 21,
        'bypass_htf_conf': False,
        'trail_pct': 25,
        'require_bos_fvg': False,
        'slippage_recovery': False,
        'anti_gap_enabled': True,
        'anti_gap_mult': 2.0,
        'min_fvg_size': 2.5,
        'min_conf': 2,
        'smt_correlated_pair': 'DXY',
        'mock_si': {
            'point': 0.001,
            'trade_tick_size': 0.001,
            'digits': 3,
            'spread': 20
        },
        'pro_flags': {
            'dol_tp': False,
            'killzone': False,
            'htf_poi': False,
            'mandatory': False,
            'regime': False,
            'ml_sizing': False,
            'ml_rank': False,
        },
        'ml_filter': False,
        'ml_min_confidence': 60.0
    }
    
    results = {}
    
    # Clear multiprocessing fork warning context if needed, but windows uses spawn
    worker_count = max(1, mp.cpu_count() // 2) if mp.cpu_count() else 2
    print(f"Starting multiprocessing pool with {worker_count} workers...")
    mp_ctx = mp.get_context('spawn')
    with mp_ctx.Pool(processes=worker_count, initializer=init_worker, initargs=(cache_path, fixed_params), maxtasksperchild=None) as pool:
        for method in methods:
            print(f"\n--- Optimizing {method} ---")
            
            grid = list(itertools.product(
                [[method]], sl_modes, trail_types, use_ote_filters, use_htf_filters, disp_only, disc_prem, sweep_only, 
                vp_vals, smt_vals, max_concurr, rrrs, daily_losses, sb_htf_bias
            ))
            
            random.seed(42)
            if len(grid) > 1000:
                grid = random.sample(grid, 1000)
                
            tasks = []
            for combo in grid:
                tasks.append(combo)
                
            print(f"Queueing {len(tasks)} tasks...")
            method_results = []
            for i, res in enumerate(pool.imap_unordered(run_combo, tasks)):
                if res.get('error'):
                    print(f"Error in combo: {res['error']}")
                if not res.get('is_waste', False) and res.get('trades', 0) >= 30:
                    method_results.append(res)
                if (i+1) % 100 == 0:
                    print(f"Processed {i+1}/{len(tasks)}")
            
            method_results.sort(key=lambda x: x['calmar'], reverse=True)
            
            if method_results:
                best = method_results[0]
                print(f"BEST {method} (IS): ROI {best['roi']}%, DD {best['dd']}%, Calmar {best['calmar']}, Trades {best['trades']}, WR {best['win_rate']}")
                if best.get('oos_metrics'):
                    oos = best['oos_metrics']
                    print(f"    --> (OOS): ROI {oos.get('roi')}%, DD {oos.get('dd')}%, Calmar {oos.get('calmar')}, Trades {oos.get('trades')}, WR {oos.get('win_rate')}")
                results[method] = method_results[:5] # Keep top 5
            else:
                print(f"No profitable configurations for {method}")
                
    # Format and save
    out_path = os.path.join(os.getcwd(), 'opt_results.json')
    # strip balance_history to save space
    for method, results_list in results.items():
        for res in results_list:
            if 'balance_history' in res:
                del res['balance_history']
                
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nDone! Results saved to {out_path}")

if __name__ == "__main__":
    main()
