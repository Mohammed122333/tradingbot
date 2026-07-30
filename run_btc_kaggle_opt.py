import sys
import types

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Create mock MetaTrader5 module for Linux environments (Kaggle/Colab)
mock_mt5 = types.ModuleType("MetaTrader5")
mock_mt5.TIMEFRAME_M1 = 1
mock_mt5.TIMEFRAME_M5 = 5
mock_mt5.TIMEFRAME_M15 = 15
mock_mt5.TIMEFRAME_M30 = 30
mock_mt5.TIMEFRAME_H1 = 16385
mock_mt5.TIMEFRAME_H4 = 16388
mock_mt5.TIMEFRAME_D1 = 16408
mock_mt5.TIMEFRAME_W1 = 32769
mock_mt5.TIMEFRAME_MN1 = 49153
mock_mt5.initialize = lambda *a, **k: True
mock_mt5.shutdown = lambda *a, **k: True
mock_mt5.last_error = lambda: (0, 'Success')
sys.modules["MetaTrader5"] = mock_mt5

import os
import json
import random
import multiprocessing as mp
import itertools
from datetime import datetime, timezone
import pandas as pd
import datetime as dt
import pickle

import logging
logging.disable(logging.CRITICAL)

import config
config.OFFLINE_BACKTESTING = True

import utils
import backtester
from opt_worker import init_worker, run_combo

import hashlib

def combo_key_for(idx, combo):
    # Python's hash() is salted per-process for str, so the old key changed on
    # every run and resume never matched anything.
    return f"{idx}_{hashlib.md5(repr(combo).encode()).hexdigest()[:12]}"

def main():
    # Load settings from KAggle.json if present
    kaggle_json_path = os.path.join(os.getcwd(), 'KAggle.json')
    kaggle_cfg = {}
    if os.path.exists(kaggle_json_path):
        print(f"Loading custom grid configuration from {kaggle_json_path}...")
        try:
            with open(kaggle_json_path, 'r') as f:
                kaggle_cfg = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read KAggle.json: {e}")

    symbol = kaggle_cfg.get("opt_symbol_entry", "XAUUSDm")
    
    date_from_str = kaggle_cfg.get("opt_date_from", "2019-01-01")
    train_date_to_str = kaggle_cfg.get("opt_date_to", "2025-01-01")
    oos_date_to_str = kaggle_cfg.get("oos_date_from", "2026-07-26")

    date_from = pd.Timestamp(date_from_str).to_pydatetime()
    train_date_to = pd.Timestamp(train_date_to_str).to_pydatetime()
    oos_date_to = pd.Timestamp(oos_date_to_str).to_pydatetime()
    
    print("==================================================================")
    print(f"       GoodBot v{config.BOT_VERSION} Headless Grid Optimization")
    print(f"       Config File: KAggle.json")
    print(f"       Symbol: {symbol}")
    print(f"       In-Sample (Train):  {date_from.date()} to {train_date_to.date()}")
    print(f"       Out-Of-Sample (Test): {train_date_to.date()} to {oos_date_to.date()}")
    print("==================================================================")
    
    # Selected strategy methods (Focused on FVG Return for systematic sweep)
    methods = ["FVG Return"]

    # SL modes (User locked: Normal SL only)
    sl_modes = [config.FVG_SL_NORMAL]

    # Trailing stop types (User enabled: ALL trailing options)
    trail_types = [config.TRAIL_TYPE_PARTIAL, config.TRAIL_TYPE_PERCENT, config.TRAIL_TYPE_ATR, "None"]

    # Binary filter sweeps (User locked: displacement=False, disc_prem=False, sweep_only=False, sb_htf_bias=False)
    use_ote_filters = [True, False] if kaggle_cfg.get("opt_sweep_ote", True) else [kaggle_cfg.get("opt_default_ote", True)]
    use_htf_filters = [True, False] if kaggle_cfg.get("opt_sweep_htf", True) else [kaggle_cfg.get("opt_default_htf", True)]
    disp_only = [False]
    disc_prem = [False]
    sweep_only = [False]
    vp_vals = [False]
    smt_vals = [False]
    sb_htf_bias = [False]

    # Max concurrent trades range (smart steps: 0, 2, 4, 6, 8, 10)
    c_min = int(kaggle_cfg.get("opt_concurr_min", 0))
    c_max = int(kaggle_cfg.get("opt_concurr_max", 10))
    if c_max - c_min >= 8:
        max_concurr = [0, 2, 4, 6, 8, 10]
    else:
        max_concurr = list(range(c_min, c_max + 1))

    # RRR range (smart steps: 1.0, 2.0, 3.0, 4.5, 6.0)
    r_min = float(kaggle_cfg.get("opt_rrr_min", 1.0))
    r_max = float(kaggle_cfg.get("opt_rrr_max", 6.0))
    r_step = float(kaggle_cfg.get("opt_rrr_step", 1.0))
    if r_max >= 6.0 and r_min <= 1.5:
        rrrs = [1.0, 2.0, 3.0, 4.5, 6.0]
    else:
        rrrs = []
        curr_r = r_min
        while curr_r <= r_max + 1e-5:
            rrrs.append(round(curr_r, 2))
            curr_r += r_step if r_step > 0 else 1.0

    # Daily loss range (smart steps: 0.0, 1.5, 3.0, 5.0)
    dl_min = float(kaggle_cfg.get("opt_dl_min", 0.0))
    dl_max = float(kaggle_cfg.get("opt_dl_max", 5.0))
    dl_step = float(kaggle_cfg.get("opt_dl_step", 1.0))
    if dl_max >= 5.0 and dl_min <= 0.0:
        daily_losses = [0.0, 1.5, 3.0, 5.0]
    else:
        daily_losses = []
        curr_dl = dl_min
        while curr_dl <= dl_max + 1e-5:
            daily_losses.append(round(curr_dl, 2))
            curr_dl += dl_step if dl_step > 0 else 1.0

    print(f"\nPre-loading history for {symbol} from history_cache...")
    TF_M1 = 1
    TF_M5 = 5
    TF_M15 = 15
    TF_H1 = 16385
    TF_H4 = 16388
    TF_D1 = 16408

    utils.get_data_by_date(symbol, TF_M1, date_from - dt.timedelta(days=1), oos_date_to + dt.timedelta(days=30))
    utils.get_data_by_date(symbol, TF_M5, date_from, oos_date_to)
    utils.get_data_by_date(symbol, TF_M15, date_from, oos_date_to)
    utils.get_data_by_date(symbol, TF_H1, date_from - dt.timedelta(days=5), oos_date_to)
    utils.get_data_by_date(symbol, TF_H4, date_from - dt.timedelta(days=20), oos_date_to)
    utils.get_data_by_date(symbol, TF_D1, date_from - dt.timedelta(days=60), oos_date_to)
    
    import indicators
    
    ict_params = indicators.get_ict_model_parameters("Default", symbol)
    fixed_params = {
        'symbol': symbol,
        'date_from': date_from,
        'date_to': train_date_to,
        'oos_date_to': oos_date_to,
        'initial_balance': float(kaggle_cfg.get("opt_balance_entry", 10000)),
        'risk_percent': float(kaggle_cfg.get("opt_risk_entry", 0.25)),
        'fixed_lot': float(kaggle_cfg.get("opt_fixed_lot_entry", 0.02)) if kaggle_cfg.get("opt_use_fixed_lot", False) else 0.0,
        'risk_mode': 'fixed' if kaggle_cfg.get("opt_use_fixed_lot", False) else 'percent',
        'trailing_methods': config.LOWER_TFS,
        'ict_params': ict_params,
        'use_dynamic_rrr': kaggle_cfg.get("opt_use_dynamic_rrr", False),
        'trade_all_tfs': kaggle_cfg.get("opt_trade_all_tfs", True),
        'use_ultra_low_tf': kaggle_cfg.get("opt_use_ultra_low_tf", False),
        'spread_cost': float(kaggle_cfg.get("opt_spread_cost", 0.0)),
        'slippage_pts': int(kaggle_cfg.get("opt_slippage", 0)),
        'commission': float(kaggle_cfg.get("opt_commission", 0.0)),
        'session_filter': kaggle_cfg.get("opt_session_filter", False),
        'session_start': int(kaggle_cfg.get("opt_session_start", 13)),
        'session_end': int(kaggle_cfg.get("opt_session_end", 21)),
        'bypass_htf_conf': kaggle_cfg.get("opt_bypass_htf_conf", True),
        'trail_pct': float(kaggle_cfg.get("opt_trail_pct", 0.5)),
        'require_bos_fvg': kaggle_cfg.get("opt_require_bos_fvg", False),
        'slippage_recovery': kaggle_cfg.get("opt_slippage_recovery", True),
        'anti_gap_enabled': kaggle_cfg.get("opt_anti_gap", True),
        'anti_gap_mult': float(kaggle_cfg.get("opt_anti_gap_mult", 2.0)),
        'min_fvg_size': float(kaggle_cfg.get("opt_min_fvg", 0.0)),
        'min_conf': int(kaggle_cfg.get("opt_min_conf", 2)),
        'smt_correlated_pair': kaggle_cfg.get("opt_smt_pair", "AUTO"),
        'mock_si': {
            'point': 0.01,
            'trade_tick_size': 0.01,
            'digits': 2,
            'spread': 3000
        },
        'pro_flags': {
            'dol_tp': kaggle_cfg.get("opt_pro_dol_tp", True),
            'killzone': kaggle_cfg.get("opt_pro_killzone", True),
            'htf_poi': kaggle_cfg.get("opt_pro_htf_poi", True),
            'mandatory': kaggle_cfg.get("opt_pro_mandatory", False),
            'regime': kaggle_cfg.get("opt_pro_regime", True),
            'ml_sizing': kaggle_cfg.get("opt_pro_ml_sizing", False),
            'ml_rank': kaggle_cfg.get("opt_pro_ml_rank", False),
        },
        'ml_filter': kaggle_cfg.get("opt_ml_filter", False),
        'ml_min_confidence': float(kaggle_cfg.get("opt_ml_min_confidence", 60.0))
    }

    # Pass in-memory history cache directly to workers without writing 300MB pickle files to disk
    cached_data_dict = getattr(utils, '_full_history_cache', {})
    
    # --- Checkpoint & Resumption setup ---
    checkpoint_file = os.path.join(os.getcwd(), 'opt_checkpoint.json')
    results = {}
    completed_combos_map = {}
    
    if os.path.exists(checkpoint_file):
        print(f"\n[RESUME] Found existing checkpoint file: {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r') as f:
                ckpt_data = json.load(f)
                results = ckpt_data.get('results', {})
                completed_combos_map = ckpt_data.get('completed_combos_map', {})
                print(f"[RESUME] Loaded {sum(len(v) for v in completed_combos_map.values())} previously evaluated combinations across methods.")
        except Exception as e:
            print(f"[RESUME WARNING] Failed to parse checkpoint file: {e}")

    MAX_RUN_SECONDS = (11 * 3600) + (45 * 60)  # 11 hours 45 minutes limit
    import time
    global_start_time = time.time()
    
    worker_count = max(1, mp.cpu_count())
    print(f"\nStarting multiprocessing pool with {worker_count} workers...")
    
    time_limit_reached = False
    
    mp_ctx = mp.get_context('spawn')
    with mp_ctx.Pool(processes=worker_count, initializer=init_worker, initargs=(cached_data_dict, fixed_params)) as pool:
        for method in methods:
            if time_limit_reached:
                break
                
            print(f"\n------------------------------------------------------------------")
            print(f" Optimizing Strategy Method: {method} (Systematic Grid)")
            print(f"------------------------------------------------------------------")
            
            grid = list(itertools.product(
                [[method]], sl_modes, trail_types, use_ote_filters, use_htf_filters, disp_only, disc_prem, sweep_only, 
                vp_vals, smt_vals, max_concurr, rrrs, daily_losses, sb_htf_bias
            ))
            
            total_grid_len = len(grid)
            print(f"Total systematic combinations for {method}: {total_grid_len}")
            
            done_keys = set(completed_combos_map.get(method, []))
            
            # Filter tasks to only those not yet evaluated
            tasks_with_index = []
            for idx, combo in enumerate(grid):
                combo_key = combo_key_for(idx, combo)
                if combo_key not in done_keys:
                    tasks_with_index.append((idx, combo_key, combo))
                    
            already_done = total_grid_len - len(tasks_with_index)
            if already_done > 0:
                print(f"[RESUME] Skipping {already_done}/{total_grid_len} combinations already evaluated for {method}.")
                
            if not tasks_with_index:
                print(f"All combinations for {method} already completed!")
                continue
                
            tasks_only = [item[2] for item in tasks_with_index]
            method_results = results.get(method, [])
            
            start_time = time.time()
            total_to_run = len(tasks_with_index)
            method_idx = methods.index(method) + 1
            total_methods = len(methods)
            
            # chunksize=1 ensures instant per-item yield without buffering latency
            chunksize = 1
            
            for i, res in enumerate(pool.imap(run_combo, tasks_only, chunksize=chunksize)):
                curr_elapsed = time.time() - global_start_time
                if curr_elapsed >= MAX_RUN_SECONDS:
                    print(f"\n[TIME LIMIT REACHED] Reached 11h 45m execution limit ({curr_elapsed/3600:.2f} hrs). Stopping safely...")
                    time_limit_reached = True
                    break

                idx, combo_key, combo = tasks_with_index[i]
                
                if method not in completed_combos_map:
                    completed_combos_map[method] = []
                completed_combos_map[method].append(combo_key)
                
                if res and res.get('error'):
                    print(f"  [COMBO ERROR] {combo}: {res['error']}", flush=True)

                MIN_TRADES = int(kaggle_cfg.get("opt_min_trades", 30))
                is_prof = (res
                           and not res.get('error')
                           and not res.get('is_waste', False)
                           and res.get('trades', 0) >= MIN_TRADES
                           and res.get('roi', 0.0) > 0.0)
                if is_prof:
                    method_results.append(res)
                
                curr_calmar = res.get('calmar', 0.0) if res else 0.0
                highest_calmar_so_far = max([r.get('calmar', 0.0) for r in method_results], default=0.0)

                done = i + 1
                elapsed = max(0.1, time.time() - start_time)
                speed = done / elapsed
                rem_combos = total_to_run - done
                eta_sec = rem_combos / speed if speed > 0 else 0
                
                elapsed_m, elapsed_s = divmod(int(elapsed), 60)
                elapsed_h, elapsed_m = divmod(elapsed_m, 60)
                elapsed_str = f"{elapsed_h}h {elapsed_m:02d}m {elapsed_s:02d}s" if elapsed_h > 0 else f"{elapsed_m:02d}m {elapsed_s:02d}s"
                
                eta_m, eta_s = divmod(int(eta_sec), 60)
                eta_h, eta_m = divmod(eta_m, 60)
                eta_str = f"{eta_h}h {eta_m:02d}m {eta_s:02d}s" if eta_h > 0 else f"{eta_m:02d}m {eta_s:02d}s"
                
                pct = ((already_done + done) / total_grid_len) * 100
                profitable_so_far = len(method_results)
                print(f" [{method_idx}/{total_methods}: {method}] {already_done + done}/{total_grid_len} ({pct:.1f}%) | Calmar: {curr_calmar:.2f} | Max Calmar: {highest_calmar_so_far:.2f} | Speed: {speed:.1f}/s | Profitable: {profitable_so_far} | Elapsed: {elapsed_str} | EST END: {eta_str}", flush=True)
                
                # Checkpoint save every 500 combinations
                if done % 500 == 0:
                    try:
                        results[method] = method_results
                        with open(checkpoint_file, 'w') as f:
                            json.dump({
                                'results': results,
                                'completed_combos_map': completed_combos_map,
                                'last_update': datetime.now(timezone.utc).isoformat()
                            }, f)
                    except Exception:
                        pass
            
            method_results.sort(key=lambda x: x.get('calmar', 0), reverse=True)
            results[method] = method_results
            
            # Save checkpoint after method ends
            try:
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'results': results,
                        'completed_combos_map': completed_combos_map,
                        'last_update': datetime.now(timezone.utc).isoformat()
                    }, f)
            except Exception:
                pass

            if method_results:
                best = method_results[0]
                print(f"\n>>> BEST {method} (In-Sample): ROI: +{best['roi']}%, Max DD: {best['dd']}%, Calmar: {best['calmar']}, Trades: {best['trades']}, Win Rate: {best['win_rate']}%")
                if best.get('oos_metrics'):
                    oos = best['oos_metrics']
                    print(f"    --> (Out-Of-Sample): ROI: +{oos.get('roi')}%, Max DD: {oos.get('dd')}%, Calmar: {oos.get('calmar')}, Trades: {oos.get('trades')}, Win Rate: {oos.get('win_rate')}%")
            else:
                print(f"No profitable configurations found for {method}")
                
    out_path = os.path.join(os.getcwd(), 'btc_opt_results.json')
    for method, results_list in results.items():
        for res in results_list:
            if isinstance(res, dict) and 'balance_history' in res:
                del res['balance_history']
                
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n==================================================================")
    print(f"   OPTIMIZATION RUN COMPLETED OR PAUSED SAFELY!")
    print(f"   Results saved to: {out_path}")
    print(f"   Checkpoint saved to: {checkpoint_file}")
    print(f"==================================================================")

if __name__ == "__main__":
    main()
