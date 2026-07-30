import random
import bisect
import heapq
import datetime
import logging
import pandas as pd
from ml_engine import ml_score_signal
import numpy as np
try:
    import MetaTrader5 as mt5
except ImportError:
    try:
        import tests.MetaTrader5 as mt5
    except ImportError:
        import sys, types
        mt5 = types.ModuleType("MetaTrader5")
        mt5.TIMEFRAME_M1 = 1
        mt5.TIMEFRAME_M5 = 5
        mt5.TIMEFRAME_M15 = 15
        mt5.TIMEFRAME_M30 = 30
        mt5.TIMEFRAME_H1 = 16385
        mt5.TIMEFRAME_H4 = 16388
        mt5.TIMEFRAME_D1 = 16408
        mt5.TIMEFRAME_W1 = 32769
        mt5.TIMEFRAME_MN1 = 49153
        mt5.initialize = lambda *a, **k: True
        mt5.shutdown = lambda *a, **k: True
        sys.modules["MetaTrader5"] = mt5
import ict_pro

import config
from config import (
    bt_stop_event, ULTRA_LOW_TFS, LOWER_TFS, HIGHER_TFS,
    FVG_SL_NORMAL, TRAIL_TYPE_PARTIAL
)
from utils import (
    get_data, get_data_by_date, get_spread, get_contract_size,
    is_crypto_symbol, is_in_trading_session, check_rrr, calculate_dynamic_lot_size,
    resolve_ict_methods, get_legacy_ob_retest_methods,
    get_symbol_profile_overrides, resolve_method_param, is_in_news_embargo,
    match_symbol_profile, _wilders_smooth_full_njit
)
from indicators import (
    get_ict_model_parameters, detect_order_blocks, detect_fair_value_gaps,
    detect_swing_points, detect_liquidity_pools, detect_all_liquidity_sweeps, detect_liquidity_sweep,
    calculate_ote_zone, check_displacement_quality, check_bos_after_mss,
    calculate_market_structure_chronologically
)
from detectors import (
    detect_silver_bullet_signals, detect_judas_swing, detect_breaker_blocks,
    detect_iofr_signals, compute_fvg_sl_price, detect_mmxm_signals,
    detect_ict_model_signals, convert_time_to_ny_hour
)
from simulation import (
    run_trade_simulation, get_structure_tp, calculate_confluence_score,
    compute_effective_entry, calculate_tp_sl, get_higher_tf_trend,
    clear_simulation_cache
)
from regime_manager import rm, classify_regime
from backtest_metrics import compute_metrics
import cost_model

_mtf_structure_cache = {}

logger = logging.getLogger()

def generate_trade_signals(symbol, timeframe_name, timeframe_value, date_from, date_to,
                           ict_params, trailing_methods=None, use_dynamic_rrr=True,
                           ict_method="ICT Retest", min_rrr=1.5, fvg_sl_mode=FVG_SL_NORMAL,
                           spread_cost=0.0, slippage_points=0, commission_per_lot=0.0,
                           session_filter=True, session_start=7, session_end=21,
                           progress_callback=None, use_htf_filter=True, use_ote_filter=True, bypass_htf_conf=False,
                           trail_type=TRAIL_TYPE_PARTIAL, trail_params=None, require_bos_fvg=False,
                           anti_gap_enabled=True, anti_gap_mult=2.0,
                           fvg_displacement_only=True, fvg_discount_premium_only=True, fvg_recent_sweep_only=False,
                           sb_require_htf_bias=False,
                           use_smt_divergence=False, smt_correlated_pair="DXY", use_volume_profile=False,
                           sym_params=None, fvg_sl_spread_buffer=2.0, limit_touch_fill=False):
    date_from = pd.Timestamp(date_from).floor('s')
    date_to = pd.Timestamp(date_to).floor('s')
    LIVE_WINDOW = 199  # Match live: 200 bars fetched, minus 1 forming = 199 closed bars
    # Fetch context bars BEFORE start date for OB detection warm-up
    context_df = get_data(symbol, timeframe_value, 200, live=False, date_to=date_from)
    main_df = get_data_by_date(symbol, timeframe_value, date_from, date_to)

    if main_df is None or main_df.empty:
        return []


    corr_dfs_dict = {}
    if use_smt_divergence and smt_correlated_pair:
        import re
        suffix_match = re.search(r'([a-z.]+)$', symbol)
        suffix = suffix_match.group(1) if suffix_match else ""
        
        if smt_correlated_pair.upper() == "AUTO":
            pairs = ["DXY"]
            if "EURUSD" in symbol or "GBPUSD" in symbol or "AUDUSD" in symbol or "NZDUSD" in symbol:
                pairs = ["DXY", "USDCAD", "USDCHF"] if "GBP" not in symbol else ["DXY", "EURUSD"]
            elif "USD" in symbol:
                pairs = ["EURUSD", "GBPUSD"]
        else:
            pairs = [p.strip() for p in smt_correlated_pair.split(',') if p.strip()]
        for p in pairs:
            pair_to_fetch = p + suffix if suffix else p
            c_df = get_data_by_date(pair_to_fetch, timeframe_value, date_from - pd.Timedelta(days=20), date_to)
            if c_df is None or c_df.empty:
                c_df = get_data_by_date(p, timeframe_value, date_from - pd.Timedelta(days=20), date_to)
            corr_dfs_dict[p] = c_df
            
        from indicators import check_advanced_smt
        
    if use_volume_profile:
        from indicators import calculate_volume_profile

    # Validate context data is actually BEFORE date_from
    if context_df is not None and not context_df.empty:
        # Filter out any bars that are AT or AFTER date_from (shouldn't happen but safety)
        context_df = context_df[context_df.index < date_from]

    if context_df is not None and not context_df.empty:
        df = pd.concat([context_df, main_df])
        df = df[~df.index.duplicated(keep='last')].sort_index()
        # trade_start_idx = first bar on or after date_from
        trade_start_idx = df.index.searchsorted(date_from)
        if trade_start_idx >= len(df):
            trade_start_idx = len(context_df)
    else:
        df = main_df
        trade_start_idx = 0

    try:
        import MetaTrader5 as mt5_local
        info = mt5_local.symbol_info(symbol)
    except Exception:
        info = None
        
    if info:
        point = info.point
    else:
        # Offline fallback based on typical symbol names
        point = cost_model.point_for_symbol(symbol)

    # --- Unified spread resolution (single source of truth: cost_model) ---
    # spread_cost > 0  -> explicit FIXED spread in POINTS
    # spread_cost == 0 -> AUTO: real spread from the dataset 'spread' column,
    #                     else a realistic per-symbol default (never a silent 0).
    current_spread = cost_model.resolve_spread(symbol, df=df, point=point, spread_cost=spread_cost)
    # Same margined cost spread is used for entry fills AND exit modelling so
    # the backtest is internally consistent and in parity with live.
    actual_spread_cost = current_spread * cost_model.spread_multiplier()

    _cached_spread = current_spread
    methods_to_run = resolve_ict_methods(ict_method)
    logger.info(f"[{symbol} {timeframe_name}] Methods to run: {methods_to_run} | SMT={use_smt_divergence} | VP={use_volume_profile}")
    completed_trades = []

    processed_ob = set()

    # Cache HTF trend to avoid repeated MT5 calls
    cached_htf_trend = {}
    htf_cache_interval = 200  # H4/D1 trends change slowly — 200 bars is safe

    _lows = df['low'].values
    _highs = df['high'].values
    _closes = df['close'].values
    _opens = df['open'].values
    _timestamps = df.index.tolist()
    _timestamps_np = df.index.values

    # --- Precalculate completely on the full DataFrame ---
    import os
    import pickle
    import hashlib
    
    param_str = f"{ict_params.get('threshold_factor')}_{ict_params.get('reversal_threshold')}_{ict_params.get('lookahead')}_{current_spread}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    # Hash the actual data content to prevent stale caches when grid opt workers
    # (using patched data functions) write caches that the single backtest reads.
    data_hash = hashlib.md5(f"{len(df)}_{df.index[0]}_{df.index[-1]}_{_highs.mean()}_{_lows.mean()}".encode()).hexdigest()[:12]
    ind_cache_file = os.path.join("history_cache", f"indicators_{symbol}_{timeframe_value}_{len(df)}_{param_hash}_{data_hash}.pkl")
    
    # Pre-fetch H1 and H4 data for MTF alignment
    global _mtf_structure_cache
    
    h1_times, h1_states_list = np.array([]), np.array([])
    h4_times, h4_states_list = np.array([]), np.array([])
    
    if len(df) > 0:
        start_dt = df.index[0].to_pydatetime()
        end_dt = df.index[-1].to_pydatetime()
        
        cache_key_h1 = (symbol, mt5.TIMEFRAME_H1, start_dt, end_dt)
        if cache_key_h1 not in _mtf_structure_cache:
            try:
                h1_df = get_data_by_date(symbol, mt5.TIMEFRAME_H1, start_dt, end_dt)
                if h1_df is not None and not h1_df.empty:
                    h1_states, _, _ = calculate_market_structure_chronologically(h1_df, lookback=5)
                    _mtf_structure_cache[cache_key_h1] = (h1_df.index.values, np.array(h1_states))
                else:
                    _mtf_structure_cache[cache_key_h1] = (np.array([]), np.array([]))
            except Exception as e:
                logger.error(f"Failed to fetch H1 for MTF alignment: {e}")
                _mtf_structure_cache[cache_key_h1] = (np.array([]), np.array([]))
        h1_times, h1_states_list = _mtf_structure_cache[cache_key_h1]

        cache_key_h4 = (symbol, mt5.TIMEFRAME_H4, start_dt, end_dt)
        if cache_key_h4 not in _mtf_structure_cache:
            try:
                h4_df = get_data_by_date(symbol, mt5.TIMEFRAME_H4, start_dt, end_dt)
                if h4_df is not None and not h4_df.empty:
                    h4_states, _, _ = calculate_market_structure_chronologically(h4_df, lookback=5)
                    _mtf_structure_cache[cache_key_h4] = (h4_df.index.values, np.array(h4_states))
                else:
                    _mtf_structure_cache[cache_key_h4] = (np.array([]), np.array([]))
            except Exception as e:
                logger.error(f"Failed to fetch H4 for MTF alignment: {e}")
                _mtf_structure_cache[cache_key_h4] = (np.array([]), np.array([]))
        h4_times, h4_states_list = _mtf_structure_cache[cache_key_h4]
    


    cached_indicators = None
    if os.path.exists(ind_cache_file):
        try:
            with open(ind_cache_file, "rb") as f:
                cached_indicators = pickle.load(f)
                logger.info(f"Loaded light-speed indicator cache for {symbol} ({len(df)} bars)")
        except Exception as e:
            pass

    if cached_indicators is not None:
        global_obs_df = cached_indicators["global_obs_df"]
        global_all_fvgs = cached_indicators["global_all_fvgs"]
        global_swing_highs = cached_indicators["global_swing_highs"]
        global_swing_lows = cached_indicators["global_swing_lows"]
        global_eqh_pools = cached_indicators.get("global_eqh_pools", detect_liquidity_pools(global_swing_highs))
        global_eql_pools = cached_indicators.get("global_eql_pools", detect_liquidity_pools(global_swing_lows))
        global_structure_states = cached_indicators["global_structure_states"]
        global_mss_events = cached_indicators["global_mss_events"]
        global_bos_events = cached_indicators["global_bos_events"]
        global_sweeps = cached_indicators["global_sweeps"]
    else:
        logger.info(f"Computing heavy indicators for {symbol} ({len(df)} bars)...")
        logger.info(f" -> Phase 1/5: Detecting Order Blocks...")
        global_obs_df = detect_order_blocks(df, threshold_factor=ict_params['threshold_factor'],
                                            reversal_threshold=ict_params['reversal_threshold'],
                                            lookahead=ict_params['lookahead'], spread=current_spread)
        if not global_obs_df.empty:
            global_obs_df['ob_bar_index'] = global_obs_df['ob_bar_index'].astype(int)
            global_obs_df['confirmation_bar'] = global_obs_df['confirmation_bar'].astype(int)
            
            # Pre-calculate first break time for breaker block speed optimization
            first_break_times = []
            
            for idx_time, row in global_obs_df.iterrows():
                ob_type = row['order_block_type']
                ob_high = row['high']
                ob_low = row['low']
                ob_bar_idx = int(row['ob_bar_index'])
                
                # Look for the first break after the OB bar index (strictly after formation)
                post_close = _closes[ob_bar_idx + 1:]
                if ob_type == 'bearish':
                    break_mask = post_close > ob_high
                else:
                    break_mask = post_close < ob_low
                    
                break_indices = np.where(break_mask)[0]
                if len(break_indices) > 0:
                    first_break_idx = ob_bar_idx + 1 + break_indices[0]
                    first_break_times.append(_timestamps[first_break_idx])
                else:
                    first_break_times.append(pd.NaT)
                    
            global_obs_df['first_break_time'] = first_break_times

        logger.info(f" -> Phase 2/5: Detecting Fair Value Gaps...")
        global_all_fvgs = detect_fair_value_gaps(df)
        # Vectorized FVG fill detection — replaces O(n²) nested loop
        for fvg in global_all_fvgs:
            fvg['filled_bar_idx'] = len(df)  # default: unfilled (use len instead of inf for int compat)
            start_j = fvg['bar_index'] + 2
            if fvg['type'] == 'bullish':
                # Find first bar where low <= mid (mitigation at consequent encroachment)
                mask = _lows[start_j:] <= fvg['mid']
            else:
                # Find first bar where high >= mid (mitigation at consequent encroachment)
                mask = _highs[start_j:] >= fvg['mid']
            filled_indices = np.where(mask)[0]
            if len(filled_indices) > 0:
                fvg['filled_bar_idx'] = start_j + filled_indices[0]

        logger.info(f" -> Phase 3/5: Detecting Swing Points...")
        global_swing_highs, global_swing_lows = detect_swing_points(df, lookback=5)
        global_eqh_pools = detect_liquidity_pools(global_swing_highs)
        global_eql_pools = detect_liquidity_pools(global_swing_lows)
        
        logger.info(f" -> Phase 4/5: Calculating Market Structure Chronologically...")
        # Pre-calculate stateful market structure chronologically (zero lookahead bias)
        global_structure_states, global_mss_events, global_bos_events = calculate_market_structure_chronologically(df, lookback=5)
        
        logger.info(f" -> Phase 5/5: Detecting Liquidity Sweeps...")
        global_sweeps = detect_all_liquidity_sweeps(df)
        logger.info(f" -> Indicators computation complete! Caching to disk...")

        try:
            with open(ind_cache_file, "wb") as f:
                pickle.dump({
                    "global_obs_df": global_obs_df,
                    "global_all_fvgs": global_all_fvgs,
                    "global_swing_highs": global_swing_highs,
                    "global_swing_lows": global_swing_lows,
                    "global_eqh_pools": global_eqh_pools,
                    "global_eql_pools": global_eql_pools,
                    "global_structure_states": global_structure_states,
                    "global_mss_events": global_mss_events,
                    "global_bos_events": global_bos_events,
                    "global_sweeps": global_sweeps
                }, f)
        except Exception as e:
            logger.error(f"Failed to save indicator cache: {e}")
    
    # Build fast O(1) lookups
    mss_lookup = {ev['break_idx']: ev['type'] for ev in global_mss_events}
    
    # Pre-calculate BOS-after-MSS confirmation states for each bar in O(N)
    bos_confirmed_after_mss_buy = [False] * len(df)
    bos_confirmed_after_mss_sell = [False] * len(df)
    
    sorted_mss = sorted(global_mss_events, key=lambda x: x['break_idx'])
    sorted_bos = sorted(global_bos_events, key=lambda x: x['break_idx'])
    
    mss_ptr = 0
    bos_ptr = 0
    
    last_bull_mss_idx = -1
    last_bear_mss_idx = -1
    bull_bos_confirmed = False
    bear_bos_confirmed = False
    
    for t in range(len(df)):
        # Process MSS events for this bar
        while mss_ptr < len(sorted_mss) and sorted_mss[mss_ptr]['break_idx'] == t:
            if sorted_mss[mss_ptr]['type'] == 'bullish':
                last_bull_mss_idx = t
                bull_bos_confirmed = False  # Reset on new MSS
            elif sorted_mss[mss_ptr]['type'] == 'bearish':
                last_bear_mss_idx = t
                bear_bos_confirmed = False  # Reset on new MSS
            mss_ptr += 1
            
        # Process BOS events for this bar
        while bos_ptr < len(sorted_bos) and sorted_bos[bos_ptr]['break_idx'] == t:
            if sorted_bos[bos_ptr]['type'] == 'bullish':
                if last_bull_mss_idx != -1 and last_bull_mss_idx < t:
                    bull_bos_confirmed = True
            elif sorted_bos[bos_ptr]['type'] == 'bearish':
                if last_bear_mss_idx != -1 and last_bear_mss_idx < t:
                    bear_bos_confirmed = True
            bos_ptr += 1
            
        bos_confirmed_after_mss_buy[t] = bull_bos_confirmed
        bos_confirmed_after_mss_sell[t] = bear_bos_confirmed

    # Pre-extract swing indices for binary search (avoids linear scan per bar)
    _sh_indices = [sh['idx'] for sh in global_swing_highs]
    _sl_indices = [sl['idx'] for sl in global_swing_lows]

    # Pre-fetch M1 data for entire test date range to optimize run_trade_simulation
    # Ensure m1_date_to is deterministic and strictly capped at date_to
    # to avoid pulling live ticks up to the exact second the backtest is run,
    # which causes non-deterministic results between runs.
    m1_date_to = date_to
        
    logger.info(f" -> Downloading M1 Tick Data for precise trade simulation (this may take a long time for large date ranges)...")
    m1_global_df = get_data_by_date(symbol, mt5.TIMEFRAME_M1, date_from - pd.Timedelta(days=1), m1_date_to)
    if m1_global_df is not None and not m1_global_df.empty:
        m1_global_df = m1_global_df.sort_index()
        m1_global_df = {
            'times': m1_global_df.index.values,
            'opens': m1_global_df['open'].values,
            'highs': m1_global_df['high'].values,
            'lows': m1_global_df['low'].values,
            'closes': m1_global_df['close'].values,
        }

    # Precalculate global ATR for speed
    global_atr = np.zeros(len(df))
    if len(df) > 1:
        _high = df['high'].values
        _low = df['low'].values
        _close = df['close'].values
        tr = np.zeros(len(df))
        tr[1:] = np.maximum(_high[1:] - _low[1:], np.maximum(np.abs(_high[1:] - _close[:-1]), np.abs(_low[1:] - _close[:-1])))
        tr[0] = _high[0] - _low[0]
        period = 14
        global_atr = _wilders_smooth_full_njit(tr, period, tr[0])

    # global_sweeps is already precalculated / loaded from cache above!
    global_sweep_idxs = [sw['idx'] for sw in global_sweeps]
    _sweep_bull_indices = [sw['idx'] for sw in global_sweeps if sw['type'] == 'bullish']
    _sweep_bear_indices = [sw['idx'] for sw in global_sweeps if sw['type'] == 'bearish']
    _sweep_bull_set = set(_sweep_bull_indices)
    _sweep_bear_set = set(_sweep_bear_indices)

    # Precalculate rolling 50-bar high/low for discount/premium filter (avoids per-bar slicing)
    _rolling_high_50 = pd.Series(_highs).rolling(window=50, min_periods=1).max().values
    _rolling_low_50 = pd.Series(_lows).rolling(window=50, min_periods=1).min().values
    
    # Precalculate rolling 15-bar high/low for liquidity sweep calculations (avoids per-bar slicing)
    _rolling_high_15 = pd.Series(_highs).rolling(window=15, min_periods=1).max().values
    _rolling_low_15 = pd.Series(_lows).rolling(window=15, min_periods=1).min().values

    # Precalculate loop-invariant min_stop_distance once (Section 2.6)
    _atr_ref = float(np.nanmedian(global_atr[global_atr > 0])) if len(global_atr) else 0.0
    _stops_level_pts = 0
    if info is not None and getattr(info, 'trade_stops_level', 0):
        _stops_level_pts = int(info.trade_stops_level)
    elif sym_params and isinstance(sym_params, dict):
        _stops_level_pts = int(sym_params.get('trade_stops_level', 0) or 0)
    min_stop_distance = max(
        _stops_level_pts * point,
        _atr_ref * float(getattr(config, 'MIN_STOP_ATR_FRAC', 0.05)),
        point * 10,
    )

    # min_stop_distance etc.

    # Precalculate displacement quality mask (min_body_ratio = 0.6)
    _bodies = np.abs(_closes - _opens)
    _ranges = _highs - _lows
    with np.errstate(divide='ignore', invalid='ignore'):
        _ratios = np.where(_ranges > 0, _bodies / _ranges, 0.0)
    _displacement_mask = _ratios >= 0.6

    # Pre-fetch HTF data ONCE for the entire backtest range to avoid repeated MT5 API calls
    _htf_prefetched = {}
    if use_htf_filter:
        for htf_name, htf_value in HIGHER_TFS:
            htf_df = get_data_by_date(symbol, htf_value, date_from - pd.Timedelta(days=200), date_to)
            if htf_df is not None and not htf_df.empty:
                _htf_prefetched[htf_name] = (htf_value, htf_df.sort_index())

    global_obs_list = []
    if not global_obs_df.empty:
        for idx_time, row in global_obs_df.iterrows():
            d = row.to_dict()
            global_obs_list.append((idx_time, d))
            
    # Speed up: Precalculate sorted indices for binary search
    _ob_bar_indices = [ob_dict['ob_bar_index'] for idx_time, ob_dict in global_obs_list]
    _fvg_bar_indices = [fvg['bar_index'] for fvg in global_all_fvgs]
    # ---------------------------------------------------

    _regime_cache_state = {'last_idx': -1, 'tradeable': True, 'size_mult': 1.0, 'conf_bump': 0, 'label': 'trend'}
    
    total_bars = len(df) - trade_start_idx
    for bar_idx, i in enumerate(range(trade_start_idx, len(df))):
        if bt_stop_event.is_set(): break
        # Progress callback
        if progress_callback and bar_idx % 20 == 0:
            progress_callback(bar_idx, total_bars, symbol, timeframe_name)

        _current_vp_poc = None
        _vp_poc_calculated = False
        def get_current_vp_poc():
            nonlocal _current_vp_poc, _vp_poc_calculated
            if not _vp_poc_calculated:
                # Remove i+1 to exclude the forming bar, preventing lookahead bias
                _current_vp_poc = calculate_volume_profile(df.iloc[max(0, i - 200):i]) if use_volume_profile else None
                _vp_poc_calculated = True
            return _current_vp_poc

        current_time = _timestamps[i]
        # Use open price as the current reference to prevent future leak on the active bar
        current_price = _opens[i]
        current_bar_low = _lows[i]
        current_bar_high = _highs[i]

        # Resolve global strict MTF alignment requirement
        global_sb_htf_bias = resolve_method_param('sb_require_htf_bias', 'Silver Bullet', sym_params, sb_require_htf_bias)
        htf_bias_direction = "neutral"
        if global_sb_htf_bias and len(h1_times) > 0 and len(h4_times) > 0:
            current_time_np = _timestamps_np[i]
            # Offset by timeframe duration to fetch fully closed bars
            h1_close_time = current_time_np - np.timedelta64(1, 'h')
            h4_close_time = current_time_np - np.timedelta64(4, 'h')
            idx_h1 = np.searchsorted(h1_times, h1_close_time, side='right') - 1
            idx_h4 = np.searchsorted(h4_times, h4_close_time, side='right') - 1
            if idx_h1 >= 0 and idx_h4 >= 0:
                s_h1 = h1_states_list[idx_h1]
                s_h4 = h4_states_list[idx_h4]
                if s_h1 == s_h4 and s_h1 != "neutral":
                    htf_bias_direction = s_h1

        # Fix #10: Session filter
        current_methods = methods_to_run
        if session_filter and not is_crypto_symbol(symbol) and not is_in_trading_session(current_time, session_start, session_end):
            if "Silver Bullet" in methods_to_run:
                current_methods = ["Silver Bullet"]
            else:
                continue

        if getattr(config, 'NEWS_EMBARGO_ENABLED', False) and not is_crypto_symbol(symbol):
            if is_in_news_embargo(current_time):
                continue
                
        # --- NEW: Macro News Calendar Filter ---
        if getattr(config, 'NEWS_FILTER_ENABLED', False):
            from news_filter import is_macro_news_embargo
            buffer_mins = getattr(config, 'NEWS_FILTER_BUFFER_MINS', 30)
            is_macro_embargo, _ = is_macro_news_embargo(symbol, current_time, is_live=False, buffer_mins=buffer_mins)
            if is_macro_embargo:
                continue

        # Match live: use a sliding window of LIVE_WINDOW closed bars ending 1 bar before current
        window_start = max(0, i - LIVE_WINDOW)
        if (i - window_start) < 20:
            continue

        # Fix 6: regime filter (parity with live). Closed bars only -> no look-ahead.
        _bt_atr = global_atr[i-1] if i > 0 else 0.0
        _regime_size_mult = 1.0
        _regime_conf_bump = 0
        if getattr(config, 'REGIME_FILTER_ENABLED', True):
            if i - _regime_cache_state['last_idx'] >= 15 or _regime_cache_state['last_idx'] == -1:
                try:
                    _fast_df = {
                        'high': _highs[window_start:i],
                        'low': _lows[window_start:i],
                        'close': _closes[window_start:i]
                    }
                    _rg = ict_pro.regime_of(_fast_df, _bt_atr)
                    if _rg is not None:
                        _regime_cache_state['tradeable'] = _rg.get('tradeable', True)
                        _regime_cache_state['size_mult'] = _rg.get('size_mult', 1.0)
                        _regime_cache_state['conf_bump'] = _rg.get('conf_bump', 0)
                        new_label = _rg.get('label', 'trend')
                        
                        if getattr(config, 'REGIME_ADAPTIVE_PARAMS_ENABLED', False):
                            if new_label != _regime_cache_state.get('label'):
                                regime_params = rm.params_for_regime(new_label)
                                rm.apply_params(regime_params)
                                
                        _regime_cache_state['label'] = new_label
                    _regime_cache_state['last_idx'] = i
                except Exception:
                    pass
            if not _regime_cache_state['tradeable']:
                continue
            _regime_size_mult = _regime_cache_state['size_mult']
            _regime_conf_bump = _regime_cache_state['conf_bump']

        # Detect OBs on same sliding window as live (not full dataset) using binary search
        valid_obs_list = []
        ob_start_idx = bisect.bisect_left(_ob_bar_indices, window_start)
        ob_end_idx = bisect.bisect_right(_ob_bar_indices, i - 1)
        for idx in range(ob_start_idx, ob_end_idx):
            idx_time, ob_dict = global_obs_list[idx]
            if ob_dict['confirmation_bar'] < i:
                valid_obs_list.append((idx_time, ob_dict))
        has_obs = len(valid_obs_list) > 0

        # Detect Fair Value Gaps (ICT's highest-probability zones) using binary search
        all_fvgs = []
        fvgs = []
        fvg_start_idx = bisect.bisect_left(_fvg_bar_indices, window_start + 1)
        fvg_end_idx = bisect.bisect_right(_fvg_bar_indices, i - 1)
        for idx in range(fvg_start_idx, fvg_end_idx):
            fvg = global_all_fvgs[idx]
            all_fvgs.append(fvg)
            if fvg['filled_bar_idx'] >= i:
                fvgs.append(fvg)

        has_independent_methods = any(m in current_methods for m in ["Silver Bullet", "Judas Swing", "Breaker Block", "IOFR"])
        if not has_obs and not fvgs and not has_independent_methods:
            continue

        # Detect Market Structure (BOS/MSS)
        # Binary search to find valid swing points in window (replaces O(n) linear scan)
        sh_lo = bisect.bisect_left(_sh_indices, window_start + 5)
        sh_hi = bisect.bisect_left(_sh_indices, i - 5)
        swing_highs = global_swing_highs[sh_lo:sh_hi]
        sl_lo = bisect.bisect_left(_sl_indices, window_start + 5)
        sl_hi = bisect.bisect_left(_sl_indices, i - 5)
        swing_lows = global_swing_lows[sl_lo:sl_hi]

        # O(1) stateful structure and MSS lookups
        structure = global_structure_states[i - 1]
        mss = mss_lookup.get(i - 1, None)

        # O(log n) lookup for recent sweeps (within last 20 bars)
        bullish_sweep = False
        bearish_sweep = False
        if _sweep_bull_indices:
            _bs_idx = bisect.bisect_right(_sweep_bull_indices, i - 1) - 1
            if _bs_idx >= 0 and _sweep_bull_indices[_bs_idx] >= i - 20:
                bullish_sweep = True
        if _sweep_bear_indices:
            _br_idx = bisect.bisect_right(_sweep_bear_indices, i - 1) - 1
            if _br_idx >= 0 and _sweep_bear_indices[_br_idx] >= i - 20:
                bearish_sweep = True

        # Cache HTF trend (Fix #6) — only if HTF filter is enabled
        if use_htf_filter:
            cache_key = i // htf_cache_interval
            if cache_key not in cached_htf_trend:
                # Use prefetched HTF data instead of MT5 API calls
                if _htf_prefetched:
                    trend_counts = {"bullish": 0, "bearish": 0}
                    for htf_name, (htf_value, htf_df) in _htf_prefetched.items():
                        try:
                            # Get bars that have fully closed before current_time
                            tf_minutes_map = {
                                mt5.TIMEFRAME_H4: 240, mt5.TIMEFRAME_D1: 1440
                            }
                            tf_min = tf_minutes_map.get(htf_value, 60)
                            cutoff = current_time - pd.Timedelta(minutes=tf_min)
                            idx = htf_df.index.searchsorted(cutoff, side='right')
                            if idx < 20:
                                continue
                            valid_htf = htf_df.iloc[max(0, idx - 100):idx]
                            if valid_htf.empty or len(valid_htf) < 20:
                                continue
                            valid_htf = valid_htf.iloc[-100:]  # Last 100 bars max
                            ote_low, ote_high, _, _ = calculate_ote_zone(
                                valid_htf, lookback=50,
                                fib_low=ict_params['fib_low'], fib_high=ict_params['fib_high'])
                            htf_bull_sweep, htf_bear_sweep, _, _ = detect_liquidity_sweep(valid_htf)
                            current_close = valid_htf.iloc[-1]['close']
                            if htf_name == "D1":
                                daily_open = valid_htf.iloc[-1]['open']
                                if current_close < daily_open and htf_bear_sweep:
                                    trend_counts["bearish"] += 2
                                elif current_close > daily_open and htf_bull_sweep:
                                    trend_counts["bullish"] += 2
                                elif current_close > daily_open:
                                    trend_counts["bullish"] += 1
                                elif current_close < daily_open:
                                    trend_counts["bearish"] += 1
                            else:
                                if htf_bull_sweep:
                                    trend_counts["bullish"] += 1
                                elif htf_bear_sweep:
                                    trend_counts["bearish"] += 1
                                elif current_close > ote_high:
                                    trend_counts["bullish"] += 1
                                elif current_close < ote_low:
                                    trend_counts["bearish"] += 1
                        except Exception:
                            pass
                    if trend_counts["bullish"] > trend_counts["bearish"]:
                        cached_htf_trend[cache_key] = "bullish"
                    elif trend_counts["bearish"] > trend_counts["bullish"]:
                        cached_htf_trend[cache_key] = "bearish"
                    else:
                        cached_htf_trend[cache_key] = "neutral"
                else:
                    cached_htf_trend[cache_key] = get_higher_tf_trend(symbol, ict_params, date_to=current_time)
            higher_tf_trend = cached_htf_trend[cache_key]
        else:
            higher_tf_trend = "neutral"

        # O(1) OTE zone calculation using precalculated rolling extremes
        swing_low_50 = _rolling_low_50[i - 1]
        swing_high_50 = _rolling_high_50[i - 1]
        ote_low = swing_low_50 + ict_params['fib_low'] * (swing_high_50 - swing_low_50)
        ote_high = swing_low_50 + ict_params['fib_high'] * (swing_high_50 - swing_low_50)

        precalc_sweep_info = None

        # === PROCESS TOP OBs (not just nearest — more trades) ===
        if has_obs:
            # Sort by recency and take up to 3 most recent unprocessed OBs
            prev_bar_time = _timestamps[i - 1]
            valid_obs_list.sort(key=lambda x: abs((x[0] - prev_bar_time).total_seconds()))
            top_3 = valid_obs_list[:3]

            for ob_time, ob_dict in top_3:
                direction = ob_dict.get('order_block_type', 'bullish')

                # === ICT STRUCTURE FILTER (replaces blind HTF filter) ===
                # ICT Rule: Trade WITH structure, or on an MSS (reversal)
                structure_ok = False
                if structure == "neutral":
                    structure_ok = True  # No clear structure = allow trade
                elif (direction == "bullish" and structure == "bullish") or \
                     (direction == "bearish" and structure == "bearish"):
                    structure_ok = True  # Trading with structure
                elif mss:
                    # MSS = structure just shifted to our direction
                    if (direction == "bullish" and mss == "bullish") or \
                       (direction == "bearish" and mss == "bearish"):
                        structure_ok = True

                if not structure_ok:
                    continue  # Strict enforcement of market structure alignment
                    
                # Check displacement quality on the OB's confirmation candle
                has_displacement = False
                if 'confirmation_bar' in ob_dict and not pd.isna(ob_dict.get('confirmation_bar', np.nan)):
                    conf_bar = int(ob_dict['confirmation_bar'])
                    if "Mock" in type(check_displacement_quality).__name__ or "mock" in type(check_displacement_quality).__name__:
                        has_displacement = check_displacement_quality(df, conf_bar)
                    else:
                        if 0 <= conf_bar < len(_displacement_mask):
                            has_displacement = _displacement_mask[conf_bar]

                if not has_displacement:
                    continue  # Strict enforcement of displacement
                    
                trade_dir = "Buy" if direction == "bullish" else "Sell"

                for method in get_legacy_ob_retest_methods(current_methods):
                    if (ob_time, method) in processed_ob:
                        continue

                    m_min_rrr = resolve_method_param('min_rrr', method, sym_params, min_rrr)
                    m_use_dynamic_rrr = resolve_method_param('use_dynamic_rrr', method, sym_params, use_dynamic_rrr)
                    m_use_htf_filter = resolve_method_param('use_htf_filter', method, sym_params, use_htf_filter)
                    m_bypass_htf_conf = resolve_method_param('bypass_htf_conf', method, sym_params, bypass_htf_conf)
                    m_use_ote_filter = resolve_method_param('use_ote_filter', method, sym_params, use_ote_filter)
                    m_anti_gap_enabled = resolve_method_param('anti_gap_enabled', method, sym_params, anti_gap_enabled)
                    m_anti_gap_mult = resolve_method_param('anti_gap_mult', method, sym_params, anti_gap_mult)
                    m_trail_type = resolve_method_param('trail_type', method, sym_params, trail_type)
                    m_trail_pct = resolve_method_param('trail_pct', method, sym_params, trail_params.get('trail_pct', 0.5) if trail_params else 0.5)
                    m_trail_params = {'trail_pct': m_trail_pct}
                    m_limit_touch_fill = resolve_method_param('limit_touch_fill', method, sym_params, limit_touch_fill)

                    entry_price = compute_effective_entry(current_price, ob_dict, symbol, method, trade_dir, fixed_spread=current_spread)

                    # Fix 4: HTF POI nesting (parity, point-in-time via date_to)
                    if not ict_pro.passes_htf_poi(symbol, trade_dir, entry_price, date_to=current_time):
                        continue

                    confluence_score, confluence_details = calculate_confluence_score(
                        ob_dict, direction, fvgs, structure, mss,
                        bullish_sweep, bearish_sweep,
                        higher_tf_trend, ote_low, ote_high, entry_price,
                        df, has_displacement, entry_time=current_time,
                        eqh_pools=global_eqh_pools, eql_pools=global_eql_pools,
                        symbol=symbol)
                        
                    # 7. Volume Profile POC alignment
                    if use_volume_profile:
                        poc = get_current_vp_poc()
                        if poc is not None:
                            dist = abs(entry_price - poc)
                            max_dist = global_atr[i-1] * 2.0 if i > 0 else current_price * 0.0025
                            if dist <= max_dist:
                                confluence_score += 1
                                confluence_details.append("POC")
                            else:
                                continue # Hard filter: skip if VP enabled but not near POC
                    m_min_confluence_score = resolve_method_param('min_confluence_score', method, sym_params, config.MIN_CONFLUENCE_SCORE) + _regime_conf_bump
                    # ICT: Minimum confluence threshold (at least 2 factors must align)
                    # Score 0-1 = skip (random noise), 2+ = valid ICT setup
                    if confluence_score < m_min_confluence_score or "Mandatory Gate Failed" in confluence_details:
                        continue

                    if m_use_htf_filter and higher_tf_trend != "neutral":
                        if not (m_bypass_htf_conf and confluence_score >= 2):
                            if higher_tf_trend == "bullish" and direction != "bullish":
                                continue
                            elif higher_tf_trend == "bearish" and direction != "bearish":
                                continue

                    # OTE filter: strict application when enabled
                    if m_use_ote_filter:
                        if not (ote_low <= entry_price <= ote_high):
                            continue

                    # Simulated ask/bid calculation using actual spread
                    limit_touch_fill_val = kwargs.get('limit_touch_fill', True) if 'kwargs' in locals() else True
                    spread_fill_req = actual_spread_cost if limit_touch_fill_val else (actual_spread_cost * 2.0)
                    if trade_dir == "Buy" and current_bar_low > (entry_price - spread_fill_req):
                        continue
                    if trade_dir == "Sell" and current_bar_high < (entry_price + (0.0 if limit_touch_fill_val else actual_spread_cost)):
                        continue

                    # Apply dynamic slippage
                    slippage = slippage_points * point
                    if trade_dir == "Buy":
                        entry_price += slippage
                    else:
                        entry_price -= slippage

                    TP_default, SL = calculate_tp_sl(symbol, df, ob_dict, trade_dir, entry_price, point, min_rrr=m_min_rrr, fixed_spread=current_spread, precalc_sweep_info=precalc_sweep_info)

                    # === DYNAMIC RRR OR FIXED TP ===
                    if m_use_dynamic_rrr:
                        struct_tp = get_structure_tp(entry_price, trade_dir, swing_highs, swing_lows, current_spread)
                        if struct_tp is not None:
                            # Use the better of structure TP or minimum RRR default TP
                            if trade_dir == "Buy":
                                TP_default = max(TP_default, struct_tp)
                            else:
                                TP_default = min(TP_default, struct_tp)

                    # Adjust SL for min stop distance, then RECALCULATE TP to maintain RRR
                    if trade_dir == "Buy":
                        if (entry_price - SL) < min_stop_distance:
                            SL = entry_price - min_stop_distance - point
                        adjusted_risk = entry_price - SL
                        min_tp = entry_price + adjusted_risk * m_min_rrr
                        TP_default = max(TP_default, min_tp)
                        if (TP_default - entry_price) < min_stop_distance:
                            TP_default = entry_price + min_stop_distance + point
                    else:
                        if (SL - entry_price) < min_stop_distance:
                            SL = entry_price + min_stop_distance + point
                        adjusted_risk = SL - entry_price
                        min_tp = entry_price - adjusted_risk * m_min_rrr
                        TP_default = min(TP_default, min_tp)
                        if (entry_price - TP_default) < min_stop_distance:
                            TP_default = entry_price - min_stop_distance - point

                    # Anti-Gap SL Protection for OBs
                    if m_anti_gap_enabled:
                        atr_val = global_atr[i-1] if i > 0 else 0
                        if atr_val > 0:
                            req_dist = atr_val * m_anti_gap_mult
                            cur_dist = abs(entry_price - SL)
                            if cur_dist < req_dist:
                                SL = (entry_price - req_dist) if trade_dir == "Buy" else (entry_price + req_dist)
                                # Recalculate TP for min_rrr
                                if trade_dir == "Buy":
                                    TP_default = max(TP_default, entry_price + req_dist * m_min_rrr)
                                else:
                                    TP_default = min(TP_default, entry_price - req_dist * m_min_rrr)

                    # Fix 1+3: Draw-on-Liquidity TP (point-in-time via date_to)
                    _dol = ict_pro.dol_target(symbol, trade_dir, entry_price, SL, m_min_rrr, date_to=current_time)
                    if _dol is not None:
                        TP_default = _dol

                    rrr_ok, rrr = check_rrr(entry_price, TP_default, SL, trade_dir, m_min_rrr)
                    if not rrr_ok:
                        continue
                    risk = (entry_price - SL) if trade_dir == "Buy" else (SL - entry_price)
                    reward = (TP_default - entry_price) if trade_dir == "Buy" else (entry_price - TP_default)

                    use_ts = (trailing_methods is None) or (method in trailing_methods)
                    use_trailing_stop = use_ts
                    if m_trail_type is None or str(m_trail_type).upper() == "NONE" or m_trail_type == "None (Disabled)":
                        use_trailing_stop = False
                        
                    if use_trailing_stop:
                        initial_risk_pts = abs(risk)
                    else:
                        initial_risk_pts = 0

                    exit_price, actual_entry_time, exit_time, outcome = run_trade_simulation(
                        df, i, trade_dir, entry_price, TP_default, SL,
                        use_trailing_stop=use_trailing_stop, initial_risk=initial_risk_pts,
                        symbol=symbol, timeframe_value=timeframe_value, m1_global_df=m1_global_df, fixed_spread=actual_spread_cost,
                        trail_type=m_trail_type, trail_params=m_trail_params, ict_method=method, limit_touch_fill=m_limit_touch_fill)

                    if outcome == "SL":
                        slippage_val = slippage_points * point
                        if trade_dir == "Buy":
                            exit_price -= slippage_val
                        else:
                            exit_price += slippage_val

                    # Calculate duration
                    duration = exit_time - actual_entry_time if isinstance(exit_time, pd.Timestamp) and isinstance(actual_entry_time, pd.Timestamp) else pd.Timedelta(0)

                    _digits = getattr(info, 'digits', None)
                    if _digits is None:
                        _digits = int(round(-np.log10(point)))
                    entry_price = round(entry_price, _digits)
                    SL = round(SL, _digits)
                    TP_default = round(TP_default, _digits)

                    trade = {
                        "symbol": symbol, "timeframe": timeframe_name,
                        "entry_time": actual_entry_time, "entry_price": entry_price,
                        "exit_time": exit_time, "exit_price": exit_price,
                        "tp_price": TP_default, "sl_price": SL,
                        "risk": risk, "reward": reward, "rrr": rrr,
                        "order_block_type": direction,
                        "htf_trend": higher_tf_trend,
                        "outcome": outcome,
                        "duration": str(duration),
                        "order_block": {k: v for k, v in ob_dict.items() if k not in ['confirmation_bar', 'ob_bar_index']},
                        "regime_size_mult": _regime_size_mult,
                        "ict_method": method, "trade_direction": trade_dir,
                        "commission_per_lot": commission_per_lot,
                        "confluence_score": confluence_score,
                        "confluence_details": ", ".join(confluence_details),
                    }

                    # Date bounds guard: reject trades with entry/exit outside backtest range
                    if isinstance(actual_entry_time, pd.Timestamp) and actual_entry_time < pd.Timestamp(date_from) - pd.Timedelta(days=1):
                        continue
                    if isinstance(exit_time, pd.Timestamp) and exit_time > pd.Timestamp(date_to) + pd.Timedelta(days=30):
                        continue

                    completed_trades.append(trade)
                    processed_ob.add((ob_time, method))

        # === FVG-ONLY TRADES (new signal source — more trades) ===
        # ICT: Unmitigated FVGs are independent high-probability zones
        if fvgs and "FVG Return" in current_methods:
            for fvg in fvgs:
                fvg_key = (f"{symbol}_{fvg['time']}_{fvg['type']}", "FVG Return")
                if fvg_key in processed_ob:
                    continue

                method_name = "FVG Return"
                m_require_bos_fvg = resolve_method_param('require_bos_fvg', method_name, sym_params, require_bos_fvg)
                m_use_htf_filter = resolve_method_param('use_htf_filter', method_name, sym_params, use_htf_filter)
                m_fvg_displacement_only = resolve_method_param('fvg_displacement_only', method_name, sym_params, fvg_displacement_only)
                m_fvg_discount_premium_only = resolve_method_param('fvg_discount_premium_only', method_name, sym_params, fvg_discount_premium_only)
                m_fvg_recent_sweep_only = resolve_method_param('fvg_recent_sweep_only', method_name, sym_params, fvg_recent_sweep_only)
                
                # Apply global mandatory sweep+dis gate to FVG trades if enabled
                if getattr(config, 'CONFLUENCE_REQUIRE_MANDATORY', False):
                    m_fvg_displacement_only = True
                    m_fvg_recent_sweep_only = True
                m_fvg_sl_mode = resolve_method_param('fvg_sl_mode', method_name, sym_params, fvg_sl_mode)
                m_anti_gap_enabled = resolve_method_param('anti_gap_enabled', method_name, sym_params, anti_gap_enabled)
                m_anti_gap_mult = resolve_method_param('anti_gap_mult', method_name, sym_params, anti_gap_mult)
                m_min_rrr = resolve_method_param('min_rrr', method_name, sym_params, min_rrr)
                m_use_dynamic_rrr = resolve_method_param('use_dynamic_rrr', method_name, sym_params, use_dynamic_rrr)
                m_trail_type = resolve_method_param('trail_type', method_name, sym_params, trail_type)
                m_trail_pct = resolve_method_param('trail_pct', method_name, sym_params, trail_params.get('trail_pct', 0.5) if trail_params else 0.5)
                m_trail_params = {'trail_pct': m_trail_pct}
                m_fvg_sl_spread_buffer = resolve_method_param('fvg_sl_spread_buffer', method_name, sym_params, fvg_sl_spread_buffer)
                m_limit_touch_fill = resolve_method_param('limit_touch_fill', method_name, sym_params, limit_touch_fill)

                fvg_dir = fvg['type']  # 'bullish' or 'bearish'
                trade_dir = "Buy" if fvg_dir == "bullish" else "Sell"

                # Strict MTF Alignment
                m_sb_require_htf_bias = resolve_method_param('sb_require_htf_bias', method_name, sym_params, sb_require_htf_bias)
                if m_sb_require_htf_bias and htf_bias_direction != "neutral" and trade_dir != htf_bias_direction:
                    continue

                # Structure filter for FVG trades
                if structure != "neutral":
                    if (fvg_dir == "bullish" and structure == "bearish") or \
                       (fvg_dir == "bearish" and structure == "bullish"):
                        if not mss:  # Only skip if no MSS supports direction
                            continue

                # Require BoS after MSS
                if m_require_bos_fvg:
                    bos_confirmed = bos_confirmed_after_mss_buy[i - 1] if trade_dir == "Buy" else bos_confirmed_after_mss_sell[i - 1]
                    if not bos_confirmed:
                        continue

                m_use_ote_filter = resolve_method_param('use_ote_filter', method_name, sym_params, use_ote_filter)

                # HTF filter for FVG trades
                if m_use_htf_filter and higher_tf_trend != "neutral":
                    if (fvg_dir == "bullish" and higher_tf_trend == "bearish") or \
                       (fvg_dir == "bearish" and higher_tf_trend == "bullish"):
                        continue

                # OTE filter for FVG trades
                if m_use_ote_filter:
                    if not (ote_low <= fvg['mid'] <= ote_high):
                        continue

                # FVG Advanced SMC/ICT Filters
                if m_fvg_displacement_only:
                    if fvg.get('body_ratio', 1.0) < 0.5 or fvg.get('atr_mult', 1.0) < 0.3:
                        continue

                if m_fvg_discount_premium_only:
                    # O(1) lookup using precalculated rolling high/low
                    if i > 0:
                        swing_h = _rolling_high_50[i - 1]
                        swing_l = _rolling_low_50[i - 1]
                        midpoint = (swing_h + swing_l) / 2.0
                        if trade_dir == "Buy" and fvg['mid'] >= midpoint:
                            continue
                        elif trade_dir == "Sell" and fvg['mid'] <= midpoint:
                            continue

                if m_fvg_recent_sweep_only:
                    # O(log n) lookup using precalculated global sweeps + binary search
                    has_recent_sweep = False
                    target_indices = _sweep_bull_indices if trade_dir == "Buy" else _sweep_bear_indices
                    # Find the rightmost sweep index <= i that is within 20 bars
                    pos = bisect.bisect_right(target_indices, i) - 1
                    if pos >= 0 and (i - target_indices[pos]) <= 20:
                        has_recent_sweep = True
                    if not has_recent_sweep:
                        continue

                # Entry at FVG midpoint (ICT consequent encroachment)
                entry_price = fvg['mid']

                # Fix #16: Filter out tiny FVGs (noise) — they have tight SLs that get hit easily
                m_min_fvg_size = resolve_method_param('min_fvg_size', method_name, sym_params, config.MIN_FVG_SIZE_SPREADS)
                if m_min_fvg_size > 0 and fvg['size'] < _cached_spread * m_min_fvg_size:
                    continue

                # Check if price reached the FVG on this bar
                if trade_dir == "Buy" and current_bar_low > entry_price - actual_spread_cost:
                    continue
                if trade_dir == "Sell" and current_bar_high < entry_price:
                    continue

                # Apply dynamic slippage
                slippage = slippage_points * point
                entry_price += slippage if trade_dir == "Buy" else -slippage

                # SL beyond the FVG zone + buffer
                buffer = _cached_spread * m_fvg_sl_spread_buffer
                SL = compute_fvg_sl_price(fvg, trade_dir, m_fvg_sl_mode, df, swing_highs, swing_lows, buffer, _cached_spread,
                                          global_i=i, global_sweeps=global_sweeps, global_sweep_idxs=global_sweep_idxs)
                
                # Anti-Gap SL Protection for FVGs
                if m_anti_gap_enabled:
                    atr_val = global_atr[i-1] if i > 0 else 0
                    if atr_val > 0:
                        req_dist = atr_val * m_anti_gap_mult
                        cur_dist = abs(entry_price - SL)
                        if cur_dist < req_dist:
                            SL = (entry_price - req_dist) if trade_dir == "Buy" else (entry_price + req_dist)

                if trade_dir == "Buy":
                    risk_pts = entry_price - SL
                else:
                    risk_pts = SL - entry_price

                if risk_pts <= 0:
                    continue

                # Fix 4: HTF POI nesting (parity, point-in-time via date_to)
                if not ict_pro.passes_htf_poi(symbol, trade_dir, entry_price, date_to=current_time):
                    continue

                # TP: structure-based or fixed
                TP_default = (entry_price + risk_pts * m_min_rrr if trade_dir == "Buy" else entry_price - risk_pts * m_min_rrr)
                if m_use_dynamic_rrr:
                    struct_tp = get_structure_tp(entry_price, trade_dir, swing_highs, swing_lows, current_spread)
                    if struct_tp is not None:
                        TP_default = max(TP_default, struct_tp) if trade_dir == "Buy" else min(TP_default, struct_tp)

                # Fix 1+3: Draw-on-Liquidity TP (point-in-time via date_to)
                _dol = ict_pro.dol_target(symbol, trade_dir, entry_price, SL, m_min_rrr, date_to=current_time)
                if _dol is not None:
                    TP_default = _dol

                # Enforce min RRR
                rrr_ok, rrr_val = check_rrr(entry_price, TP_default, SL, trade_dir, m_min_rrr)
                if not rrr_ok:
                    continue

                # === Institutional Filters (must match independent method loop) ===
                # SMT Divergence hard filter
                if use_smt_divergence and corr_dfs_dict:
                    fvg_direction = "bullish" if trade_dir == "Buy" else "bearish"
                    has_smt = check_advanced_smt(df, corr_dfs_dict, i, fvg_direction, lookback=20)
                    if not has_smt:
                        continue

                # Volume Profile POC hard filter was removed here and moved to confluence below



                risk = risk_pts
                reward = abs(TP_default - entry_price)

                use_ts = (trailing_methods is None) or ("FVG Return" in trailing_methods)
                use_trailing_stop = use_ts
                if m_trail_type is None or str(m_trail_type).upper() == "NONE" or m_trail_type == "None (Disabled)":
                    use_trailing_stop = False
                    
                if use_trailing_stop:
                    initial_risk_pts = abs(risk)
                else:
                    initial_risk_pts = 0

                exit_price, actual_entry_time, exit_time, outcome = run_trade_simulation(
                    df, i, trade_dir, entry_price, TP_default, SL,
                    use_trailing_stop=use_trailing_stop, initial_risk=initial_risk_pts,
                    symbol=symbol, timeframe_value=timeframe_value, m1_global_df=m1_global_df, fixed_spread=actual_spread_cost,
                    trail_type=m_trail_type, trail_params=m_trail_params, ict_method="FVG Return", limit_touch_fill=m_limit_touch_fill)

                if outcome == "SL":
                    slippage_val = slippage_points * point
                    if trade_dir == "Buy":
                        exit_price -= slippage_val
                    else:
                        exit_price += slippage_val

                duration = exit_time - actual_entry_time if isinstance(exit_time, pd.Timestamp) and isinstance(actual_entry_time, pd.Timestamp) else pd.Timedelta(0)

                # Calculate dynamic confluence score for FVG Return
                fvg_conf_score = 0
                fvg_conf_details = []
                fvg_sig_dir = "bullish" if trade_dir == "Buy" else "bearish"
                
                # 1. Structure alignment
                if structure == fvg_sig_dir:
                    fvg_conf_score += 2
                    fvg_conf_details.append("Structure")
                
                # 2. HTF Trend alignment
                if higher_tf_trend == fvg_sig_dir:
                    fvg_conf_score += 1
                    fvg_conf_details.append("HTF")
                    
                # 3. OTE Zone alignment
                if ote_low <= entry_price <= ote_high:
                    fvg_conf_score += 1
                    fvg_conf_details.append("OTE")
                    
                # 4. Sweep component
                if (trade_dir == "Buy" and bullish_sweep) or (trade_dir == "Sell" and bearish_sweep):
                    fvg_conf_score += 2
                    fvg_conf_details.append("Sweep")
                    
                # 5. POC Alignment
                if use_volume_profile:
                    poc = get_current_vp_poc()
                    if poc is not None:
                        dist = abs(entry_price - poc)
                        max_dist = global_atr[i-1] * 2.0 if i > 0 else current_price * 0.0025
                        if dist <= max_dist:
                            fvg_conf_score += 1
                            fvg_conf_details.append("POC")
                    
                # 5. Displacement
                if fvg.get('body_ratio', 1.0) >= 0.6:
                    fvg_conf_score += 1
                    fvg_conf_details.append("Displacement")
                    
                # 6. Multi-TF Gate & Confluence
                try:
                    from multi_tf import hierarchical_confluence
                    mtf_result = hierarchical_confluence(symbol, fvg_sig_dir, date_to=current_time) if hierarchical_confluence else {}
                    if getattr(config, 'MULTI_TF_CONFLUENCE_ENABLED', False):
                        aligned = mtf_result.get('score', 0)
                        total_layers = len(mtf_result.get('layers', []))
                        if aligned > 0:
                            fvg_conf_score += min(aligned, 3)
                            fvg_conf_details.append(f"MultiTF({aligned}/{total_layers})")
                    if getattr(config, 'MULTI_TF_GATE_ENABLED', False) and mtf_result.get('gated', False):
                        continue # Multi-TF Gate Failed
                except Exception:
                    pass

                if fvg_conf_score == 0:
                    fvg_conf_score = 1
                    fvg_conf_details.append("FVG Imbalance")
                    
                m_min_confluence_score = resolve_method_param('min_confluence_score', "FVG Return", sym_params, config.MIN_CONFLUENCE_SCORE) + _regime_conf_bump
                if fvg_conf_score < m_min_confluence_score:
                    continue
                    
                fvg_conf_details_str = ", ".join(fvg_conf_details)

                idx = fvg['bar_index']
                disp_high = _highs[idx]
                disp_low = _lows[idx]
                disp_open = _opens[idx]
                disp_close = _closes[idx]
                disp_range = max(disp_high - disp_low, 1e-8)
                
                fvg_upper_wick_ratio = (disp_high - max(disp_open, disp_close)) / disp_range
                fvg_lower_wick_ratio = (min(disp_open, disp_close) - disp_low) / disp_range
                fvg_gap_ratio = fvg['size'] / disp_range
                
                fvg_hour_ny = convert_time_to_ny_hour(fvg['time'])
                fvg_is_sb_window = 1.0 if (fvg_hour_ny == 10) or (fvg_hour_ny == 14) or (fvg_hour_ny == 3) else 0.0
                
                if idx + 1 < len(df):
                    c3_high = _highs[idx + 1]
                    c3_low = _lows[idx + 1]
                    c3_open = _opens[idx + 1]
                    c3_close = _closes[idx + 1]
                    c3_range = max(c3_high - c3_low, 1e-8)
                    fvg_c3_body_ratio = abs(c3_close - c3_open) / c3_range
                else:
                    fvg_c3_body_ratio = 0.5

                trade = {
                    "symbol": symbol, "timeframe": timeframe_name,
                    "entry_time": actual_entry_time, "entry_price": entry_price,
                    "exit_time": exit_time, "exit_price": exit_price,
                    "tp_price": TP_default, "sl_price": SL,
                    "risk": risk, "reward": reward, "rrr": rrr_val,
                    "order_block_type": fvg_dir,
                    "htf_trend": higher_tf_trend,
                    "outcome": outcome,
                    "duration": str(duration),
                    "order_block": {"type": "FVG", "top": fvg['top'], "bottom": fvg['bottom'], "mid": fvg['mid']},
                    "regime_size_mult": _regime_size_mult,
                    "ict_method": "FVG Return", "trade_direction": trade_dir,
                    "commission_per_lot": commission_per_lot,
                    "confluence_score": fvg_conf_score,
                    "confluence_details": fvg_conf_details_str,
                    "fvg_size": fvg['size'],
                    "fvg_body_ratio": fvg.get('body_ratio', 1.0),
                    "fvg_atr_mult": fvg.get('atr_mult', 1.0),
                    "fvg_upper_wick_ratio": fvg_upper_wick_ratio,
                    "fvg_lower_wick_ratio": fvg_lower_wick_ratio,
                    "fvg_gap_ratio": fvg_gap_ratio,
                    "fvg_is_sb_window": fvg_is_sb_window,
                    "fvg_c3_body_ratio": fvg_c3_body_ratio,
                }

                if isinstance(actual_entry_time, pd.Timestamp) and actual_entry_time < pd.Timestamp(date_from) - pd.Timedelta(days=1):
                    continue
                if isinstance(exit_time, pd.Timestamp) and exit_time > pd.Timestamp(date_to) + pd.Timedelta(days=30):
                    continue
                if outcome == "CANCELLED":
                    continue

                completed_trades.append(trade)
                processed_ob.add(fvg_key)

        # === INDEPENDENT METHOD DETECTORS ===
        real_signals = []
        current_atr = global_atr[i-1] if i > 0 else (global_atr[0] if len(global_atr) > 0 else _cached_spread * 5)

        # Always define sb_htf_bias_val: the MMXM / ICT Model 2022 / 2025 detector
        # blocks below reference it via use_mtf_alignment even when "Silver Bullet"
        # is not among the selected methods (previously an UnboundLocalError).
        sb_htf_bias_val = resolve_method_param('sb_require_htf_bias', 'Silver Bullet', sym_params, sb_require_htf_bias)

        if "Silver Bullet" in current_methods:
            sb_fvg_sl_mode = resolve_method_param('fvg_sl_mode', 'Silver Bullet', sym_params, fvg_sl_mode)
            sb_fvg_displacement_only = resolve_method_param('fvg_displacement_only', 'Silver Bullet', sym_params, fvg_displacement_only)
            sb_fvg_discount_premium_only = resolve_method_param('fvg_discount_premium_only', 'Silver Bullet', sym_params, fvg_discount_premium_only)
            sb_fvg_recent_sweep_only = resolve_method_param('fvg_recent_sweep_only', 'Silver Bullet', sym_params, fvg_recent_sweep_only)
            sb_htf_bias_val = resolve_method_param('sb_require_htf_bias', 'Silver Bullet', sym_params, sb_require_htf_bias)
            
            real_signals.extend(detect_silver_bullet_signals(
                df, current_time, fvgs, structure, mss, spread=_cached_spread, broker_tz_offset=0,
                fvg_sl_mode=sb_fvg_sl_mode,
                fvg_displacement_only=sb_fvg_displacement_only,
                fvg_discount_premium_only=sb_fvg_discount_premium_only,
                fvg_recent_sweep_only=sb_fvg_recent_sweep_only,
                sb_require_htf_bias=sb_htf_bias_val,
                swing_highs=swing_highs, swing_lows=swing_lows,
                analysis_df=df, global_i=i,
                rolling_high_50=_rolling_high_50, rolling_low_50=_rolling_low_50,
                sweep_bull_indices=_sweep_bull_indices, sweep_bear_indices=_sweep_bear_indices,
                global_sweeps=global_sweeps, global_sweep_idxs=global_sweep_idxs,
                htf_bias_direction=htf_bias_direction
            ))
        if "Judas Swing" in current_methods:
            real_signals.extend(detect_judas_swing(
                df, current_time, structure, spread=_cached_spread, window=15, broker_tz_offset=0,
                swing_highs=swing_highs, swing_lows=swing_lows, atr_val=current_atr,
                global_i=i, rolling_high_50=_rolling_high_50, rolling_low_50=_rolling_low_50
            ))
        if "Breaker Block" in current_methods:
            real_signals.extend(detect_breaker_blocks(
                df, global_obs_df, spread=_cached_spread, current_bar_idx=i, atr_val=current_atr,
                global_i=i, rolling_high_50=_rolling_high_50, rolling_low_50=_rolling_low_50
            ))
        if "IOFR" in current_methods:
            real_signals.extend(detect_iofr_signals(
                df, spread=_cached_spread,
                swing_highs=swing_highs, swing_lows=swing_lows, atr_val=current_atr,
                global_i=i
            ))
        if "MMXM" in current_methods:
            real_signals.extend(detect_mmxm_signals(
                df, current_time, fvgs, structure, mss, spread=_cached_spread,
                swing_highs=swing_highs, swing_lows=swing_lows, analysis_df=df, global_i=i,
                rolling_high_50=_rolling_high_50, rolling_low_50=_rolling_low_50,
                sweep_bull_indices=_sweep_bull_indices, sweep_bear_indices=_sweep_bear_indices,
                global_sweeps=global_sweeps, global_sweep_idxs=global_sweep_idxs,
                atr_val=current_atr,
                use_mtf_alignment=sb_htf_bias_val, htf_bias_direction=htf_bias_direction
            ))
        if "ICT Model 2022" in current_methods:
            real_signals.extend(detect_ict_model_signals(
                df, current_time, fvgs, structure, mss, spread=_cached_spread, model_version="2022",
                swing_highs=swing_highs, swing_lows=swing_lows, analysis_df=df, global_i=i,
                rolling_high_50=_rolling_high_50, rolling_low_50=_rolling_low_50,
                sweep_bull_indices=_sweep_bull_indices, sweep_bear_indices=_sweep_bear_indices,
                global_sweeps=global_sweeps, global_sweep_idxs=global_sweep_idxs,
                atr_val=current_atr,
                use_mtf_alignment=sb_htf_bias_val, htf_bias_direction=htf_bias_direction
            ))
        if "ICT Model 2025" in current_methods:
            real_signals.extend(detect_ict_model_signals(
                df, current_time, fvgs, structure, mss, spread=_cached_spread, model_version="2025",
                swing_highs=swing_highs, swing_lows=swing_lows, analysis_df=df, global_i=i,
                rolling_high_50=_rolling_high_50, rolling_low_50=_rolling_low_50,
                sweep_bull_indices=_sweep_bull_indices, sweep_bear_indices=_sweep_bear_indices,
                global_sweeps=global_sweeps, global_sweep_idxs=global_sweep_idxs,
                atr_val=current_atr,
                use_mtf_alignment=sb_htf_bias_val, htf_bias_direction=htf_bias_direction
            ))

        # Fix 7: rank candidate setups so the best ideas claim limited slots (parity with live)
        for _s in real_signals:
            try:
                _sd = "bullish" if _s.get('trade_dir') == "Buy" else "bearish"
                _s['_rank'] = (2 if structure == _sd else 0) + (1 if higher_tf_trend == _sd else 0)
            except Exception:
                _s['_rank'] = 0
        real_signals = ict_pro.rank_setups(real_signals, prob_key='_rank', conf_key='_rank')

        for sig in real_signals:
            method = sig['method']
            m_use_htf_filter = resolve_method_param('use_htf_filter', method, sym_params, use_htf_filter)
            m_anti_gap_enabled = resolve_method_param('anti_gap_enabled', method, sym_params, anti_gap_enabled)
            m_anti_gap_mult = resolve_method_param('anti_gap_mult', method, sym_params, anti_gap_mult)
            m_min_rrr = resolve_method_param('min_rrr', method, sym_params, min_rrr)
            m_use_dynamic_rrr = resolve_method_param('use_dynamic_rrr', method, sym_params, use_dynamic_rrr)
            m_trail_type = resolve_method_param('trail_type', method, sym_params, trail_type)
            m_trail_pct = resolve_method_param('trail_pct', method, sym_params, trail_params.get('trail_pct', 0.5) if trail_params else 0.5)
            m_trail_params = {'trail_pct': m_trail_pct}
            m_limit_touch_fill = resolve_method_param('limit_touch_fill', method, sym_params, limit_touch_fill)

            trade_dir = sig['trade_dir']
            entry_price = sig['entry']
            sl_price = sig['sl']
            
            # Apply dynamic slippage
            slippage = slippage_points * point
            if trade_dir == "Buy":
                entry_price += slippage
            else:
                entry_price -= slippage
            
            # --- SMT Divergence Filter ---
            if use_smt_divergence and corr_dfs_dict:
                # Use signal_idx which is the index of the swing point in the dataframe
                signal_idx = sig.get('global_i', i)
                direction = "bullish" if trade_dir == "Buy" else "bearish"
                has_smt = check_advanced_smt(df, corr_dfs_dict, signal_idx, direction, lookback=20)
                if not has_smt:
                    continue
                    
            if use_volume_profile:
                poc = get_current_vp_poc()
                if poc is not None:
                    sig['vp_poc'] = poc

            # HTF filter
            if m_use_htf_filter and higher_tf_trend != "neutral":
                if (trade_dir == "Buy" and higher_tf_trend == "bearish") or \
                   (trade_dir == "Sell" and higher_tf_trend == "bullish"):
                    continue

            # OTE filter
            m_use_ote_filter = resolve_method_param('use_ote_filter', method, sym_params, use_ote_filter)
            if m_use_ote_filter:
                if not (ote_low <= entry_price <= ote_high):
                    continue

            # Fix 4: HTF POI nesting (parity, point-in-time via date_to)
            if not ict_pro.passes_htf_poi(symbol, trade_dir, entry_price, date_to=current_time):
                continue

            # Target TP
            struct_tp = get_structure_tp(entry_price, trade_dir, swing_highs, swing_lows, _cached_spread)
            
            # Anti-Gap SL Protection for signals
            if m_anti_gap_enabled:
                atr_val = global_atr[i-1] if i > 0 else 0
                if atr_val > 0:
                    req_dist = atr_val * m_anti_gap_mult
                    cur_dist = abs(entry_price - sl_price)
                    if cur_dist < req_dist:
                        sl_price = (entry_price - req_dist) if trade_dir == "Buy" else (entry_price + req_dist)

            risk_pts = abs(entry_price - sl_price)
            tp_price = entry_price + risk_pts * m_min_rrr if trade_dir == "Buy" else entry_price - risk_pts * m_min_rrr
            if m_use_dynamic_rrr and struct_tp is not None:
                tp_price = max(tp_price, struct_tp) if trade_dir == "Buy" else min(tp_price, struct_tp)

            # Fix 1+3: Draw-on-Liquidity TP (point-in-time via date_to)
            _dol = ict_pro.dol_target(symbol, trade_dir, entry_price, sl_price, m_min_rrr, date_to=current_time)
            if _dol is not None:
                tp_price = _dol

            rrr_ok, rrr_val = check_rrr(entry_price, tp_price, sl_price, trade_dir, m_min_rrr)
            if not rrr_ok:
                continue

            use_ts = (trailing_methods is None) or (method in trailing_methods)
            use_trailing_stop = use_ts
            if m_trail_type is None or str(m_trail_type).upper() == "NONE" or m_trail_type == "None (Disabled)":
                use_trailing_stop = False
                
            initial_risk_pts = risk_pts if use_trailing_stop else 0

            # SL, TP, Entry logic is handled by standard simulation
            exit_price, actual_entry_time, exit_time, outcome = run_trade_simulation(
                df, i, trade_dir, entry_price, tp_price, sl_price,
                use_trailing_stop=use_ts, initial_risk=initial_risk_pts,
                symbol=symbol, timeframe_value=timeframe_value, m1_global_df=m1_global_df, fixed_spread=actual_spread_cost,
                trail_type=m_trail_type, trail_params=m_trail_params, ict_method=method, limit_touch_fill=m_limit_touch_fill)

            duration = exit_time - actual_entry_time if isinstance(exit_time, pd.Timestamp) and isinstance(actual_entry_time, pd.Timestamp) else pd.Timedelta(0)

            # Calculate dynamic confluence score for independent strategies
            conf_score = 0
            conf_details = []
            sig_dir = "bullish" if trade_dir == "Buy" else "bearish"
            
            # 1. Sweep component
            is_sweep_strategy = method in ["MMXM", "ICT Model 2022", "ICT Model 2025", "Judas Swing", "IOFR"]
            if is_sweep_strategy or (trade_dir == "Buy" and bullish_sweep) or (trade_dir == "Sell" and bearish_sweep):
                conf_score += 2
                conf_details.append("Sweep")
                
            # 2. MSS component
            is_mss_strategy = method in ["MMXM", "ICT Model 2022", "ICT Model 2025", "IOFR"]
            if is_mss_strategy or (mss and mss == sig_dir):
                conf_score += 2
                conf_details.append("MSS")
                
            # 3. Structure alignment
            if structure == sig_dir:
                conf_score += 2
                conf_details.append("Structure")
                
            # 4. HTF Trend alignment
            if higher_tf_trend == sig_dir:
                conf_score += 1
                conf_details.append("HTF")
                
            # 5. OTE Zone alignment
            if ote_low <= entry_price <= ote_high:
                conf_score += 1
                conf_details.append("OTE")
                
            # 6. Displacement
            if method in ["MMXM", "ICT Model 2022", "ICT Model 2025", "Breaker Block", "IOFR"]:
                conf_score += 1
                conf_details.append("Displacement")
            elif sig.get('is_displacement') or sig.get('is_displacement_quality'):
                conf_score += 1
                conf_details.append("Displacement")
                
            if use_volume_profile and sig.get('vp_poc') is not None:
                dist = abs(entry_price - sig['vp_poc'])
                max_dist = global_atr[i-1] * 2.0 if i > 0 else current_price * 0.0025
                if dist <= max_dist:
                    conf_score += 1
                    conf_details.append("POC")
                else:
                    continue # Hard filter: if VP is enabled, trades must be near POC
            if conf_score == 0:
                conf_score = 2
                conf_details.append("SMC Pattern")
                
            # 7. Multi-TF Gate & Confluence
            try:
                from multi_tf import hierarchical_confluence
                mtf_result = hierarchical_confluence(symbol, sig_dir, date_to=current_time) if hierarchical_confluence else {}
                if getattr(config, 'MULTI_TF_CONFLUENCE_ENABLED', False):
                    aligned = mtf_result.get('score', 0)
                    total_layers = len(mtf_result.get('layers', []))
                    if aligned > 0:
                        conf_score += min(aligned, 3)
                        conf_details.append(f"MultiTF({aligned}/{total_layers})")
                if getattr(config, 'MULTI_TF_GATE_ENABLED', False) and mtf_result.get('gated', False):
                    continue # Multi-TF Gate Failed
            except Exception:
                pass

            m_min_confluence_score = resolve_method_param('min_confluence_score', method, sym_params, config.MIN_CONFLUENCE_SCORE) + _regime_conf_bump
            if conf_score < m_min_confluence_score:
                continue

            conf_details_str = ", ".join(conf_details)

            fvg_dict = sig.get('fvg', {}) if isinstance(sig.get('fvg'), dict) else {}
            
            # Extract FVG shape pattern features if fvg_dict is present
            fvg_upper_wick_ratio = 0.1
            fvg_lower_wick_ratio = 0.1
            fvg_gap_ratio = 0.3
            fvg_is_sb_window = 0.0
            fvg_c3_body_ratio = 0.5
            
            if fvg_dict and 'bar_index' in fvg_dict:
                idx = fvg_dict['bar_index']
                if idx < len(df):
                    disp_high = _highs[idx]
                    disp_low = _lows[idx]
                    disp_open = _opens[idx]
                    disp_close = _closes[idx]
                    disp_range = max(disp_high - disp_low, 1e-8)
                    
                    fvg_upper_wick_ratio = (disp_high - max(disp_open, disp_close)) / disp_range
                    fvg_lower_wick_ratio = (min(disp_open, disp_close) - disp_low) / disp_range
                    fvg_gap_ratio = fvg_dict['size'] / disp_range
                    
                    fvg_hour_ny = convert_time_to_ny_hour(fvg_dict['time'])
                    fvg_is_sb_window = 1.0 if (fvg_hour_ny == 10) or (fvg_hour_ny == 14) or (fvg_hour_ny == 3) else 0.0
                    
                    if idx + 1 < len(df):
                        c3_high = _highs[idx + 1]
                        c3_low = _lows[idx + 1]
                        c3_open = _opens[idx + 1]
                        c3_close = _closes[idx + 1]
                        c3_range = max(c3_high - c3_low, 1e-8)
                        fvg_c3_body_ratio = abs(c3_close - c3_open) / c3_range
            
            trade = {
                "symbol": symbol, "timeframe": timeframe_name,
                "entry_time": actual_entry_time, "entry_price": entry_price,
                "exit_time": exit_time, "exit_price": exit_price,
                "tp_price": tp_price, "sl_price": sl_price,
                "risk": risk_pts, "reward": abs(tp_price - entry_price), "rrr": rrr_val,
                "order_block_type": sig['type'],
                "htf_trend": higher_tf_trend,
                "outcome": outcome,
                "duration": str(duration),
                "order_block": {"type": method},
                "regime_size_mult": _regime_size_mult,
                "ict_method": method, "trade_direction": trade_dir,
                "commission_per_lot": commission_per_lot,
                "confluence_score": conf_score,
                "confluence_details": conf_details_str,
                "fvg_size": fvg_dict.get('size', 0.0),
                "fvg_body_ratio": fvg_dict.get('body_ratio', 1.0),
                "fvg_atr_mult": fvg_dict.get('atr_mult', 1.0),
                "fvg_upper_wick_ratio": fvg_upper_wick_ratio,
                "fvg_lower_wick_ratio": fvg_lower_wick_ratio,
                "fvg_gap_ratio": fvg_gap_ratio,
                "fvg_is_sb_window": fvg_is_sb_window,
                "fvg_c3_body_ratio": fvg_c3_body_ratio,
            }
            if isinstance(actual_entry_time, pd.Timestamp) and actual_entry_time < pd.Timestamp(date_from) - pd.Timedelta(days=1):
                continue
            if isinstance(exit_time, pd.Timestamp) and exit_time > pd.Timestamp(date_to) + pd.Timedelta(days=30):
                continue
            if outcome == "CANCELLED":
                continue

            completed_trades.append(trade)

    return completed_trades


def combined_backtest(symbols, date_from, date_to, initial_balance, risk_percent, fixed_lot, risk_mode,
                      trailing_methods, ict_params,
                      ict_method="ICT Retest", min_rrr=1.5, use_dynamic_rrr=True,
                      trade_on_all_tfs=True, use_ultra_low_tf=False, fvg_sl_mode=FVG_SL_NORMAL,
                      spread_cost=0.0, slippage_points=0, commission_per_lot=0.0,
                      session_filter=True, session_start=7, session_end=21,
                      progress_callback=None, use_htf_filter=True, use_ote_filter=True, bypass_htf_conf=False,
                      trail_type=TRAIL_TYPE_PARTIAL, trail_params=None, require_bos_fvg=False, enable_slippage_recovery=False,
                      anti_gap_enabled=True, anti_gap_mult=2.0, fvg_sl_spread_buffer=2.0, limit_touch_fill=False,
                      fvg_displacement_only=True, fvg_discount_premium_only=True, fvg_recent_sweep_only=False,
                      sb_require_htf_bias=False,
                      use_symbol_profiles=False, clear_cache=True, ml_filter=False, ml_min_confidence=0.60,
                      use_smt_divergence=False, smt_correlated_pair="DXY", use_volume_profile=False,
                      use_institutional_orderflow=False, institutional_lookback=20, institutional_threshold=0.5,
                      max_concurrent_trades=None, daily_loss_limit=None, min_confluence_score=None, min_fvg_size=None,
                      mock_si=None, **kwargs):
    # Explicit risk/filter overrides keep the OPT grid and single Backtest in sync.


    # When provided, these take precedence over the mutable config globals so a grid
    # combo and a manual Backtest run with identical inputs behave the same.
    date_from = pd.Timestamp(date_from).floor('s')
    date_to = pd.Timestamp(date_to).floor('s')
    
    offline = getattr(config, 'OFFLINE_BACKTESTING', False)
    
    if not offline:
        try:
            if not mt5.initialize():
                logger.error("Failed to initialize/connect MetaTrader 5 inside combined_backtest.")
                return [], {
                    'final_balance': initial_balance, 'net_profit': 0.0, 'total_trades': 0,
                    'win_rate': 0.0, 'profit_factor': 0.0, 'max_drawdown': 0.0,
                    'avg_trade_profit': 0.0, 'average_rrr': 0.0, 'max_runup': 0.0
                }
        except Exception as e:
            logger.error(f"MT5 initialization exception: {e}")
            return [], {
                'final_balance': initial_balance, 'net_profit': 0.0, 'total_trades': 0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'max_drawdown': 0.0,
                'avg_trade_profit': 0.0, 'average_rrr': 0.0, 'max_runup': 0.0
            }

    # ── Regime-adaptive parameter loading ───────────────────────────────
    if getattr(config, 'REGIME_ADAPTIVE_PARAMS_ENABLED', False):
        try:
            sample_sym = symbols[0] if symbols else config.DEFAULT_SYMBOL
            sample_df = get_data(sample_sym, mt5.TIMEFRAME_H1, 100, live=False, date_to=date_from)
            if sample_df is not None and len(sample_df) >= 20:
                # Strip any bar whose close is after date_from (lookahead protection)
                while len(sample_df) > 0:
                    bar_close = sample_df.index[-1] + pd.Timedelta(hours=1)
                    if bar_close > date_from:
                        sample_df = sample_df.iloc[:-1]
                    else:
                        break
                regime_info = classify_regime(sample_df)
                regime_params = rm.params_for_regime(regime_info['label'])
                rm.apply_params(regime_params)
                logger.info("Regime set: %s (params: %s)", regime_info['label'],
                            {k: v for k, v in regime_params.items()
                             if k in ('max_concurrent_trades', 'min_confluence_score', 'min_fvg_size_spreads')})
        except Exception as exc:
            logger.debug("regime_manager: could not classify at start — %s", exc)

    # Re-apply explicit kwargs so they take precedence over RegimeManager
    if max_concurrent_trades is not None:
        config.MAX_CONCURRENT_TRADES = int(max_concurrent_trades)
    if daily_loss_limit is not None:
        config.DAILY_LOSS_LIMIT_PCT = float(daily_loss_limit)
    if min_confluence_score is not None:
        config.MIN_CONFLUENCE_SCORE = int(min_confluence_score)
    if min_fvg_size is not None:
        config.MIN_FVG_SIZE_SPREADS = float(min_fvg_size)
    if clear_cache:
        clear_simulation_cache()

    all_trades = []
    if use_ultra_low_tf:
        timeframes = ULTRA_LOW_TFS[:]
    else:
        timeframes = LOWER_TFS[:] if trade_on_all_tfs else [("M15", mt5.TIMEFRAME_M15)]

    total_tasks = len(symbols) * len(timeframes)
    task_idx = 0
    for symbol in symbols:
        if bt_stop_event.is_set(): break
        # Instrument-aware parameters (adapt OB thresholds per symbol class)
        ict_params = get_ict_model_parameters("Default", symbol)
        
        # Load profile overrides for this specific symbol
        sym_defaults = {
            'min_rrr': min_rrr,
            'use_dynamic_rrr': use_dynamic_rrr,
            'fvg_sl_mode': fvg_sl_mode,
            'session_filter': session_filter,
            'session_start': session_start,
            'session_end': session_end,
            'use_htf_filter': use_htf_filter,
            'use_ote_filter': use_ote_filter,
            'bypass_htf_conf': bypass_htf_conf,
            'trail_type': trail_type,
            'trail_pct': trail_params.get('trail_pct', 0.5) if (trail_params and 'trail_pct' in trail_params) else 0.5,
            'require_bos_fvg': require_bos_fvg,
            'anti_gap_enabled': anti_gap_enabled,
            'anti_gap_mult': anti_gap_mult,
            'fvg_sl_spread_buffer': fvg_sl_spread_buffer,
            'limit_touch_fill': limit_touch_fill,
            'fvg_displacement_only': fvg_displacement_only,
            'fvg_discount_premium_only': fvg_discount_premium_only,
            'fvg_recent_sweep_only': fvg_recent_sweep_only,
            'sb_require_htf_bias': sb_require_htf_bias,
            'spread_cost': spread_cost,
            'slippage_points': slippage_points,
            'commission_per_lot': commission_per_lot,
            'ict_method': ict_method,
            'trail_methods': trailing_methods
        }
        if use_symbol_profiles:
            sym_params = get_symbol_profile_overrides(symbol, sym_defaults)
        else:
            sym_params = sym_defaults
        
        for tf_name, tf_value in timeframes:
            if bt_stop_event.is_set(): break
            task_idx += 1
            if progress_callback:
                progress_callback(0, 1, symbol, tf_name, phase=f"Scanning {symbol} {tf_name} ({task_idx}/{total_tasks})")
            trades = generate_trade_signals(symbol, tf_name, tf_value, date_from, date_to,
                ict_params, trailing_methods=sym_params['trail_methods'], use_dynamic_rrr=sym_params['use_dynamic_rrr'],
                ict_method=sym_params['ict_method'], min_rrr=sym_params['min_rrr'],
                spread_cost=sym_params['spread_cost'], slippage_points=sym_params['slippage_points'],
                commission_per_lot=sym_params['commission_per_lot'],
                session_filter=sym_params['session_filter'], session_start=sym_params['session_start'], session_end=sym_params['session_end'],
                progress_callback=progress_callback, use_htf_filter=sym_params['use_htf_filter'], use_ote_filter=sym_params['use_ote_filter'],
                fvg_sl_mode=sym_params['fvg_sl_mode'], bypass_htf_conf=sym_params['bypass_htf_conf'],
                trail_type=sym_params['trail_type'], trail_params={'trail_pct': sym_params['trail_pct']}, require_bos_fvg=sym_params['require_bos_fvg'],
                anti_gap_enabled=sym_params['anti_gap_enabled'], anti_gap_mult=sym_params['anti_gap_mult'],
                fvg_sl_spread_buffer=sym_params['fvg_sl_spread_buffer'], limit_touch_fill=sym_params['limit_touch_fill'],
                fvg_displacement_only=sym_params['fvg_displacement_only'],
                fvg_discount_premium_only=sym_params['fvg_discount_premium_only'],
                fvg_recent_sweep_only=sym_params['fvg_recent_sweep_only'],
                sb_require_htf_bias=sym_params.get('sb_require_htf_bias', sb_require_htf_bias),
                use_smt_divergence=use_smt_divergence, smt_correlated_pair=smt_correlated_pair,
                use_volume_profile=use_volume_profile,
                sym_params=sym_params)
            all_trades.extend(trades)

    if progress_callback:
        progress_callback(0, 1, "", "", phase="Calculating PnL & metrics...")

    all_trades.sort(key=lambda x: x["entry_time"])

    # Fix #8: Allow overlapping trades (concurrent execution, perfectly matching live)
    executed_trades = []

    # === SPEED FIX: Cache contract_size and lot info per symbol (avoid MT5 API calls per trade) ===
    _contract_size_cache = {}
    _lot_info_cache = {}  # {symbol: (min_lot, max_lot, lot_step)}
    for sym in set(t["symbol"] for t in all_trades):
        _contract_size_cache[sym] = get_contract_size(sym)
        try:
            import MetaTrader5 as mt5_local
            sym_info = mt5_local.symbol_info(sym)
        except Exception:
            sym_info = None
        if sym_info:
            min_v = sym_info.volume_min if sym_info.volume_min > 0 else 0.0001
            max_v = sym_info.volume_max if sym_info.volume_max > 0 else 1000.0
            step_v = sym_info.volume_step if sym_info.volume_step > 0 else 0.0001
            _lot_info_cache[sym] = (min_v, max_v, step_v)
        else:
            _lot_info_cache[sym] = (0.01, 100.0, 0.01)

    def _fast_normalize_lots(symbol, lot_size):
        """Fast lot normalization without MT5 API calls - uses cached data."""
        min_lot, max_lot, lot_step = _lot_info_cache[symbol]
        
        # Absolute failsafe against infinite loops
        if min_lot <= 0: min_lot = 0.0001
        if max_lot <= 0: max_lot = 1000.0
            
        if lot_size < min_lot:
            return []
            
        lots = []
        remaining = float(lot_size)
        
        # Safety counter to absolutely prevent any infinite loop freeze
        max_iterations = 10000 
        iters = 0
        
        while remaining >= min_lot - 1e-8 and iters < max_iterations:
            iters += 1
            chunk = min(remaining, max_lot)
            
            # Round to lot_step
            if lot_step > 0:
                chunk = round(round(chunk / lot_step) * lot_step, 8)
                
            if chunk < min_lot - 1e-8 or chunk <= 0.0:
                break
                
            lots.append(chunk)
            remaining -= chunk
            
        return lots

    # === SPEED FIX: Incremental balance calculation (O(n log n) instead of O(n²)) ===
    exit_heap = []
    realized_pnl = 0.0  # Sum of PnL from trades that have fully exited

    bt_recovery_tracker = {
        'excess_loss': 0.0,
        'trades_remaining': 0,
        'spread_trades': 5
    }

    # Method-specific risk limit tracking per (symbol, method)
    _daily_loss_current_day = None
    _daily_loss_day_start_balance = initial_balance
    _daily_loss_today_by_key = {}       # {(symbol, method): total_loss_amount}

    # Track open trades for max concurrent limit per symbol and method + global portfolio cap
    _open_trades_by_symbol_and_method = {} # {(symbol, method): heap}
    _open_global_heap = []

    ml_total_signals = 0
    ml_passed = 0
    ml_rejected = 0

    batch_ml_probas = []
    if ml_filter and all_trades:
        from ml_engine import ml_score_signals_batch
        batch_ml_probas = ml_score_signals_batch(all_trades)

    _sym_profile_cache = {}
    
    for i, trade in enumerate(all_trades):
        # Drain the exit heap: add PnL from any trades that exited before this trade's entry
        while exit_heap and exit_heap[0][0] <= trade["entry_time"]:
            exit_t, pnl_val, ex_symbol, ex_method = heapq.heappop(exit_heap)
            realized_pnl += pnl_val
            
            trade_day = exit_t.date() if hasattr(exit_t, 'date') else None
            if pnl_val < 0 and trade_day is not None:
                loss_key = (trade_day, ex_symbol, ex_method)
                _daily_loss_today_by_key[loss_key] = _daily_loss_today_by_key.get(loss_key, 0.0) + abs(pnl_val)

        if ml_filter:
            ml_total_signals += 1
            if i < len(batch_ml_probas):
                proba = batch_ml_probas[i]
            else:
                proba = 0.0
            
            if proba < ml_min_confidence:
                ml_rejected += 1
                continue
            ml_passed += 1

        realized_balance = initial_balance + realized_pnl
        
        if progress_callback and i % 100 == 0:
            progress_callback(i, len(all_trades), "", "", phase=f"Simulating trades... Balance: ${realized_balance:,.2f}")
            
        if realized_balance <= 0 or bt_stop_event.is_set():
            logger.info("Backtest simulation stopped due to zero balance or user stop request.")
            break

        # Resolve symbol profile parameters (reset sym_params per loop iteration to avoid leak)
        sym = trade["symbol"]
        method = trade["ict_method"]
        concur_key = (sym, method)
        sym_params = None
        
        sym_defaults = {
            'risk_percent': risk_percent,
            'fixed_lot': fixed_lot,
            'commission_per_lot': commission_per_lot,
            'max_concurrent_trades': config.MAX_CONCURRENT_TRADES,
            'daily_loss_limit': config.DAILY_LOSS_LIMIT_PCT
        }
        if use_symbol_profiles:
            if sym not in _sym_profile_cache:
                _sym_profile_cache[sym] = get_symbol_profile_overrides(sym, sym_defaults)
            sym_params = _sym_profile_cache[sym]
            
            # Resolve parameter with proper fallback hierarchy
            daily_loss_limit_pct = 0.0
            # 1. Try method-specific override from the symbol profile
            method_override_found = False
            method_override_val = 0.0
            if sym_params and "methods" in sym_params and isinstance(sym_params["methods"], dict):
                method_overrides = sym_params["methods"].get(method)
                if method_overrides and isinstance(method_overrides, dict) and 'daily_loss_limit' in method_overrides:
                    method_override_found = True
                    method_override_val = float(method_overrides['daily_loss_limit'])
            
            if method_override_found:
                daily_loss_limit_pct = method_override_val
            else:
                prof = match_symbol_profile(sym)
                if prof and 'daily_loss_limit' in prof:
                    daily_loss_limit_pct = float(prof['daily_loss_limit'])
                else:
                    daily_loss_limit_pct = config.DAILY_LOSS_LIMIT_PCT
        else:
            daily_loss_limit_pct = config.DAILY_LOSS_LIMIT_PCT

        # Daily loss limit check (per symbol and method for entry day)
        entry_day = trade["entry_time"].date() if hasattr(trade["entry_time"], 'date') else None
        if entry_day != _daily_loss_current_day:
            _daily_loss_current_day = entry_day
            _daily_loss_day_start_balance = realized_balance

        if daily_loss_limit_pct > 0.0 and entry_day is not None:
            daily_limit = _daily_loss_day_start_balance * (daily_loss_limit_pct / 100.0)
            today_loss = _daily_loss_today_by_key.get((entry_day, sym, method), 0.0)
            if today_loss >= daily_limit:
                continue

        # Global portfolio position cap check
        while _open_global_heap and _open_global_heap[0] <= trade["entry_time"]:
            heapq.heappop(_open_global_heap)
        
        global_max_concurr = getattr(config, 'MAX_PORTFOLIO_CONCURRENT_TRADES', 0)
        if global_max_concurr > 0 and len(_open_global_heap) >= global_max_concurr:
            continue

        # Max concurrent trades (per symbol and method)
        sym_max_concurr = resolve_method_param('max_concurrent_trades', method, sym_params, config.MAX_CONCURRENT_TRADES)

        if concur_key not in _open_trades_by_symbol_and_method:
            _open_trades_by_symbol_and_method[concur_key] = []

        method_heap = _open_trades_by_symbol_and_method[concur_key]
        # Remove trades of this symbol/method that have exited by now
        while method_heap and method_heap[0] <= trade["entry_time"]:
            heapq.heappop(method_heap)

        if sym_max_concurr > 0:
            if len(method_heap) >= sym_max_concurr:
                continue  # Skip — too many positions open for this method

        contract_size = _contract_size_cache[trade["symbol"]]

        # Load profile overrides for this specific symbol
        trade_risk_pct = resolve_method_param('risk_percent', method, sym_params, risk_percent)
        trade_fixed_lot = resolve_method_param('fixed_lot', method, sym_params, fixed_lot)
        trade_use_fixed_lot = resolve_method_param('use_fixed_lot', method, sym_params, risk_mode == "Fixed")
        trade_commission = resolve_method_param('commission_per_lot', method, sym_params, commission_per_lot)

        # Lot size calculation matches live using exact historical equity
        if not trade_use_fixed_lot:
            risk_amount = realized_balance * (trade_risk_pct / 100)
            risk_points = (trade["entry_price"] - trade["sl_price"]) if trade["trade_direction"] == "Buy" \
                else (trade["sl_price"] - trade["entry_price"])
            if risk_points <= 0:
                continue
            lot_size = calculate_dynamic_lot_size(trade['symbol'], risk_amount, risk_points)

            # Regime + ML conviction sizing
            lot_size *= trade.get('regime_size_mult', 1.0)
            if ml_filter:
                lot_size *= ict_pro.ml_size_multiplier(proba)
            
            # --- SLIPPAGE RECOVERY: Adjust Lot Size ---
            if enable_slippage_recovery and bt_recovery_tracker['trades_remaining'] > 0:
                recovery_per_trade = bt_recovery_tracker['excess_loss'] / bt_recovery_tracker['spread_trades']
                reduction_factor = max(0.7, 1.0 - (recovery_per_trade / (risk_amount + 1e-9)))
                lot_size *= reduction_factor
                bt_recovery_tracker['trades_remaining'] -= 1
                if bt_recovery_tracker['trades_remaining'] <= 0:
                    bt_recovery_tracker['excess_loss'] = 0.0 # reset
        else:
            lot_size = trade_fixed_lot

        if lot_size <= 0:
            continue

        target_lots = _fast_normalize_lots(trade["symbol"], lot_size)
        if not target_lots:
            continue
        lot_size = sum(target_lots)

        # ONLY AFTER lot size is verified and trade is accepted, push to concurrency heaps
        heapq.heappush(method_heap, trade["exit_time"])
        heapq.heappush(_open_global_heap, trade["exit_time"])

        # Calculate PnL
        if trade["trade_direction"] == "Buy":
            pnl = (trade["exit_price"] - trade["entry_price"]) * lot_size * contract_size
        else:
            pnl = (trade["entry_price"] - trade["exit_price"]) * lot_size * contract_size

        # Subtract commission (both entry and exit)
        total_commission = trade_commission * lot_size * 2
        pnl -= total_commission

        # Deduct swap/financing cost for multi-day holds (Audit §4.1)
        from cost_model import estimate_swap_cost
        hold_days = (trade["exit_time"] - trade["entry_time"]).total_seconds() / 86400.0 if (trade.get("exit_time") and trade.get("entry_time")) else 0.0
        swap_cost = estimate_swap_cost(trade["symbol"], trade["trade_direction"], hold_days, lot_size)
        pnl -= swap_cost
        trade["swap_cost"] = swap_cost

        # --- SLIPPAGE RECOVERY: Track Excess Loss ---
        if enable_slippage_recovery and trade.get('outcome') == 'SL':
             risk_points = (trade["entry_price"] - trade["sl_price"]) if trade["trade_direction"] == "Buy" \
                 else (trade["sl_price"] - trade["entry_price"])
             intended_loss = - (risk_points * lot_size * contract_size) - total_commission
             if pnl < intended_loss:
                 excess_loss = intended_loss - pnl
                 bt_recovery_tracker['excess_loss'] += excess_loss
                 bt_recovery_tracker['trades_remaining'] = bt_recovery_tracker['spread_trades']

        trade_record = {**trade, "lot_size": lot_size, "pnl": pnl, "commission": total_commission}
        executed_trades.append(trade_record)

        # Push this trade's exit into the heap for future balance calculations
        heapq.heappush(exit_heap, (trade["exit_time"], pnl, trade["symbol"], trade["ict_method"]))

    metrics = compute_metrics(
        executed_trades, initial_balance,
        ml_filter=ml_filter,
        ml_total_signals=ml_total_signals if ml_filter else 0,
        ml_passed=ml_passed if ml_filter else 0,
        ml_rejected=ml_rejected if ml_filter else 0,
    )

    return executed_trades, metrics
