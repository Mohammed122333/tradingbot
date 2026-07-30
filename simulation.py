import pandas as pd
import numpy as np
import logging
try:
    import MetaTrader5 as mt5
except ImportError:
    import tests.MetaTrader5 as mt5
import config

# Graceful fallback wrapper for Numba JIT compiler
try:
    import numba
    _has_numba = True
except ImportError:
    _has_numba = False
    class numba_dummy:
        @staticmethod
        def njit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator
    numba = numba_dummy
from config import (
    TRAIL_TYPE_PARTIAL, TRAIL_TYPE_ATR, TRAIL_TYPE_PERCENT, DEFAULT_TRAIL_PERCENT,
    HIGHER_TFS
)
from utils import get_spread, get_data, get_data_by_date
from indicators import (
    detect_swing_points, calculate_ote_zone, detect_liquidity_sweep, check_ob_fvg_confluence,
    detect_fair_value_gaps, detect_all_liquidity_sweeps
)

# Confluence scoring, TP/SL, HTF trend — extracted to keep simulation.py focused
from simulation_scoring import (
    calculate_confluence_score,
    calculate_tp_sl,
    get_higher_tf_trend,
)

_sim_cache = {}
_df_array_cache = {}

def clear_simulation_cache():
    _sim_cache.clear()
    _df_array_cache.clear()
    # Clear MTF structure cache to prevent cross-run contamination
    try:
        import backtester as _bt
        _bt._mtf_structure_cache.clear()
    except Exception:
        pass
    # Clear ICT Pro TTL caches (HTF POI / liquidity pools)
    try:
        import ict_pro as _ip
        _ip._htf_cache.clear()
        _ip._pool_cache.clear()
    except Exception:
        pass

logger = logging.getLogger()

def get_structure_tp(entry_price, direction, swing_highs, swing_lows, spread):
    """ICT: Target the next liquidity pool (previous swing high/low).
    More realistic than fixed RRR — these are levels where actual liquidity sits."""
    if direction == "Buy":
        if not isinstance(swing_highs, (list, tuple)):
            return None
        targets = [sh['price'] for sh in swing_highs if isinstance(sh, dict) and 'price' in sh and sh['price'] > entry_price + spread * 5]
        if targets:
            return min(targets)
    else:
        if not isinstance(swing_lows, (list, tuple)):
            return None
        targets = [sl['price'] for sl in swing_lows if isinstance(sl, dict) and 'price' in sl and sl['price'] < entry_price - spread * 5]
        if targets:
            return max(targets)
    return None

# -------------------------
# Confluence Scoring — ICT multi-factor probability assessment
# -------------------------
# (calculate_confluence_score moved to simulation_scoring.py)

# -------------------------
# V2 Entry Computation (simplified — only used for legacy OB path)
# -------------------------
def compute_effective_entry(current_price, ob, symbol, method, direction="Buy", fixed_spread=None):
    """V2: Each real method has its own entry logic in its detect_* function.
    This is only used as fallback for any OB-based signals still in the pipeline."""
    if not isinstance(ob, dict) or not ob:
        return current_price
        
    spread = fixed_spread if fixed_spread is not None else get_spread(symbol)
    high = ob.get('high', current_price)
    low = ob.get('low', current_price)
    rng = high - low
    if direction == "Buy":
        return high - (rng * 0.5)  # Default: OB midpoint
    else:
        return low + (rng * 0.5)

# -------------------------
# V2: Trade Simulation — TP/SL Only + Partial Take-Profit
# No trailing stops. Eliminates live/backtest divergence.
# -------------------------
@numba.njit
def _detect_swing_points_t_jit(highs, lows, lookback):
    n = len(highs)
    last_sh_price = -1.0
    last_sl_price = -1.0
    
    if n < 2 * lookback + 1:
        return last_sh_price, last_sl_price
        
    for center in range(lookback, n - lookback):
        val_h = highs[center]
        val_l = lows[center]
        
        is_high = True
        is_low = True
        
        for offset in range(-lookback, lookback + 1):
            if offset == 0:
                continue
            idx = center + offset
            if highs[idx] > val_h:
                is_high = False
            if lows[idx] < val_l:
                is_low = False
                
        if is_high:
            last_sh_price = val_h
        if is_low:
            last_sl_price = val_l
            
    return last_sh_price, last_sl_price

@numba.njit
def _run_trade_simulation_jit(
    m1_times, m1_opens, m1_highs, m1_lows, m1_closes,
    direction_val, entry_price, tp_price, sl_price,
    use_trailing_stop, initial_risk, spread, slippage, cancel_limit,
    trail_type_val, trail_pct, min_stop_distance,
    df_highs, df_lows, df_times_int64,
    entry_time_int64, first_time_int64, limit_touch_fill
):
    n_m1 = len(m1_times)
    fill_idx = -1
    
    m1_times_int64 = m1_times.view(np.int64)
    start_k = np.searchsorted(m1_times_int64, entry_time_int64)
    entered = False
    
    for k in range(start_k, n_m1):
        bar_time = m1_times_int64[k]
        bar_open = m1_opens[k]
        bar_high = m1_highs[k]
        bar_low = m1_lows[k]
        
        if not entered:
            if bar_time < entry_time_int64:
                continue
            req_spread = 0.0 if limit_touch_fill else spread
            if direction_val == 1 and (bar_low + req_spread) <= entry_price:
                entered = True
                fill_idx = k
            elif direction_val == -1 and bar_high >= entry_price:
                entered = True
                fill_idx = k
            elif bar_time == entry_time_int64 and (
                (direction_val == 1 and (bar_open + req_spread) <= entry_price) or
                (direction_val == -1 and bar_open >= entry_price)):
                entered = True
                fill_idx = k
                
            if not entered:
                cancel_limit_ns = cancel_limit * 1000000000
                if (bar_time - first_time_int64) > cancel_limit_ns:
                    return entry_price, -1, k, 3
                else:
                    continue
            break
            
    if not entered:
        return entry_price, -1, n_m1 - 1, 3
        
    current_sl = sl_price
    original_sl = sl_price
    best_price = entry_price
    breakeven_moved = False
    partial_taken = False
    partial_price = 0.0
    _last_df_idx = -1
    _cached_trail_sl = -1.0
    
    for k in range(fill_idx, n_m1):
        bar_time = m1_times_int64[k]
        bar_open = m1_opens[k]
        bar_high = m1_highs[k]
        bar_low = m1_lows[k]
        bar_close = m1_closes[k]
        
        is_fill_bar = (k == fill_idx)
        reference_price = bar_open
        
        if direction_val == 1:
            sl_hit = bar_low <= current_sl
            if is_fill_bar and sl_hit:
                exit_p = min(current_sl, reference_price) - slippage
                return exit_p, fill_idx, k, 1
                
            if use_trailing_stop and initial_risk > 0:
                if trail_type_val == 1:
                    # True Partial TP System: 50% closed at 1R, SL to Breakeven + 50% profit lock above 1R
                    if not partial_taken and bar_high >= entry_price + initial_risk:
                        partial_taken = True
                        partial_price = entry_price + initial_risk
                        current_sl = max(current_sl, entry_price + spread * 5)
                    if partial_taken:
                        best_price = max(best_price, bar_high)
                        price_at_1r = entry_price + initial_risk
                        if best_price > price_at_1r:
                            points_above_1r = best_price - price_at_1r
                            be_level = entry_price + spread * 5
                            new_sl = be_level + (points_above_1r / 2.0)
                            if new_sl > current_sl and (bar_high - new_sl) >= min_stop_distance:
                                current_sl = new_sl
                else:
                    # Other trailing types require standard Breakeven move first
                    if not breakeven_moved:
                        if bar_high - entry_price >= initial_risk:
                            breakeven_moved = True
                            current_sl = max(current_sl, entry_price + spread * 5)
                    else:
                        best_price = max(best_price, bar_high)
                        if trail_type_val == 2:
                            price_at_1r = entry_price + initial_risk
                            if best_price > price_at_1r:
                                points_above_1r = best_price - price_at_1r
                                be_level = entry_price + spread * 5
                                new_sl = be_level + (points_above_1r / 2.0)
                                if new_sl > current_sl and (bar_high - new_sl) >= min_stop_distance:
                                    current_sl = new_sl
                        elif trail_type_val == 3:
                            trail_pct_val = trail_pct / 100.0
                            new_sl = best_price * (1.0 - trail_pct_val)
                            if new_sl > current_sl and (bar_high - new_sl) >= min_stop_distance:
                                current_sl = new_sl
                            
            tp_hit = bar_high >= tp_price
            if tp_hit and sl_hit:
                dist_to_tp = tp_price - reference_price
                dist_to_sl = reference_price - current_sl
                if dist_to_sl <= dist_to_tp:
                    outcome_code = 5 if current_sl > original_sl else 1
                    exit_p = min(current_sl, reference_price) - slippage
                    if partial_taken: exit_p = (partial_price + exit_p) / 2.0
                    return exit_p, fill_idx, k, outcome_code
                else:
                    exit_p = max(tp_price, reference_price)
                    if partial_taken: exit_p = (partial_price + exit_p) / 2.0
                    return exit_p, fill_idx, k, 2
            elif tp_hit:
                exit_p = max(tp_price, reference_price)
                if partial_taken: exit_p = (partial_price + exit_p) / 2.0
                return exit_p, fill_idx, k, 2
            elif sl_hit:
                outcome_code = 5 if current_sl > original_sl else 1
                exit_p = min(current_sl, reference_price) - slippage
                if partial_taken: exit_p = (partial_price + exit_p) / 2.0
                return exit_p, fill_idx, k, outcome_code
        else:
            sl_hit = (bar_low + spread) >= current_sl
            if is_fill_bar and sl_hit:
                exit_p = max(current_sl, reference_price + spread) + slippage
                return exit_p, fill_idx, k, 1
                
            if use_trailing_stop and initial_risk > 0:
                if trail_type_val == 1:
                    # True Partial TP System: 50% closed at 1R, SL to Breakeven + 50% profit lock above 1R
                    if not partial_taken and bar_low <= entry_price - initial_risk:
                        partial_taken = True
                        partial_price = entry_price - initial_risk
                        current_sl = min(current_sl, entry_price - spread * 5)
                    if partial_taken:
                        best_price = min(best_price, bar_low)
                        price_at_1r = entry_price - initial_risk
                        if best_price < price_at_1r:
                            points_below_1r = price_at_1r - best_price
                            be_level = entry_price - spread * 5
                            new_sl = be_level - (points_below_1r / 2.0)
                            if new_sl < current_sl and (new_sl - bar_low) >= min_stop_distance:
                                current_sl = new_sl
                else:
                    # Other trailing types require standard Breakeven move first
                    if not breakeven_moved:
                        if entry_price - (bar_low + spread) >= initial_risk:
                            breakeven_moved = True
                            current_sl = min(current_sl, entry_price - spread * 5)
                    else:
                        best_price = min(best_price, bar_low)
                        if trail_type_val == 2:
                            price_at_1r = entry_price - initial_risk
                            if best_price < price_at_1r:
                                points_below_1r = price_at_1r - best_price
                                be_level = entry_price - spread * 5
                                new_sl = be_level - (points_below_1r / 2.0)
                                if new_sl < current_sl and (new_sl - bar_low) >= min_stop_distance:
                                    current_sl = new_sl
                        elif trail_type_val == 3:
                            trail_pct_val = trail_pct / 100.0
                            new_sl = best_price * (1.0 + trail_pct_val)
                            if new_sl < current_sl and (new_sl - bar_low) >= min_stop_distance:
                                current_sl = new_sl
                            
            tp_hit = (bar_low + spread) <= tp_price
            if tp_hit and sl_hit:
                dist_to_tp = reference_price - tp_price
                dist_to_sl = current_sl - reference_price
                if dist_to_sl <= dist_to_tp:
                    outcome_code = 5 if current_sl < original_sl else 1
                    exit_p = max(current_sl, reference_price + spread) + slippage
                    if partial_taken: exit_p = (partial_price + exit_p) / 2.0
                    return exit_p, fill_idx, k, outcome_code
                else:
                    exit_p = min(tp_price, reference_price + spread)
                    if partial_taken: exit_p = (partial_price + exit_p) / 2.0
                    return exit_p, fill_idx, k, 2
            elif tp_hit:
                exit_p = min(tp_price, reference_price + spread)
                if partial_taken: exit_p = (partial_price + exit_p) / 2.0
                return exit_p, fill_idx, k, 2
            elif sl_hit:
                outcome_code = 5 if current_sl < original_sl else 1
                exit_p = max(current_sl, reference_price + spread) + slippage
                if partial_taken: exit_p = (partial_price + exit_p) / 2.0
                return exit_p, fill_idx, k, outcome_code
                
    timeout_price = m1_closes[-1] + (spread if direction_val == -1 else 0)
    timeout_price = timeout_price - slippage if direction_val == 1 else timeout_price + slippage
    return timeout_price, fill_idx, n_m1 - 1, 4

def run_trade_simulation(df, start_index, direction, entry_price, tp_price, sl_price,
                         use_trailing_stop=False, initial_risk=0, symbol=None,
                         timeframe_value=None, m1_global_df=None, fixed_spread=None,
                         trail_type=TRAIL_TYPE_PARTIAL, trail_params=None, ict_method=None, min_stop_distance=0.0, limit_touch_fill=False):
    if trail_params is None:
        trail_params = {}
    trail_pct = trail_params.get('trail_pct', DEFAULT_TRAIL_PERCENT)
    cache_key = (
        symbol, timeframe_value, start_index, direction, round(entry_price, 5), round(tp_price, 5), round(sl_price, 5),
        use_trailing_stop, round(initial_risk, 5), trail_type, round(trail_pct, 4), ict_method, round(min_stop_distance, 5), limit_touch_fill
    )
    if cache_key in _sim_cache:
        return _sim_cache[cache_key]
        
    res = _run_trade_simulation_impl(
        df, start_index, direction, entry_price, tp_price, sl_price,
        use_trailing_stop, initial_risk, symbol, timeframe_value,
        m1_global_df, fixed_spread, trail_type, trail_params, ict_method, min_stop_distance, limit_touch_fill
    )
    _sim_cache[cache_key] = res
    
    # Prevent memory leak and garbage collection slowdown over massive backtests
    if len(_sim_cache) > 50000:
        _sim_cache.clear()
        
    return res

def get_df_precalc_arrays(df):
    # NEVER key on id(): the cache holds only derived arrays, so `df` gets
    # garbage-collected and CPython reuses the address. A different frame with
    # the same length + first timestamp then silently receives stale arrays.
    if len(df) == 0:
        df_id = ('empty', 0)
    else:
        df_id = (
            len(df),
            df.index[0].value,
            df.index[-1].value,
            float(df['high'].values[0]),
            float(df['low'].values[-1]),
            float(df['close'].values[-1]),
        )
    if df_id not in _df_array_cache:
        if len(_df_array_cache) > 64:
            _df_array_cache.clear()
        _df_array_cache[df_id] = {
            'highs': df['high'].values,
            'lows': df['low'].values,
            'times_int64': df.index.values.view(np.int64),
            'index_timestamps': df.index.tolist()
        }
    return _df_array_cache[df_id]

def _run_trade_simulation_impl(df, start_index, direction, entry_price, tp_price, sl_price,
                               use_trailing_stop=False, initial_risk=0, symbol=None,
                               timeframe_value=None, m1_global_df=None, fixed_spread=None,
                               trail_type=TRAIL_TYPE_PARTIAL, trail_params=None, ict_method=None, min_stop_distance=0.0, limit_touch_fill=False):
    """Pure TP/SL simulation with trailing stops.
    Supports high-speed JIT path when m1_global_df is passed as a pre-extracted dict of arrays.
    """
    if trail_params is None:
        trail_params = {}
    
    if fixed_spread is not None:
        spread = fixed_spread
    else:
        spread = get_spread(symbol) if symbol else 0.0001
        
    import config
    import cost_model
    if getattr(config, 'SIM_APPLY_SLIPPAGE_ON_STOP', True):
        _pt = cost_model.point_for_symbol(symbol)
        slippage = float(getattr(config, 'SLIPPAGE_MAX_POINTS', 5)) * _pt
    else:
        slippage = 0.0
    cancel_limit = float(getattr(config, 'PENDING_LIMIT_EXPIRY_HOURS', 2.0)) * 3600.0
    use_m1_drilldown = symbol and (timeframe_value is None or timeframe_value != mt5.TIMEFRAME_M1)
    
    if use_m1_drilldown and isinstance(m1_global_df, dict):
        cached_arrays = get_df_precalc_arrays(df)
        entry_time = cached_arrays['index_timestamps'][start_index]
        start_time_m1 = entry_time - pd.Timedelta(minutes=5)
        _max_hold_h = float(getattr(config, 'SIM_MAX_HOLD_HOURS', 72.0))
        end_time_m1 = entry_time + pd.Timedelta(hours=_max_hold_h)
        
        m1_times_all = m1_global_df['times']
        start_idx = np.searchsorted(m1_times_all, np.datetime64(start_time_m1))
        end_idx = np.searchsorted(m1_times_all, np.datetime64(end_time_m1), side='right')
        
        m1_times = m1_times_all[start_idx:end_idx]
        m1_opens = m1_global_df['opens'][start_idx:end_idx]
        m1_highs = m1_global_df['highs'][start_idx:end_idx]
        m1_lows = m1_global_df['lows'][start_idx:end_idx]
        m1_closes = m1_global_df['closes'][start_idx:end_idx]
        
        if len(m1_times) > 0:
            direction_val = 1 if direction == "Buy" else -1
            trail_type_map = {
                TRAIL_TYPE_PARTIAL: 1,
                TRAIL_TYPE_ATR: 2,
                TRAIL_TYPE_PERCENT: 3,
            }
            trail_type_val = trail_type_map.get(trail_type, 0)
            trail_pct = float(trail_params.get('trail_pct', DEFAULT_TRAIL_PERCENT))
            
            df_highs = cached_arrays['highs']
            df_lows = cached_arrays['lows']
            df_times_int64 = cached_arrays['times_int64']
            
            entry_time_int64 = df_times_int64[start_index]
            first_time_int64 = df_times_int64[start_index]
            
            exit_price, fill_idx, exit_idx, outcome_code = _run_trade_simulation_jit(
                m1_times, m1_opens, m1_highs, m1_lows, m1_closes,
                direction_val, entry_price, tp_price, sl_price,
                use_trailing_stop, initial_risk, spread, slippage, cancel_limit,
                trail_type_val, trail_pct, min_stop_distance,
                df_highs, df_lows, df_times_int64,
                entry_time_int64, first_time_int64, limit_touch_fill
            )
            
            outcome_map = {
                1: "SL",
                2: "TP",
                3: "CANCELLED",
                4: "TIMEOUT",
                5: "TRAIL"
            }
            outcome = outcome_map.get(outcome_code, "TIMEOUT")
            
            fill_time = pd.Timestamp(m1_times[fill_idx]) if fill_idx >= 0 else entry_time
            exit_time = pd.Timestamp(m1_times[exit_idx])
            
            return exit_price, fill_time, exit_time, outcome

    original_sl = sl_price
    current_sl = sl_price
    entry_time = pd.Timestamp(df.index[start_index])
    breakeven_moved = False
    best_price = entry_price  # Track best price for percentage trail
    
    # Advanced: M1 precision drilldown — only for timeframes > M1
    if use_m1_drilldown:
        start_time_m1 = entry_time - pd.Timedelta(minutes=5)
        end_time_m1 = entry_time + pd.Timedelta(days=3)
        
        has_m1 = False
        if isinstance(m1_global_df, dict):
            # Fast NumPy path
            m1_times_all = m1_global_df['times']
            start_idx = np.searchsorted(m1_times_all, np.datetime64(start_time_m1))
            end_idx = np.searchsorted(m1_times_all, np.datetime64(end_time_m1), side='right')
            
            m1_times = m1_times_all[start_idx:end_idx]
            m1_opens = m1_global_df['opens'][start_idx:end_idx]
            m1_highs = m1_global_df['highs'][start_idx:end_idx]
            m1_lows = m1_global_df['lows'][start_idx:end_idx]
            m1_closes = m1_global_df['closes'][start_idx:end_idx]
            has_m1 = len(m1_times) > 0
        else:
            # Fallback Pandas path
            if m1_global_df is not None and not m1_global_df.empty:
                m1_df = m1_global_df.loc[start_time_m1:end_time_m1]
            else:
                m1_df = get_data_by_date(symbol, mt5.TIMEFRAME_M1, start_time_m1, end_time_m1)
                
            if m1_df is not None and not m1_df.empty:
                m1_times = m1_df.index.tolist()
                m1_opens = m1_df['open'].values
                m1_highs = m1_df['high'].values
                m1_lows = m1_df['low'].values
                m1_closes = m1_df['close'].values
                has_m1 = True
            
        if has_m1:
            # High-Speed Vectorized Simulation (when trailing stops are disabled)
            if not use_trailing_stop:
                if isinstance(m1_global_df, dict):
                    # NumPy execution path
                    entry_time_np = np.datetime64(entry_time)
                    valid_time_mask = m1_times >= entry_time_np
                    req_spread = 0.0 if limit_touch_fill else spread
                    fill_mask = (m1_lows + req_spread) <= entry_price if direction == "Buy" else m1_highs >= entry_price
                    fill_mask = fill_mask & valid_time_mask
                    
                    fill_indices = np.where(fill_mask)[0]
                    fill_idx = fill_indices[0] if len(fill_indices) > 0 else -1
                    
                    first_time = pd.Timestamp(df.index[start_index])
                    
                    if fill_idx != -1:
                        fill_time = pd.Timestamp(m1_times[fill_idx])
                        time_to_fill = (fill_time - first_time).total_seconds()
                        if time_to_fill > cancel_limit:
                            return entry_price, first_time, first_time + pd.Timedelta(seconds=cancel_limit), "CANCELLED"
                    else:
                        cancel_time = first_time + pd.Timedelta(seconds=cancel_limit)
                        last_m1_time = pd.Timestamp(m1_times[-1])
                        if cancel_time > last_m1_time:
                            cancel_time = last_m1_time
                        return entry_price, first_time, cancel_time, "CANCELLED"
                            
                    _highs_sub = m1_highs[fill_idx:]
                    _lows_sub = m1_lows[fill_idx:]
                    _opens_sub = m1_opens[fill_idx:]
                    _times_sub = m1_times[fill_idx:]
                    
                    if direction == "Buy":
                        sl_mask = _lows_sub <= current_sl
                        tp_mask = _highs_sub >= tp_price
                    else:
                        sl_mask = (_highs_sub + spread) >= current_sl
                        tp_mask = (_lows_sub + spread) <= tp_price
                        
                    sl_hit_indices = np.where(sl_mask)[0]
                    tp_hit_indices = np.where(tp_mask)[0]
                    
                    sub_len = len(_times_sub)
                    sl_hit_idx = sl_hit_indices[0] if len(sl_hit_indices) > 0 else sub_len
                    tp_hit_idx = tp_hit_indices[0] if len(tp_hit_indices) > 0 else sub_len
                    
                    if sl_hit_idx < tp_hit_idx:
                        hit_time = pd.Timestamp(_times_sub[sl_hit_idx])
                        ref_p = _opens_sub[sl_hit_idx]
                        exit_p = min(current_sl, ref_p) if direction == "Buy" else max(current_sl, ref_p + spread)
                        return exit_p, fill_time, hit_time, "SL"
                    elif tp_hit_idx < sl_hit_idx:
                        hit_time = pd.Timestamp(_times_sub[tp_hit_idx])
                        ref_p = _opens_sub[tp_hit_idx]
                        exit_p = max(tp_price, ref_p) if direction == "Buy" else min(tp_price, ref_p + spread)
                        return exit_p, fill_time, hit_time, "TP"
                    elif sl_hit_idx == tp_hit_idx and sl_hit_idx < sub_len:
                        hit_time = pd.Timestamp(_times_sub[sl_hit_idx])
                        ref_p = _opens_sub[sl_hit_idx]
                        
                        if direction == "Buy":
                            dist_to_tp = tp_price - ref_p
                            dist_to_sl = ref_p - current_sl
                            if dist_to_sl <= dist_to_tp:
                                return min(current_sl, ref_p), fill_time, hit_time, "SL"
                            else:
                                return max(tp_price, ref_p), fill_time, hit_time, "TP"
                        else:
                            dist_to_tp = ref_p - tp_price
                            dist_to_sl = current_sl - ref_p
                            if dist_to_sl <= dist_to_tp:
                                return max(current_sl, ref_p + spread), fill_time, hit_time, "SL"
                            else:
                                return min(tp_price, ref_p + spread), fill_time, hit_time, "TP"
                    else:
                        timeout_price = m1_closes[-1] + (spread if direction == "Sell" else 0)
                        return timeout_price, fill_time, pd.Timestamp(m1_times[-1]), "TIMEOUT"
                else:
                    # Pandas fallback execution path
                    m1_df_sub = m1_df[m1_df.index >= entry_time]
                    if not m1_df_sub.empty:
                        req_spread = 0.0 if limit_touch_fill else spread
                        fill_mask = (m1_df_sub['low'] + req_spread) <= entry_price if direction == "Buy" else m1_df_sub['high'] >= entry_price
                        fill_indices = np.where(fill_mask)[0]
                        fill_idx = fill_indices[0] if len(fill_indices) > 0 else -1
                        
                        first_time = pd.Timestamp(df.index[start_index])
                        
                        if fill_idx != -1:
                            fill_time = m1_df_sub.index[fill_idx]
                            time_to_fill = (fill_time - first_time).total_seconds()
                            if time_to_fill > cancel_limit:
                                return entry_price, first_time, first_time + pd.Timedelta(seconds=cancel_limit), "CANCELLED"
                        else:
                            cancel_time = first_time + pd.Timedelta(seconds=cancel_limit)
                            if cancel_time > m1_df_sub.index[-1]:
                                cancel_time = m1_df_sub.index[-1]
                            return entry_price, first_time, cancel_time, "CANCELLED"
                                
                        post_fill_df = m1_df_sub.iloc[fill_idx:]
                        if not post_fill_df.empty:
                            _highs_sub = post_fill_df['high'].values
                            _lows_sub = post_fill_df['low'].values
                            _opens_sub = post_fill_df['open'].values
                            _times_sub = post_fill_df.index
                            
                            if direction == "Buy":
                                sl_mask = _lows_sub <= current_sl
                                tp_mask = _highs_sub >= tp_price
                            else:
                                sl_mask = (_highs_sub + spread) >= current_sl
                                tp_mask = (_lows_sub + spread) <= tp_price
                                
                            sl_hit_indices = np.where(sl_mask)[0]
                            tp_hit_indices = np.where(tp_mask)[0]
                            
                            sl_hit_idx = sl_hit_indices[0] if len(sl_hit_indices) > 0 else len(post_fill_df)
                            tp_hit_idx = tp_hit_indices[0] if len(tp_hit_indices) > 0 else len(post_fill_df)
                            
                            if sl_hit_idx < tp_hit_idx:
                                hit_time = _times_sub[sl_hit_idx]
                                ref_p = _opens_sub[sl_hit_idx]
                                exit_p = min(current_sl, ref_p) if direction == "Buy" else max(current_sl, ref_p + spread)
                                return exit_p, fill_time, hit_time, "SL"
                            elif tp_hit_idx < sl_hit_idx:
                                hit_time = _times_sub[tp_hit_idx]
                                ref_p = _opens_sub[tp_hit_idx]
                                exit_p = max(tp_price, ref_p) if direction == "Buy" else min(tp_price, ref_p + spread)
                                return exit_p, fill_time, hit_time, "TP"
                            elif sl_hit_idx == tp_hit_idx and sl_hit_idx < len(post_fill_df):
                                hit_time = _times_sub[sl_hit_idx]
                                ref_p = _opens_sub[sl_hit_idx]
                                
                                if direction == "Buy":
                                    dist_to_tp = tp_price - ref_p
                                    dist_to_sl = ref_p - current_sl
                                    if dist_to_sl <= dist_to_tp:
                                        return min(current_sl, ref_p), fill_time, hit_time, "SL"
                                    else:
                                        return max(tp_price, ref_p), fill_time, hit_time, "TP"
                                else:
                                    dist_to_tp = ref_p - tp_price
                                    dist_to_sl = current_sl - ref_p
                                    if dist_to_sl <= dist_to_tp:
                                        return max(current_sl, ref_p + spread), fill_time, hit_time, "SL"
                                    else:
                                        return min(tp_price, ref_p + spread), fill_time, hit_time, "TP"
                            else:
                                timeout_price = post_fill_df['close'].values[-1] + (spread if direction == "Sell" else 0)
                                return timeout_price, fill_time, post_fill_df.index[-1], "TIMEOUT"

            entered = False
            is_fill_bar = False  # Track the limit order fill bar
            _cached_trail_sl = None
            _last_df_idx = -1
            
            if isinstance(m1_global_df, dict):
                # High-speed numpy comparisons
                entry_time_np = np.datetime64(entry_time)
                first_time_np = np.datetime64(df.index[start_index])
                cancel_limit_td = np.timedelta64(cancel_limit, 's')
                
                for k in range(len(m1_times)):
                    bar_time = m1_times[k]
                    bar_open = m1_opens[k]
                    bar_high = m1_highs[k]
                    bar_low = m1_lows[k]
                    
                    if not entered:
                        if bar_time < entry_time_np:
                            continue
                        req_spread = 0.0 if limit_touch_fill else spread
                        if direction == "Buy" and (bar_low + req_spread) <= entry_price:
                            entered = True
                            entry_time_np = bar_time
                        elif direction == "Sell" and bar_high >= entry_price:
                            entered = True
                            entry_time_np = bar_time
                        elif bar_time == entry_time_np and (
                            (direction == "Buy" and (bar_open + req_spread) <= entry_price) or
                            (direction == "Sell" and bar_open >= entry_price)):
                            entered = True
                            entry_time_np = bar_time
                        
                        if not entered:
                            if (bar_time - first_time_np) > cancel_limit_td:
                                return entry_price, pd.Timestamp(first_time_np), pd.Timestamp(bar_time), "CANCELLED"
                            else:
                                continue
                        is_fill_bar = True
                    
                    reference_price = bar_open
                    if direction == "Buy":
                        sl_hit = bar_low <= current_sl
                        if is_fill_bar and sl_hit:
                            is_fill_bar = False
                            exit_p = min(current_sl, reference_price)
                            return exit_p, pd.Timestamp(entry_time_np), pd.Timestamp(bar_time), "SL"
                        is_fill_bar = False
                        
                        if use_trailing_stop and initial_risk > 0:
                            if not breakeven_moved:
                                if bar_high - entry_price >= initial_risk:
                                    breakeven_moved = True
                                    current_sl = max(current_sl, entry_price + spread * 5)
                            else:
                                best_price = max(best_price, bar_high)
                                if trail_type == TRAIL_TYPE_PARTIAL:
                                    bar_time_ts = pd.Timestamp(bar_time)
                                    df_idx = df.index.searchsorted(bar_time_ts)
                                    if df_idx >= len(df) or df.index[df_idx] > bar_time_ts:
                                        df_idx -= 1
                                    if df_idx != _last_df_idx:
                                        _last_df_idx = df_idx
                                        lookback_window = df.iloc[max(0, df_idx-100):df_idx]
                                        _cached_trail_sl = None
                                        if len(lookback_window) >= 15:
                                            _, swing_lows_t = detect_swing_points(lookback_window, lookback=2)
                                            if swing_lows_t:
                                                _cached_trail_sl = swing_lows_t[-1]['price'] - spread
                                    if _cached_trail_sl is not None:
                                        if current_sl < _cached_trail_sl < bar_open:
                                            current_sl = _cached_trail_sl
                                elif trail_type == TRAIL_TYPE_ATR:
                                    price_at_1r = entry_price + initial_risk
                                    if best_price > price_at_1r:
                                        points_above_1r = best_price - price_at_1r
                                        be_level = entry_price + spread * 5
                                        new_sl = be_level + (points_above_1r / 2.0)
                                        if new_sl > current_sl and new_sl < bar_open:
                                            current_sl = new_sl
                                elif trail_type == TRAIL_TYPE_PERCENT:
                                    trail_pct = trail_params.get('trail_pct', DEFAULT_TRAIL_PERCENT) / 100.0
                                    new_sl = best_price * (1.0 - trail_pct)
                                    if new_sl > current_sl and new_sl < bar_open:
                                        current_sl = new_sl

                        tp_hit = bar_high >= tp_price
                        if tp_hit and sl_hit:
                            dist_to_tp = tp_price - reference_price
                            dist_to_sl = reference_price - current_sl
                            if dist_to_sl <= dist_to_tp:
                                outcome = "TRAIL" if current_sl > original_sl else "SL"
                                exit_p = min(current_sl, reference_price)
                                return exit_p, pd.Timestamp(entry_time_np), pd.Timestamp(bar_time), outcome
                            else:
                                exit_p = max(tp_price, reference_price)
                                return exit_p, pd.Timestamp(entry_time_np), pd.Timestamp(bar_time), "TP"
                        elif tp_hit:
                            exit_p = max(tp_price, reference_price)
                            return exit_p, pd.Timestamp(entry_time_np), pd.Timestamp(bar_time), "TP"
                        elif sl_hit:
                            outcome = "TRAIL" if current_sl > original_sl else "SL"
                            exit_p = min(current_sl, reference_price)
                            return exit_p, pd.Timestamp(entry_time_np), pd.Timestamp(bar_time), outcome
                    else:
                        sl_hit = (bar_high + spread) >= current_sl
                        if is_fill_bar and sl_hit:
                            is_fill_bar = False
                            exit_p = max(current_sl, reference_price + spread)
                            return exit_p, pd.Timestamp(entry_time_np), pd.Timestamp(bar_time), "SL"
                        is_fill_bar = False
                        
                        if use_trailing_stop and initial_risk > 0:
                            if not breakeven_moved:
                                if entry_price - (bar_low + spread) >= initial_risk:
                                    breakeven_moved = True
                                    current_sl = min(current_sl, entry_price - spread * 5)
                            else:
                                best_price = min(best_price, bar_low)
                                if trail_type == TRAIL_TYPE_PARTIAL:
                                    bar_time_ts = pd.Timestamp(bar_time)
                                    df_idx = df.index.searchsorted(bar_time_ts)
                                    if df_idx >= len(df) or df.index[df_idx] > bar_time_ts:
                                        df_idx -= 1
                                    if df_idx != _last_df_idx:
                                        _last_df_idx = df_idx
                                        lookback_window = df.iloc[max(0, df_idx-100):df_idx]
                                        _cached_trail_sl = None
                                        if len(lookback_window) >= 15:
                                            swing_highs_t, _ = detect_swing_points(lookback_window, lookback=2)
                                            if swing_highs_t:
                                                _cached_trail_sl = swing_highs_t[-1]['price'] + spread
                                    if _cached_trail_sl is not None:
                                        if current_sl > _cached_trail_sl > bar_open:
                                            current_sl = _cached_trail_sl
                                elif trail_type == TRAIL_TYPE_ATR:
                                    price_at_1r = entry_price - initial_risk
                                    if best_price < price_at_1r:
                                        points_below_1r = price_at_1r - best_price
                                        be_level = entry_price - spread * 5
                                        new_sl = be_level - (points_below_1r / 2.0)
                                        if new_sl < current_sl and new_sl > bar_open:
                                            current_sl = new_sl
                                elif trail_type == TRAIL_TYPE_PERCENT:
                                    trail_pct = trail_params.get('trail_pct', DEFAULT_TRAIL_PERCENT) / 100.0
                                    new_sl = best_price * (1.0 + trail_pct)
                                    if new_sl < current_sl and new_sl > bar_open:
                                        current_sl = new_sl

                        tp_hit = (bar_low + spread) <= tp_price
                        if tp_hit and sl_hit:
                            dist_to_tp = reference_price - tp_price
                            dist_to_sl = current_sl - reference_price
                            if dist_to_sl <= dist_to_tp:
                                outcome = "TRAIL" if current_sl < original_sl else "SL"
                                exit_p = max(current_sl, reference_price + spread)
                                return exit_p, pd.Timestamp(entry_time_np), pd.Timestamp(bar_time), outcome
                            else:
                                exit_p = min(tp_price, reference_price + spread)
                                return exit_p, pd.Timestamp(entry_time_np), pd.Timestamp(bar_time), "TP"
                        elif tp_hit:
                            exit_p = min(tp_price, reference_price + spread)
                            return exit_p, pd.Timestamp(entry_time_np), pd.Timestamp(bar_time), "TP"
                        elif sl_hit:
                            outcome = "TRAIL" if current_sl < original_sl else "SL"
                            exit_p = max(current_sl, reference_price + spread)
                            return exit_p, pd.Timestamp(entry_time_np), pd.Timestamp(bar_time), outcome
                timeout_price = m1_closes[-1] + (spread if direction == "Sell" else 0)
                return timeout_price, pd.Timestamp(entry_time_np), pd.Timestamp(m1_times[-1]), "TIMEOUT"
            else:
                # Pandas loop
                for k in range(len(m1_df)):
                    bar_time = m1_times[k]
                    bar_open = m1_opens[k]
                    bar_high = m1_highs[k]
                    bar_low = m1_lows[k]
                    
                    if not entered:
                        if bar_time < entry_time:
                            continue
                        if direction == "Buy" and (bar_low + spread) <= entry_price:
                            entered = True
                            entry_time = bar_time
                        elif direction == "Sell" and bar_high >= entry_price:
                            entered = True
                            entry_time = bar_time
                        elif bar_time == entry_time and (
                            (direction == "Buy" and (bar_open + spread) <= entry_price) or
                            (direction == "Sell" and bar_open >= entry_price)):
                            entered = True
                            entry_time = bar_time
                        
                        if not entered:
                            if (bar_time - pd.Timestamp(df.index[start_index])).total_seconds() > cancel_limit:
                                return entry_price, pd.Timestamp(df.index[start_index]), bar_time, "CANCELLED"
                            else:
                                continue
                        is_fill_bar = True
                    
                    reference_price = bar_open
                    if direction == "Buy":
                        sl_hit = bar_low <= current_sl
                        if is_fill_bar and sl_hit:
                            is_fill_bar = False
                            exit_p = min(current_sl, reference_price)
                            return exit_p, entry_time, bar_time, "SL"
                        is_fill_bar = False
                        
                        if use_trailing_stop and initial_risk > 0:
                            if not breakeven_moved:
                                if bar_high - entry_price >= initial_risk:
                                    breakeven_moved = True
                                    current_sl = max(current_sl, entry_price + spread * 5)
                            else:
                                best_price = max(best_price, bar_high)
                                if trail_type == TRAIL_TYPE_PARTIAL:
                                    df_idx = df.index.searchsorted(bar_time)
                                    if df_idx >= len(df) or df.index[df_idx] > bar_time:
                                        df_idx -= 1
                                    if df_idx >= _last_df_idx + 15 or _last_df_idx == -1:
                                        _last_df_idx = df_idx
                                        lookback_window = df.iloc[max(0, df_idx-100):df_idx]
                                        if len(lookback_window) >= 15:
                                            _, swing_lows_t = detect_swing_points(lookback_window, lookback=2)
                                            if swing_lows_t:
                                                _cached_trail_sl = swing_lows_t[-1]['price'] - spread
                                    if _cached_trail_sl is not None:
                                        if current_sl < _cached_trail_sl < bar_open:
                                            current_sl = _cached_trail_sl
                                elif trail_type == TRAIL_TYPE_ATR:
                                    price_at_1r = entry_price + initial_risk
                                    if best_price > price_at_1r:
                                        points_above_1r = best_price - price_at_1r
                                        be_level = entry_price + spread * 5
                                        new_sl = be_level + (points_above_1r / 2.0)
                                        if new_sl > current_sl and new_sl < bar_open:
                                            current_sl = new_sl
                                elif trail_type == TRAIL_TYPE_PERCENT:
                                    trail_pct = trail_params.get('trail_pct', DEFAULT_TRAIL_PERCENT) / 100.0
                                    new_sl = best_price * (1.0 - trail_pct)
                                    if new_sl > current_sl and new_sl < bar_open:
                                        current_sl = new_sl

                        tp_hit = bar_high >= tp_price
                        if tp_hit and sl_hit:
                            dist_to_tp = tp_price - reference_price
                            dist_to_sl = reference_price - current_sl
                            if dist_to_sl <= dist_to_tp:
                                outcome = "TRAIL" if current_sl > original_sl else "SL"
                                exit_p = min(current_sl, reference_price)
                                return exit_p, entry_time, bar_time, outcome
                            else:
                                exit_p = max(tp_price, reference_price)
                                return exit_p, entry_time, bar_time, "TP"
                        elif tp_hit:
                            exit_p = max(tp_price, reference_price)
                            return exit_p, entry_time, bar_time, "TP"
                        elif sl_hit:
                            outcome = "TRAIL" if current_sl > original_sl else "SL"
                            exit_p = min(current_sl, reference_price)
                            return exit_p, entry_time, bar_time, outcome
                    else:
                        sl_hit = (bar_high + spread) >= current_sl
                        if is_fill_bar and sl_hit:
                            is_fill_bar = False
                            exit_p = max(current_sl, reference_price + spread)
                            return exit_p, entry_time, bar_time, "SL"
                        is_fill_bar = False
                        
                        if use_trailing_stop and initial_risk > 0:
                            if not breakeven_moved:
                                if entry_price - (bar_low + spread) >= initial_risk:
                                    breakeven_moved = True
                                    current_sl = min(current_sl, entry_price - spread * 5)
                            else:
                                best_price = min(best_price, bar_low)
                                if trail_type == TRAIL_TYPE_PARTIAL:
                                    df_idx = df.index.searchsorted(bar_time)
                                    if df_idx >= len(df) or df.index[df_idx] > bar_time:
                                        df_idx -= 1
                                    if df_idx >= _last_df_idx + 15 or _last_df_idx == -1:
                                        _last_df_idx = df_idx
                                        lookback_window = df.iloc[max(0, df_idx-100):df_idx]
                                        if len(lookback_window) >= 15:
                                            swing_highs_t, _ = detect_swing_points(lookback_window, lookback=2)
                                            if swing_highs_t:
                                                _cached_trail_sl = swing_highs_t[-1]['price'] + spread
                                    if _cached_trail_sl is not None:
                                        if current_sl > _cached_trail_sl > bar_open:
                                            current_sl = _cached_trail_sl
                                elif trail_type == TRAIL_TYPE_ATR:
                                    price_at_1r = entry_price - initial_risk
                                    if best_price < price_at_1r:
                                        points_below_1r = price_at_1r - best_price
                                        be_level = entry_price - spread * 5
                                        new_sl = be_level - (points_below_1r / 2.0)
                                        if new_sl < current_sl and new_sl > bar_open:
                                            current_sl = new_sl
                                elif trail_type == TRAIL_TYPE_PERCENT:
                                    trail_pct = trail_params.get('trail_pct', DEFAULT_TRAIL_PERCENT) / 100.0
                                    new_sl = best_price * (1.0 + trail_pct)
                                    if new_sl < current_sl and new_sl > bar_open:
                                        current_sl = new_sl

                        tp_hit = (bar_low + spread) <= tp_price
                        if tp_hit and sl_hit:
                            dist_to_tp = reference_price - tp_price
                            dist_to_sl = current_sl - reference_price
                            if dist_to_sl <= dist_to_tp:
                                outcome = "TRAIL" if current_sl < original_sl else "SL"
                                exit_p = max(current_sl, reference_price + spread)
                                return exit_p, entry_time, bar_time, outcome
                            else:
                                exit_p = min(tp_price, reference_price + spread)
                                return exit_p, entry_time, bar_time, "TP"
                        elif tp_hit:
                            exit_p = min(tp_price, reference_price + spread)
                            return exit_p, entry_time, bar_time, "TP"
                        elif sl_hit:
                            outcome = "TRAIL" if current_sl < original_sl else "SL"
                            exit_p = max(current_sl, reference_price + spread)
                            return exit_p, entry_time, bar_time, outcome
                timeout_price = m1_closes[-1] + (spread if direction == "Sell" else 0)
                return timeout_price, entry_time, pd.Timestamp(m1_times[-1]), "TIMEOUT"

    # Fallback to current TF if M1 drilldown is skipped or fails
    entered = False
    is_first_bar = True  # First bar after entry = fill bar (pessimistic SL)
    fallback_df = df.iloc[start_index + 1 : start_index + 500]
    for bars_passed, row in enumerate(fallback_df.itertuples(), start=1):
        bar = {'open': row.open, 'high': row.high, 'low': row.low, 'close': row.close}
        bar_time = row.Index
        
        if not entered:
            if direction == "Buy" and (bar['low'] + spread) <= entry_price:
                entered = True
                entry_time = bar_time
            elif direction == "Sell" and bar['high'] >= entry_price:
                entered = True
                entry_time = bar_time
            
            if not entered:
                tf_bar_duration = (df.index[1] - df.index[0]).total_seconds() if len(df) > 1 else 900
                tf_bar_duration = max(1.0, tf_bar_duration)
                cancel_limit = 3600 if ict_method == "FVG Return" else 7200
                bars_limit = max(1, int(cancel_limit / tf_bar_duration))
                if bars_passed > bars_limit:
                    return entry_price, pd.Timestamp(df.index[start_index]), bar_time, "CANCELLED"
                else:
                    continue

        reference_price = bar['open']
        
        if direction == "Buy":
            sl_hit = bar['low'] <= current_sl
            
            # On fill bar (first bar after entry), SL hit = immediate SL
            if is_first_bar and sl_hit:
                exit_p = min(current_sl, reference_price)
                return exit_p, entry_time, bar_time, "SL"
            
            if use_trailing_stop and initial_risk > 0:
                if not breakeven_moved:
                    if bar['high'] - entry_price >= initial_risk:
                        breakeven_moved = True
                        current_sl = max(current_sl, entry_price + spread * 5)
                else:
                    best_price = max(best_price, bar['high'])
                    if trail_type == TRAIL_TYPE_PARTIAL:
                        # Fixed j lookup by index search
                        df_idx = df.index.searchsorted(bar_time)
                        if df_idx >= len(df) or df.index[df_idx] > bar_time:
                            df_idx -= 1
                        lookback_window = df.iloc[max(0, df_idx-100):df_idx]
                        if len(lookback_window) >= 15:
                            _, swing_lows = detect_swing_points(lookback_window, lookback=2)
                            if swing_lows:
                                recent_sl = swing_lows[-1]['price'] - spread
                                if current_sl < recent_sl < bar['open']:
                                    current_sl = recent_sl
                    elif trail_type == TRAIL_TYPE_ATR:
                        price_at_1r = entry_price + initial_risk
                        if best_price > price_at_1r:
                            points_above_1r = best_price - price_at_1r
                            be_level = entry_price + spread * 5
                            new_sl = be_level + (points_above_1r / 2.0)
                            if new_sl > current_sl and new_sl < bar['open']:
                                current_sl = new_sl
                    elif trail_type == TRAIL_TYPE_PERCENT:
                        trail_pct = trail_params.get('trail_pct', DEFAULT_TRAIL_PERCENT) / 100.0
                        new_sl = best_price * (1.0 - trail_pct)
                        if new_sl > current_sl and new_sl < bar['open']:
                            current_sl = new_sl

            tp_hit = bar['high'] >= tp_price
            if tp_hit and sl_hit:
                dist_to_tp = tp_price - reference_price
                dist_to_sl = reference_price - current_sl
                if dist_to_sl <= dist_to_tp:
                    outcome = "TRAIL" if current_sl > original_sl else "SL"
                    exit_p = min(current_sl, reference_price)
                    return exit_p, entry_time, bar_time, outcome
                else:
                    exit_p = max(tp_price, reference_price)
                    return exit_p, entry_time, bar_time, "TP"
            elif tp_hit:
                exit_p = max(tp_price, reference_price)
                return exit_p, entry_time, bar_time, "TP"
            elif sl_hit:
                outcome = "TRAIL" if current_sl > original_sl else "SL"
                exit_p = min(current_sl, reference_price)
                return exit_p, entry_time, bar_time, outcome
        else:
            sl_hit = (bar['high'] + spread) >= current_sl
            
            # On fill bar (first bar after entry), SL hit = immediate SL
            if is_first_bar and sl_hit:
                exit_p = max(current_sl, reference_price + spread)
                return exit_p, entry_time, bar_time, "SL"
            
            if use_trailing_stop and initial_risk > 0:
                if not breakeven_moved:
                    if entry_price - (bar['low'] + spread) >= initial_risk:
                        breakeven_moved = True
                        current_sl = min(current_sl, entry_price - spread * 5)
                else:
                    best_price = min(best_price, bar['low'])
                    if trail_type == TRAIL_TYPE_PARTIAL:
                        df_idx = df.index.searchsorted(bar_time)
                        if df_idx >= len(df) or df.index[df_idx] > bar_time:
                            df_idx -= 1
                        lookback_window = df.iloc[max(0, df_idx-100):df_idx]
                        if len(lookback_window) >= 15:
                            swing_highs, _ = detect_swing_points(lookback_window, lookback=2)
                            if swing_highs:
                                recent_sl = swing_highs[-1]['price'] + spread
                                if current_sl > recent_sl > bar['open']:
                                    current_sl = recent_sl
                    elif trail_type == TRAIL_TYPE_ATR:
                        price_at_1r = entry_price - initial_risk
                        if best_price < price_at_1r:
                            points_below_1r = price_at_1r - best_price
                            be_level = entry_price - spread * 5
                            new_sl = be_level - (points_below_1r / 2.0)
                            if new_sl < current_sl and new_sl > bar['open']:
                                current_sl = new_sl
                    elif trail_type == TRAIL_TYPE_PERCENT:
                        trail_pct = trail_params.get('trail_pct', DEFAULT_TRAIL_PERCENT) / 100.0
                        new_sl = best_price * (1.0 + trail_pct)
                        if new_sl < current_sl and new_sl > bar['open']:
                            current_sl = new_sl

            tp_hit = (bar['low'] + spread) <= tp_price
            if tp_hit and sl_hit:
                dist_to_tp = reference_price - tp_price
                dist_to_sl = current_sl - reference_price
                if dist_to_sl <= dist_to_tp:
                    outcome = "TRAIL" if current_sl < original_sl else "SL"
                    exit_p = max(current_sl, reference_price + spread)
                    return exit_p, entry_time, bar_time, outcome
                else:
                    exit_p = min(tp_price, reference_price + spread)
                    return exit_p, entry_time, bar_time, "TP"
            elif tp_hit:
                exit_p = min(tp_price, reference_price + spread)
                return exit_p, entry_time, bar_time, "TP"
            elif sl_hit:
                outcome = "TRAIL" if current_sl < original_sl else "SL"
                exit_p = max(current_sl, reference_price + spread)
                return exit_p, entry_time, bar_time, outcome
        is_first_bar = False  # Only the first bar after entry gets pessimistic treatment
    if not fallback_df.empty:
        timeout_price = fallback_df.iloc[-1]['close'] + (spread if direction == "Sell" else 0)
        return timeout_price, entry_time, fallback_df.index[-1], "TIMEOUT"
    else:
        timeout_price = df.iloc[-1]['close'] + (spread if direction == "Sell" else 0)
        return timeout_price, entry_time, df.index[-1], "TIMEOUT"

# -------------------------
# TP/SL Calculation
# -------------------------
# (calculate_tp_sl / get_higher_tf_trend moved to simulation_scoring.py)
