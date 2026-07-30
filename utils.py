import pandas as pd
import numpy as np
import datetime
import logging
from broker import get_broker, TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, \
    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1
from config import CORE_METHODS, ICT_METHODS, LEGACY_OB_RETEST_METHODS, stop_event, bt_stop_event

logger = logging.getLogger()

import threading

cache_locks = {}
cache_locks_lock = threading.Lock()

def get_cache_lock(cache_path):
    with cache_locks_lock:
        if cache_path not in cache_locks:
            cache_locks[cache_path] = threading.Lock()
        return cache_locks[cache_path]


def resolve_ict_methods(ict_method):
    """Normalize UI/API method selection into concrete detector names."""
    if isinstance(ict_method, (list, tuple, set)):
        requested = [m for m in ict_method if m]
    elif ict_method:
        requested = [ict_method]
    else:
        requested = []

    if "Core Methods" in requested:
        return CORE_METHODS[:]
    if "All Methods" in requested:
        return ICT_METHODS[:]

    allowed = set(ICT_METHODS) | LEGACY_OB_RETEST_METHODS
    methods, unknown = [], []
    for method in requested:
        if method in allowed:
            if method not in methods:
                methods.append(method)
        else:
            unknown.append(method)
    if unknown:
        # Silently falling back to CORE_METHODS made four "optimization runs"
        # in run_headless_opt secretly identical. Never again.
        raise ValueError(
            f"Unknown ICT method(s): {unknown}. Valid: {sorted(allowed)}"
        )
    if not methods:
        raise ValueError("resolve_ict_methods received no valid methods")
    return methods

def get_legacy_ob_retest_methods(methods_to_run):
    """Return only explicit legacy OB methods; avoids cloning every method over OB entries."""
    return [m for m in methods_to_run if m in LEGACY_OB_RETEST_METHODS]

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit
    def _wilders_smooth_full_njit(tr, period, initial_atr):
        atr = np.zeros(len(tr))
        atr[0] = initial_atr
        alpha = 1.0 / period
        for j in range(1, len(tr)):
            atr[j] = atr[j-1] * (1 - alpha) + tr[j] * alpha
        return atr
else:
    def _wilders_smooth_full_njit(tr, period, initial_atr):
        atr = np.zeros(len(tr))
        atr[0] = initial_atr
        alpha = 1.0 / period
        for j in range(1, len(tr)):
            atr[j] = atr[j-1] * (1 - alpha) + tr[j] * alpha
        return atr


def calculate_atr(df, period=14):
    from atr_context import atr_last
    return atr_last(df, period)

def fvg_size_ok(fvg_size, spread, atr, spread_mult, atr_fraction=None):
    """Volatility-aware FVG-size gate (shared live+backtest).
    atr_fraction now defaults to config.FVG_MIN_ATR_FRACTION rather than 0.0,
    because every caller was silently getting the spread-only legacy path.
    """
    import config
    if atr_fraction is None:
        atr_fraction = float(getattr(config, 'FVG_MIN_ATR_FRACTION', 0.20))
    thresholds = []
    if spread_mult and spread_mult > 0 and spread and spread > 0:
        thresholds.append(spread * spread_mult)
    if atr_fraction > 0 and atr and atr > 0:
        thresholds.append(atr * atr_fraction)
    if not thresholds:
        return True
    return fvg_size >= max(thresholds)

def get_contract_size(symbol):
    import config
    if not getattr(config, 'OFFLINE_BACKTESTING', False):
        broker = get_broker()
        if hasattr(broker, 'symbol_info'):
            symbol_info = broker.symbol_info(symbol)
            if symbol_info is not None:
                return symbol_info.trade_contract_size
    
    # Offline fallback
    sym_u = symbol.upper()
    if "XAU" in sym_u or "XAG" in sym_u:
        return 100.0
    elif "BTC" in sym_u or "ETH" in sym_u:
        return 1.0
    elif "USD" not in sym_u[-4:] and "JPY" not in sym_u[-4:]:
        return 1.0
    else:
        return 100000.0

OFFLINE_VOLUME_SPECS = {
    'XAU': {'min': 0.01, 'step': 0.01, 'max': 50.0},
    'XAG': {'min': 0.01, 'step': 0.01, 'max': 50.0},
    'BTC': {'min': 0.01, 'step': 0.01, 'max': 10.0},
    'ETH': {'min': 0.01, 'step': 0.01, 'max': 20.0},
    'US30': {'min': 0.10, 'step': 0.10, 'max': 100.0},
    'US500': {'min': 0.10, 'step': 0.10, 'max': 100.0},
    'USTEC': {'min': 0.10, 'step': 0.10, 'max': 100.0},
    'DEFAULT': {'min': 0.01, 'step': 0.01, 'max': 200.0},
}

def _offline_volume_spec(symbol):
    sym_u = (symbol or "").upper()
    for key, spec in OFFLINE_VOLUME_SPECS.items():
        if key != 'DEFAULT' and key in sym_u:
            return spec
    return OFFLINE_VOLUME_SPECS['DEFAULT']

def _split_to_chunks(raw_lot_size, vol_min, vol_step, vol_max):
    """Shared quantisation + chunking. Returns [] when below vol_min so that
    live and backtest SKIP the trade identically (never a 0-lot phantom)."""
    if vol_step and vol_step > 0:
        step_str = f"{vol_step:.8f}".rstrip('0')
        decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
        lot = round(round(raw_lot_size / vol_step) * vol_step, decimals)
    else:
        decimals = 2
        lot = round(raw_lot_size, 2)

    if lot < vol_min:
        return []

    lots = []
    while lot >= vol_min:
        chunk = min(lot, vol_max)
        if vol_step and vol_step > 0:
            chunk = round(round(chunk / vol_step) * vol_step, decimals)
        else:
            chunk = round(chunk, 2)
        if chunk < vol_min:
            break
        lots.append(chunk)
        lot = round(lot - chunk, decimals)
    return lots

def normalize_and_split_lots(symbol, raw_lot_size):
    import config
    if raw_lot_size is None or raw_lot_size <= 0:
        return []

    if getattr(config, 'OFFLINE_BACKTESTING', False):
        spec = _offline_volume_spec(symbol)
        return _split_to_chunks(raw_lot_size, spec['min'], spec['step'], spec['max'])

    broker = get_broker()
    try:
        symbol_info = broker.symbol_info(symbol)
        if symbol_info is None:
            broker.symbol_select(symbol, True)
            symbol_info = broker.symbol_info(symbol)
    except Exception:
        symbol_info = None

    if not symbol_info:
        spec = _offline_volume_spec(symbol)
        return _split_to_chunks(raw_lot_size, spec['min'], spec['step'], spec['max'])

    return _split_to_chunks(
        raw_lot_size,
        getattr(symbol_info, 'volume_min', 0.01),
        getattr(symbol_info, 'volume_step', 0.01),
        getattr(symbol_info, 'volume_max', 100.0),
    )

def wait_for_seconds(seconds):
    for _ in range(seconds):
        if stop_event.is_set() or bt_stop_event.is_set():
            return True
        if stop_event.wait(0.5) or bt_stop_event.wait(0.5):
            return True
    return False

CACHE_DIR = "history_cache"

def get_data(symbol, timeframe, num_bars, live=False, date_to=None):
    import config
    offline = getattr(config, 'OFFLINE_BACKTESTING', False)
    
    if live:
        broker = get_broker()
        if not offline:
            try:
                broker.symbol_select(symbol, True)
            except Exception:
                pass
        rates = broker.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    else:
        if date_to:
            import config
            offline = getattr(config, 'OFFLINE_BACKTESTING', False)
            cache_mode = "offline" if offline else "online"
            full_cache_key = (symbol, timeframe, cache_mode)
            if offline:
                if full_cache_key not in _full_history_cache:
                    get_data_by_date(symbol, timeframe, date_to, date_to, suppress_warning=True)
                
                if full_cache_key in _full_history_cache:
                    cached_df = _full_history_cache[full_cache_key]
                    dt = pd.Timestamp(date_to).replace(tzinfo=None)
                    idx = cached_df.index.searchsorted(dt, side='right')
                    if idx > 0:
                        return cached_df.iloc[max(0, idx - num_bars):idx]
                    return None

            tf_minutes_map = {
                TIMEFRAME_M1: 1, TIMEFRAME_M5: 5, TIMEFRAME_M15: 15,
                TIMEFRAME_H1: 60, TIMEFRAME_H4: 240, TIMEFRAME_D1: 1440,
                32769: 10080, 49153: 43200
            }
            tf_min = tf_minutes_map.get(timeframe, 15)
            lookback_minutes = int(tf_min * num_bars * 3.0)
            start_date = date_to - datetime.timedelta(minutes=lookback_minutes)
            df = get_data_by_date(symbol, timeframe, start_date, date_to)
            if df is not None and len(df) > num_bars:
                df = df[-num_bars:]
            return df
        else:
            rates = broker.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
            if rates is None or len(rates) == 0:
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df

_full_history_cache = {}
_last_mt5_sync_time = {}

def _load_sync_limits():
    global _last_mt5_sync_time
    import os
    import json
    path = os.path.join(CACHE_DIR, "mt5_sync_limits.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                for k, v in data.items():
                    parts = k.split(",")
                    if len(parts) == 2:
                        _last_mt5_sync_time[(parts[0], int(parts[1]))] = v
        except Exception:
            pass

def _save_sync_limits():
    import os
    import json
    path = os.path.join(CACHE_DIR, "mt5_sync_limits.json")
    try:
        data = {f"{k[0]},{k[1]}": v for k, v in _last_mt5_sync_time.items()}
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

# Load limits on module import
_load_sync_limits()

def get_data_by_date(symbol, timeframe, date_from, date_to, suppress_warning=False):
    """Fetch data between two dates, utilizing a local CSV and binary PKL history cache with MT5 fallback."""
    import os
    import pandas as pd
    import datetime
    import numpy as np
    import config
    
    offline = getattr(config, 'OFFLINE_BACKTESTING', False)
    cache_mode = "offline" if offline else "online"
    full_cache_key = (symbol, timeframe, cache_mode)
        
    # Ensure date_from and date_to are standard timezone-naive datetime.datetime objects
    # to avoid compatibility issues with the MT5 Python C-extension
    if hasattr(date_from, 'to_pydatetime'):
        date_from = date_from.to_pydatetime()
    elif isinstance(date_from, pd.Timestamp):
        date_from = date_from.to_pydatetime()
    elif isinstance(date_from, np.datetime64):
        date_from = pd.Timestamp(date_from).to_pydatetime()
        
    if hasattr(date_to, 'to_pydatetime'):
        date_to = date_to.to_pydatetime()
    elif isinstance(date_to, pd.Timestamp):
        date_to = date_to.to_pydatetime()
    elif isinstance(date_to, np.datetime64):
        date_to = pd.Timestamp(date_to).to_pydatetime()

    if isinstance(date_from, datetime.datetime):
        date_from = date_from.replace(tzinfo=None)
    if isinstance(date_to, datetime.datetime):
        date_to = date_to.replace(tzinfo=None)

    # Cap date_to at current local time + 1 day to handle future ranges gracefully without breaking cache checks or expecting future bars
    now = datetime.datetime.now()
    if date_to > now + datetime.timedelta(days=1):
        date_to = now + datetime.timedelta(days=1)

    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR)
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            
    timeframe_names = {
        1: "M1", 2: "M2", 3: "M3", 4: "M4", 5: "M5", 6: "M6", 10: "M10", 12: "M12",
        15: "M15", 20: "M20", 30: "M30", 16385: "H1", 16386: "H2", 16387: "H3",
        16388: "H4", 16390: "H6", 16392: "H8", 16396: "H12", 16408: "D1",
        32769: "W1", 49153: "MN1"
    }
    tf_str = timeframe_names.get(timeframe, str(timeframe))

    if offline:
        cache_file_csv = os.path.join(CACHE_DIR, f"{symbol}_{tf_str}.csv")
        cache_file_pkl = os.path.join(CACHE_DIR, f"{symbol}_{tf_str}.pkl")
        
        if not os.path.exists(cache_file_pkl) and not os.path.exists(cache_file_csv):
            for alt_dir in ["QuantHistoryFull", "QuantHistory"]:
                alt_csv = os.path.join(alt_dir, f"{symbol}_{tf_str}.csv")
                alt_pkl = os.path.join(alt_dir, f"{symbol}_{tf_str}.pkl")
                if os.path.exists(alt_pkl) or os.path.exists(alt_csv):
                    cache_file_csv = alt_csv
                    cache_file_pkl = alt_pkl
                    break
        save_cache_file_csv = os.path.join(CACHE_DIR, f"{symbol}_{tf_str}.csv")
        save_cache_file_pkl = os.path.join(CACHE_DIR, f"{symbol}_{tf_str}.pkl")
    else:
        online_dir = os.path.join(CACHE_DIR, "Online")
        if not os.path.exists(online_dir):
            try:
                os.makedirs(online_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create Online cache directory: {e}")
                
        cache_file_csv = os.path.join(online_dir, f"{symbol}_{tf_str}.csv")
        cache_file_pkl = os.path.join(online_dir, f"{symbol}_{tf_str}.pkl")
        
        if not os.path.exists(cache_file_pkl) and not os.path.exists(cache_file_csv):
            for alt_dir in [
                os.path.join("data", "Online"),
                os.path.join(CACHE_DIR, "Online"),
                os.path.join("QuantHistoryFull", "Online"),
                os.path.join("QuantHistory", "Online"),
                "Online"
            ]:
                alt_csv = os.path.join(alt_dir, f"{symbol}_{tf_str}.csv")
                alt_pkl = os.path.join(alt_dir, f"{symbol}_{tf_str}.pkl")
                if os.path.exists(alt_pkl) or os.path.exists(alt_csv):
                    cache_file_csv = alt_csv
                    cache_file_pkl = alt_pkl
                    break
        save_cache_file_csv = os.path.join(online_dir, f"{symbol}_{tf_str}.csv")
        save_cache_file_pkl = os.path.join(online_dir, f"{symbol}_{tf_str}.pkl")
    
    cached_df = _full_history_cache.get(full_cache_key)
    cache_start = None
    cache_end = None
    
    if cached_df is None:
        if os.path.exists(cache_file_pkl):
            try:
                with get_cache_lock(cache_file_pkl):
                    cached_df = pd.read_pickle(cache_file_pkl)
                    if not cached_df.index.is_monotonic_increasing:
                        cached_df.sort_index(inplace=True)
                    _full_history_cache[full_cache_key] = cached_df
            except Exception as e:
                logger.error(f"Failed to read PKL cache {cache_file_pkl}: {e}")
                try:
                    os.remove(cache_file_pkl)
                except Exception:
                    pass
    
        if cached_df is None and os.path.exists(cache_file_csv):
            try:
                with get_cache_lock(cache_file_csv):
                    logger.info(f"Loading large CSV file {cache_file_csv}...")
                    cached_df = pd.read_csv(cache_file_csv, parse_dates=['time'], index_col='time')
                    cached_df.sort_index(inplace=True)
                    # Convert to PKL for ultra-fast subsequent loads
                    logger.info(f"Converting {cache_file_csv} to PKL for light-speed access...")
                    cached_df.to_pickle(cache_file_pkl)
                    _full_history_cache[full_cache_key] = cached_df
            except Exception as e:
                logger.error(f"Failed to read CSV cache {cache_file_csv}: {e}")

    # If we successfully loaded the cache (either via PKL or CSV), check the range
    if cached_df is not None and not cached_df.empty:
        import config
        if getattr(config, 'OFFLINE_BACKTESTING', False):
            try:
                # Ensure date_from and date_to are naive timestamps
                d_from = pd.Timestamp(date_from).replace(tzinfo=None)
                d_to = pd.Timestamp(date_to).replace(tzinfo=None)
                
                # Check if index is tz-aware and make it naive for slicing
                df_to_slice = cached_df.copy(deep=False)
                if df_to_slice.index.tz is not None:
                    df_to_slice.index = df_to_slice.index.tz_convert(None)
                    
                # Use boolean masking instead of .loc to bypass all non-unique/missing key edge cases
                mask = (df_to_slice.index >= d_from) & (df_to_slice.index <= d_to)
                sliced = df_to_slice.loc[mask]
                
                if not sliced.empty:
                    return sliced
                else:
                    if not suppress_warning:
                        logger.warning(f"[Offline] No local history for {symbol} {tf_str} in requested range ({d_from} to {d_to}). Cache bounds: {df_to_slice.index[0]} to {df_to_slice.index[-1]}")
                    return None
            except Exception as e:
                logger.error(f"[Offline] Error slicing {symbol} {tf_str}: {e}")
                return None

        cache_start = cached_df.index[0].to_pydatetime().replace(tzinfo=None)
        cache_end = cached_df.index[-1].to_pydatetime().replace(tzinfo=None)
        
        if cache_start <= date_from and cache_end >= date_to:
            sliced = cached_df.loc[date_from:date_to]
            if not sliced.empty:
                return sliced

    import config
    if getattr(config, 'OFFLINE_BACKTESTING', False):
        logger.warning(f"[Offline] No cache file found for {symbol} {tf_str}, and MT5 fetch is disabled.")
        return None

    # Rate limit MT5 queries to once every 10 minutes (600 seconds) for the same symbol/timeframe
    import time
    sync_key = (symbol, timeframe)
    current_time_sec = time.time()
    
    last_sync = _last_mt5_sync_time.get(sync_key, 0)
    if current_time_sec - last_sync < 60:
        if cached_df is not None and not cached_df.empty:
            sliced = cached_df.loc[date_from:date_to]
            if not sliced.empty:
                return sliced
        return None

    broker = get_broker()
    # Ensure symbol is selected in Market Watch
    broker.symbol_select(symbol, True)
    
    # Adjust query_from if cache covers the beginning
    query_from = date_from
    if cached_df is not None and not cached_df.empty:
        if cache_start <= date_from and cache_end < date_to:
            query_from = cache_end

    # Estimate the expected number of active trading bars to detect incomplete downloads (forex active ~71.4% of elapsed time)
    tf_minutes_map = {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 10: 10, 12: 12,
        15: 15, 20: 20, 30: 30, 16385: 60, 16386: 120, 16387: 180,
        16388: 240, 16390: 360, 16392: 480, 16396: 720, 16408: 1440,
        32769: 10080, 49153: 43200
    }
    tf_min = tf_minutes_map.get(timeframe, 15)
    total_minutes = (date_to - query_from).total_seconds() / 60
    expected_bars = int((total_minutes / tf_min) * 0.62)
    
    rates = None
    max_retries = 8
    prev_len = -1
    no_growth_attempts = 0
    
    # Check if broker is connected before requesting data. If disconnected, skip query and fallback to cache.
    is_connected = broker.connected
    
    if is_connected:
        _last_mt5_sync_time[sync_key] = current_time_sec
        _save_sync_limits()
        # Try fetching from broker with a retry loop to allow history download in the background
        for attempt in range(max_retries):
            rates = broker.copy_rates_range(symbol, timeframe, query_from, date_to)
            current_len = len(rates) if rates is not None else 0
            
            if current_len > 0:
                # If we fetched close to expected, or if expected_bars is small, or if it has reached a robust count
                if expected_bars <= 50 or current_len >= int(expected_bars * 0.85):
                    break
                    
                # Early break if successive attempts show no growth (sync already complete or unavailable)
                if current_len == prev_len:
                    no_growth_attempts += 1
                    if no_growth_attempts >= 2:
                        break
                else:
                    no_growth_attempts = 0
                    
                prev_len = current_len
                logger.info(f"[MT5 Sync] Symbol {symbol} {tf_str} returned {current_len}/{expected_bars} estimated bars. Retrying to allow background sync (attempt {attempt + 1}/{max_retries})...")
            else:
                logger.info(f"[MT5 Sync] Symbol {symbol} {tf_str} returned no data. Falling back to cache.")
                break
                
            # Call copy_rates_from_pos to trigger background history retrieval from the broker server
            broker.copy_rates_from_pos(symbol, timeframe, 0, min(15000, max(2000, expected_bars)))
            
            # Sleep for 1 second (respecting stop event)
            if wait_for_seconds(1):
                break
    else:
        logger.info(f"[MT5 Sync] Symbol {symbol} {tf_str} terminal is offline or disconnected. Skipping sync query and falling back to cache.")
            
    if rates is not None and len(rates) > 0:
        df_new = pd.DataFrame(rates)
        df_new['time'] = pd.to_datetime(df_new['time'], unit='s')
        df_new.set_index('time', inplace=True)
        
        # Merge new data with cached data
        if cached_df is not None and not cached_df.empty:
            df_merged = pd.concat([cached_df, df_new])
            df_merged = df_merged[~df_merged.index.duplicated(keep='last')].sort_index()
        else:
            df_merged = df_new
            
        _full_history_cache[full_cache_key] = df_merged
        
        # Save cache asynchronously in a background thread to prevent blocking the UI/backtester
        # Save cache asynchronously in a background thread to prevent blocking the UI/backtester
        def save_cache_async(df_to_save, path_csv, path_pkl):
            import pandas as pd
            import os
            try:
                cached_bg = None
                if os.path.exists(path_pkl):
                    try:
                        with get_cache_lock(path_pkl):
                            cached_bg = pd.read_pickle(path_pkl)
                    except Exception as e:
                        logger.error(f"Async cache: Failed to read PKL {path_pkl}: {e}")
                        try:
                            os.remove(path_pkl)
                        except Exception:
                            pass
                
                if cached_bg is None and os.path.exists(path_csv):
                    try:
                        with get_cache_lock(path_csv):
                            cached_bg = pd.read_csv(path_csv, parse_dates=['time'], index_col='time')
                    except Exception as e:
                        logger.error(f"Async cache: Failed to read CSV {path_csv}: {e}")
                
                if cached_bg is not None and not cached_bg.empty:
                    new_ts = df_to_save.index.difference(cached_bg.index)
                    if not new_ts.empty:
                        df_merged_bg = pd.concat([cached_bg, df_to_save])
                        df_merged_bg = df_merged_bg[~df_merged_bg.index.duplicated(keep='last')].sort_index()
                    else:
                        df_merged_bg = cached_bg
                else:
                    df_merged_bg = df_to_save.sort_index()
                
                with get_cache_lock(path_pkl):
                    df_merged_bg.to_pickle(path_pkl)
                with get_cache_lock(path_csv):
                    df_merged_bg.to_csv(path_csv)
            except Exception as e:
                logger.error(f"Async cache save failed: {e}")

        import threading
        threading.Thread(target=save_cache_async, args=(df_new.copy(), save_cache_file_csv, save_cache_file_pkl), daemon=True).start()
        
        sliced_df = df_merged.loc[date_from:date_to]
        return sliced_df
        
    else:
        # Fallback to cache.
        if is_connected:
            logger.warning(f"MT5 copy_rates_range returned no data for {symbol}. Attempting cache fallback...")
        else:
            logger.info(f"Using local cache fallback for {symbol} {tf_str} since MT5 is offline.")
            
        if cached_df is not None and not cached_df.empty:
            sliced = cached_df.loc[date_from:date_to]
            if not sliced.empty:
                return sliced
        return None

def get_spread(symbol):
    """Current spread in PRICE units.

    Delegates to the shared cost_model so offline runs get a realistic
    per-symbol default instead of 0.0 (which used to silently disable every
    spread-based cost and safety buffer).
    """
    import config
    from cost_model import resolve_spread
    if getattr(config, 'OFFLINE_BACKTESTING', False):
        return resolve_spread(symbol)
    broker = get_broker()
    sp = broker.spread_from_tick(symbol)
    if sp and sp > 0:
        return sp
    return resolve_spread(symbol)

def check_rrr(entry_price, tp, sl, direction, min_rrr):
    if direction == "Buy":
        risk = entry_price - sl
        reward = tp - entry_price
    else:
        risk = sl - entry_price
        reward = entry_price - tp
    # Rounding to `digits` can collapse a tight setup to zero risk.
    if risk <= 0 or reward <= 0:
        return False, 0.0
    rrr = reward / risk
    # Use 1e-5 tolerance to prevent floating point inaccuracies (e.g. 5.999999 printed as 6.00 < 6.00)
    if rrr < (min_rrr - 1e-5):
        return False, rrr
    return True, rrr

def is_in_trading_session(timestamp, session_start=7, session_end=21):
    """Check if timestamp is within trading session (NY hours)."""
    import pytz
    import config
    try:
        broker_tz = pytz.timezone(config.BROKER_TIMEZONE)
        ny_tz = pytz.timezone('America/New_York')
        if getattr(timestamp, 'tzinfo', None) is not None:
            localized_time = timestamp.astimezone(broker_tz)
        else:
            localized_time = broker_tz.localize(timestamp)
        ny_time = localized_time.astimezone(ny_tz)
        hour = ny_time.hour
    except Exception:
        hour = timestamp.hour
        
    if session_start <= session_end:
        return session_start <= hour < session_end
    else:
        return hour >= session_start or hour < session_end

def is_in_news_embargo(timestamp):
    """Check if the current broker timestamp falls into ICT News Embargo windows.
    - 08:20 to 08:35 NY
    - 09:50 to 10:10 NY
    """
    try:
        from detectors import convert_time_to_ny_hour
        ny_hour = convert_time_to_ny_hour(timestamp)
        # Embargo 1: 08:20 - 08:35 AM NY (Major News)
        if (8.0 + 20.0 / 60.0) <= ny_hour < (8.0 + 35.0 / 60.0):
            return True
        # Embargo 2: 09:50 - 10:10 AM NY (Silver Bullet / Open Macro)
        if (9.0 + 50.0 / 60.0) <= ny_hour < (10.0 + 10.0 / 60.0):
            return True
        return False
    except Exception:
        return False
        logger.warning("[NEWS-EMBARGO] Check failed (%s); failing closed (blocking trade).", e)
        return True

def is_crypto_symbol(symbol):
    """Crypto trades 24/7 — session filter should not apply."""
    sym = symbol.upper()
    return any(c in sym for c in ['BTC', 'ETH', 'LTC', 'XRP', 'BNB', 'SOL', 'DOGE', 'ADA', 'DOT'])

_symbol_profiles_cache = {"mtime": None, "path": None, "data": {}}


def _load_symbol_profiles():
    """Load and cache symbol_profiles.json, reloading only when the file changes."""
    import os
    import json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbol_profiles.json")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _symbol_profiles_cache.update({"mtime": None, "path": path, "data": {}})
        return {}
    if _symbol_profiles_cache["mtime"] != mtime or _symbol_profiles_cache["path"] != path:
        try:
            with open(path, "r") as f:
                _symbol_profiles_cache["data"] = json.load(f)
            _symbol_profiles_cache["mtime"] = mtime
            _symbol_profiles_cache["path"] = path
        except Exception as e:
            logger.error(f"Error loading symbol_profiles.json: {e}")
            _symbol_profiles_cache["data"] = {}
    return _symbol_profiles_cache["data"]


def match_symbol_profile(symbol, profiles=None):
    """Return the raw profile dict for a symbol, or None.

    Matching order (safe; no dangerous catch-all / substring pollution):
      1. case-insensitive exact key match
      2. longest profile key that the symbol *starts with* (e.g. 'BTC' -> 'BTCUSDm')
      3. GOLD / BTC aliases
    """
    if profiles is None:
        profiles = _load_symbol_profiles()
    if not profiles:
        return None
    su = str(symbol).upper()
    # 1. exact (case-insensitive)
    for k, v in profiles.items():
        if str(k).upper() == su:
            return v
    # 2. longest key that is a prefix of the symbol
    best, best_len = None, 0
    for k, v in profiles.items():
        ku = str(k).upper()
        if len(ku) >= 3 and su.startswith(ku) and len(ku) > best_len:
            best, best_len = v, len(ku)
    if best is not None:
        return best
    # 3. safe aliases
    for k, v in profiles.items():
        ku = str(k).upper()
        if ku == "GOLD" and "XAU" in su:
            return v
        if ku == "BTC" and "BTC" in su:
            return v
    return None


def get_symbol_profile_overrides(symbol, default_params):
    """
    Load overrides from symbol_profiles.json for the given symbol and merge with default_params.
    Supports case-insensitive matching, partial matches, and special fallbacks.
    """
    import os
    import json
    
    profiles = _load_symbol_profiles()
    profile = match_symbol_profile(symbol, profiles)
                
    if not profile:
        res = default_params.copy()
        if "methods" not in res:
            res["methods"] = {}
        return res
        
    res = default_params.copy()
    if "methods" not in res:
        res["methods"] = {}
        
    for field, val in profile.items():
        if field in ('ict_method', 'trail_methods'):
            continue
        if field == "fvg_sl_mode" and isinstance(val, str):
            val_upper = val.upper()
            import config
            if "NORMAL" in val_upper:
                val = config.FVG_SL_NORMAL
            elif "SWEEP" in val_upper:
                val = config.FVG_SL_SWEEP
            elif "CANDLE" in val_upper or "EXTREME" in val_upper:
                val = config.FVG_SL_CANDLE
            elif "BOS" in val_upper:
                val = config.FVG_SL_BOS
            elif "MSS" in val_upper:
                val = config.FVG_SL_MSS
        
        if field == "trail_type" and isinstance(val, str) and val.upper() == "NONE":
            val = None
            
        if field == "methods" and isinstance(val, dict):
            normalized_methods = {}
            for method_name, method_overrides in val.items():
                if isinstance(method_overrides, dict):
                    normalized_override = {}
                    for m_field, m_val in method_overrides.items():
                        if m_field == "fvg_sl_mode" and isinstance(m_val, str):
                            m_val_upper = m_val.upper()
                            import config
                            if "NORMAL" in m_val_upper:
                                m_val = config.FVG_SL_NORMAL
                            elif "SWEEP" in m_val_upper:
                                m_val = config.FVG_SL_SWEEP
                            elif "CANDLE" in m_val_upper or "EXTREME" in m_val_upper:
                                m_val = config.FVG_SL_CANDLE
                            elif "BOS" in m_val_upper:
                                m_val = config.FVG_SL_BOS
                            elif "MSS" in m_val_upper:
                                m_val = config.FVG_SL_MSS
                        if m_field == "trail_type" and isinstance(m_val, str) and m_val.upper() == "NONE":
                            m_val = None
                        normalized_override[m_field] = m_val
                    normalized_methods[method_name] = normalized_override
            val = normalized_methods
            
        res[field] = val
        
    return res


def resolve_method_param(param_name, method_name, sym_params, default_val):
    """
    Resolve parameter value using method-specific override, fallback to symbol override,
    fallback to default_val.
    """
    if sym_params and "methods" in sym_params and isinstance(sym_params["methods"], dict):
        method_overrides = sym_params["methods"].get(method_name)
        if method_overrides and isinstance(method_overrides, dict) and param_name in method_overrides:
            return method_overrides[param_name]
    if sym_params and param_name in sym_params:
        return sym_params[param_name]
    return default_val


def calculate_dynamic_lot_size(symbol, risk_amount, risk_points):
    """Calculate lot size dynamically accounting for cross-pair tick values."""
    broker = get_broker()
    if risk_points <= 0 or risk_amount <= 0:
        return 0.01
    
    import config
    offline = getattr(config, 'OFFLINE_BACKTESTING', False)
    
    if offline:
        contract_size = get_contract_size(symbol)
        lot = (risk_amount / risk_points) / contract_size
        return max(0.01, min(100.0, lot))
        
    try:
        symbol_info = broker.symbol_info(symbol)
    except Exception:
        symbol_info = None

    if symbol_info is None:
        contract_size = get_contract_size(symbol)
        lot_size = (risk_amount / risk_points) / contract_size
    else:
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        
        if tick_value == 0 or tick_size == 0:
            lot_size = (risk_amount / risk_points) / symbol_info.trade_contract_size
        else:
            ticks_at_risk = risk_points / tick_size
            loss_per_lot = ticks_at_risk * tick_value
            if loss_per_lot <= 0:
                lot_size = 0.01
            else:
                lot_size = risk_amount / loss_per_lot

    vol_min = getattr(symbol_info, 'volume_min', 0.01) if symbol_info else 0.01
    vol_max = getattr(symbol_info, 'volume_max', 100.0) if symbol_info else 100.0
    vol_step = getattr(symbol_info, 'volume_step', 0.01) if symbol_info else 0.01
    
    lot_size = max(vol_min, min(vol_max, lot_size))
    import math
    if vol_step > 0:
        lot_size = math.floor(lot_size / vol_step + 1e-9) * vol_step
        _dec = max(0, int(round(-math.log10(vol_step)))) if vol_step < 1 else 0
        lot_size = round(lot_size, _dec)
    lot_size = max(vol_min, min(vol_max, lot_size))
    return lot_size


from collections import OrderedDict

class BoundedCache(OrderedDict):
    """LRU dict with a hard cap. Drop-in for the plain dicts used as caches."""
    def __init__(self, maxsize=256):
        super().__init__()
        self.maxsize = int(maxsize)

    def __setitem__(self, key, value):
        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, value)
        while len(self) > self.maxsize:
            self.popitem(last=False)

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return super().__getitem__(key)
        return default


def validate_trailing_methods(trailing_methods):
    """trailing_methods is a list of METHOD NAMES. Passing timeframes (or
    (name, value) tuples) silently disabled trailing for every trade."""
    if trailing_methods is None:
        return None
    bad = [m for m in trailing_methods
           if not isinstance(m, str) or m not in (set(ICT_METHODS) | LEGACY_OB_RETEST_METHODS)]
    if bad:
        raise ValueError(f"trailing_methods must be ICT method names; got {bad}")
    return list(trailing_methods)


