import pandas as pd
import numpy as np
import bisect
import logging
import config
from config import FVG_SL_NORMAL, FVG_SL_CANDLE, FVG_SL_BOS, FVG_SL_MSS, FVG_SL_SWEEP
from indicators import detect_market_structure, detect_all_liquidity_sweeps, detect_swing_points
from utils import calculate_atr
from functools import lru_cache


logger = logging.getLogger()

try:
    import pytz
    _has_pytz = True
except ImportError:
    _has_pytz = False


def compute_fvg_sl_price(fvg, trade_dir, fvg_sl_mode, analysis_df, swing_highs, swing_lows, buffer, spread,
                         global_i=None, global_sweeps=None, global_sweep_idxs=None, swing_lookback=None):
    """Compute SL price for FVG Return trades based on the selected SL mode.
    Only swings formed before global_i - swing_lookback are visible (no lookahead).
    """
    import config
    if swing_lookback is None:
        swing_lookback = int(getattr(config, 'SWING_LOOKBACK', 5))
    cutoff = None
    if global_i is not None:
        cutoff = global_i - swing_lookback

    def _visible(swings):
        if cutoff is None:
            return swings
        return [s for s in swings
                if isinstance(s, dict) and s.get('idx') is not None
                and s['idx'] <= cutoff]

    swing_highs = _visible(swing_highs or [])
    swing_lows = _visible(swing_lows or [])

    entry_price = fvg['mid']
    fallback_buy = fvg['bottom'] - buffer
    fallback_sell = fvg['top'] + buffer

    if fvg_sl_mode == FVG_SL_NORMAL:
        return fallback_buy if trade_dir == "Buy" else fallback_sell

    if fvg_sl_mode == FVG_SL_CANDLE:
        if trade_dir == "Buy":
            return fvg['candle_low'] - buffer
        else:
            return fvg['candle_high'] + buffer

    if fvg_sl_mode == FVG_SL_BOS:
        if not swing_highs or not swing_lows:
            return fallback_buy if trade_dir == "Buy" else fallback_sell
        # Collect all BoS structural levels
        bos_levels = []
        # Bullish BoS (HH after HH): broken level = the prior swing high
        for i in range(2, len(swing_highs)):
            if swing_highs[i]['price'] > swing_highs[i-1]['price'] and \
               swing_highs[i-1]['price'] > swing_highs[i-2]['price']:
                bos_levels.append(swing_highs[i-1]['price'])
        # Bearish BoS (LL after LL): broken level = the prior swing low
        for i in range(2, len(swing_lows)):
            if swing_lows[i]['price'] < swing_lows[i-1]['price'] and \
               swing_lows[i-1]['price'] < swing_lows[i-2]['price']:
                bos_levels.append(swing_lows[i-1]['price'])

        if trade_dir == "Buy":
            valid = sorted([l for l in bos_levels if l < entry_price])
            if valid:
                sl = valid[-1] - buffer
                if sl < entry_price:
                    return sl
        else:
            valid = sorted([l for l in bos_levels if l > entry_price])
            if valid:
                sl = valid[0] + buffer
                if sl > entry_price:
                    return sl
        return fallback_buy if trade_dir == "Buy" else fallback_sell

    if fvg_sl_mode == FVG_SL_MSS:
        if not swing_highs or not swing_lows:
            return fallback_buy if trade_dir == "Buy" else fallback_sell
        # Collect all MSS reference levels
        mss_levels = []
        # Bullish MSS precondition: was making LH (downtrend) — level = prior swing high
        for i in range(1, len(swing_highs)):
            if swing_highs[i]['price'] < swing_highs[i-1]['price']:
                mss_levels.append(swing_highs[i-1]['price'])
        # Bearish MSS precondition: was making HL (uptrend) — level = prior swing low
        for i in range(1, len(swing_lows)):
            if swing_lows[i]['price'] > swing_lows[i-1]['price']:
                mss_levels.append(swing_lows[i-1]['price'])

        if trade_dir == "Buy":
            valid = sorted([l for l in mss_levels if l < entry_price])
            if valid:
                sl = valid[-1] - buffer
                if sl < entry_price:
                    return sl
        else:
            valid = sorted([l for l in mss_levels if l > entry_price])
            if valid:
                sl = valid[0] + buffer
                if sl > entry_price:
                    return sl
        return fallback_buy if trade_dir == "Buy" else fallback_sell

    if fvg_sl_mode == FVG_SL_SWEEP:
        if global_sweeps is not None and global_i is not None and global_sweep_idxs is not None:
            pos = bisect.bisect_left(global_sweep_idxs, global_i)
            sweeps = global_sweeps[:pos]
        else:
            sweeps = detect_all_liquidity_sweeps(analysis_df)
        if trade_dir == "Buy":
            buy_sweeps = [s for s in sweeps if s['type'] == 'bullish' and s['price'] < entry_price]
            if buy_sweeps:
                sl = buy_sweeps[-1]['price'] - buffer
                if sl < entry_price:
                    return sl
        else:
            sell_sweeps = [s for s in sweeps if s['type'] == 'bearish' and s['price'] > entry_price]
            if sell_sweeps:
                sl = sell_sweeps[-1]['price'] + buffer
                if sl > entry_price:
                    return sl
        return fallback_buy if trade_dir == "Buy" else fallback_sell

    # Unknown mode — fallback to normal
    return fallback_buy if trade_dir == "Buy" else fallback_sell

@lru_cache(maxsize=200000)
def _convert_time_to_ny_hour_cached(dt, broker_tz_name):
    """Convert a timestamp to New York hour in a DST-aware way, respecting configured broker timezone."""
    if hasattr(dt, 'to_pydatetime'):
        dt = dt.to_pydatetime()
    elif isinstance(dt, (int, np.integer)):
        dt = pd.to_datetime(dt, unit='s').to_pydatetime()
    if _has_pytz:
        try:
            if broker_tz_name != 'Europe/Helsinki':
                broker_tz = pytz.timezone(broker_tz_name)
                ny_tz = pytz.timezone('America/New_York')
                if dt.tzinfo is None:
                    dt_broker = broker_tz.localize(dt)
                else:
                    dt_broker = dt
                dt_ny = dt_broker.astimezone(ny_tz)
                return dt_ny.hour
        except Exception:
            pass

    year = dt.year
    dst_start_ny = pd.Timestamp(year=year, month=3, day=8) + pd.Timedelta(days=(13 - pd.Timestamp(year=year, month=3, day=8).dayofweek) % 7)
    dst_end_ny = pd.Timestamp(year=year, month=11, day=1) + pd.Timedelta(days=(7 - pd.Timestamp(year=year, month=11, day=1).dayofweek) % 7)
    
    is_ny_dst = (dst_start_ny <= dt < dst_end_ny)
    
    if broker_tz_name.upper() in ('UTC', 'GMT'):
        offset = 4 if is_ny_dst else 5
    else:
        offset = 7
    return (dt.hour - offset) % 24

def convert_time_to_ny_hour(dt):
    return _convert_time_to_ny_hour_cached(
        dt, getattr(config, 'BROKER_TIMEZONE', 'Europe/Helsinki'))

convert_time_to_ny_hour.cache_clear = _convert_time_to_ny_hour_cached.cache_clear


@lru_cache(maxsize=200000)
def _convert_time_to_ny_dt_cached(dt, broker_tz_name):
    """Convert broker timestamp to full New York datetime."""
    if _has_pytz:
        try:
            if broker_tz_name != 'Europe/Helsinki':
                broker_tz = pytz.timezone(broker_tz_name)
                ny_tz = pytz.timezone('America/New_York')
                if dt.tzinfo is None:
                    dt_broker = broker_tz.localize(dt)
                else:
                    dt_broker = dt
                return dt_broker.astimezone(ny_tz).replace(tzinfo=None)
        except Exception:
            pass

    year = dt.year
    dst_start_ny = pd.Timestamp(year=year, month=3, day=8) + pd.Timedelta(days=(13 - pd.Timestamp(year=year, month=3, day=8).dayofweek) % 7)
    dst_end_ny = pd.Timestamp(year=year, month=11, day=1) + pd.Timedelta(days=(7 - pd.Timestamp(year=year, month=11, day=1).dayofweek) % 7)
    
    is_ny_dst = (dst_start_ny <= dt < dst_end_ny)
    
    if broker_tz_name.upper() in ('UTC', 'GMT'):
        offset = 4 if is_ny_dst else 5
    else:
        offset = 7
            
    return dt - pd.Timedelta(hours=offset)

def convert_time_to_ny_dt(dt):
    return _convert_time_to_ny_dt_cached(
        dt, getattr(config, 'BROKER_TIMEZONE', 'Europe/Helsinki'))

convert_time_to_ny_dt.cache_clear = _convert_time_to_ny_dt_cached.cache_clear


def get_session_extremes(analysis_df, global_i=None):
    if global_i is None:
        global_i = len(analysis_df)
    if global_i == 0:
        return {}
        
    # Extract only the last ~2 days of data for extreme speed using Numpy
    start_idx = max(0, global_i - 300)
    times = analysis_df.index.values[start_idx:global_i]
    highs = analysis_df['high'].values[start_idx:global_i]
    lows = analysis_df['low'].values[start_idx:global_i]
    
    if len(times) == 0:
        return {}
        
    current_broker_time = pd.Timestamp(times[-1])
    current_ny_time = convert_time_to_ny_dt(current_broker_time)
    
    if current_ny_time.hour < 20:
        asian_start_ny = current_ny_time.replace(hour=20, minute=0, second=0, microsecond=0) - pd.Timedelta(days=1)
    else:
        asian_start_ny = current_ny_time.replace(hour=20, minute=0, second=0, microsecond=0)
    asian_end_ny = asian_start_ny + pd.Timedelta(hours=4)
    
    _day = current_ny_time.replace(hour=0, minute=0, second=0, microsecond=0)
    london_start_ny = _day.replace(hour=2, minute=0)
    london_end_ny = _day.replace(hour=5, minute=0)
    ny_am_start_ny = _day.replace(hour=8, minute=30)
    ny_am_end_ny = _day.replace(hour=12, minute=0)
    if london_start_ny > current_ny_time:
        london_start_ny -= pd.Timedelta(days=1)
        london_end_ny -= pd.Timedelta(days=1)
    if ny_am_start_ny > current_ny_time:
        ny_am_start_ny -= pd.Timedelta(days=1)
        ny_am_end_ny -= pd.Timedelta(days=1)
    
    prev_day_start_ny = asian_start_ny.replace(hour=0, minute=0) - pd.Timedelta(days=1)
    prev_day_end_ny = prev_day_start_ny + pd.Timedelta(days=1)
    
    offset = current_broker_time - current_ny_time
    
    asian_start_br = asian_start_ny + offset
    asian_end_br = asian_end_ny + offset
    london_start_br = london_start_ny + offset
    london_end_br = london_end_ny + offset
    ny_am_start_br = ny_am_start_ny + offset
    ny_am_end_br = ny_am_end_ny + offset
    prev_day_start_br = prev_day_start_ny + offset
    prev_day_end_br = prev_day_end_ny + offset
    
    current_day_start_br = prev_day_end_ny + offset
    current_day_end_br = current_ny_time + pd.Timedelta(hours=1) + offset
    
    def get_high_low(start_br, end_br):
        start_ns = pd.Timestamp(start_br).to_datetime64()
        end_ns = pd.Timestamp(end_br).to_datetime64()
        mask = (times >= start_ns) & (times < end_ns)
        if mask.any():
            return float(highs[mask].max()), float(lows[mask].min())
        return None, None
        
    asian_high, asian_low = get_high_low(asian_start_br, asian_end_br)
    london_high, london_low = get_high_low(london_start_br, london_end_br)
    ny_am_high, ny_am_low = get_high_low(ny_am_start_br, ny_am_end_br)
    pd_high, pd_low = get_high_low(prev_day_start_br, prev_day_end_br)
    cd_high, cd_low = get_high_low(current_day_start_br, current_day_end_br)
    
    return {
        'asian_high': asian_high, 'asian_low': asian_low,
        'london_high': london_high, 'london_low': london_low,
        'ny_am_high': ny_am_high, 'ny_am_low': ny_am_low,
        'pd_high': pd_high, 'pd_low': pd_low,
        'cd_high': cd_high, 'cd_low': cd_low
    }

def detect_silver_bullet_signals(df, current_time, fvgs, structure, mss, spread, broker_tz_offset=0,
                                fvg_sl_mode=FVG_SL_NORMAL, fvg_displacement_only=False, 
                                fvg_discount_premium_only=False, fvg_recent_sweep_only=False,
                                swing_highs=None, swing_lows=None, analysis_df=None, global_i=None,
                                rolling_high_50=None, rolling_low_50=None, sweep_bull_indices=None, sweep_bear_indices=None,
                                global_sweeps=None, global_sweep_idxs=None, sb_strict_window=False, sb_require_htf_bias=False, htf_bias_direction="neutral"):
    """Silver Bullet (Professional Version): Trades high-probability FVGs with strict liquidity sweeps and targeting rules."""
    signals = []
    if not fvgs:
        return signals

    ny_hour = convert_time_to_ny_hour(current_time)

    # Always enforce exact 60-minute strict macro window for Silver Bullet
    in_window = (ny_hour == 10) or (ny_hour == 14) or (ny_hour == 3)
        
    if not in_window:
        return signals

    if swing_highs is None or swing_lows is None:
        swing_highs, swing_lows = detect_swing_points(df)

    active_analysis_df = analysis_df if analysis_df is not None else df
    session_extremes = get_session_extremes(active_analysis_df, global_i)

    window_type = None
    if 9 <= ny_hour <= 12: window_type = "AM"
    elif 13 <= ny_hour <= 16: window_type = "PM"
    elif 2 <= ny_hour <= 5: window_type = "LONDON"
    
    # --- HOISTED COMPUTATIONS ---
    current_broker_time = current_time
    offset = current_broker_time - convert_time_to_ny_dt(current_broker_time)
    
    if window_type == "AM":
        window_start_br = convert_time_to_ny_dt(current_broker_time).replace(hour=8, minute=30) + offset
    elif window_type == "PM":
        window_start_br = convert_time_to_ny_dt(current_broker_time).replace(hour=13, minute=0) + offset
    elif window_type == "LONDON":
        window_start_br = convert_time_to_ny_dt(current_broker_time).replace(hour=2, minute=0) + offset
    else:
        window_start_br = None
        
    start_ns = pd.Timestamp(window_start_br).to_datetime64() if window_start_br is not None else None

    end_idx = global_i if global_i is not None else len(active_analysis_df)
    search_start = max(0, end_idx - 200)
    
    times_slice = active_analysis_df.index.values[search_start:end_idx]
    highs_slice = active_analysis_df['high'].values[search_start:end_idx]
    lows_slice = active_analysis_df['low'].values[search_start:end_idx]
    
    if start_ns is not None:
        mask = times_slice >= start_ns
        if mask.any():
            current_window_high = float(highs_slice[mask].max())
            current_window_low = float(lows_slice[mask].min())
        else:
            current_window_high = None
            current_window_low = None
    else:
        current_window_high = None
        current_window_low = None
    # ----------------------------

    for fvg in fvgs:
        fvg_time = fvg['time']
        fvg_ny_dt = convert_time_to_ny_dt(fvg_time)
        fvg_time_val = fvg_ny_dt.hour + (fvg_ny_dt.minute / 60.0)
        
        fvg_ny_hour = fvg_ny_dt.hour
        
        fvg_in_window = (fvg_ny_hour == 10) or (fvg_ny_hour == 14) or (fvg_ny_hour == 3)
            
        if not fvg_in_window:
            continue

        fvg_dir = fvg['type']
        trade_dir = "Buy" if fvg_dir == "bullish" else "Sell"
        entry = fvg['mid']

        if sb_require_htf_bias:
            if htf_bias_direction == "neutral" or trade_dir != htf_bias_direction:
                continue

        if structure != "neutral":
            if fvg_dir != structure:
                if not (mss and mss == fvg_dir):
                    continue

        # 2. Daily Dealing Range (Discount / Premium)
        # Silver Bullet trades intraday volatility, strict daily DR equilibrium check removed.

        # 3. Breaker Block / Mitigation Block Alignment (Removed: Too restrictive for basic Silver Bullet)

        # 4. Professional Sweep & DOL Targeting
        valid_setup = False
        
        if current_window_high is None or current_window_low is None:
            continue

        if trade_dir == "Buy":
            swept_low = False
            pools_to_check_low = ['london_low', 'asian_low', 'pd_low', 'ny_am_low']
            if window_type == "LONDON" and 'london_low' in pools_to_check_low:
                pools_to_check_low.remove('london_low')

            for pool in pools_to_check_low:
                p_val = session_extremes.get(pool)
                if p_val is not None and current_window_low <= p_val:
                    swept_low = True
                    break
            
            target_intact = False
            for pool in ['pd_high', 'asian_high', 'london_high', 'ny_am_high']:
                p_val = session_extremes.get(pool)
                if p_val is not None and current_window_high < p_val:
                    target_intact = True
                    break

            if swept_low and target_intact:
                valid_setup = True

        else:
            swept_high = False
            pools_to_check_high = ['london_high', 'asian_high', 'pd_high', 'ny_am_high']
            if window_type == "LONDON" and 'london_high' in pools_to_check_high:
                pools_to_check_high.remove('london_high')

            for pool in pools_to_check_high:
                p_val = session_extremes.get(pool)
                if p_val is not None and current_window_high >= p_val:
                    swept_high = True
                    break
            
            target_intact = False
            for pool in ['pd_low', 'asian_low', 'london_low', 'ny_am_low']:
                p_val = session_extremes.get(pool)
                if p_val is not None and current_window_low > p_val:
                    target_intact = True
                    break

            if swept_high and target_intact:
                valid_setup = True
                
        if not valid_setup:
            continue

        if fvg_displacement_only:
            if fvg.get('body_ratio', 1.0) < 0.6 or fvg.get('atr_mult', 1.0) < 0.4:
                continue

        entry = fvg['mid']
        buffer = spread * 2

        sl = compute_fvg_sl_price(
            fvg, trade_dir, fvg_sl_mode, active_analysis_df, swing_highs, swing_lows, buffer, spread,
            global_i=global_i, global_sweeps=global_sweeps, global_sweep_idxs=global_sweep_idxs
        )
        
        risk = entry - sl if trade_dir == "Buy" else sl - entry
        if risk <= 0:
            continue

        signals.append({
            'type': fvg_dir,
            'trade_dir': trade_dir,
            'entry': entry,
            'sl': sl,
            'risk': risk,
            'method': 'Silver Bullet',
            'fvg': fvg,
            'window_hour': ny_hour,
        })
    return signals

def detect_judas_swing(df, current_time, structure, spread, window=60, broker_tz_offset=0, swing_highs=None, swing_lows=None, atr_val=None,
                       global_i=None, rolling_high_50=None, rolling_low_50=None):
    """Judas Swing: False breakout during session opening (ICT kill zones).
    Upgraded to sweep major structural session liquidity extremes with wick rejection and close-inside confirmation.
    """
    if global_i is not None:
        if global_i < window + 20:
            return []
    else:
        if len(df) < window + 20:
            return []
            
    signals = []

    # Kill zone openings (NY time mapped dynamically based on broker timezone)
    ny_hour = convert_time_to_ny_hour(current_time)

    # Widened kill zones (London 1-5, NY 6-11)
    kill_zones = [(1, 5), (6, 11)]
    in_kill_zone = any(start <= ny_hour <= end for start, end in kill_zones)
    if not in_kill_zone:
        return signals

    if swing_highs is None or swing_lows is None:
        swing_highs, swing_lows = detect_swing_points(df)

    if global_i is not None:
        ref_idx = global_i
        _highs = df['high'].values
        _lows = df['low'].values
        _opens = df['open'].values
        _closes = df['close'].values
        last_bar = {'high': _highs[ref_idx - 1], 'low': _lows[ref_idx - 1], 'open': _opens[ref_idx - 1], 'close': _closes[ref_idx - 1]}
    else:
        last_bar = df.iloc[-1]

    # Calculate 14-bar ATR for noise buffer
    if atr_val is not None:
        atr = atr_val
    else:
        if global_i is not None:
            # Simple fallback or fast ATR
            atr = spread * 5
        else:
            tr = np.maximum(df['high'] - df['low'], np.maximum(np.abs(df['high'] - df['close'].shift(1)), np.abs(df['low'] - df['close'].shift(1))))
            atr = tr.rolling(14, min_periods=1).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = spread * 5
    sl_buffer = max(spread * 3, atr * 0.4)

    # 50-bar Premium/Discount midpoint
    if global_i is not None and rolling_high_50 is not None and rolling_low_50 is not None:
        midpoint_50 = (rolling_high_50[global_i - 1] + rolling_low_50[global_i - 1]) / 2.0
    else:
        recent_50 = df.iloc[-50:]
        midpoint_50 = (recent_50['high'].max() + recent_50['low'].min()) / 2.0

    total_range = last_bar['high'] - last_bar['low']
    if total_range <= 0:
        return signals

    # Find structural swing highs/lows in the major lookback window
    if global_i is not None:
        recent_sh = [sh for sh in swing_highs if (global_i - window - 10) <= sh['idx'] <= (global_i - 2)]
        recent_sl = [sl for sl in swing_lows if (global_i - window - 10) <= sl['idx'] <= (global_i - 2)]
    else:
        idx_start = df.index[-window - 10]
        idx_end = df.index[-2]
        if isinstance(idx_start, (int, np.integer)):
            idx_start = pd.to_datetime(idx_start, unit='s')
            idx_end = pd.to_datetime(idx_end, unit='s')
            
        recent_sh = [sh for sh in swing_highs if sh['time'] >= idx_start and sh['time'] <= idx_end]
        recent_sl = [sl for sl in swing_lows if sl['time'] >= idx_start and sl['time'] <= idx_end]

    # Bearish Judas: swept structural swing high → rejected high, closed bearish below the swept high (reversal in Premium)
    if recent_sh:
        swept_high = max(sh['price'] for sh in recent_sh)
        if last_bar['high'] > swept_high:
            upper_wick = last_bar['high'] - max(last_bar['open'], last_bar['close'])
            has_wick_rejection = (upper_wick / total_range >= 0.40)
            is_close_inside = (last_bar['close'] <= swept_high)
            
            if last_bar['close'] < last_bar['open'] and last_bar['close'] >= midpoint_50 and has_wick_rejection and is_close_inside:
                entry = last_bar['close']
                sl = last_bar['high'] + sl_buffer
                risk = sl - entry
                if risk > spread * 2:
                    signals.append({
                        'type': 'bearish',
                        'trade_dir': 'Sell',
                        'entry': entry,
                        'sl': sl,
                        'risk': risk,
                        'method': 'Judas Swing',
                        'sweep_price': swept_high,
                        'is_displacement': True,
                    })

    # Bullish Judas: swept structural swing low → rejected low, closed bullish above the swept low (reversal in Discount)
    if recent_sl:
        swept_low = min(sl['price'] for sl in recent_sl)
        if last_bar['low'] < swept_low:
            lower_wick = min(last_bar['open'], last_bar['close']) - last_bar['low']
            has_wick_rejection = (lower_wick / total_range >= 0.40)
            is_close_inside = (last_bar['close'] >= swept_low)
            
            if last_bar['close'] > last_bar['open'] and last_bar['close'] <= midpoint_50 and has_wick_rejection and is_close_inside:
                entry = last_bar['close']
                sl = last_bar['low'] - sl_buffer
                risk = entry - sl
                if risk > spread * 2:
                    signals.append({
                        'type': 'bullish',
                        'trade_dir': 'Buy',
                        'entry': entry,
                        'sl': sl,
                        'risk': risk,
                        'method': 'Judas Swing',
                        'sweep_price': swept_low,
                        'is_displacement': True,
                    })
    return signals

def detect_breaker_blocks(df, order_blocks_df, spread, current_bar_idx=None, atr_val=None,
                          global_i=None, rolling_high_50=None, rolling_low_50=None):
    """Breaker Block: A failed Order Block that flips polarity.
    Upgraded to enter at OB boundaries with candlestick confirmation and structural swing stop-losses.
    """
    signals = []
    
    if global_i is not None:
        if order_blocks_df is None or order_blocks_df.empty or global_i < 50:
            return signals
    else:
        if order_blocks_df is None or order_blocks_df.empty or len(df) < 50:
            return signals

    if global_i is not None:
        ref_idx = global_i
        _highs = df['high'].values
        _lows = df['low'].values
        _opens = df['open'].values
        _closes = df['close'].values
        _times = df.index
        last_bar = {'high': _highs[ref_idx - 1], 'low': _lows[ref_idx - 1], 'open': _opens[ref_idx - 1], 'close': _closes[ref_idx - 1]}
        prev_bar = {'high': _highs[ref_idx - 2], 'low': _lows[ref_idx - 2], 'open': _opens[ref_idx - 2], 'close': _closes[ref_idx - 2]}
        current_time = _times[ref_idx - 1]
    else:
        # FIX: in the live branch global_i is None, so ref_idx was never defined
        # before being used below (past_obs filter) -> NameError that silently
        # killed every live Breaker Block scan. Anchor it to the latest bar.
        ref_idx = len(df)
        last_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]
        current_time = df.index[-1]
    
    # Calculate 14-bar ATR for noise buffer
    if atr_val is not None:
        atr = atr_val
    else:
        if global_i is not None:
            atr = spread * 5
        else:
            tr = np.maximum(df['high'] - df['low'], np.maximum(np.abs(df['high'] - df['close'].shift(1)), np.abs(df['low'] - df['close'].shift(1))))
            atr = tr.rolling(14, min_periods=1).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = spread * 5
    sl_buffer = max(spread * 3, atr * 0.4)

    # 50-bar Premium/Discount midpoint
    if global_i is not None and rolling_high_50 is not None and rolling_low_50 is not None:
        midpoint_50 = (rolling_high_50[global_i - 1] + rolling_low_50[global_i - 1]) / 2.0
    else:
        recent_50 = df.iloc[-50:]
        midpoint_50 = (recent_50['high'].max() + recent_50['low'].min()) / 2.0

    # Filter OBs to only those that formed on or before current time and within the last 50 bars
    if global_i is not None:
        lookback_time = df.index[global_i - 50]
    else:
        lookback_time = df.index[-50] if len(df) >= 50 else df.index[0]
        
    start_idx = order_blocks_df.index.searchsorted(lookback_time)
    end_idx = order_blocks_df.index.searchsorted(current_time, side='right')
    past_obs = order_blocks_df.iloc[start_idx:end_idx]
    
    # Anti-Lookahead Bias: Only use Order Blocks that have actually been confirmed
    past_obs = past_obs[past_obs['confirmation_bar'] <= ref_idx]
    
    if past_obs.empty:
        return signals

    # Ensure first_break_time is present (fallback for live trading if not pre-calculated)
    if 'first_break_time' not in past_obs.columns:
        first_break_times = []
        for ob_time, row in past_obs.iterrows():
            ob_type = row['order_block_type']
            ob_high = row['high']
            ob_low = row['low']
            
            mask = (df.index > ob_time) & (df.index <= current_time)
            if not mask.any():
                first_break_times.append(pd.NaT)
                continue
            
            post_closes = df.loc[mask, 'close']
            if ob_type == 'bearish':
                break_mask = post_closes > ob_high
            else:
                break_mask = post_closes < ob_low
            
            break_indices = np.where(break_mask)[0]
            if len(break_indices) > 0:
                first_break_times.append(post_closes.index[break_indices[0]])
            else:
                first_break_times.append(pd.NaT)
        past_obs = past_obs.copy()
        past_obs['first_break_time'] = first_break_times

    # Vectorized filter to only keep OBs that have already been broken
    past_obs = past_obs[past_obs['first_break_time'] <= current_time]
    if past_obs.empty:
        return signals

    for ob_time, ob in past_obs.iterrows():
        ob_type = ob.get('order_block_type', '')
        ob_high = ob['high']
        ob_low = ob['low']
        midpoint = (ob_high + ob_low) / 2.0

        if ob_type == 'bearish':
            # Was bearish, now bullish breaker: buy when price returns to old OB high zone
            if midpoint > midpoint_50:
                continue
            
            # Retest condition: price touched the breaker block high boundary
            touched = (last_bar['low'] <= ob_high) or (prev_bar['low'] <= ob_high)
            if not touched:
                continue
                
            # Reversal confirmation: last bar closed bullish
            if last_bar['close'] <= last_bar['open']:
                continue
                
            # Find the true structural swing low since the OB was formed
            if global_i is not None and 'ob_bar_index' in ob:
                lowest_low = _lows[int(ob['ob_bar_index']) : global_i].min()
            elif current_bar_idx is not None and 'ob_bar_index' in ob:
                rel_ob_idx = int(ob['ob_bar_index']) - (current_bar_idx - len(df))
                rel_ob_idx = max(0, min(rel_ob_idx, len(df) - 1))
                lowest_low = df['low'].values[rel_ob_idx:].min()
            else:
                lowest_low = df.loc[ob_time:current_time, 'low'].min()
            
            # Invalidation: close must not be below this lowest structural low
            if last_bar['close'] < lowest_low:
                continue
                
            entry = last_bar['close']
            sl = lowest_low - sl_buffer
            risk = entry - sl
            
            if risk > spread * 2:
                signals.append({
                    'type': 'bullish',
                    'trade_dir': 'Buy',
                    'entry': entry,
                    'sl': sl,
                    'risk': risk,
                    'method': 'Breaker Block',
                    'original_ob_type': ob_type,
                    'ob': ob,
                })
                
        elif ob_type == 'bullish':
            # Was bullish, now bearish breaker: sell when price returns to old OB low zone
            if midpoint < midpoint_50:
                continue
                
            # Retest condition: price touched the breaker block low boundary
            touched = (last_bar['high'] >= ob_low) or (prev_bar['high'] >= ob_low)
            if not touched:
                continue
                
            # Reversal confirmation: last bar closed bearish
            if last_bar['close'] >= last_bar['open']:
                continue
                
            # Find the true structural swing high since the OB was formed
            if global_i is not None and 'ob_bar_index' in ob:
                highest_high = _highs[int(ob['ob_bar_index']) : global_i].max()
            elif current_bar_idx is not None and 'ob_bar_index' in ob:
                rel_ob_idx = int(ob['ob_bar_index']) - (current_bar_idx - len(df))
                rel_ob_idx = max(0, min(rel_ob_idx, len(df) - 1))
                highest_high = df['high'].values[rel_ob_idx:].max()
            else:
                highest_high = df.loc[ob_time:current_time, 'high'].max()
            
            # Invalidation: close must not be above highest_high
            if last_bar['close'] > highest_high:
                continue
                
            entry = last_bar['close']
            sl = highest_high + sl_buffer
            risk = sl - entry
            
            if risk > spread * 2:
                signals.append({
                    'type': 'bearish',
                    'trade_dir': 'Sell',
                    'entry': entry,
                    'sl': sl,
                    'risk': risk,
                    'method': 'Breaker Block',
                    'original_ob_type': ob_type,
                    'ob': ob,
                })
    return signals

def detect_iofr_signals(df, spread, swing_highs=None, swing_lows=None, atr_val=None,
                        global_i=None):
    """Institutional Order Flow Reclaim (IOFR):
    A unique ICT-based strategy designed to capture major institutional sweeps with high precision.
    
    Setup:
      1. Identify a recent structural swing point (swing high or low) from the last 30 bars.
      2. A candle (the "sweep candle", usually bar -2) sweeps this swing level (penetrates it with its wick)
         but fails to close beyond it, demonstrating a liquidity sweep.
      3. The next candle (the "reclaim candle", bar -1) shows strong displacement in the opposite direction:
         - Bullish reclaim: Closes above the open and high of the sweep candle with high body ratio (>= 0.60).
         - Bearish reclaim: Closes below the open and low of the sweep candle with high body ratio (>= 0.60).
      4. Place a market entry at the close of the reclaim candle (guaranteeing fill on strong reversals) OR limit at open.
      5. Set a tight SL just beyond the sweep candle's extreme wick, and TP at the opposing swing structure (min 2.5 RRR).
    """
    signals = []
    
    if global_i is not None:
        if global_i < 50:
            return signals
    else:
        if len(df) < 50:
            return signals

    # Extract raw arrays for fast numpy indexing (1000x faster than pandas iloc/Series)
    _lows = df['low'].values
    _highs = df['high'].values
    _opens = df['open'].values
    _closes = df['close'].values
    _times = df.index
     
    if global_i is not None:
        ref_idx = global_i
        last_bar = {'open': _opens[ref_idx - 1], 'high': _highs[ref_idx - 1], 'low': _lows[ref_idx - 1], 'close': _closes[ref_idx - 1], 'name': _times[ref_idx - 1]}
        prev_bar = {'open': _opens[ref_idx - 2], 'high': _highs[ref_idx - 2], 'low': _lows[ref_idx - 2], 'close': _closes[ref_idx - 2], 'name': _times[ref_idx - 2]}
    else:
        last_bar = {'open': _opens[-1], 'high': _highs[-1], 'low': _lows[-1], 'close': _closes[-1], 'name': _times[-1]}
        prev_bar = {'open': _opens[-2], 'high': _highs[-2], 'low': _lows[-2], 'close': _closes[-2], 'name': _times[-2]}

    # Fetch swings if not provided
    if swing_highs is None or swing_lows is None:
        swing_highs, swing_lows = detect_swing_points(df)

    # 14-bar ATR for noise buffer
    if atr_val is not None:
        atr = atr_val
    else:
        if global_i is not None:
            atr = spread * 5
        else:
            tr = np.maximum(df['high'] - df['low'], np.maximum(np.abs(df['high'] - df['close'].shift(1)), np.abs(df['low'] - df['close'].shift(1))))
            atr = tr.rolling(14, min_periods=1).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = spread * 5
    sl_buffer = max(spread * 2, atr * 0.15)

    # Filter structural swing points in the last 30 bars using robust absolute time comparisons
    if global_i is not None:
        cutoff_idx_start = max(0, global_i - 30)
        cutoff_idx_end = max(0, global_i - 3)
        recent_sh = [sh for sh in swing_highs if cutoff_idx_start <= sh['idx'] <= cutoff_idx_end]
        recent_sl = [sl for sl in swing_lows if cutoff_idx_start <= sl['idx'] <= cutoff_idx_end]
    else:
        cutoff_time_start = _times[-30]
        cutoff_time_end = _times[-3]
        if isinstance(cutoff_time_start, (int, np.integer)):
            cutoff_time_start = pd.to_datetime(cutoff_time_start, unit='s')
            cutoff_time_end = pd.to_datetime(cutoff_time_end, unit='s')
        recent_sh = [sh for sh in swing_highs if sh.get('time') >= cutoff_time_start and sh.get('time') <= cutoff_time_end]
        recent_sl = [sl for sl in swing_lows if sl.get('time') >= cutoff_time_start and sl.get('time') <= cutoff_time_end]

    # Reclaim candle displacement checks
    last_range = last_bar['high'] - last_bar['low']
    last_body = abs(last_bar['close'] - last_bar['open'])
    last_body_ratio = last_body / max(last_range, 1e-8)
    
    if last_body_ratio < 0.60:
        return signals

    # BULLISH IOFR: Prev bar swept a recent structural low, last bar reclaimed order flow bullishly
    if last_bar['close'] > last_bar['open'] and recent_sl:
        # Find the lowest swing low swept by prev_bar's low
        swept_lows = [sl for sl in recent_sl if prev_bar['low'] < sl['price'] and prev_bar['close'] > sl['price']]
        if swept_lows:
            # Sort to get the most recent swept swing low
            target_sl = swept_lows[-1]
            
            # Check displacement and reclaim rules:
            # - Reclaim candle must close above the sweep candle's open and high
            if last_bar['close'] > prev_bar['open'] and last_bar['close'] > prev_bar['high']:
                entry = last_bar['close']  # REVAMPED: Market execution at the reclaim candle's close (guarantees fill)
                sl = prev_bar['low'] - sl_buffer
                risk = entry - sl
                
                if risk > spread * 2:
                    # Find target TP at the nearest recent swing high above entry
                    tp_target = None
                    valid_highs = [sh for sh in recent_sh if sh['price'] > entry + risk * 2.5]
                    if valid_highs:
                        tp_target = valid_highs[-1]['price']
                    else:
                        tp_target = entry + risk * 3.0  # Fallback to 3R
                        
                    signals.append({
                        'type': 'bullish',
                        'trade_dir': 'Buy',
                        'entry': entry,
                        'sl': sl,
                        'risk': risk,
                        'tp': tp_target,
                        'method': 'IOFR',
                        'swept_level': target_sl['price'],
                        'sweep_idx': prev_bar['name']
                    })

    # BEARISH IOFR: Prev bar swept a recent structural high, last bar reclaimed order flow bearishly
    elif last_bar['close'] < last_bar['open'] and recent_sh:
        # Find the highest swing high swept by prev_bar's high
        swept_highs = [sh for sh in recent_sh if prev_bar['high'] > sh['price'] and prev_bar['close'] < sh['price']]
        if swept_highs:
            # Sort to get the most recent swept swing high
            target_sh = swept_highs[-1]
            
            # Check displacement and reclaim rules:
            # - Reclaim candle must close below the sweep candle's open and low
            if last_bar['close'] < prev_bar['open'] and last_bar['close'] < prev_bar['low']:
                entry = last_bar['close']  # REVAMPED: Market execution at the reclaim candle's close (guarantees fill)
                sl = prev_bar['high'] + sl_buffer
                risk = sl - entry
                
                if risk > spread * 2:
                    # Find target TP at the nearest recent swing low below entry
                    tp_target = None
                    valid_lows = [sl_pt for sl_pt in recent_sl if sl_pt['price'] < entry - risk * 2.5]
                    if valid_lows:
                        tp_target = valid_lows[-1]['price']
                    else:
                        tp_target = entry - risk * 3.0  # Fallback to 3R
                        
                    signals.append({
                        'type': 'bearish',
                        'trade_dir': 'Sell',
                        'entry': entry,
                        'sl': sl,
                        'risk': risk,
                        'tp': tp_target,
                        'method': 'IOFR',
                        'swept_level': target_sh['price'],
                        'sweep_idx': prev_bar['name']
                    })

    return signals


def detect_mmxm_signals(df, current_time, fvgs, structure, mss, spread,
                        swing_highs=None, swing_lows=None, analysis_df=None, global_i=None,
                        rolling_high_50=None, rolling_low_50=None, sweep_bull_indices=None, sweep_bear_indices=None,
                        global_sweeps=None, global_sweep_idxs=None, atr_val=None,
                        use_mtf_alignment=False, htf_bias_direction="neutral"):
    """MMXM: Market Maker Buy/Sell Model.
    - Market Maker Buy Model (MMBM):
      1. A swing low is swept in discount (Smart Money Reversal).
      2. Bullish MSS shift occurs (displacement).
      3. Retracement into a bullish FVG/OB (reclaiming a left-side consolidation level / in discount).
    - Market Maker Sell Model (MMSM):
      1. A swing high is swept in premium (Smart Money Reversal).
      2. Bearish MSS shift occurs (displacement).
      3. Retracement into a bearish FVG/OB (reclaiming a left-side consolidation level / in premium).
    """
    signals = []
    if len(df) < 50 or not fvgs:
        return signals

    if swing_highs is None or swing_lows is None:
        swing_highs, swing_lows = detect_swing_points(df)

    active_analysis_df = analysis_df if analysis_df is not None else df
    
    if atr_val is not None:
        current_atr = atr_val
    else:
        # Calculate 14-bar ATR for noise buffer
        _highs = active_analysis_df['high'].values
        _lows = active_analysis_df['low'].values
        _closes = active_analysis_df['close'].values
        if len(active_analysis_df) > 1:
            tr = np.maximum(_highs[1:] - _lows[1:], np.maximum(np.abs(_highs[1:] - _closes[:-1]), np.abs(_lows[1:] - _closes[:-1])))
            current_atr = tr[-14:].mean() if len(tr) >= 14 else tr.mean()
        else:
            current_atr = spread * 5
    if pd.isna(current_atr) or current_atr <= 0:
        current_atr = spread * 5
        
    sl_buffer = max(spread * 3, current_atr * 0.4)

    # 1. Sweep Check
    has_bull_sweep = False
    has_bear_sweep = False
    bull_sweep_idx = None
    bear_sweep_idx = None
    
    if global_i is not None and sweep_bull_indices is not None:
        pos = bisect.bisect_right(sweep_bull_indices, global_i) - 1
        if pos >= 0 and (global_i - sweep_bull_indices[pos]) <= 25:
            has_bull_sweep = True
            bull_sweep_idx = sweep_bull_indices[pos]
    else:
        sweeps = detect_all_liquidity_sweeps(active_analysis_df)
        bull_sws = [s for s in sweeps if s['type'] == 'bullish']
        if bull_sws and (len(active_analysis_df) - bull_sws[-1]['idx']) <= 25:
            has_bull_sweep = True
            bull_sweep_idx = bull_sws[-1]['idx']

    if global_i is not None and sweep_bear_indices is not None:
        pos = bisect.bisect_right(sweep_bear_indices, global_i) - 1
        if pos >= 0 and (global_i - sweep_bear_indices[pos]) <= 25:
            has_bear_sweep = True
            bear_sweep_idx = sweep_bear_indices[pos]
    else:
        sweeps = detect_all_liquidity_sweeps(active_analysis_df)
        bear_sws = [s for s in sweeps if s['type'] == 'bearish']
        if bear_sws and (len(active_analysis_df) - bear_sws[-1]['idx']) <= 25:
            has_bear_sweep = True
            bear_sweep_idx = bear_sws[-1]['idx']

    for fvg in fvgs:
        fvg_dir = fvg['type']
        trade_dir = "Buy" if fvg_dir == "bullish" else "Sell"

        # Market structure must align: bullish for MMBM, bearish for MMSM
        if use_mtf_alignment and htf_bias_direction != "neutral" and trade_dir != htf_bias_direction:
            continue
            
        if trade_dir == "Buy" and not (structure == "bullish" or mss == "bullish"):
            continue
        if trade_dir == "Sell" and not (structure == "bearish" or mss == "bearish"):
            continue

        if trade_dir == "Buy" and not has_bull_sweep:
            continue
        if trade_dir == "Sell" and not has_bear_sweep:
            continue

        # D-fix: FVG must form on/after the originating sweep (post-sweep displacement leg),
        # not merely coincide with an unrelated sweep elsewhere in the lookback window.
        _sweep_idx = bull_sweep_idx if trade_dir == "Buy" else bear_sweep_idx
        if _sweep_idx is not None and fvg.get('bar_index', -1) < _sweep_idx:
            continue

        # Premium/Discount check of the overall 50-bar range: MMBM is in discount, MMSM in premium
        if global_i is not None and rolling_high_50 is not None and rolling_low_50 is not None:
            swing_h = rolling_high_50[global_i - 1]
            swing_l = rolling_low_50[global_i - 1]
        else:
            recent_50 = active_analysis_df.iloc[-50:]
            swing_h = recent_50['high'].max()
            swing_l = recent_50['low'].min()
        
        midpoint = (swing_h + swing_l) / 2.0
        if trade_dir == "Buy" and fvg['mid'] >= midpoint:
            continue
        if trade_dir == "Sell" and fvg['mid'] <= midpoint:
            continue

        # Check left-side consolidation reclamation alignment
        left_side_aligned = False
        ref_idx = global_i if global_i is not None else len(active_analysis_df)
        if trade_dir == "Buy":
            left_highs = [sh for sh in swing_highs if sh['idx'] < (ref_idx - 15)]
            if left_highs:
                for lh in left_highs:
                    if abs(lh['price'] - fvg['mid']) < current_atr * 2.0:
                        left_side_aligned = True
                        break
        else:
            left_lows = [sl for sl in swing_lows if sl['idx'] < (ref_idx - 15)]
            if left_lows:
                for ll in left_lows:
                    if abs(ll['price'] - fvg['mid']) < current_atr * 2.0:
                        left_side_aligned = True
                        break

        # Filter out low-displacement setups if left-side is not aligned
        if not left_side_aligned and fvg.get('body_ratio', 1.0) < 0.6:
            continue

        entry = fvg['mid']
        sl = compute_fvg_sl_price(
            fvg, trade_dir, "Last Sweep", active_analysis_df, swing_highs, swing_lows, sl_buffer, spread,
            global_i=global_i, global_sweeps=global_sweeps, global_sweep_idxs=global_sweep_idxs
        )
        
        if trade_dir == "Buy" and sl >= entry:
            continue
        if trade_dir == "Sell" and sl <= entry:
            continue
            
        risk = abs(entry - sl)
        if risk <= 0:
            continue

        signals.append({
            'type': fvg_dir,
            'trade_dir': trade_dir,
            'entry': entry,
            'sl': sl,
            'risk': risk,
            'method': 'MMXM',
            'fvg': fvg
        })

    return signals


def detect_ict_model_signals(df, current_time, fvgs, structure, mss, spread, model_version="2022",
                             swing_highs=None, swing_lows=None, analysis_df=None, global_i=None,
                             rolling_high_50=None, rolling_low_50=None, sweep_bull_indices=None, sweep_bear_indices=None,
                             global_sweeps=None, global_sweep_idxs=None, atr_val=None,
                             use_mtf_alignment=False, htf_bias_direction="neutral"):
    """ICT Mentorship Model 2022 / 2025.
    1. Liquidity sweep of a key swing level (within last 15 bars).
    2. Market Structure Shift (MSS) in opposite direction with displacement (within last 15 bars).
    3. Retrace into a FVG formed during that displacement leg.
    4. Stop Loss placed at the extreme of the sweep candle / swing point.
    
    Model 2025 updates:
    - Enforces higher-tf trend alignment.
    - Enforces discount/premium alignment (Buy in Discount, Sell in Premium of 50-bar range).
    """
    signals = []
    if len(df) < 50 or not fvgs:
        return signals

    if swing_highs is None or swing_lows is None:
        swing_highs, swing_lows = detect_swing_points(df)

    active_analysis_df = analysis_df if analysis_df is not None else df
    
    if atr_val is not None:
        current_atr = atr_val
    else:
        # Calculate 14-bar ATR for noise buffer
        _highs = active_analysis_df['high'].values
        _lows = active_analysis_df['low'].values
        _closes = active_analysis_df['close'].values
        if len(active_analysis_df) > 1:
            tr = np.maximum(_highs[1:] - _lows[1:], np.maximum(np.abs(_highs[1:] - _closes[:-1]), np.abs(_lows[1:] - _closes[:-1])))
            current_atr = tr[-14:].mean() if len(tr) >= 14 else tr.mean()
        else:
            current_atr = spread * 5
    if pd.isna(current_atr) or current_atr <= 0:
        current_atr = spread * 5
        
    sl_buffer = max(spread * 3, current_atr * 0.4)

    # 1. Sweep Check
    has_bull_sweep = False
    has_bear_sweep = False
    bull_sweep_idx = None
    bear_sweep_idx = None
    
    if global_i is not None and sweep_bull_indices is not None:
        pos = bisect.bisect_right(sweep_bull_indices, global_i) - 1
        if pos >= 0 and (global_i - sweep_bull_indices[pos]) <= 15:
            has_bull_sweep = True
            bull_sweep_idx = sweep_bull_indices[pos]
    else:
        sweeps = detect_all_liquidity_sweeps(active_analysis_df)
        bull_sws = [s for s in sweeps if s['type'] == 'bullish']
        if bull_sws and (len(active_analysis_df) - bull_sws[-1]['idx']) <= 15:
            has_bull_sweep = True
            bull_sweep_idx = bull_sws[-1]['idx']

    if global_i is not None and sweep_bear_indices is not None:
        pos = bisect.bisect_right(sweep_bear_indices, global_i) - 1
        if pos >= 0 and (global_i - sweep_bear_indices[pos]) <= 15:
            has_bear_sweep = True
            bear_sweep_idx = sweep_bear_indices[pos]
    else:
        sweeps = detect_all_liquidity_sweeps(active_analysis_df)
        bear_sws = [s for s in sweeps if s['type'] == 'bearish']
        if bear_sws and (len(active_analysis_df) - bear_sws[-1]['idx']) <= 15:
            has_bear_sweep = True
            bear_sweep_idx = bear_sws[-1]['idx']

    for fvg in fvgs:
        fvg_dir = fvg['type']
        trade_dir = "Buy" if fvg_dir == "bullish" else "Sell"

        if use_mtf_alignment and htf_bias_direction != "neutral" and trade_dir != htf_bias_direction:
            continue

        # Requires displacement quality
        if fvg.get('body_ratio', 1.0) < 0.5 or fvg.get('atr_mult', 1.0) < 0.3:
            continue

        # 2. MSS / Structure Shift Check
        if trade_dir == "Buy" and not (mss == "bullish" or structure == "bullish"):
            continue
        if trade_dir == "Sell" and not (mss == "bearish" or structure == "bearish"):
            continue

        if trade_dir == "Buy" and not has_bull_sweep:
            continue
        if trade_dir == "Sell" and not has_bear_sweep:
            continue

        # D-fix: FVG must form on/after the originating sweep (post-sweep displacement leg),
        # not merely coincide with an unrelated sweep elsewhere in the lookback window.
        _sweep_idx = bull_sweep_idx if trade_dir == "Buy" else bear_sweep_idx
        if _sweep_idx is not None and fvg.get('bar_index', -1) < _sweep_idx:
            continue

        # 3. Model 2025 Filters: Premium/Discount check
        if model_version == "2025":
            if global_i is not None and rolling_high_50 is not None and rolling_low_50 is not None:
                swing_h = rolling_high_50[global_i - 1]
                swing_l = rolling_low_50[global_i - 1]
            else:
                recent_50 = active_analysis_df.iloc[-50:]
                swing_h = recent_50['high'].max()
                swing_l = recent_50['low'].min()
            
            midpoint = (swing_h + swing_l) / 2.0
            if trade_dir == "Buy" and fvg['mid'] >= midpoint:
                continue
            if trade_dir == "Sell" and fvg['mid'] <= midpoint:
                continue

        entry = fvg['mid']
        sl = compute_fvg_sl_price(
            fvg, trade_dir, "Last Sweep", active_analysis_df, swing_highs, swing_lows, sl_buffer, spread,
            global_i=global_i, global_sweeps=global_sweeps, global_sweep_idxs=global_sweep_idxs
        )

        if trade_dir == "Buy" and sl >= entry:
            continue
        if trade_dir == "Sell" and sl <= entry:
            continue

        risk = abs(entry - sl)
        if risk <= 0:
            continue

        signals.append({
            'type': fvg_dir,
            'trade_dir': trade_dir,
            'entry': entry,
            'sl': sl,
            'risk': risk,
            'method': f'ICT Model {model_version}',
            'fvg': fvg
        })

    return signals




