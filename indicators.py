try:
    import MetaTrader5 as mt5
except ImportError:
    import tests.MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import config
import utils

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

logger = logging.getLogger()

# -------------------------
# OTE Zone
# -------------------------
def calculate_ote_zone(df, lookback=50, fib_low=0.618, fib_high=0.786):
    if df is None or df.empty:
        return 0.0, 0.0, 0.0, 0.0
    if len(df) < lookback:
        lookback = len(df)
    recent = df.iloc[-lookback:]
    swing_low = recent['low'].min()
    swing_high = recent['high'].max()
    fib_zone_low = swing_low + fib_low * (swing_high - swing_low)
    fib_zone_high = swing_low + fib_high * (swing_high - swing_low)
    return fib_zone_low, fib_zone_high, swing_low, swing_high

def get_ict_model_parameters(model, symbol=None):
    """ONE parameter set for every instrument.
    The old per-asset-class table existed only to paper over spread-relative
    thresholds. Now that displacement is ATR-relative, ICT's claim that the
    concepts are instrument-agnostic actually holds in code.
    """
    import config
    return {
        "threshold_factor": getattr(config, 'OB_RANGE_ATR_MIN', 0.35),
        "reversal_threshold": getattr(config, 'OB_DISPLACEMENT_ATR_MULT', 1.5),
        "lookahead": getattr(config, 'OB_LOOKAHEAD', 8),
        "fib_low": getattr(config, 'OTE_FIB_LOW', 0.618),
        "fib_high": getattr(config, 'OTE_FIB_HIGH', 0.786),
    }

# -------------------------
# Fix #1: OB Detection with confirmation_bar tracking
# -------------------------
@numba.njit
def _detect_order_blocks_jit(opens, closes, highs, lows, atr,
                            ob_range_atr_min, disp_atr_mult, lookahead,
                            require_fvg):
    """ICT order block: the LAST OPPOSING CANDLE before displacement.
    Emphasis is on the DISPLACEMENT LEG, not on the OB candle being large.
    All thresholds are ATR-relative, so behaviour is identical on gold, FX,
    crypto and indices -- and identical live vs backtest, because spread no
    longer enters signal generation at all.
    """
    n = len(opens)
    ob_indices = []
    confirmation_bars = []
    types = []
    disp_atr = []

    for i in range(n - 1):
        a = atr[i]
        if a <= 0.0:
            continue
        o = opens[i]; c = closes[i]; h = highs[i]; l = lows[i]
        bar_range = h - l
        # Loose sanity floor only -- NOT a "must be big" requirement.
        if bar_range < ob_range_atr_min * a:
            continue
        reversal_dist = disp_atr_mult * a

        if o > c:  # down candle -> potential BULLISH order block
            for j in range(1, min(lookahead + 1, n - i)):
                k = i + j
                if closes[k] > h + reversal_dist:
                    # Optional: the displacement must leave an FVG behind
                    # (ICT's highest-grade confirmation). Gap between the
                    # candle before the leg and the candle after it.
                    if require_fvg == 1:
                        found = 0
                        for q in range(i + 1, k + 1):
                            if q - 1 >= 0 and q + 1 < n:
                                if highs[q - 1] < lows[q + 1]:
                                    found = 1
                                    break
                        if found == 0:
                            break
                    ob_indices.append(i)
                    confirmation_bars.append(k)
                    types.append(1)
                    disp_atr.append((closes[k] - h) / a)
                    break
        elif o < c:  # up candle -> potential BEARISH order block
            for j in range(1, min(lookahead + 1, n - i)):
                k = i + j
                if closes[k] < l - reversal_dist:
                    if require_fvg == 1:
                        found = 0
                        for q in range(i + 1, k + 1):
                            if q - 1 >= 0 and q + 1 < n:
                                if lows[q - 1] > highs[q + 1]:
                                    found = 1
                                    break
                        if found == 0:
                            break
                    ob_indices.append(i)
                    confirmation_bars.append(k)
                    types.append(-1)
                    disp_atr.append((l - closes[k]) / a)
                    break

    return ob_indices, confirmation_bars, types, disp_atr


def detect_order_blocks(df, threshold_factor=None, reversal_threshold=None,
                        lookahead=None, spread=0.0, atr=None):
    """Detect ICT order blocks."""
    import config
    from atr_context import atr_series

    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame()

    ob_range_atr_min = float(threshold_factor if threshold_factor is not None
                            else getattr(config, 'OB_RANGE_ATR_MIN', 0.35))
    disp_atr_mult = float(reversal_threshold if reversal_threshold is not None
                          else getattr(config, 'OB_DISPLACEMENT_ATR_MULT', 1.5))
    lookahead = int(lookahead if lookahead is not None
                    else getattr(config, 'OB_LOOKAHEAD', 8))
    require_fvg = 1 if getattr(config, 'OB_REQUIRE_FVG', True) else 0

    df = df.copy().sort_index()
    atr_arr = atr_series(df, 14) if atr is None else np.asarray(atr, dtype=float)
    atr_arr = np.round(atr_arr, 6)

    ob_indices, confirmation_bars, types, disp_atr = _detect_order_blocks_jit(
        df['open'].values.astype(float), df['close'].values.astype(float),
        df['high'].values.astype(float), df['low'].values.astype(float),
        atr_arr, ob_range_atr_min, disp_atr_mult, lookahead, require_fvg,
    )

    cols = df.columns.tolist() + ['order_block_type', 'confirmation_bar',
                                'ob_bar_index', 'displacement_atr']
    if not ob_indices:
        return pd.DataFrame(columns=cols)

    res_df = df.iloc[ob_indices].copy()
    res_df['order_block_type'] = ['bullish' if t == 1 else 'bearish' for t in types]
    res_df['confirmation_bar'] = confirmation_bars
    res_df['ob_bar_index'] = ob_indices
    res_df['displacement_atr'] = disp_atr
    return res_df

# -------------------------
# Liquidity Sweep Detection
# -------------------------
def detect_liquidity_sweep(df, window=None):
    """Detect whether the CURRENT bar swept a recent structural point.
    `window` now defaults to config.SWEEP_LOOKBACK so live and backtest agree.
    It is passed straight to _detect_swing_points_jit as the fractal lookback.
    """
    import config
    if window is None:
        window = int(getattr(config, 'SWEEP_LOOKBACK', 5))
    if len(df) < 5:
        return False, False, None, None
        
    highs = df['high'].values[:-1] # Exclude current bar from swing calculation
    lows = df['low'].values[:-1]
    
    ith_indices, itl_indices = _detect_swing_points_jit(highs, lows, window)
    
    last_bar = df.iloc[-1]
    prev_low = df.iloc[-2]['low']
    prev_high = df.iloc[-2]['high']
    
    bullish_sweep = False
    bearish_sweep = False
    
    swing_low = lows.min() if len(lows) > 0 else last_bar['low']
    if len(itl_indices) > 0:
        swing_low = lows[itl_indices[-1]]
        if last_bar['low'] < swing_low and last_bar['close'] > swing_low:
            bullish_sweep = True
            
    swing_high = highs.max() if len(highs) > 0 else last_bar['high']
    if len(ith_indices) > 0:
        swing_high = highs[ith_indices[-1]]
        if last_bar['high'] > swing_high and last_bar['close'] < swing_high:
            bearish_sweep = True
            
    # True extremes must include the sweep's wick so SL calculations don't stop you out inside your own signal
    true_swing_high = max(swing_high, last_bar['high']) if bearish_sweep else swing_high
    true_swing_low = min(swing_low, last_bar['low']) if bullish_sweep else swing_low
    
    return bullish_sweep, bearish_sweep, true_swing_high, true_swing_low

@numba.njit
def _detect_all_liquidity_sweeps_jit(highs, lows, closes, sh_indices, sl_indices, lookback):
    n = len(highs)
    types = [] # 1 for bullish, -1 for bearish
    indices = []
    prices = []
    
    for i in range(1, n):
        b_low = lows[i]
        b_high = highs[i]
        b_close = closes[i]
        
        recent_sl = -1
        for j in range(len(sl_indices) - 1, -1, -1):
            if sl_indices[j] + lookback <= i:
                recent_sl = sl_indices[j]
                break
                
        if recent_sl != -1:
            s_low = lows[recent_sl]
            if b_low < s_low and b_close > s_low:
                types.append(1)
                indices.append(i)
                prices.append(b_low)
                continue
                
        recent_sh = -1
        for j in range(len(sh_indices) - 1, -1, -1):
            if sh_indices[j] + lookback <= i:
                recent_sh = sh_indices[j]
                break
                
        if recent_sh != -1:
            s_high = highs[recent_sh]
            if b_high > s_high and b_close < s_high:
                types.append(-1)
                indices.append(i)
                prices.append(b_high)
                
    return types, indices, prices

def detect_all_liquidity_sweeps(df, lookback=5):
    """Detect ALL liquidity sweeps across the entire dataframe. JIT optimized."""
    if len(df) < 5:
        return []
    
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    index = df.index
    
    sh_indices, sl_indices = _detect_swing_points_jit(highs, lows, lookback)
    
    sh_arr = np.array(sh_indices, dtype=np.int32)
    sl_arr = np.array(sl_indices, dtype=np.int32)
    
    types, indices, prices = _detect_all_liquidity_sweeps_jit(
        highs, lows, closes, sh_arr, sl_arr, lookback
    )
    
    sweeps = []
    for t, idx, price in zip(types, indices, prices):
        sweeps.append({
            'type': 'bullish' if t == 1 else 'bearish',
            'time': index[idx],
            'price': price,
            'idx': idx
        })
    # Attach ICT fractal tier of the swept swing (1=ST, 2=IT, 3=LT)
    _sh_swings = [{'price': highs[k], 'idx': k} for k in sh_indices]
    _sl_swings = [{'price': lows[k], 'idx': k} for k in sl_indices]
    _sh_tier = {sw['idx']: tr for sw, tr in zip(_sh_swings, classify_swing_tiers(_sh_swings, True))}
    _sl_tier = {sw['idx']: tr for sw, tr in zip(_sl_swings, classify_swing_tiers(_sl_swings, False))}
    _sh_list = list(sh_indices)
    _sl_list = list(sl_indices)
    for sweep in sweeps:
        _si = sweep['idx']
        _tier = 1
        if sweep['type'] == 'bullish':
            for j in range(len(_sl_list) - 1, -1, -1):
                if _sl_list[j] + lookback <= _si:
                    _tier = _sl_tier.get(_sl_list[j], 1)
                    break
        else:
            for j in range(len(_sh_list) - 1, -1, -1):
                if _sh_list[j] + lookback <= _si:
                    _tier = _sh_tier.get(_sh_list[j], 1)
                    break
        sweep['tier'] = _tier
    return sweeps

# -------------------------
# Fair Value Gap (FVG) Detection — Core ICT concept
# -------------------------
@numba.njit
def _detect_fair_value_gaps_jit(highs, lows, opens, closes):
    n = len(highs)
    
    atr = np.zeros(n)
    if n > 0:
        tr = np.zeros(n)
        for i in range(1, n):
            val1 = highs[i] - lows[i]
            val2 = abs(highs[i] - closes[i - 1])
            val3 = abs(lows[i] - closes[i - 1])
            tr[i] = max(val1, max(val2, val3))
        tr[0] = highs[0] - lows[0]
        
        if n >= 14:
            sum_tr = 0.0
            for i in range(14):
                sum_tr += tr[i]
            for i in range(14):
                atr[i] = sum_tr / 14.0
            for i in range(14, n):
                atr[i] = atr[i-1] * (13.0/14.0) + tr[i] * (1.0/14.0)
        else:
            sum_tr = 0.0
            for i in range(n):
                sum_tr += tr[i]
            avg_tr = sum_tr / n if n > 0 else 0.0
            for i in range(n):
                atr[i] = avg_tr if avg_tr > 0 else 0.00001

    types = [] # 1 for bullish, -1 for bearish
    tops = []
    bottoms = []
    bar_indices = []
    sizes = []
    body_ratios = []
    atr_mults = []
    is_vi = []
    
    for i in range(2, n):
        c1_high = highs[i - 2]
        c1_low = lows[i - 2]
        c3_low = lows[i]
        c3_high = highs[i]
        
        c_atr = atr[i - 1] if atr[i - 1] > 0 else 0.00001
        candle_range = highs[i - 1] - lows[i - 1]
        body_range = abs(closes[i - 1] - opens[i - 1])
        body_ratio = body_range / max(candle_range, 1e-8)
        atr_mult = candle_range / c_atr
        
        is_fvg = False
        # Bullish FVG
        if c1_high < c3_low:
            types.append(1)
            tops.append(c3_low)
            bottoms.append(c1_high)
            bar_indices.append(i - 1)
            sizes.append(c3_low - c1_high)
            body_ratios.append(body_ratio)
            atr_mults.append(atr_mult)
            is_vi.append(0)
            is_fvg = True
        # Bearish FVG
        elif c1_low > c3_high:
            types.append(-1)
            tops.append(c1_low)
            bottoms.append(c3_high)
            bar_indices.append(i - 1)
            sizes.append(c1_low - c3_high)
            body_ratios.append(body_ratio)
            atr_mults.append(atr_mult)
            is_vi.append(0)
            is_fvg = True
            
        if not is_fvg:
            # Real displacement metrics for the VI-forming candle (candle i)
            vi_candle_range = highs[i] - lows[i]
            vi_body_range = abs(closes[i] - opens[i])
            vi_body_ratio = vi_body_range / max(vi_candle_range, 1e-8)
            vi_atr_mult = vi_candle_range / c_atr
            # Volume Imbalance (Bullish)
            prev_body_high = max(opens[i-1], closes[i-1])
            curr_body_low = min(opens[i], closes[i])
            if prev_body_high < curr_body_low:
                types.append(1)
                tops.append(curr_body_low)
                bottoms.append(prev_body_high)
                bar_indices.append(i)
                sizes.append(curr_body_low - prev_body_high)
                body_ratios.append(vi_body_ratio)
                atr_mults.append(vi_atr_mult)
                is_vi.append(1)
                
            # Volume Imbalance (Bearish)
            prev_body_low = min(opens[i-1], closes[i-1])
            curr_body_high = max(opens[i], closes[i])
            if prev_body_low > curr_body_high:
                types.append(-1)
                tops.append(prev_body_low)
                bottoms.append(curr_body_high)
                bar_indices.append(i)
                sizes.append(prev_body_low - curr_body_high)
                body_ratios.append(vi_body_ratio)
                atr_mults.append(vi_atr_mult)
                is_vi.append(1)
            
    return types, tops, bottoms, bar_indices, sizes, body_ratios, atr_mults, is_vi

def detect_fair_value_gaps(df):
    """Detect Fair Value Gaps (imbalances). ICT's highest-probability zones. JIT optimized."""
    if len(df) < 3:
        return []
    
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    closes = df['close'].values
    index = df.index
    
    types, tops, bottoms, bar_indices, sizes, body_ratios, atr_mults, is_vi = \
        _detect_fair_value_gaps_jit(highs, lows, opens, closes)
        
    fvgs = []
    for t, top, bottom, bar_idx, size, body_ratio, atr_mult, vi_flag in zip(types, tops, bottoms, bar_indices, sizes, body_ratios, atr_mults, is_vi):
        fvgs.append({
            'type': 'bullish' if t == 1 else 'bearish',
            'top': top,
            'bottom': bottom,
            'mid': (top + bottom) / 2,
            'time': index[bar_idx],
            'bar_index': bar_idx,
            'size': size,
            'candle_high': highs[bar_idx],
            'candle_low': lows[bar_idx],
            'body_ratio': body_ratio,
            'atr_mult': atr_mult,
            'is_vi': bool(vi_flag)
        })
    return fvgs

def get_unmitigated_fvgs(df, fvgs):
    """Filter FVGs to only unmitigated (unfilled) ones. ICT: price returns to fill FVGs."""
    if not fvgs or df.empty:
        return []
        
    highs = df['high'].values
    lows = df['low'].values
    df_len = len(df)
    
    unmitigated = []
    for fvg in fvgs:
        filled = False
        start_idx = fvg['bar_index'] + 2
        if start_idx >= df_len:
            unmitigated.append(fvg)
            continue
            
        if fvg['type'] == 'bullish':
            if np.any(lows[start_idx:] <= fvg['mid']):
                filled = True
        elif fvg['type'] == 'bearish':
            if np.any(highs[start_idx:] >= fvg['mid']):
                filled = True
                
        if not filled:
            unmitigated.append(fvg)
    return unmitigated

# -------------------------
# Market Structure (BOS / MSS) — ICT structural analysis
# -------------------------
@numba.njit
def _detect_swing_points_jit(highs, lows, lookback):
    n = len(highs)
    sth_indices = []
    stl_indices = []
    
    if lookback <= 0:
        lookback = 5

    if n < 2 * lookback + 1:
        return sth_indices, stl_indices

    for center in range(lookback, n - lookback):
        val_h = highs[center]
        val_l = lows[center]
        is_high = True
        is_low = True
        for offset in range(-lookback, lookback + 1):
            if offset == 0:
                continue
            idx = center + offset
            if offset < 0:
                # LEFT side: strict. A plateau resolves to its LAST bar.
                if highs[idx] > val_h:
                    is_high = False
                if lows[idx] < val_l:
                    is_low = False
            else:
                # RIGHT side: inclusive. Ties do not veto the swing.
                if highs[idx] >= val_h:
                    is_high = False
                if lows[idx] <= val_l:
                    is_low = False
        if is_high:
            sth_indices.append(center)
        if is_low:
            stl_indices.append(center)
    return sth_indices, stl_indices

def classify_swing_tiers(swings, is_high):
    """ICT fractal swing hierarchy. Assigns each swing a tier:
    1 = Short-Term (ST), 2 = Intermediate-Term (IT), 3 = Long-Term (LT).
    An IT swing is more extreme than its neighbouring ST swings; an LT swing
    is more extreme than its neighbouring IT swings. swings must be ordered
    chronologically (by bar index). Returns a list of tiers aligned to swings.
    """
    n = len(swings)
    tiers = [1] * n
    if n < 3:
        return tiers
    prices = [s['price'] for s in swings]
    it_positions = []
    for i in range(2, n):
        if is_high:
            if prices[i - 1] > prices[i - 2] and prices[i - 1] > prices[i]:
                tiers[i - 1] = 2
                it_positions.append(i - 1)
        else:
            if prices[i - 1] < prices[i - 2] and prices[i - 1] < prices[i]:
                tiers[i - 1] = 2
                it_positions.append(i - 1)
    m = len(it_positions)
    for k in range(2, m):
        i = it_positions[k - 1]
        ip = it_positions[k - 2]
        cur = it_positions[k]
        if is_high:
            if prices[i] > prices[ip] and prices[i] > prices[cur]:
                tiers[i] = 3
        else:
            if prices[i] < prices[ip] and prices[i] < prices[cur]:
                tiers[i] = 3
    return tiers


def detect_swing_points(df, lookback=None):
    """Find swing highs and swing lows using N-bar lookback. JIT optimized."""
    import config
    if lookback is None:
        lookback = int(getattr(config, 'SWING_LOOKBACK', 5))
    highs = df['high'].values
    lows = df['low'].values
    
    sh_indices, sl_indices = _detect_swing_points_jit(highs, lows, lookback)
    
    swing_highs = []
    swing_lows = []
    index = df.index
    for idx in sh_indices:
        swing_highs.append({'price': highs[idx], 'time': index[idx], 'idx': idx})
    for idx in sl_indices:
        swing_lows.append({'price': lows[idx], 'time': index[idx], 'idx': idx})
    # ICT fractal tier (1=ST, 2=IT, 3=LT) attached as metadata
    for _s, _t in zip(swing_highs, classify_swing_tiers(swing_highs, True)):
        _s['tier'] = _t
    for _s, _t in zip(swing_lows, classify_swing_tiers(swing_lows, False)):
        _s['tier'] = _t
    return swing_highs, swing_lows

def detect_liquidity_pools(swing_points, threshold_points=2.0):
    """
    Scans a list of swing points (highs or lows) to find Equal Highs (EQH) or Equal Lows (EQL).
    threshold_points: Maximum price difference between swings to be considered 'equal'.
    Returns a list of liquidity pools.
    """
    pools = []
    if len(swing_points) < 2:
        return pools
        
    # We look for adjacent or nearby swings of the same type that are at the same price level.
    # To prevent O(N^2) explosion, we only look back at the last 10 swings for each new swing.
    for i in range(1, len(swing_points)):
        current_swing = swing_points[i]
        
        # Look back up to 10 previous swings
        start_idx = max(0, i - 10)
        for j in range(i - 1, start_idx - 1, -1):
            prev_swing = swing_points[j]
            
            # If price difference is within threshold, it's an equal high/low
            if abs(current_swing['price'] - prev_swing['price']) <= threshold_points:
                pools.append({
                    'price': (current_swing['price'] + prev_swing['price']) / 2,
                    'start_time': prev_swing['time'],
                    'end_time': current_swing['time'],
                    'start_idx': prev_swing['idx'],
                    'end_idx': current_swing['idx'],
                    'point_diff': abs(current_swing['price'] - prev_swing['price'])
                })
                break # Found the closest matching equal swing
                
    return pools

@numba.njit
def _calculate_market_structure_jit(closes, highs, lows, sh_indices, sl_indices, lookback):
    n = len(closes)
    # Track states: 0 = neutral, 1 = bullish, -1 = bearish
    structure_states = np.zeros(n, dtype=np.int32)
    
    mss_types = []
    mss_swing_idxs = []
    mss_break_idxs = []
    
    bos_types = []
    bos_swing_idxs = []
    bos_break_idxs = []
    
    current_state = 0
    
    broken_sh = np.zeros(n, dtype=np.bool_)
    broken_sl = np.zeros(n, dtype=np.bool_)
    
    sh_ptr = 0
    sl_ptr = 0
    confirmed_sh_idx = []
    confirmed_sl_idx = []
    
    for t in range(n):
        sh_limit = t - lookback
        sl_limit = t - lookback
        
        while sh_ptr < len(sh_indices) and sh_indices[sh_ptr] <= sh_limit:
            confirmed_sh_idx.append(sh_indices[sh_ptr])
            sh_ptr += 1
            
        while sl_ptr < len(sl_indices) and sl_indices[sl_ptr] <= sl_limit:
            confirmed_sl_idx.append(sl_indices[sl_ptr])
            sl_ptr += 1
            
        last_sh_idx = -1
        for j in range(len(confirmed_sh_idx) - 1, -1, -1):
            idx = confirmed_sh_idx[j]
            if not broken_sh[idx]:
                last_sh_idx = idx
                break
        if last_sh_idx == -1 and len(confirmed_sh_idx) > 0:
            last_sh_idx = confirmed_sh_idx[-1]
            
        last_sl_idx = -1
        for j in range(len(confirmed_sl_idx) - 1, -1, -1):
            idx = confirmed_sl_idx[j]
            if not broken_sl[idx]:
                last_sl_idx = idx
                break
        if last_sl_idx == -1 and len(confirmed_sl_idx) > 0:
            last_sl_idx = confirmed_sl_idx[-1]
            
        close_t = closes[t]
        
        if last_sh_idx != -1 and last_sl_idx != -1:
            sh_price = highs[last_sh_idx]
            sl_price = lows[last_sl_idx]
            
            if current_state == 0:
                if close_t > sh_price and not broken_sh[last_sh_idx]:
                    current_state = 1
                    broken_sh[last_sh_idx] = True
                    mss_types.append(1)
                    mss_swing_idxs.append(last_sh_idx)
                    mss_break_idxs.append(t)
                elif close_t < sl_price and not broken_sl[last_sl_idx]:
                    current_state = -1
                    broken_sl[last_sl_idx] = True
                    mss_types.append(-1)
                    mss_swing_idxs.append(last_sl_idx)
                    mss_break_idxs.append(t)
            elif current_state == 1:
                if close_t < sl_price and not broken_sl[last_sl_idx]:
                    current_state = -1
                    broken_sl[last_sl_idx] = True
                    mss_types.append(-1)
                    mss_swing_idxs.append(last_sl_idx)
                    mss_break_idxs.append(t)
                elif close_t > sh_price and not broken_sh[last_sh_idx]:
                    broken_sh[last_sh_idx] = True
                    bos_types.append(1)
                    bos_swing_idxs.append(last_sh_idx)
                    bos_break_idxs.append(t)
            elif current_state == -1:
                if close_t > sh_price and not broken_sh[last_sh_idx]:
                    current_state = 1
                    broken_sh[last_sh_idx] = True
                    mss_types.append(1)
                    mss_swing_idxs.append(last_sh_idx)
                    mss_break_idxs.append(t)
                elif close_t < sl_price and not broken_sl[last_sl_idx]:
                    broken_sl[last_sl_idx] = True
                    bos_types.append(-1)
                    bos_swing_idxs.append(last_sl_idx)
                    bos_break_idxs.append(t)
                    
        structure_states[t] = current_state
        
    return structure_states, mss_types, mss_swing_idxs, mss_break_idxs, bos_types, bos_swing_idxs, bos_break_idxs

def calculate_market_structure_chronologically(df, lookback=5):
    """Chronologically calculates market structure states (bullish/bearish/neutral),
    BOS, and MSS events for the entire DataFrame with ZERO lookahead bias. JIT optimized.
    """
    swing_highs, swing_lows = detect_swing_points(df, lookback=lookback)
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    sh_indices = np.array([sh['idx'] for sh in swing_highs], dtype=np.int32)
    sl_indices = np.array([sl['idx'] for sl in swing_lows], dtype=np.int32)
    
    structure_states_raw, mss_types, mss_swing_idxs, mss_break_idxs, bos_types, bos_swing_idxs, bos_break_idxs = \
        _calculate_market_structure_jit(closes, highs, lows, sh_indices, sl_indices, lookback)
        
    state_map = {0: "neutral", 1: "bullish", -1: "bearish"}
    structure_states = [state_map[s] for s in structure_states_raw]
    
    index = df.index
    mss_events = []
    for t, swing_idx, break_idx in zip(mss_types, mss_swing_idxs, mss_break_idxs):
        mss_events.append({
            'type': 'bullish' if t == 1 else 'bearish',
            'time': index[swing_idx],
            'price': highs[swing_idx] if t == 1 else lows[swing_idx],
            'idx': swing_idx,
            'break_time': index[break_idx],
            'break_idx': break_idx
        })
        
    bos_events = []
    for t, swing_idx, break_idx in zip(bos_types, bos_swing_idxs, bos_break_idxs):
        bos_events.append({
            'type': 'bullish' if t == 1 else 'bearish',
            'time': index[swing_idx],
            'price': highs[swing_idx] if t == 1 else lows[swing_idx],
            'idx': swing_idx,
            'break_time': index[break_idx],
            'break_idx': break_idx
        })
        
    return structure_states, mss_events, bos_events

def detect_market_structure(df, swing_lookback=5):
    """Detect current market structure and Market Structure Shifts (MSS).
    ICT MSS = Price breaks past a significant prior swing point, signaling a reversal.
    - Bullish MSS: During a downtrend (LH+LL), price breaks ABOVE a prior swing high.
    - Bearish MSS: During an uptrend (HH+HL), price breaks BELOW a prior swing low.
    Returns: (structure, mss_direction, swing_highs, swing_lows)
    """
    swing_highs, swing_lows = detect_swing_points(df, swing_lookback)
    
    # Run the robust stateful chronological calculator
    states, mss_events, _ = calculate_market_structure_chronologically(df, swing_lookback)
    
    structure = states[-1] if states else "neutral"
    
    # A market structure shift is active if it occurred on the last closed bar
    mss = None
    if mss_events:
        last_mss = mss_events[-1]
        # Check if the break happened recently (within last 2 closed bars to catch entries)
        if last_mss['break_idx'] >= len(df) - 2:
            mss = last_mss['type']

    return structure, mss, swing_highs, swing_lows

def compute_bos_after_mss_states(mss_events, bos_events, n):
    """Per-bar BoS-after-MSS confirmation states (buy, sell).
    Identical O(N) algorithm used by the backtester so live and backtest agree.
    A BoS confirms only if it occurs strictly after the most recent MSS of the
    same direction; a new MSS resets confirmation. Returns two lists len n.
    """
    buy_states = [False] * n
    sell_states = [False] * n
    sorted_mss = sorted(mss_events, key=lambda x: x['break_idx'])
    sorted_bos = sorted(bos_events, key=lambda x: x['break_idx'])
    mss_ptr = 0
    bos_ptr = 0
    last_bull_mss_idx = -1
    last_bear_mss_idx = -1
    bull_ok = False
    bear_ok = False
    for t in range(n):
        while mss_ptr < len(sorted_mss) and sorted_mss[mss_ptr]['break_idx'] == t:
            if sorted_mss[mss_ptr]['type'] == 'bullish':
                last_bull_mss_idx = t
                bull_ok = False
            elif sorted_mss[mss_ptr]['type'] == 'bearish':
                last_bear_mss_idx = t
                bear_ok = False
            mss_ptr += 1
        while bos_ptr < len(sorted_bos) and sorted_bos[bos_ptr]['break_idx'] == t:
            if sorted_bos[bos_ptr]['type'] == 'bullish':
                if last_bull_mss_idx != -1 and last_bull_mss_idx < t:
                    bull_ok = True
            elif sorted_bos[bos_ptr]['type'] == 'bearish':
                if last_bear_mss_idx != -1 and last_bear_mss_idx < t:
                    bear_ok = True
            bos_ptr += 1
        buy_states[t] = bull_ok
        sell_states[t] = bear_ok
    return buy_states, sell_states


def check_bos_after_mss(swing_highs, swing_lows, direction, df=None):
    """Check if there is a Break of Structure (BoS) confirming after an MSS.
    ICT: After an MSS breaks a key swing level, we need a subsequent BoS
    (new HH for bullish, new LL for bearish) to confirm the shift is real.
    When df is provided, uses the rigorous chronological MSS/BoS calculation
    (identical algorithm to the backtester) for live/backtest parity. The
    legacy swing-sequence heuristic remains as a fallback when df is None.
    """
    if df is not None and len(df) > 0:
        _, _mss_events, _bos_events = calculate_market_structure_chronologically(df)
        _buy, _sell = compute_bos_after_mss_states(_mss_events, _bos_events, len(df))
        if not _buy:
            return False
        return _buy[-1] if direction in ("bullish", "Buy") else _sell[-1]
    if direction == "bullish" or direction == "Buy":
        if len(swing_highs) < 4: return False
        prices = [h['price'] for h in swing_highs[-6:]]
        # Find where MSS happened: a HH after LH sequence
        for i in range(2, len(prices)):
            was_downtrend = prices[i-1] < prices[i-2]  # LH = downtrend
            broke_above = prices[i] > prices[i-1]       # HH = MSS
            if was_downtrend and broke_above:
                # Now look for BoS: another HH after the MSS point
                for j in range(i+1, len(prices)):
                    if prices[j] > prices[j-1]:  # Continuation HH = BoS confirmed
                        return True
        return False
    else:
        if len(swing_lows) < 4: return False
        prices = [l['price'] for l in swing_lows[-6:]]
        for i in range(2, len(prices)):
            was_uptrend = prices[i-1] > prices[i-2]   # HL = uptrend
            broke_below = prices[i] < prices[i-1]       # LL = MSS
            if was_uptrend and broke_below:
                for j in range(i+1, len(prices)):
                    if prices[j] < prices[j-1]:  # Continuation LL = BoS confirmed
                        return True
        return False

# -------------------------
# Displacement Quality — ICT confirmation candle quality
# -------------------------
def check_displacement_quality(df, bar_index, min_body_ratio=0.6):
    """ICT displacement: strong full-bodied candle with minimal wicks.
    A valid displacement has body >= 60% of total range."""
    if bar_index >= len(df) or bar_index < 0:
        return False
    bar = df.iloc[bar_index]
    body = abs(bar['close'] - bar['open'])
    total_range = bar['high'] - bar['low']
    if total_range == 0:
        return False
    return (body / total_range) >= min_body_ratio

# -------------------------
# OB + FVG Confluence — ICT's strongest setup
# -------------------------
def check_ob_fvg_confluence(ob, fvgs, direction):
    """Check if an Order Block overlaps with any Fair Value Gap.
    ICT: OB + FVG overlap = institutional interest confirmed from two independent signals."""
    ob_high = ob['high']
    ob_low = ob['low']
    for fvg in fvgs:
        if direction == "bullish" and fvg['type'] == 'bullish':
            if ob_low <= fvg['top'] and ob_high >= fvg['bottom']:
                return True, fvg
        elif direction == "bearish" and fvg['type'] == 'bearish':
            if ob_high >= fvg['bottom'] and ob_low <= fvg['top']:
                return True, fvg
    return False, None

# -------------------------
# Volume Profile (POC / HVN)
# -------------------------
def calculate_volume_profile(df, lookback=200, bins=20):
    """
    Calculates the Point of Control (POC) for the last `lookback` bars.
    Distributes tick_volume across the high-low range of each bar into price bins.
    Returns the POC price level.
    """
    if len(df) == 0:
        return None
    
    recent_df = df.iloc[-lookback:] if len(df) > lookback else df
    min_price = recent_df['low'].min()
    max_price = recent_df['high'].max()
    
    if max_price == min_price:
        return min_price
        
    bin_size = (max_price - min_price) / bins
    if bin_size == 0:
        return min_price
        
    price_bins = np.zeros(bins)
    
    highs = recent_df['high'].values
    lows = recent_df['low'].values
    vols = recent_df['tick_volume'].values if 'tick_volume' in recent_df.columns else np.ones(len(recent_df))
    
    start_bins = np.clip(((lows - min_price) / bin_size).astype(int), 0, bins - 1)
    end_bins = np.clip(((highs - min_price) / bin_size).astype(int), 0, bins - 1)
    
    for i in range(len(highs)):
        sb = start_bins[i]
        eb = end_bins[i]
        if sb == eb:
            price_bins[sb] += vols[i]
        else:
            vol_per_bin = vols[i] / (eb - sb + 1)
            price_bins[sb:eb+1] += vol_per_bin
                    
    poc_bin_idx = np.argmax(price_bins)
    poc_price = min_price + (poc_bin_idx + 0.5) * bin_size
    return poc_price

# -------------------------
# SMT Divergence Scaffold
# -------------------------
def check_advanced_smt(main_df, correlated_dfs_dict, current_idx, direction, lookback=20):
    """
    Advanced SMT: Evaluates multiple correlated pairs dynamically.
    Returns True if ANY of the correlated assets show SMT divergence.
    """
    if not correlated_dfs_dict:
        return False
        
    divergences_found = 0
    for symbol_name, corr_df in correlated_dfs_dict.items():
        if corr_df is not None and not corr_df.empty:
            if check_smt_divergence(main_df, corr_df, current_idx, direction, lookback, symbol_name):
                divergences_found += 1
                
    # If at least one correlated pair shows SMT divergence, we have confirmation
    return divergences_found > 0

def check_smt_divergence(main_df, correlated_df, current_idx, direction, lookback=20, corr_symbol_name="DXY"):
    """
    Checks if there is an SMT divergence between the main asset and a correlated asset.
    
    True SMT Divergence:
    - Bullish: Main asset makes a LOWER low, but correlated asset makes a HIGHER low 
      (or inverse: DXY makes a LOWER high).
    - Bearish: Main asset makes a HIGHER high, but correlated asset makes a HIGHER high
      (or inverse: DXY makes a HIGHER low).
    """
    if correlated_df is None or len(correlated_df) == 0:
        return False
        
    try:
        # Increase effective lookback to ensure we capture two distinct structural swings
        effective_lookback = max(80, lookback)
        start_idx = max(0, current_idx - effective_lookback)
        main_window = main_df.iloc[start_idx:current_idx+1].copy()
        
        if len(main_window) < 20:
            return False
            
        inverse_pairs = ["DXY", "USDCHF", "USDCAD", "USDJPY"]
        is_inverse = any(p in corr_symbol_name.upper() for p in inverse_pairs)
        
        import numpy as np
        import pandas as pd
        
        # Helper to get the absolute time for an index in main_window
        def get_time(idx):
            if 'time' in main_window.columns:
                return main_window.loc[idx, 'time']
            else:
                return idx  # Assuming it's already a DatetimeIndex
                
        # Helper to find extreme in correlated asset around a specific time
        def get_corr_extreme(target_time, is_max):
            if 'time' in correlated_df.columns:
                corr_times = correlated_df['time'].values
            else:
                corr_times = correlated_df.index.values
            
            # Find closest index
            idx = np.searchsorted(corr_times, np.datetime64(target_time), side='left')
            idx = min(idx, len(correlated_df) - 1)
            
            # Look in a +/- 5 bar window to account for slight data feed desyncs
            c_start = max(0, idx - 5)
            c_end = min(len(correlated_df), idx + 6)
            c_window = correlated_df.iloc[c_start:c_end]
            
            if len(c_window) == 0:
                return None
                
            return c_window['high'].max() if is_max else c_window['low'].min()

        if direction == "bullish":
            # 1. Find absolute lowest point
            min1_idx = main_window['low'].idxmin()
            min1_time = get_time(min1_idx)
            min1_price = main_window.loc[min1_idx, 'low']
            
            # 2. Mask out +/- 10 bars around min1 to find the second distinct swing low
            min1_pos = main_window.index.get_loc(min1_idx)
            mask = abs(np.arange(len(main_window)) - min1_pos) > 10
            if not mask.any(): return False
            
            main_window_masked = main_window.iloc[mask]
            min2_idx = main_window_masked['low'].idxmin()
            min2_time = get_time(min2_idx)
            min2_price = main_window_masked.loc[min2_idx, 'low']
            
            # 3. Order chronologically
            if min1_time > min2_time:
                recent_time, recent_price = min1_time, min1_price
                prev_time, prev_price = min2_time, min2_price
            else:
                recent_time, recent_price = min2_time, min2_price
                prev_time, prev_price = min1_time, min1_price
                
            # Main asset must have made a lower low
            if recent_price >= prev_price:
                return False
                
            # 4. Check correlated asset
            if is_inverse:
                # DXY should make a higher high to confirm. Divergence = DXY fails to make higher high.
                corr_recent_high = get_corr_extreme(recent_time, is_max=True)
                corr_prev_high = get_corr_extreme(prev_time, is_max=True)
                
                if corr_recent_high is None or corr_prev_high is None: return False
                return corr_recent_high < corr_prev_high
            else:
                # Correlated should make a lower low. Divergence = Correlated fails to make lower low.
                corr_recent_low = get_corr_extreme(recent_time, is_max=False)
                corr_prev_low = get_corr_extreme(prev_time, is_max=False)
                
                if corr_recent_low is None or corr_prev_low is None: return False
                return corr_recent_low > corr_prev_low

        else:
            # Bearish
            # 1. Find absolute highest point
            max1_idx = main_window['high'].idxmax()
            max1_time = get_time(max1_idx)
            max1_price = main_window.loc[max1_idx, 'high']
            
            # 2. Mask out +/- 10 bars around max1
            max1_pos = main_window.index.get_loc(max1_idx)
            mask = abs(np.arange(len(main_window)) - max1_pos) > 10
            if not mask.any(): return False
            
            main_window_masked = main_window.iloc[mask]
            max2_idx = main_window_masked['high'].idxmax()
            max2_time = get_time(max2_idx)
            max2_price = main_window_masked.loc[max2_idx, 'high']
            
            # 3. Order chronologically
            if max1_time > max2_time:
                recent_time, recent_price = max1_time, max1_price
                prev_time, prev_price = max2_time, max2_price
            else:
                recent_time, recent_price = max2_time, max2_price
                prev_time, prev_price = max1_time, max1_price
                
            # Main asset must have made a higher high
            if recent_price <= prev_price:
                return False
                
            # 4. Check correlated asset
            if is_inverse:
                # DXY should make a lower low to confirm. Divergence = DXY fails to make lower low.
                corr_recent_low = get_corr_extreme(recent_time, is_max=False)
                corr_prev_low = get_corr_extreme(prev_time, is_max=False)
                
                if corr_recent_low is None or corr_prev_low is None: return False
                return corr_recent_low > corr_prev_low
            else:
                # Correlated should make a higher high. Divergence = Correlated fails to make higher high.
                corr_recent_high = get_corr_extreme(recent_time, is_max=True)
                corr_prev_high = get_corr_extreme(prev_time, is_max=True)
                
                if corr_recent_high is None or corr_prev_high is None: return False
                return corr_recent_high < corr_prev_high

    except Exception as e:
        logger.debug(f"SMT divergence error: {e}")
        return False


CORRELATED_PAIRS = {
    'EURUSDm': 'GBPUSDm', 'GBPUSDm': 'EURUSDm',
    'EURUSD': 'GBPUSD', 'GBPUSD': 'EURUSD',
    'AUDUSDm': 'NZDUSDm', 'NZDUSDm': 'AUDUSDm',
    'USDJPYm': 'USDCHFm', 'USDCHFm': 'USDJPYm',
    'US30m': 'USTECm', 'USTECm': 'US30m', 'US500m': 'USTECm',
    'XAUUSDm': 'XAGUSDm', 'XAGUSDm': 'XAUUSDm',
    'BTCUSDm': 'ETHUSDm', 'ETHUSDm': 'BTCUSDm',
}

def get_correlated_pair(symbol: str) -> str:
    s = str(symbol).strip()
    return CORRELATED_PAIRS.get(s, CORRELATED_PAIRS.get(s.upper(), ''))


def detect_smt_divergence(df_primary, df_correlated, lookback=20, mode='all'):
    """Detect ICT SMT (Smart Money Tool) Divergence between correlated assets.

    Correlated pairs:
    - Bullish SMT: Primary pair makes a Lower Low (LL) while Correlated pair fails
      to make a LL (makes a Higher Low, HL).
    - Bearish SMT: Primary pair makes a Higher High (HH) while Correlated pair fails
      to make a HH (makes a Lower High, LH).

    Returns a list of SMT divergence events dict:
      {'type': 'bullish'/'bearish', 'idx': int, 'time': Timestamp, 'strength': float}
    """
    if df_primary is None or df_correlated is None or len(df_primary) < 20 or len(df_correlated) < 20:
        return []

    common_idx = df_primary.index.intersection(df_correlated.index)
    if len(common_idx) < 20:
        return []

    p_df = df_primary.loc[common_idx]
    c_df = df_correlated.loc[common_idx]

    sh_p, sl_p = detect_swing_points(p_df, lookback=5)
    sh_c, sl_c = detect_swing_points(c_df, lookback=5)

    events = []
    if len(sl_p) >= 2 and len(sl_c) >= 2:
        for i in range(1, len(sl_p)):
            idx_p2 = sl_p[i]
            idx_p1 = sl_p[i - 1]
            if (idx_p2 - idx_p1) > lookback:
                continue

            low_p2 = p_df['low'].iloc[idx_p2]
            low_p1 = p_df['low'].iloc[idx_p1]

            matching_c2 = [k for k in sl_c if abs(k - idx_p2) <= 3]
            matching_c1 = [k for k in sl_c if abs(k - idx_p1) <= 3]

            if matching_c2 and matching_c1:
                idx_c2 = matching_c2[-1]
                idx_c1 = matching_c1[-1]
                low_c2 = c_df['low'].iloc[idx_c2]
                low_c1 = c_df['low'].iloc[idx_c1]

                if low_p2 < low_p1 and low_c2 > low_c1:
                    events.append({
                        'type': 'bullish',
                        'idx': idx_p2,
                        'time': common_idx[idx_p2],
                        'strength': float((low_c2 - low_c1) / (c_df['close'].iloc[idx_c2] * 0.001) if c_df['close'].iloc[idx_c2] > 0 else 1.0)
                    })

    if len(sh_p) >= 2 and len(sh_c) >= 2:
        for i in range(1, len(sh_p)):
            idx_p2 = sh_p[i]
            idx_p1 = sh_p[i - 1]
            if (idx_p2 - idx_p1) > lookback:
                continue

            high_p2 = p_df['high'].iloc[idx_p2]
            high_p1 = p_df['high'].iloc[idx_p1]

            matching_c2 = [k for k in sh_c if abs(k - idx_p2) <= 3]
            matching_c1 = [k for k in sh_c if abs(k - idx_p1) <= 3]

            if matching_c2 and matching_c1:
                idx_c2 = matching_c2[-1]
                idx_c1 = matching_c1[-1]
                high_c2 = c_df['high'].iloc[idx_c2]
                high_c1 = c_df['high'].iloc[idx_c1]

                if high_p2 > high_p1 and high_c2 < high_c1:
                    events.append({
                        'type': 'bearish',
                        'idx': idx_p2,
                        'time': common_idx[idx_p2],
                        'strength': float((high_c1 - high_c2) / (c_df['close'].iloc[idx_c2] * 0.001) if c_df['close'].iloc[idx_c2] > 0 else 1.0)
                    })

    if mode == 'bullish':
        return [e for e in events if e['type'] == 'bullish']
    elif mode == 'bearish':
        return [e for e in events if e['type'] == 'bearish']
    return events


def detect_mmxm_model(df, lookback=100, atr=None):
    """Detect ICT Market Maker Buy Model (MMBM) and Market Maker Sell Model (MMSM).

    MMBM:
    1. Consolidation (Original Accumulation)
    2. Engineering Liquidity (Low Resistance Liquidity Run / Sell Side Liquidity sweep)
    3. Smart Money Reversal (MSS bullish + FVG/OB)
    4. Re-accumulation -> Target Original Accumulation High

    MMSM:
    1. Consolidation (Original Accumulation)
    2. Engineering Liquidity (Buy Side Liquidity sweep)
    3. Smart Money Reversal (MSS bearish + FVG/OB)
    4. Re-distribution -> Target Original Accumulation Low
    """
    if df is None or len(df) < 50:
        return []

    from atr_context import atr_last
    if atr is None:
        atr = atr_last(df)

    sh, sl = detect_swing_points(df, lookback=5)
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    models = []
    n = len(df)

    for i in range(max(30, n - lookback), n - 5):
        recent_sl = [k for k in sl if k < i and (i - k) <= 30]
        if not recent_sl:
            continue

        orig_accum_idx = max(0, i - 60)
        orig_accum_high = highs[orig_accum_idx:i - 15].max()
        orig_accum_low = lows[orig_accum_idx:i - 15].min()

        ssl_swept = lows[i] < orig_accum_low and closes[i] > orig_accum_low
        if ssl_swept and (i + 3) < n:
            disp_range = highs[i:i + 4].max() - lows[i]
            if disp_range >= 1.2 * atr:
                models.append({
                    'type': 'MMBM',
                    'direction': 'bullish',
                    'reversal_idx': i,
                    'reversal_time': df.index[i],
                    'orig_high': orig_accum_high,
                    'orig_low': orig_accum_low,
                    'target_price': orig_accum_high,
                    'stop_price': lows[i] - 0.5 * atr,
                    'quality_score': float(disp_range / atr)
                })

        recent_sh = [k for k in sh if k < i and (i - k) <= 30]
        if not recent_sh:
            continue

        bsl_swept = highs[i] > orig_accum_high and closes[i] < orig_accum_high
        if bsl_swept and (i + 3) < n:
            disp_range = highs[i] - lows[i:i + 4].min()
            if disp_range >= 1.2 * atr:
                models.append({
                    'type': 'MMSM',
                    'direction': 'bearish',
                    'reversal_idx': i,
                    'reversal_time': df.index[i],
                    'orig_high': orig_accum_high,
                    'orig_low': orig_accum_low,
                    'target_price': orig_accum_low,
                    'stop_price': highs[i] + 0.5 * atr,
                    'quality_score': float(disp_range / atr)
                })

    return models





