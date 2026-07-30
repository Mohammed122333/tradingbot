"""Market regime classification — pure price action (no lagging indicators).

Shared by BOTH backtest and live so the 'stand aside in chop / size down in
high volatility' behavior is IDENTICAL in both (parity).

A discretionary ICT trader does NOT slap ADX or oscillators on the chart.
They read:
  1. SWING STRUCTURE — clean HH/HL = uptrend, LH/LL = downtrend,
     overlapping swings = range/chop.
  2. DISPLACEMENT EFFICIENCY — did price actually go somewhere, or just
     oscillate?  |close[-1]-close[-N]| / sum(bar ranges). Near 1 = trend,
     near 0 = chop.
  3. INSIDE BAR RATIO — high % of inside bars = tight consolidation = chop.
  4. RANGE EXPANSION / CONTRACTION — recent bar ranges vs. older ranges.
     Blowing out = high volatility (size down). Compressing = chop (stand aside).

All thresholds are config-driven and the whole filter can be disabled with
REGIME_FILTER_ENABLED = False.
"""
import numpy as np
import config


def _detect_swings(highs, lows, lookback=3):
    """Detect swing highs/lows using fractal logic (no indicators).

    A swing high is a bar whose high exceeds `lookback` bars on each side.
    A swing low is a bar whose low is below `lookback` bars on each side.
    Returns (swing_highs, swing_lows) as lists of (index, price).
    """
    n = len(highs)
    sh, sl = [], []
    for i in range(lookback, n - lookback):
        is_sh = all(highs[i] >= highs[i - j] for j in range(1, lookback + 1)) and \
                all(highs[i] >= highs[i + j] for j in range(1, lookback + 1))
        is_sl = all(lows[i] <= lows[i - j] for j in range(1, lookback + 1)) and \
                all(lows[i] <= lows[i + j] for j in range(1, lookback + 1))
        if is_sh:
            sh.append((i, highs[i]))
        if is_sl:
            sl.append((i, lows[i]))
    return sh, sl


def _classify_swing_structure(sh, sl):
    """Classify swing structure: 'up', 'down', or 'range'.

    Up   = HH + HL (higher highs AND higher lows)
    Down = LH + LL (lower highs AND lower lows)
    Range = anything else (overlapping, mixed)
    """
    if len(sh) < 2 or len(sl) < 2:
        return 'range'
    hh = sh[-1][1] > sh[-2][1]
    lh = sh[-1][1] < sh[-2][1]
    hl = sl[-1][1] > sl[-2][1]
    ll = sl[-1][1] < sl[-2][1]
    if hh and hl:
        return 'up'
    if lh and ll:
        return 'down'
    return 'range'


def _displacement_efficiency(closes, lookback):
    """Directional conviction: |net move| / total path length.

    1.0 = price moved in a straight line (strong trend).
    0.0 = price went nowhere (pure chop / mean reversion).
    """
    if len(closes) < lookback + 1:
        return 0.0
    recent = closes[-lookback:]
    net = abs(recent[-1] - recent[0])
    path = np.abs(np.diff(recent)).sum()
    return float(net / path) if path > 0 else 0.0


def _inside_bar_ratio(highs, lows, lookback):
    """Fraction of recent bars that are inside bars (consolidation signature).

    An inside bar is entirely within the previous bar's range.
    High ratio = tight consolidation = chop.
    """
    if lookback < 2:
        return 0.0
    h = highs[-lookback:]
    l = lows[-lookback:]
    inside = 0
    for i in range(1, len(h)):
        if h[i] <= h[i - 1] and l[i] >= l[i - 1]:
            inside += 1
    return inside / (len(h) - 1) if len(h) > 1 else 0.0


def _range_expansion_ratio(highs, lows, lookback):
    """Recent bar ranges vs. older bar ranges.

    > 1 = expanding (volatile, trending hard).
    < 1 = compressing (chop, consolidation).
    Returns ratio of mean(recent half ranges) / mean(older half ranges).
    """
    if lookback < 8:
        return 1.0
    half = lookback // 2
    ranges = highs[-lookback:] - lows[-lookback:]
    recent_mean = float(ranges[-half:].mean()) if half > 0 else 0.0
    older_mean = float(ranges[:-half].mean()) if half > 0 else 0.0
    return recent_mean / older_mean if older_mean > 0 else 1.0


def classify_regime(df, atr_val=None):
    """Classify the regime for a window of CLOSED bars (no look-ahead).

    Pure price-action: swing structure + displacement efficiency + inside-bar
    ratio + range expansion/contraction. No ADX, no oscillators.

    Returns a dict:
      tradeable  -> False in chop/range when REGIME_CHOP_BLOCK is on
      size_mult  -> < 1.0 in high-volatility / blow-out regimes (size down)
      conf_bump  -> extra confluence points required in weak/range regimes
      label      -> 'trend', 'range', 'chop', 'high-vol', 'compression'
      structure  -> 'up', 'down', or 'range'
      efficiency -> displacement efficiency ratio (0-1)
      inside_bar_pct -> inside bar percentage (0-1)
      range_expansion -> recent/older range ratio
      atr        -> passed-through atr (for diagnostics only)

    Fails OPEN (tradeable=True, size_mult=1.0) on any error or thin data so a
    regime glitch can never silently halt the whole bot.
    """
    res = {
        'tradeable': True, 'size_mult': 1.0, 'conf_bump': 0,
        'label': 'trend', 'structure': 'range', 'efficiency': 0.0,
        'inside_bar_pct': 0.0, 'range_expansion': 1.0, 'atr': 0.0,
    }
    if not getattr(config, 'REGIME_FILTER_ENABLED', False):
        return res
    if df is None:
        return res

    lookback = int(getattr(config, 'REGIME_LOOKBACK', 50))
    swing_lb = int(getattr(config, 'REGIME_SWING_LOOKBACK', 3))
    eff_min = float(getattr(config, 'REGIME_DISPLACEMENT_MIN', 0.35))
    ib_max = float(getattr(config, 'REGIME_INSIDE_BAR_MAX', 0.35))
    contraction_max = float(getattr(config, 'REGIME_RANGE_CONTRACTION_MAX', 0.6))
    expansion_mult = float(getattr(config, 'REGIME_RANGE_EXPANSION_MULT', 2.0))
    hv_size_mult = float(getattr(config, 'REGIME_HIGH_VOL_SIZE_MULT', 0.5))

    try:
        if isinstance(df, dict):
            highs = df['high'].astype(float)
            lows = df['low'].astype(float)
            closes = df['close'].astype(float)
            data_len = len(highs)
        else:
            highs = df['high'].values.astype(float)
            lows = df['low'].values.astype(float)
            closes = df['close'].values.astype(float)
            data_len = len(df)
    except Exception:
        return res

    if data_len < 30:
        return res

    lb = min(lookback, data_len)

    # 1) Swing structure (market structure read)
    try:
        sh, sl = _detect_swings(highs, lows, swing_lb)
        structure = _classify_swing_structure(sh, sl)
    except Exception:
        structure = 'range'
    res['structure'] = structure

    # 2) Displacement efficiency (directional conviction)
    try:
        efficiency = _displacement_efficiency(closes, lb)
    except Exception:
        efficiency = 0.0
    res['efficiency'] = efficiency

    # 3) Inside bar ratio (consolidation signature)
    try:
        ib_ratio = _inside_bar_ratio(highs, lows, lb)
    except Exception:
        ib_ratio = 0.0
    res['inside_bar_pct'] = ib_ratio

    # 4) Range expansion / contraction
    try:
        range_exp = _range_expansion_ratio(highs, lows, lb)
    except Exception:
        range_exp = 1.0
    res['range_expansion'] = range_exp

    # Store atr for diagnostics only (NOT used for regime decision)
    if atr_val is not None and atr_val > 0:
        res['atr'] = float(atr_val)

    # --- DECISION (pure price-action, like a human reading the chart) ---

    is_trending_structure = structure in ('up', 'down')
    has_displacement = efficiency >= eff_min
    is_consolidating = ib_ratio >= ib_max
    is_compressing = range_exp <= contraction_max
    is_blowing_out = range_exp >= expansion_mult

    # High volatility / blow-out: size down regardless of structure
    if is_blowing_out:
        res['label'] = 'high-vol'
        res['size_mult'] = min(res['size_mult'], hv_size_mult)
        # Still tradeable if structure + displacement support it, just smaller
        if not (is_trending_structure and has_displacement):
            res['conf_bump'] = int(getattr(config, 'REGIME_CHOP_CONF_BUMP', 1))

    # Chop / range / consolidation: stand aside
    elif is_consolidating or is_compressing or (not has_displacement and not is_trending_structure):
        res['label'] = 'chop' if is_consolidating else 'range'
        res['conf_bump'] = int(getattr(config, 'REGIME_CHOP_CONF_BUMP', 1))
        if getattr(config, 'REGIME_CHOP_BLOCK', True):
            res['tradeable'] = False

    # Weak trend (structure ok but efficiency marginal): require more confluence
    elif is_trending_structure and not has_displacement:
        res['label'] = 'weak-trend'
        res['conf_bump'] = int(getattr(config, 'REGIME_CHOP_CONF_BUMP', 1))

    # Clean trend: tradeable, no bump
    else:
        res['label'] = 'trend'

    return res
