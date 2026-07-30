"""Confluence scoring, TP/SL calculation, and HTF trend detection.

Extracted from simulation.py to keep each file focused.
"""
import logging
import os

import pandas as pd
import numpy as np

import config
from broker import get_broker, TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, \
    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1
from config import HIGHER_TFS
from utils import get_spread, get_data
from indicators import (
    check_ob_fvg_confluence, detect_fair_value_gaps, detect_liquidity_sweep,
)


try:
    from multi_tf import hierarchical_confluence
except ImportError:
    hierarchical_confluence = None

logger = logging.getLogger(__name__)


def calculate_confluence_score(ob, direction, fvgs, structure, mss,
                                bullish_sweep, bearish_sweep,
                                higher_tf_trend, ote_low, ote_high, entry_price,
                                df, has_displacement=False, vp_poc=None,
                                eqh_pools=None, eql_pools=None,
                                symbol=None, **kwargs):
    if df is None or df.empty:
        return 0, []

    _cw = getattr(config, 'CONFLUENCE_WEIGHTS', {}) or {}

    if getattr(config, 'CONFLUENCE_REQUIRE_MANDATORY', False):
        if not ((direction == "bullish" and bullish_sweep) or
                (direction == "bearish" and bearish_sweep)) or not has_displacement:
            return -1, ["Mandatory Gate Failed"]

    score = 0
    details = []

    has_fvg, _ = check_ob_fvg_confluence(ob, fvgs, direction)
    if has_fvg:
        score += _cw.get('fvg_ob', 3)
        details.append("FVG+OB")

    # ── Structure, tier-weighted
    if (direction in ("bullish", "Buy") and structure == "bullish") or \
       (direction in ("bearish", "Sell") and structure == "bearish"):
        _st_tier = int(kwargs.get('structure_tier', 1) or 1)
        _st_pts = getattr(config, 'STRUCTURE_TIER_POINTS', {1: 1, 2: 2, 3: 3})
        score += _st_pts.get(_st_tier, _cw.get('structure', 2))
        details.append(f"Structure(T{_st_tier})")

    if mss:
        if (direction in ("bullish", "Buy") and mss == "bullish") or \
           (direction in ("bearish", "Sell") and mss == "bearish"):
            _mss_tier = int(kwargs.get('mss_tier', 1) or 1)
            _st_pts = getattr(config, 'STRUCTURE_TIER_POINTS', {1: 1, 2: 2, 3: 3})
            score += _st_pts.get(_mss_tier, _cw.get('mss', 2))
            details.append(f"MSS(T{_mss_tier})")

    # ── Sweep, tier-weighted
    if (direction in ("bullish", "Buy") and bullish_sweep) or \
       (direction in ("bearish", "Sell") and bearish_sweep):
        _sw_tier = int(kwargs.get('sweep_tier', 1) or 1)
        _sw_pts = getattr(config, 'SWEEP_TIER_POINTS', {1: 1, 2: 2, 3: 4})
        score += _sw_pts.get(_sw_tier, _cw.get('sweep', 2))
        details.append(f"Sweep(T{_sw_tier})")

    if ote_low <= entry_price <= ote_high:
        score += _cw.get('ote', 1)
        details.append("OTE")

    if vp_poc is not None:
        dist = abs(entry_price - vp_poc)
        if entry_price * 0.001 >= dist:
            score += _cw.get('poc', 1)
            details.append("POC")

    if higher_tf_trend != "neutral":
        if (direction in ("bullish", "Buy") and higher_tf_trend == "bullish") or \
           (direction in ("bearish", "Sell") and higher_tf_trend == "bearish"):
            score += _cw.get('htf', 1)
            details.append("HTF")

    if has_displacement:
        score += _cw.get('displacement', 1)
        details.append("Displacement")

    # ── Multi-TF Hierarchical Confluence ────────────────────────────────
    if ((getattr(config, 'MULTI_TF_CONFLUENCE_ENABLED', False) or getattr(config, 'MULTI_TF_GATE_ENABLED', False))
            and symbol is not None
            and hierarchical_confluence is not None):
        try:
            mtf_dir = 'bullish' if direction in ('bullish', 'Buy') else 'bearish'
            mtf_date_to = kwargs.get('entry_time')
            mtf_result = hierarchical_confluence(symbol, mtf_dir, date_to=mtf_date_to)
            
            if getattr(config, 'MULTI_TF_CONFLUENCE_ENABLED', False):
                aligned = mtf_result.get('score', 0)
                total_layers = len(mtf_result.get('layers', []))
                if aligned > 0:
                    pts = min(aligned, _cw.get('multi_tf_aligned', 3))
                    score += pts
                    details.append(f"MultiTF({aligned}/{total_layers})")

            gated = mtf_result.get('gated', False)
            if gated and getattr(config, 'MULTI_TF_GATE_ENABLED', False):
                return -1, [mtf_result.get('gate_msg', 'Multi-TF Gate')]
        except Exception as _mtf_err:
            logger.debug("multi_tf: %s", _mtf_err)

    if getattr(config, 'KILLZONE_ENABLED', False) and 'entry_time' in kwargs:
        try:
            import ict_pro
            kz_pts, kz_name = ict_pro.killzone_bonus(kwargs['entry_time'])
            if kz_pts > 0:
                score += kz_pts
                details.append(f"Killzone(+{kz_pts} {kz_name})")
        except Exception:
            pass

    # ── Causality gate: pools formed by FUTURE swings must not influence the
    # score, and MUST NOT drive the veto below. `bar_index` is the current bar.
    _bar_i = kwargs.get('bar_index')
    _swing_lb = int(getattr(config, 'SWING_LOOKBACK', 5))

    def _visible_pools(pools):
        if not pools or _bar_i is None:
            return pools or []
        out = []
        for p in pools:
            if not isinstance(p, dict):
                out.append(p)
                continue
            end_idx = p.get('end_idx')
            if end_idx is None or (end_idx + _swing_lb) <= _bar_i:
                out.append(p)
        return out

    eqh_pools = _visible_pools(eqh_pools)
    eql_pools = _visible_pools(eql_pools)

    if direction in ("bullish", "Buy"):
        if eqh_pools:
            _cands = [p for p in eqh_pools
                      if isinstance(p, dict) and p.get('price', 0) > entry_price]
            if _cands:
                _best = min(_cands, key=lambda p: p['price'])
                if (_best['price'] - entry_price) > (entry_price * 0.0005):
                    _base = _cw.get('eq_target', 2)
                    _bonus = 1 if (_best.get('tightness', 0) > 0.7
                                   and _best.get('tier', 1) >= 2) else 0
                    score += _base + _bonus
                    details.append(f"Targeting EQH(T{_best.get('tier',1)})")
        if eql_pools:
            _cands = [p for p in eql_pools
                      if isinstance(p, dict) and p.get('price', 0) < entry_price]
            if _cands:
                _best = max(_cands, key=lambda p: p['price'])
                if (entry_price - _best['price']) < (entry_price * 0.001):
                    return -1, ["Avoided (EQL Sweep Risk)"]
    else:
        if eql_pools:
            _cands = [p for p in eql_pools
                      if isinstance(p, dict) and p.get('price', 0) < entry_price]
            if _cands:
                _best = max(_cands, key=lambda p: p['price'])
                if (entry_price - _best['price']) > (entry_price * 0.0005):
                    _base = _cw.get('eq_target', 2)
                    _bonus = 1 if (_best.get('tightness', 0) > 0.7
                                   and _best.get('tier', 1) >= 2) else 0
                    score += _base + _bonus
                    details.append(f"Targeting EQL(T{_best.get('tier',1)})")
        if eqh_pools:
            _cands = [p for p in eqh_pools
                      if isinstance(p, dict) and p.get('price', 0) > entry_price]
            if _cands:
                _best = min(_cands, key=lambda p: p['price'])
                if (_best['price'] - entry_price) < (entry_price * 0.001):
                    return -1, ["Avoided (EQH Sweep Risk)"]

    return score, details


def calculate_tp_sl(symbol, df, ob, direction, current_price, point, buffer=None, min_rrr=1.5, fixed_spread=None,
                    precalc_sweep_info=None):
    broker = get_broker()
    spread = fixed_spread if fixed_spread is not None else get_spread(symbol)
    if buffer is None:
        buffer = spread * 2
    if precalc_sweep_info is not None:
        bullish_sweep, bearish_sweep, swing_high, swing_low = precalc_sweep_info
    else:
        bullish_sweep, bearish_sweep, swing_high, swing_low = detect_liquidity_sweep(df)
    tp_multiplier = max(min_rrr, 1.5)
    ob_low = ob.get('low') if isinstance(ob, dict) else None
    ob_high = ob.get('high') if isinstance(ob, dict) else None

    if direction == "Buy":
        if swing_low is not None and ob_low is not None:
            sl_candidate = min(swing_low, ob_low)
        elif swing_low is not None:
            sl_candidate = swing_low
        elif ob_low is not None:
            sl_candidate = ob_low
        else:
            sl_candidate = current_price

        sl_price = sl_candidate - buffer
        if sl_price >= current_price:
            sl_price = current_price - max(abs(current_price - sl_price), point * 10)
        risk = current_price - sl_price
        if risk <= 0:
            risk = max(abs(current_price - sl_price), point * 10)

        if bullish_sweep and swing_high:
            tp_price = swing_high + (swing_high - swing_low) * getattr(config, 'TP_LIQUIDITY_EXTENSION', 0.3)
            min_tp = current_price + risk * tp_multiplier
            tp_price = max(tp_price, min_tp)
        else:
            tp_price = current_price + risk * tp_multiplier
    else:
        if swing_high is not None and ob_high is not None:
            sl_candidate = max(swing_high, ob_high)
        elif swing_high is not None:
            sl_candidate = swing_high
        elif ob_high is not None:
            sl_candidate = ob_high
        else:
            sl_candidate = current_price

        sl_price = sl_candidate + buffer
        if sl_price <= current_price:
            sl_price = current_price + max(abs(current_price - sl_price), point * 10)
        risk = sl_price - current_price
        if risk <= 0:
            risk = max(abs(current_price - sl_price), point * 10)

        if bearish_sweep and swing_low:
            tp_price = swing_low - (swing_high - swing_low) * getattr(config, 'TP_LIQUIDITY_EXTENSION', 0.3)
            min_tp = current_price - risk * tp_multiplier
            tp_price = min(tp_price, min_tp)
        else:
            tp_price = current_price - risk * tp_multiplier
    min_dist = max(spread * 3, point * 10)
    if direction == "Buy":
        if tp_price - current_price < min_dist:
            tp_price = current_price + min_dist
        if current_price - sl_price < min_dist:
            sl_price = current_price - min_dist
    else:
        if current_price - tp_price < min_dist:
            tp_price = current_price - min_dist
        if sl_price - current_price < min_dist:
            sl_price = current_price + min_dist

    info = broker.symbol_info(symbol)
    if info is not None:
        sym_digits = info.digits
        tp_price = round(tp_price, sym_digits)
        sl_price = round(sl_price, sym_digits)

    return tp_price, sl_price


def get_higher_tf_trend(symbol, ict_params, date_to=None):
    trend_counts = {"bullish": 0, "bearish": 0}
    for tf_name, tf_value in HIGHER_TFS:
        try:
            if date_to:
                df = get_data(symbol, tf_value, 100, live=False, date_to=date_to)
                if df is not None and not df.empty:
                    tf_minutes_map = {
                        TIMEFRAME_M1: 1, TIMEFRAME_M5: 5, TIMEFRAME_M15: 15,
                        TIMEFRAME_H1: 60, TIMEFRAME_H4: 240, TIMEFRAME_D1: 1440
                    }
                    tf_min = tf_minutes_map.get(tf_value, 60)
                    while not df.empty:
                        idx = df.index[-1]
                        if isinstance(idx, pd.Timestamp):
                            bar_close_time = idx + pd.Timedelta(minutes=tf_min)
                        elif isinstance(idx, (int, np.integer)):
                            bar_close_time = pd.to_datetime(idx, unit='s') + pd.Timedelta(minutes=tf_min)
                        else:
                            bar_close_time = pd.to_datetime(idx) + pd.Timedelta(minutes=tf_min)
                            
                        if pd.to_datetime(bar_close_time) > pd.to_datetime(date_to):
                            df = df.iloc[:-1]
                        else:
                            break
            else:
                df = get_data(symbol, tf_value, 100, live=True)
                if df is not None and not df.empty:
                    df = df.iloc[:-1]

            if df is None or len(df) < 5:
                continue

            current_close = df.iloc[-1]['close']
            fvgs = detect_fair_value_gaps(df)

            nearest_bullish_fvg = None
            nearest_bearish_fvg = None

            for f in reversed(fvgs):
                if f['time'] >= df.index[-1]:
                    continue
                if f['type'] == 'bullish' and current_close > f['top']:
                    subset = df.loc[f['time']:]
                    if len(subset) > 0 and subset['low'].min() > f['top']:
                        if nearest_bullish_fvg is None:
                            nearest_bullish_fvg = f
                if f['type'] == 'bearish' and current_close < f['bottom']:
                    subset = df.loc[f['time']:]
                    if len(subset) > 0 and subset['high'].max() < f['bottom']:
                        if nearest_bearish_fvg is None:
                            nearest_bearish_fvg = f
                if nearest_bullish_fvg and nearest_bearish_fvg:
                    break

            dist_to_bull_fvg = (current_close - nearest_bullish_fvg['top']) if nearest_bullish_fvg else float('inf')
            dist_to_bear_fvg = (nearest_bearish_fvg['bottom'] - current_close) if nearest_bearish_fvg else float('inf')

            if dist_to_bear_fvg < dist_to_bull_fvg:
                trend_counts["bullish"] += 2
            elif dist_to_bull_fvg < dist_to_bear_fvg:
                trend_counts["bearish"] += 2

            bullish_sweep, bearish_sweep, _, _ = detect_liquidity_sweep(df)

            if tf_name == "D1":
                daily_open = df.iloc[-1]['open']
                amd_weight = 2
                try:
                    m15_df = get_data(symbol, TIMEFRAME_M15, 100, live=(date_to is None), date_to=date_to)
                    if m15_df is not None and len(m15_df) >= 16:
                        avg_m15_range = (m15_df['high'] - m15_df['low']).mean()
                        recent_16 = m15_df.iloc[-16:]
                        consolidation_size = recent_16['high'].max() - recent_16['low'].min()
                        if consolidation_size < avg_m15_range * 6:
                            amd_weight = 4
                except Exception:
                    pass

                if current_close < daily_open and bearish_sweep:
                    trend_counts["bearish"] += amd_weight
                elif current_close > daily_open and bullish_sweep:
                    trend_counts["bullish"] += amd_weight
                elif current_close > daily_open:
                    trend_counts["bullish"] += 1
                elif current_close < daily_open:
                    trend_counts["bearish"] += 1
            else:
                if bullish_sweep:
                    trend_counts["bullish"] += 1
                elif bearish_sweep:
                    trend_counts["bearish"] += 1

        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {tf_name}: {str(e)}")

    if trend_counts["bullish"] > trend_counts["bearish"]:
        return "bullish"
    elif trend_counts["bearish"] > trend_counts["bullish"]:
        return "bearish"
    return "neutral"
