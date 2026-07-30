"""
Regime-adaptive parameter switching.

Stores optimal parameter sets per market regime and dynamically loads
the right set when the regime changes — the single biggest edge for
closing the gap between a static bot and a human reading context.

Usage:
    from regime_manager import rm
    rm.learn_from_grid_results("opt_results.csv")
    ...
    regime = classify_regime(df)
    params = rm.params_for_regime(regime['label'])
    rm.apply_params(params)
"""
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd

import config
from state import app_state

logger = logging.getLogger(__name__)

# Default params (fallback when no grid data is available per regime)
_DEFAULT_PARAMS: dict[str, Any] = {
    'max_concurrent_trades': 2,
    'daily_loss_limit_pct': 0.0,
    'min_fvg_size_spreads': 2.5,
    'min_confluence_score': 2,
    'use_htf_filter': True,
    'use_ote_filter': True,
    'fvg_displacement_only': True,
    'fvg_discount_premium_only': True,
}


class RegimeManager:
    """Stores and applies best parameters per market regime.

    Populated either from grid-optimisation CSV output or from a
    curated JSON config.  Each regime gets its own parameter set.
    """

    REGIMES = ('trend', 'chop', 'range', 'high-vol', 'weak-trend')

    def __init__(self, grid_results_path: Optional[str] = None):
        self._params: dict[str, dict] = {r: dict(_DEFAULT_PARAMS) for r in self.REGIMES}
        self._current_regime: str = 'trend'
        if grid_results_path and os.path.exists(grid_results_path):
            self.learn_from_grid_results(grid_results_path)

    # ── Public API ──────────────────────────────────────────────────

    def params_for_regime(self, regime_label: str) -> dict:
        """Return the best parameter dict for a regime label.

        Falls back to 'trend' (generally safest) for unknown labels.
        """
        return self._params.get(regime_label, self._params['trend']).copy()

    def apply_params(self, params: dict) -> None:
        """Write a parameter dict into the live StateManager / config.

        Only touches keys that map to a known state attribute or config
        variable (silently skips the rest).
        """
        mapping = {
            'max_concurrent_trades': ('max_concurrent_trades', int),
            'max_concurrent': ('max_concurrent_trades', int),
            'concurr': ('max_concurrent_trades', int),
            'daily_loss_limit_pct': ('daily_loss_limit_pct', float),
            'daily_loss_limit': ('daily_loss_limit_pct', float),
            'dl_val': ('daily_loss_limit_pct', float),
            'min_fvg_size_spreads': ('min_fvg_size_spreads', float),
            'min_fvg_size': ('min_fvg_size_spreads', float),
            'min_confluence_score': ('min_confluence_score', int),
            'min_conf': ('min_confluence_score', int),
            'use_htf_filter': ('htf_poi_enabled', bool),
            'use_ote_filter': None,  # handled separately in backtester args
            'fvg_displacement_only': None,
            'fvg_discount_premium_only': None,
            'fvg_recent_sweep_only': None,
            'RRR': None,
            'rrr': None,
            'min_rrr': None,
        }
        for csv_key, val_info in mapping.items():
            if val_info is None:
                continue
            state_attr, cast = val_info
            if csv_key in params:
                val = params[csv_key]
                if pd.isna(val) or val is None:
                    continue
                try:
                    setattr(app_state, state_attr, cast(val))
                except Exception:
                    pass

        # Also apply via config module-level for backward compat
        config_vars = {
            'max_concurrent_trades': 'MAX_CONCURRENT_TRADES',
            'daily_loss_limit_pct': 'DAILY_LOSS_LIMIT_PCT',
            'min_fvg_size_spreads': 'MIN_FVG_SIZE_SPREADS',
            'min_confluence_score': 'MIN_CONFLUENCE_SCORE',
            'htf_poi_enabled': 'HTF_POI_ENABLED',
        }
        for state_key, cfg_name in config_vars.items():
            val = getattr(app_state, state_key, None)
            if val is not None:
                setattr(config, cfg_name, val)

        logger.info("regime_manager: applied params → %s", {k: v for k, v in params.items() if k in mapping})

    def learn_from_grid_results(self, csv_path: str) -> int:
        """Parse grid-optimisation CSV and compute the best parameter set per regime.

        Expects columns: (regime_label, roi, dd, win_rate, profit_factor, ...)
        plus all parameter columns.

        Returns the number of combos loaded (0 on failure).
        """
        if not os.path.exists(csv_path):
            logger.warning("regime_manager: grid results not found at %s", csv_path)
            return 0
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error("regime_manager: failed to read CSV: %s", e)
            return 0

        # Detect regime column (may be 'regime', 'regime_label', 'market_regime')
        regime_col = None
        for candidate in ('regime_label', 'regime', 'market_regime', 'label'):
            if candidate in df.columns:
                regime_col = candidate
                break
        if regime_col is None:
            logger.warning("regime_manager: no regime column found, using all data")
            df['_regime'] = 'trend'
            regime_col = '_regime'

        # Score combos by Calmar ratio, then load best per regime
        if 'calmar' not in df.columns:
            if 'roi_pct' in df.columns:
                df['calmar'] = df['roi_pct'] / df['max_drawdown_pct'].clip(lower=0.1)
            elif 'roi' in df.columns:
                df['calmar'] = df['roi'] / df['dd'].clip(lower=0.1)

        loaded = 0
        for regime in self.REGIMES:
            subset = df[df[regime_col].str.lower().str.strip() == regime]
            if subset.empty:
                continue
            best = subset.loc[subset['calmar'].idxmax()] if 'calmar' in subset.columns else subset.iloc[0]
            self._params[regime] = best.to_dict()
            loaded += 1

        logger.info("regime_manager: loaded %d regime parameter sets from %s", loaded, csv_path)
        return loaded

    def save_json(self, path: str = "regime_params.json") -> None:
        """Export learned parameters to a human-readable JSON file."""
        try:
            # Convert non-serializable values (e.g. numpy floats)
            serializable = {}
            for regime, params in self._params.items():
                serializable[regime] = {
                    k: float(v) if isinstance(v, (np.floating,)) else
                       int(v) if isinstance(v, (np.integer,)) else v
                    for k, v in params.items()
                }
            with open(path, 'w') as f:
                json.dump(serializable, f, indent=2)
            logger.info("regime_manager: saved to %s", path)
        except Exception as e:
            logger.error("regime_manager: failed to save JSON: %s", e)


# ── Standalone regime classifier ────────────────────────────────────────

def classify_regime(df: pd.DataFrame, lookback: int = 50) -> dict:
    """Classify a DataFrame slice into a market regime label.

    Uses volatility ratio, range index, trend strength, and directional
    consistency to determine the regime.  Matches the schema returned by
    *ict_pro.regime_of* so it can be used as a drop-in replacement or for
    the RegimeManager flow.

    Returns
    -------
    dict with keys: label, tradeable, size_mult, conf_bump, and diagnostics.
    Falls back safely when *df* is too short or missing.
    """
    safe = {'label': 'trend', 'tradeable': True, 'size_mult': 1.0, 'conf_bump': 0}
    try:
        if df is None or len(df) < 20:
            return safe
    
        df = df.tail(lookback)
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        closes = df['close'].values.astype(float)
    
        atr = float(np.mean(highs - lows))
        atr_sma = float(pd.Series(highs - lows).rolling(20).mean().iloc[-1]) if len(df) >= 20 else atr
        atr_sma = float(np.nan_to_num(atr_sma, nan=1e-10, posinf=1e-10, neginf=1e-10))
        atr_sma = max(atr_sma, 1e-10)
        vol_ratio = atr / atr_sma
        vol_ratio = float(np.nan_to_num(vol_ratio, nan=1.0, posinf=1.0, neginf=1.0))
    
        price_range = float(max(highs) - min(lows))
        avg_candle = float(np.mean(highs - lows)) + 1e-10
        range_idx = price_range / avg_candle
    
        sma50 = float(pd.Series(closes).rolling(min(50, len(closes))).mean().iloc[-1])
        trend_str = abs(closes[-1] - sma50) / max(atr, 1e-10)
    
        deltas = np.diff(closes)
        up = float(np.sum(deltas > 0))
        down = float(np.sum(deltas < 0))
        dir_ratio = max(up, down) / max(len(deltas), 1.0)
    
        if vol_ratio > 1.5:
            label = 'high-vol'
            tradeable = vol_ratio <= 2.5
            size_mult = 0.5 if vol_ratio > 2.0 else 0.75
            conf_bump = 1 if vol_ratio > 2.0 else 0
        elif range_idx < 4.0 and trend_str < 0.8:
            label = 'chop'
            tradeable = False
            size_mult = 0.0
            conf_bump = 2
        elif trend_str > 1.5 and dir_ratio > 0.65:
            label = 'trend'
            tradeable = True
            size_mult = 1.0
            conf_bump = 0
        elif trend_str > 0.8:
            label = 'weak-trend'
            tradeable = True
            size_mult = 0.8
            conf_bump = 1
        else:
            label = 'range'
            tradeable = True
            size_mult = 0.9
            conf_bump = 0
    
        return {
            'label': label,
            'tradeable': tradeable,
            'size_mult': size_mult,
            'conf_bump': conf_bump,
            'volatility_ratio': round(vol_ratio, 2),
            'range_index': round(range_idx, 2),
            'trend_strength': round(trend_str, 2),
        }
    except Exception:
        return safe


# Singleton
rm = RegimeManager()
