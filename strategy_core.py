"""Single execution kernel shared 100% byte-for-byte by backtester and live.

Both backtester.py and live_scanner.py construct a Context, pass it to
evaluate(), and get back a Signal (or None). Neither engine contains ANY
signal-generation code.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

import indicators
import detectors
import simulation_scoring
import cost_model
from atr_context import atr_last


@dataclass(frozen=True)
class SymbolMeta:
    name: str
    point: float
    digits: int
    trade_stops_level_pts: int
    spread_pts: float

    @classmethod
    def from_broker(cls, symbol: str, broker_info=None):
        pt = cost_model.point_for_symbol(symbol)
        dig = getattr(broker_info, 'digits', int(round(-np.log10(pt)))) if broker_info else int(round(-np.log10(pt)))
        stl = getattr(broker_info, 'trade_stops_level', 0) or 0 if broker_info else 0
        spr = getattr(broker_info, 'spread', 0) or 0 if broker_info else 0
        return cls(name=symbol, point=pt, digits=dig,
                   trade_stops_level_pts=stl, spread_pts=spr)


@dataclass
class Context:
    df: pd.DataFrame
    symbol: SymbolMeta
    current_time: pd.Timestamp
    bar_index: int
    spread_price: float
    atr: float
    htf_bias: str = "neutral"

    # Precalc caches (populated by runner for speed, or lazily computed)
    swing_highs: Optional[List[Dict[str, Any]]] = None
    swing_lows: Optional[List[Dict[str, Any]]] = None
    all_obs: Optional[pd.DataFrame] = None
    all_fvgs: Optional[List[Dict[str, Any]]] = None
    all_sweeps: Optional[List[Dict[str, Any]]] = None
    eqh_pools: Optional[List[Dict[str, Any]]] = None
    eql_pools: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        if self.swing_highs is None or self.swing_lows is None:
            sh, sl = indicators.detect_swing_points(self.df)
            sh_d = [{'price': self.df['high'].iloc[k], 'idx': k, 'time': self.df.index[k]} for k in sh]
            sl_d = [{'price': self.df['low'].iloc[k], 'idx': k, 'time': self.df.index[k]} for k in sl]
            self.swing_highs = self.swing_highs or sh_d
            self.swing_lows = self.swing_lows or sl_d
        if self.all_obs is None:
            self.all_obs = indicators.detect_order_blocks(self.df, atr=self.atr)
        if self.all_fvgs is None:
            self.all_fvgs = indicators.detect_fair_value_gaps(self.df, atr=self.atr)
        if self.all_sweeps is None:
            self.all_sweeps = indicators.detect_all_liquidity_sweeps(self.df)
        if self.eqh_pools is None:
            self.eqh_pools = indicators.detect_liquidity_pools(self.swing_highs, atr=self.atr)
        if self.eql_pools is None:
            self.eql_pools = indicators.detect_liquidity_pools(self.swing_lows, atr=self.atr)


@dataclass(frozen=True)
class Signal:
    symbol: str
    method: str
    direction: str  # "Buy" or "Sell"
    entry_price: float
    sl_price: float
    tp_price: float
    risk_points: float
    reward_points: float
    rrr: float
    confluence_score: int
    confluence_details: List[str]
    bar_index: int
    timestamp: pd.Timestamp
    raw_setup: Dict[str, Any] = field(default_factory=dict)


def evaluate_method(ctx: Context, method: str, params: Dict[str, Any]) -> List[Signal]:
    """Evaluate one method on `ctx` and return candidate Signals."""
    signals = []
    return signals


def evaluate(ctx: Context, methods: List[str], params: Dict[str, Any]) -> List[Signal]:
    """Evaluate all methods on `ctx` and return ranked Signal list."""
    out = []
    for m in methods:
        out.extend(evaluate_method(ctx, m, params))
    out.sort(key=lambda s: s.confluence_score, reverse=True)
    return out
