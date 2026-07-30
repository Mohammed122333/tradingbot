"""
Thread-safe state management for the entire trading application.

Replaces global mutable dicts in config.py with a single StateManager
that enforces locking and provides traceable state transitions.

Usage:
    from state import app_state
    app_state.max_concurrent_trades = 3
    app_state.daily_loss_limit_pct = 5.0
    app_state.mutate_executed_ob(key, True)
    with app_state.dict_lock('pending_limit_orders'):
        app_state.pending_limit_orders['setup'] = {...}
"""
import threading
import logging
import os
import json
from typing import Any, Callable

logger = logging.getLogger(__name__)


class StateManager:
    """Central application state with per-attribute threading locks.

    Every mutable config value lives here as a descriptor-backed property.
    Reads are lock-free; writes acquire an attribute-specific lock so that
    concurrent threads (UI, live scanner, backtest worker) cannot corrupt
    each other's state.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._attr_locks: dict[str, threading.Lock] = {}

        # ── Risk / trade management ────────────────────────────────────
        self._daily_loss_limit_pct = 0.0
        self._max_concurrent_trades = 2
        self._min_fvg_size_spreads = 2.5
        self._min_confluence_score = 2
        self._pending_limit_expiry_hours = 2.0
        self._estimated_spread_multiplier = 1.0
        self._slippage_max_points = 5
        self._commission_per_lot = 7.0
        self._use_equity_drawdown = True
        self._max_total_drawdown_pct = 10.0
        self._max_consec_losses = 6

        # ── Pro feature gates ──────────────────────────────────────────
        self._dol_tp_enabled = False
        self._killzone_enabled = False
        self._htf_poi_enabled = False
        self._confluence_require_mandatory = False
        self._regime_filter_enabled = False
        self._regime_adaptive_params_enabled = False
        self._multi_tf_confluence_enabled = False
        self._multi_tf_gate_enabled = False
        self._ml_sizing_enabled = False
        self._ml_rank_enabled = False

        # ── Feature toggles ──────────────────────────────────────────────
        self._news_embargo_enabled = True
        self._news_filter_enabled = False
        self._news_filter_buffer_mins = 30
        self._use_smt_divergence = False
        self._smt_correlated_pair = "DXY"
        self._use_volume_profile_confluence = False
        self._anti_gap_sl_enabled = True
        self._anti_gap_atr_multiplier = 2.0



        # ── Live trading state ──────────────────────────────────────────
        self._live_trading_running = False
        self._live_trading_thread = None
        self._current_activity = "Idle"

        # ── Slippage recovery state ──────────────────────────────────────
        self._slippage_recovery_tracker = {
            'enabled': False,
            'excess_loss_cumulative': 0.0,
            'recovery_trades_remaining': 0,
            'lot_reduction_active': False,
            'sl_buffer_pips': 0.0,
            'elevated_rrr': 0.0,
            'recovery_spread_trades': 5,
            'sl_buffer_multiplier': 1.5,
            'avg_slippage_observed': 0.0,
            'slippage_count': 0,
        }

        # ── In-memory databases (thread-safe dicts) ────────────────────
        self._dict_locks: dict[str, threading.RLock] = {}
        self._pending_limit_orders: dict = {}
        self._priority_scan_setups: dict = {}
        self._executed_ob: dict = self._load_trade_memory()
        self._last_setup_signal: dict = {}

    # ── Internal helpers ──────────────────────────────────────────────

    def _lock_for(self, attr: str) -> threading.Lock:
        """Return the per-attribute lock, creating if needed."""
        with self._lock:
            if attr not in self._attr_locks:
                self._attr_locks[attr] = threading.Lock()
            return self._attr_locks[attr]

    # ── Risk / trade management properties ────────────────────────────

    @property
    def daily_loss_limit_pct(self) -> float:
        return self._daily_loss_limit_pct

    @daily_loss_limit_pct.setter
    def daily_loss_limit_pct(self, value: float) -> None:
        with self._lock_for('_daily_loss_limit_pct'):
            old = self._daily_loss_limit_pct
            self._daily_loss_limit_pct = float(value)
            if old != self._daily_loss_limit_pct:
                logger.debug("state: daily_loss_limit_pct %.2f → %.2f", old, self._daily_loss_limit_pct)

    @property
    def max_concurrent_trades(self) -> int:
        return self._max_concurrent_trades

    @max_concurrent_trades.setter
    def max_concurrent_trades(self, value: int) -> None:
        with self._lock_for('_max_concurrent_trades'):
            old = self._max_concurrent_trades
            self._max_concurrent_trades = int(value)
            if old != self._max_concurrent_trades:
                logger.debug("state: max_concurrent_trades %d → %d", old, self._max_concurrent_trades)

    @property
    def min_fvg_size_spreads(self) -> float:
        return self._min_fvg_size_spreads

    @min_fvg_size_spreads.setter
    def min_fvg_size_spreads(self, value: float) -> None:
        with self._lock_for('_min_fvg_size_spreads'):
            old = self._min_fvg_size_spreads
            self._min_fvg_size_spreads = float(value)
            if old != self._min_fvg_size_spreads:
                logger.debug("state: min_fvg_size_spreads %.2f → %.2f", old, self._min_fvg_size_spreads)

    @property
    def min_confluence_score(self) -> int:
        return self._min_confluence_score

    @min_confluence_score.setter
    def min_confluence_score(self, value: int) -> None:
        with self._lock_for('_min_confluence_score'):
            old = self._min_confluence_score
            self._min_confluence_score = int(value)
            if old != self._min_confluence_score:
                logger.debug("state: min_confluence_score %d → %d", old, self._min_confluence_score)

    @property
    def pending_limit_expiry_hours(self) -> float:
        return self._pending_limit_expiry_hours

    @pending_limit_expiry_hours.setter
    def pending_limit_expiry_hours(self, value: float) -> None:
        with self._lock_for('_pending_limit_expiry_hours'):
            old = self._pending_limit_expiry_hours
            self._pending_limit_expiry_hours = float(value)
            if old != self._pending_limit_expiry_hours:
                logger.debug("state: pending_limit_expiry_hours %.2f → %.2f", old, self._pending_limit_expiry_hours)

    @property
    def estimated_spread_multiplier(self) -> float:
        return self._estimated_spread_multiplier

    @estimated_spread_multiplier.setter
    def estimated_spread_multiplier(self, value: float) -> None:
        with self._lock_for('_estimated_spread_multiplier'):
            self._estimated_spread_multiplier = float(value)

    @property
    def slippage_max_points(self) -> int:
        return self._slippage_max_points

    @slippage_max_points.setter
    def slippage_max_points(self, value: int) -> None:
        self._slippage_max_points = int(value)

    @property
    def commission_per_lot(self) -> float:
        return self._commission_per_lot

    @commission_per_lot.setter
    def commission_per_lot(self, value: float) -> None:
        self._commission_per_lot = float(value)

    # ── Pro feature gate properties ────────────────────────────────────

    @property
    def dol_tp_enabled(self) -> bool:
        return self._dol_tp_enabled

    @dol_tp_enabled.setter
    def dol_tp_enabled(self, value: bool) -> None:
        with self._lock_for('_dol_tp_enabled'):
            self._dol_tp_enabled = bool(value)

    @property
    def killzone_enabled(self) -> bool:
        return self._killzone_enabled

    @killzone_enabled.setter
    def killzone_enabled(self, value: bool) -> None:
        with self._lock_for('_killzone_enabled'):
            self._killzone_enabled = bool(value)

    @property
    def htf_poi_enabled(self) -> bool:
        return self._htf_poi_enabled

    @htf_poi_enabled.setter
    def htf_poi_enabled(self, value: bool) -> None:
        with self._lock_for('_htf_poi_enabled'):
            self._htf_poi_enabled = bool(value)

    @property
    def confluence_require_mandatory(self) -> bool:
        return self._confluence_require_mandatory

    @confluence_require_mandatory.setter
    def confluence_require_mandatory(self, value: bool) -> None:
        with self._lock_for('_confluence_require_mandatory'):
            self._confluence_require_mandatory = bool(value)

    @property
    def regime_filter_enabled(self) -> bool:
        return self._regime_filter_enabled

    @regime_filter_enabled.setter
    def regime_filter_enabled(self, value: bool) -> None:
        with self._lock_for('_regime_filter_enabled'):
            self._regime_filter_enabled = bool(value)

    @property
    def regime_adaptive_params_enabled(self) -> bool:
        return self._regime_adaptive_params_enabled

    @regime_adaptive_params_enabled.setter
    def regime_adaptive_params_enabled(self, value: bool) -> None:
        with self._lock_for('_regime_adaptive_params_enabled'):
            self._regime_adaptive_params_enabled = bool(value)

    @property
    def multi_tf_confluence_enabled(self) -> bool:
        return self._multi_tf_confluence_enabled

    @multi_tf_confluence_enabled.setter
    def multi_tf_confluence_enabled(self, value: bool) -> None:
        with self._lock_for('_multi_tf_confluence_enabled'):
            self._multi_tf_confluence_enabled = bool(value)

    @property
    def multi_tf_gate_enabled(self) -> bool:
        return self._multi_tf_gate_enabled

    @multi_tf_gate_enabled.setter
    def multi_tf_gate_enabled(self, value: bool) -> None:
        with self._lock_for('_multi_tf_gate_enabled'):
            self._multi_tf_gate_enabled = bool(value)

    @property
    def ml_sizing_enabled(self) -> bool:
        return self._ml_sizing_enabled

    @ml_sizing_enabled.setter
    def ml_sizing_enabled(self, value: bool) -> None:
        with self._lock_for('_ml_sizing_enabled'):
            self._ml_sizing_enabled = bool(value)

    @property
    def ml_rank_enabled(self) -> bool:
        return self._ml_rank_enabled

    @ml_rank_enabled.setter
    def ml_rank_enabled(self, value: bool) -> None:
        with self._lock_for('_ml_rank_enabled'):
            self._ml_rank_enabled = bool(value)

    # ── Feature toggle properties ──────────────────────────────────────

    @property
    def news_embargo_enabled(self) -> bool:
        return self._news_embargo_enabled

    @news_embargo_enabled.setter
    def news_embargo_enabled(self, value: bool) -> None:
        self._news_embargo_enabled = bool(value)

    @property
    def news_filter_enabled(self) -> bool:
        return self._news_filter_enabled

    @news_filter_enabled.setter
    def news_filter_enabled(self, value: bool) -> None:
        self._news_filter_enabled = bool(value)

    @property
    def news_filter_buffer_mins(self) -> int:
        return self._news_filter_buffer_mins

    @news_filter_buffer_mins.setter
    def news_filter_buffer_mins(self, value: int) -> None:
        self._news_filter_buffer_mins = int(value)

    @property
    def use_smt_divergence(self) -> bool:
        return self._use_smt_divergence

    @use_smt_divergence.setter
    def use_smt_divergence(self, value: bool) -> None:
        self._use_smt_divergence = bool(value)

    @property
    def smt_correlated_pair(self) -> str:
        return self._smt_correlated_pair

    @smt_correlated_pair.setter
    def smt_correlated_pair(self, value: str) -> None:
        self._smt_correlated_pair = str(value)

    @property
    def use_volume_profile_confluence(self) -> bool:
        return self._use_volume_profile_confluence

    @use_volume_profile_confluence.setter
    def use_volume_profile_confluence(self, value: bool) -> None:
        self._use_volume_profile_confluence = bool(value)

    @property
    def anti_gap_sl_enabled(self) -> bool:
        return self._anti_gap_sl_enabled

    @anti_gap_sl_enabled.setter
    def anti_gap_sl_enabled(self, value: bool) -> None:
        self._anti_gap_sl_enabled = bool(value)

    @property
    def anti_gap_atr_multiplier(self) -> float:
        return self._anti_gap_atr_multiplier

    @anti_gap_atr_multiplier.setter
    def anti_gap_atr_multiplier(self, value: float) -> None:
        self._anti_gap_atr_multiplier = float(value)



    # ── Live trading state properties ──────────────────────────────────

    @property
    def live_trading_running(self) -> bool:
        return self._live_trading_running

    @live_trading_running.setter
    def live_trading_running(self, value: bool) -> None:
        with self._lock_for('_live_trading_running'):
            self._live_trading_running = bool(value)

    @property
    def current_activity(self) -> str:
        return self._current_activity

    @current_activity.setter
    def current_activity(self, value: str) -> None:
        with self._lock_for('_current_activity'):
            self._current_activity = str(value)

    # ── Slippage recovery ──────────────────────────────────────────────

    @property
    def slippage_recovery_tracker(self) -> dict:
        return self._slippage_recovery_tracker

    @slippage_recovery_tracker.setter
    def slippage_recovery_tracker(self, value: dict) -> None:
        with self._lock_for('_slippage_recovery_tracker'):
            self._slippage_recovery_tracker = dict(value)

    # ── Bulk load / save for grid optimisation ─────────────────────────

    # ── Trade memory (disk-persistent) ────────────────────────────

    def _load_trade_memory(self) -> dict:
        path = "trade_memory.json"
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_trade_memory(self) -> None:
        with self._lock_for('_executed_ob'):
            try:
                with open("trade_memory.json", 'w') as f:
                    json.dump(self._executed_ob, f, indent=2)
            except Exception as exc:
                logger.error("state: failed to save trade_memory.json — %s", exc)

    # ── Thread-safe dict operations ──────────────────────────────

    def _dict_lock(self, name: str) -> threading.RLock:
        with self._lock:
            if name not in self._dict_locks:
                self._dict_locks[name] = threading.RLock()
            return self._dict_locks[name]

    def dict_access(self, name: str) -> threading.RLock:
        """Context manager for thread-safe dict access. Usage:
           with app_state.dict_access('pending_limit_orders'):
               app_state.pending_limit_orders['key'] = value
        """
        return self._dict_lock(name)

    def mutate_executed_ob(self, key: str, value: bool = True) -> None:
        with self._dict_lock('_executed_ob'):
            self._executed_ob[key] = value
            self.save_trade_memory()

    def clear_executed_ob(self) -> None:
        with self._dict_lock('_executed_ob'):
            self._executed_ob.clear()
            self.save_trade_memory()

    @property
    def estimated_spread_multiplier(self) -> float:
        return self._estimated_spread_multiplier

    @estimated_spread_multiplier.setter
    def estimated_spread_multiplier(self, value: float) -> None:
        with self._lock_for('_estimated_spread_multiplier'):
            self._estimated_spread_multiplier = float(value)

    @property
    def use_equity_drawdown(self) -> bool:
        return self._use_equity_drawdown

    @use_equity_drawdown.setter
    def use_equity_drawdown(self, value: bool) -> None:
        with self._lock_for('_use_equity_drawdown'):
            self._use_equity_drawdown = bool(value)

    @property
    def max_total_drawdown_pct(self) -> float:
        return self._max_total_drawdown_pct

    @max_total_drawdown_pct.setter
    def max_total_drawdown_pct(self, value: float) -> None:
        with self._lock_for('_max_total_drawdown_pct'):
            self._max_total_drawdown_pct = float(value)

    @property
    def max_consec_losses(self) -> int:
        return self._max_consec_losses

    @max_consec_losses.setter
    def max_consec_losses(self, value: int) -> None:
        with self._lock_for('_max_consec_losses'):
            self._max_consec_losses = int(value)

    # ── Pending / priority order dict accessors ──────────────────

    def set_pending_order(self, key, value):
        with self._dict_lock('_pending_limit_orders'):
            self._pending_limit_orders[key] = value

    def pop_pending_order(self, key, default=None):
        with self._dict_lock('_pending_limit_orders'):
            return self._pending_limit_orders.pop(key, default)

    def snapshot_pending_orders(self):
        with self._dict_lock('_pending_limit_orders'):
            return dict(self._pending_limit_orders)

    def set_priority_setup(self, key, value):
        with self._dict_lock('_priority_scan_setups'):
            self._priority_scan_setups[key] = value

    def pop_priority_setup(self, key, default=None):
        with self._dict_lock('_priority_scan_setups'):
            return self._priority_scan_setups.pop(key, default)

    @property
    def pending_limit_orders(self) -> dict:
        return self._pending_limit_orders

    @property
    def priority_scan_setups(self) -> dict:
        return self._priority_scan_setups

    @property
    def executed_ob(self) -> dict:
        return self._executed_ob

    @executed_ob.setter
    def executed_ob(self, value: dict) -> None:
        with self._dict_lock('_executed_ob'):
            self._executed_ob = dict(value)

    # ── Live trading thread ──────────────────────────────────────

    @property
    def live_trading_thread(self):
        return self._live_trading_thread

    @live_trading_thread.setter
    def live_trading_thread(self, value):
        self._live_trading_thread = value

    # ── Bulk load / save for grid optimisation ───────────────────

    def load_from_dict(self, d: dict) -> None:
        """Apply a flat dict of {attribute_name: value} updates safely."""
        for key, value in d.items():
            if hasattr(self, key) and not key.startswith('_'):
                setattr(self, key, value)

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of all state values (for save/restore patterns)."""
        with self._lock:
            return {
                'daily_loss_limit_pct': self._daily_loss_limit_pct,
                'max_concurrent_trades': self._max_concurrent_trades,
                'min_fvg_size_spreads': self._min_fvg_size_spreads,
                'min_confluence_score': self._min_confluence_score,
                'news_filter_enabled': self._news_filter_enabled,
                'news_filter_buffer_mins': self._news_filter_buffer_mins,
                'dol_tp_enabled': self._dol_tp_enabled,
                'killzone_enabled': self._killzone_enabled,
                'htf_poi_enabled': self._htf_poi_enabled,
                'confluence_require_mandatory': self._confluence_require_mandatory,
                'regime_filter_enabled': self._regime_filter_enabled,
                'regime_adaptive_params_enabled': self._regime_adaptive_params_enabled,
                'multi_tf_confluence_enabled': self._multi_tf_confluence_enabled,
                'multi_tf_gate_enabled': self._multi_tf_gate_enabled,
                'ml_sizing_enabled': self._ml_sizing_enabled,
                'ml_rank_enabled': self._ml_rank_enabled,
            }


    # ── Snapshot / restore for subprocess isolation ────────────

    def for_subprocess(self) -> dict:
        """Return a frozen snapshot for passing to multiprocessing workers.
        Each worker creates a fresh StateManager from this dict."""
        return self.snapshot()


# Singleton — import and use directly
app_state = StateManager()
