"""
Multiprocessing worker for grid optimization.
Each worker process initialises MT5 independently, receives cached DataFrames
via Pool initializer, and patches data‑access functions so every
combined_backtest call reads from memory (zero MT5 bar fetches).

Performance optimizations:
  - mt5.initialize() → no-op after first init (eliminates IPC per combo)
  - mt5.symbol_info() → cached (eliminates IPC per combo per TF)
  - mt5.symbol_info_tick() → cached (eliminates IPC per spread lookup)
  - get_symbol_profile_overrides() → reads JSON once, cached in memory
  - get_contract_size() / get_spread() → use cached symbol_info
  - Indicator results memoized by DataFrame length
  - Data access functions serve pre-loaded DataFrames with smart slicing
"""
import logging
import traceback

logger = logging.getLogger(__name__)

# ── Per-process global state, set once by init_worker ────────────────────────
_worker_state = {}


def init_worker(cached_data_source, fixed_params):
    """Called exactly once per Pool worker at process start.

    Parameters
    ----------
    cached_data_source : str or dict
        Path to pickled cached data, or the dict directly.
    fixed_params : dict
        All backtest parameters that do **not** change across grid combos.
    """
    global _worker_state

    import logging, os
    # Don't disable logging outright — that's how thousands of failing combos
    # looked like "no profitable configurations found".
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(f"worker_{os.getpid()}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    root.addHandler(fh)
    root.setLevel(logging.WARNING)

    # Lower process priority (niceness) so GUI/OS remain smooth and responsive
    import os
    try:
        if hasattr(os, 'nice'):
            os.nice(10)
    except Exception:
        pass
    
    # Load cached data from disk to avoid multiprocessing IPC Pickling limits
    if isinstance(cached_data_source, str) and os.path.exists(cached_data_source):
        import pickle
        with open(cached_data_source, 'rb') as f:
            cached_data = pickle.load(f)
    else:
        cached_data = cached_data_source
        
    _worker_state['cached_data'] = cached_data
    _worker_state['fixed_params'] = fixed_params

    # 1. Connect this process to the running MT5 terminal ──────────────────
    # We DO NOT call mt5.initialize() to avoid STATUS_BREAKPOINT crashes caused
    # by multiple python processes concurrently invoking MT5 C++ extensions.

    # 2. Patch get_data / get_data_by_date to serve cached DataFrames ──────
    import MetaTrader5 as mt5
    import pandas as pd
    import numpy as np
    import utils
    import simulation
    import backtester as bt_mod

    cached = cached_data

    import config
    config.NEWS_FILTER_ENABLED = fixed_params.get('news_filter_enabled', False)
    config.NEWS_FILTER_BUFFER_MINS = fixed_params.get('news_filter_buffer', 30)
    if 'offline_backtesting' in fixed_params:
        config.OFFLINE_BACKTESTING = fixed_params['offline_backtesting']

    # ── 2.1 SPEED: Cache mt5.symbol_info results (avoid IPC per combo) ────
    _si_cache = {}
    
    # Pre-cache the symbol we're optimizing using the injected mock_si
    sym = fixed_params.get('symbol', '')
    if sym and 'mock_si' in fixed_params:
        class FakeSI:
            def __init__(self, d):
                self.spread = d.get('spread', 0)
                self.point = d.get('point', 0.00001)
                self.trade_stops_level = d.get('trade_stops_level', 0)
                self.trade_calc_mode = d.get('trade_calc_mode', 0)
                self.currency_margin = d.get('currency_margin', 'USD')
                self.trade_contract_size = d.get('trade_contract_size', 100000)
                self.trade_tick_value = d.get('trade_tick_value', 0)
                self.trade_tick_size = d.get('trade_tick_size', self.point)
                self.volume_step = d.get('volume_step', 0.01)
                self.volume_min = d.get('volume_min', 0.01)
                self.volume_max = d.get('volume_max', 1000.0)
        _si_cache[sym] = FakeSI(fixed_params['mock_si'])
    else:
        # Fallback if no mock_si was provided (which shouldn't happen)
        class FakeSI:
            def __init__(self):
                self.spread = 10
                self.point = 0.00001
                self.trade_stops_level = 0
                self.trade_calc_mode = 0
                self.currency_margin = 'USD'
                self.trade_contract_size = 100000
                self.trade_tick_value = 0
                self.trade_tick_size = 0.00001
                self.volume_step = 0.01
                self.volume_min = 0.01
                self.volume_max = 1000.0
        _si_cache[sym] = FakeSI()

    def _cached_symbol_info(symbol):
        return _si_cache.get(symbol, None)

    def _cached_symbol_info_tick(symbol):
        """Return a fake tick from cached symbol_info (avoids live IPC)."""
        si = _cached_symbol_info(symbol)
        if si:
            # Return a minimal object with ask/bid spread estimation
            class FakeTick:
                def __init__(self, spread_val):
                    self.ask = spread_val
                    self.bid = 0.0
            return FakeTick(si.spread * si.point)
        return None

    # Patch mt5 module-wide (all importers share the same module object)
    mt5.symbol_info = _cached_symbol_info
    mt5.symbol_info_tick = _cached_symbol_info_tick

    # ── 2.2 SPEED: Make mt5.initialize() a no-op (already connected) ──────
    mt5.initialize = lambda *a, **kw: True

    # ── 2.3 Pre-merge cached DataFrames ONCE (eliminates pd.concat IPC per combo)
    _premerged_dfs = {}
    if isinstance(cached, dict):
        for k, df in cached.items():
            if len(k) >= 2 and df is not None and getattr(df, 'empty', False) is False:
                sym_tf_key = (k[0], k[1])
                if sym_tf_key not in _premerged_dfs:
                    _premerged_dfs[sym_tf_key] = []
                _premerged_dfs[sym_tf_key].append(df)

        for key, frames in list(_premerged_dfs.items()):
            if frames:
                comb = pd.concat(frames).sort_index()
                _premerged_dfs[key] = comb[~comb.index.duplicated(keep='last')]
            else:
                _premerged_dfs[key] = None

    def patched_get_data(sym, timeframe, count, live=False, date_to=None):
        if date_to is None:
            return None
        combined = _premerged_dfs.get((sym, timeframe))
        if combined is None or combined.empty:
            return None
        dt = pd.Timestamp(date_to).replace(tzinfo=None)
        idx = combined.index.searchsorted(dt, side='right')
        if idx > 0:
            return combined.iloc[max(0, idx - count):idx]
        return None

    def patched_get_data_by_date(sym, timeframe, d_from, d_to):
        combined = _premerged_dfs.get((sym, timeframe))
        if combined is None or combined.empty:
            return None
        dt_from = pd.Timestamp(d_from).replace(tzinfo=None)
        dt_to = pd.Timestamp(d_to).replace(tzinfo=None)
        idx_start = combined.index.searchsorted(dt_from, side='left')
        idx_end = combined.index.searchsorted(dt_to, side='right')
        if idx_start < idx_end:
            return combined.iloc[idx_start:idx_end]
        return None

    utils.get_data = patched_get_data
    utils.get_data_by_date = patched_get_data_by_date
    simulation.get_data = patched_get_data
    simulation.get_data_by_date = patched_get_data_by_date
    bt_mod.get_data = patched_get_data
    bt_mod.get_data_by_date = patched_get_data_by_date
    
    import multi_tf

    multi_tf.get_data = patched_get_data
    multi_tf.get_data_by_date = patched_get_data_by_date
    
    import simulation_scoring
    simulation_scoring.get_data = patched_get_data

    # ── 2.4 SPEED: Cache symbol_profiles.json in memory ──────────────────
    #    get_symbol_profile_overrides reads this JSON from DISK on every call.
    #    For 1000 combos × 100 trades = 300,000+ file reads eliminated.
    import json, os
    _profiles_data = {}
    profile_path = os.path.join(os.path.dirname(os.path.abspath(utils.__file__)), "symbol_profiles.json")
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r') as f:
                _profiles_data = json.load(f)
        except Exception:
            pass

    # Pre-resolve the profile for our optimization symbol
    _profile_cache = {}

    def _resolve_profile(symbol):
        if symbol in _profile_cache:
            return _profile_cache[symbol]
        profile = _profiles_data.get(symbol)
        if not profile:
            symbol_upper = symbol.upper()
            for k, v in _profiles_data.items():
                k_upper = k.upper()
                if k_upper in symbol_upper or symbol_upper in k_upper:
                    profile = v
                    break
                if k_upper == "GOLD" and "XAU" in symbol_upper:
                    profile = v
                    break
                if k_upper == "BTC" and "BTC" in symbol_upper:
                    profile = v
                    break
        _profile_cache[symbol] = profile
        return profile

    def _cached_get_overrides(symbol, default_params):
        import config as _cfg
        profile = _resolve_profile(symbol)
        if not profile:
            res = default_params.copy()
            if "methods" not in res:
                res["methods"] = {}
            return res
        res = default_params.copy()
        if "methods" not in res:
            res["methods"] = {}
        for field, val in profile.items():
            if field == "fvg_sl_mode" and isinstance(val, str):
                val_upper = val.upper()
                if "NORMAL" in val_upper:
                    val = _cfg.FVG_SL_NORMAL
                elif "SWEEP" in val_upper:
                    val = _cfg.FVG_SL_SWEEP
                elif "CANDLE" in val_upper or "EXTREME" in val_upper:
                    val = _cfg.FVG_SL_CANDLE
                elif "BOS" in val_upper:
                    val = _cfg.FVG_SL_BOS
                elif "MSS" in val_upper:
                    val = _cfg.FVG_SL_MSS
            if field == "trail_type" and isinstance(val, str) and (val.upper() == "NONE" or val == "None (Disabled)"):
                val = None
            if field == "methods" and isinstance(val, dict):
                normalized_methods = {}
                for method_name, method_overrides in val.items():
                    if isinstance(method_overrides, dict):
                        normalized_override = {}
                        for m_field, m_val in method_overrides.items():
                            if m_field == "fvg_sl_mode" and isinstance(m_val, str):
                                m_val_upper = m_val.upper()
                                if "NORMAL" in m_val_upper:
                                    m_val = _cfg.FVG_SL_NORMAL
                                elif "SWEEP" in m_val_upper:
                                    m_val = _cfg.FVG_SL_SWEEP
                                elif "CANDLE" in m_val_upper or "EXTREME" in m_val_upper:
                                    m_val = _cfg.FVG_SL_CANDLE
                                elif "BOS" in m_val_upper:
                                    m_val = _cfg.FVG_SL_BOS
                                elif "MSS" in m_val_upper:
                                    m_val = _cfg.FVG_SL_MSS
                            if m_field == "trail_type" and isinstance(m_val, str) and (m_val.upper() == "NONE" or m_val == "None (Disabled)"):
                                m_val = None
                            normalized_override[m_field] = m_val
                        normalized_methods[method_name] = normalized_override
                val = normalized_methods
            res[field] = val
        return res

    utils.get_symbol_profile_overrides = _cached_get_overrides
    bt_mod.get_symbol_profile_overrides = _cached_get_overrides

    # ── 2.5 Patch indicators with memoized caches ─────────────────────────
    import indicators
    orig_ob = indicators.detect_order_blocks
    orig_fvg = indicators.detect_fair_value_gaps
    orig_struct = indicators.calculate_market_structure_chronologically
    orig_swings = indicators.detect_swing_points
    orig_sweeps = indicators.detect_all_liquidity_sweeps

    indicator_cache = {}
    def patched_ob(df, *args, **kwargs):
        k = ('ob', len(df), str(args), str(kwargs))
        if k not in indicator_cache: indicator_cache[k] = orig_ob(df, *args, **kwargs)
        return indicator_cache[k].copy()
        
    def patched_fvg(df, *args, **kwargs):
        k = ('fvg', len(df), str(args), str(kwargs))
        if k not in indicator_cache: indicator_cache[k] = orig_fvg(df, *args, **kwargs)
        return indicator_cache[k]
        
    def patched_struct(df, *args, **kwargs):
        k = ('struct', len(df), str(args), str(kwargs))
        if k not in indicator_cache: indicator_cache[k] = orig_struct(df, *args, **kwargs)
        return indicator_cache[k]
        
    def patched_swings(df, *args, **kwargs):
        k = ('swings', len(df), str(args), str(kwargs))
        if k not in indicator_cache: indicator_cache[k] = orig_swings(df, *args, **kwargs)
        return indicator_cache[k]
        
    def patched_sweeps(df, *args, **kwargs):
        k = ('sweeps', len(df), str(args), str(kwargs))
        if k not in indicator_cache: indicator_cache[k] = orig_sweeps(df, *args, **kwargs)
        return indicator_cache[k]


    bt_mod.detect_order_blocks = orig_ob
    bt_mod.detect_fair_value_gaps = orig_fvg
    bt_mod.calculate_market_structure_chronologically = orig_struct
    bt_mod.detect_swing_points = orig_swings
    bt_mod.detect_all_liquidity_sweeps = orig_sweeps

    # ── 2.6 SPEED: Signal-level caching for generate_trade_signals ────────
    #    generate_trade_signals iterates over thousands of bars per call.
    #    Its output depends only on signal-affecting params (methods, rrr,
    #    sl_mode, filters, trail_type), NOT on concurr or daily_loss.
    #    When the grid sweeps concurrent/daily_loss, we skip the bar loop
    #    entirely and return cached signals.
    _orig_gen_signals = bt_mod.generate_trade_signals
    _signal_cache = {}
    MAX_SIGNAL_CACHE_SIZE = 10

    def _cached_gen_signals(symbol, tf_name, tf_value, date_from, date_to,
                            ict_params, **kwargs):
        # Build a hashable cache key from ALL signal-affecting parameters
        methods = kwargs.get('ict_method', '')
        if isinstance(methods, list):
            methods = tuple(sorted(methods))
        trail_methods = kwargs.get('trailing_methods', None)
        if isinstance(trail_methods, list):
            trail_methods = tuple(sorted(trail_methods))

        key = (
            symbol, tf_value,
            methods,
            kwargs.get('min_rrr', 1.5),
            kwargs.get('fvg_sl_mode', 0),
            kwargs.get('use_ote_filter', True),
            kwargs.get('use_htf_filter', True),
            kwargs.get('trail_type', None),
            trail_methods,
            kwargs.get('use_dynamic_rrr', True),
            kwargs.get('bypass_htf_conf', False),
            kwargs.get('require_bos_fvg', False),
            kwargs.get('anti_gap_enabled', True),
            kwargs.get('anti_gap_mult', 2.0),
            kwargs.get('fvg_displacement_only', True),
            kwargs.get('fvg_discount_premium_only', True),
            kwargs.get('fvg_recent_sweep_only', False),
            kwargs.get('session_filter', True),
            kwargs.get('session_start', 7),
            kwargs.get('session_end', 21),
            kwargs.get('use_volume_profile', False),
            kwargs.get('use_smt_divergence', False),
            kwargs.get('sb_require_htf_bias', False),
            kwargs.get('min_confluence_score', None),
            kwargs.get('min_fvg_size', None),
            kwargs.get('use_institutional_orderflow', False),
            kwargs.get('institutional_lookback', 20),
            kwargs.get('institutional_threshold', 0.5),
        )

        if key in _signal_cache:
            return list(_signal_cache[key])  # Shallow copy of the list

        from config import bt_stop_event
        result = _orig_gen_signals(symbol, tf_name, tf_value, date_from, date_to,
                                   ict_params, **kwargs)

        # Only cache complete results (not cancelled mid-way)
        if not bt_stop_event.is_set():
            if len(_signal_cache) >= MAX_SIGNAL_CACHE_SIZE:
                _signal_cache.pop(next(iter(_signal_cache)))
            _signal_cache[key] = result

        return result

    bt_mod.generate_trade_signals = _cached_gen_signals

    # 3. Set fixed config values ───────────────────────────────────────────
    import config as cfg
    cfg.MIN_FVG_SIZE_SPREADS = fixed_params['min_fvg_size']
    cfg.MIN_CONFLUENCE_SCORE = fixed_params['min_conf']

    # Sync remaining fixed config
    cfg.NEWS_FILTER_ENABLED = fixed_params.get('news_filter_enabled', False)
    try:
        cfg.NEWS_FILTER_BUFFER_MINS = int(fixed_params.get('news_filter_buffer', 30))
    except (ValueError, TypeError):
        cfg.NEWS_FILTER_BUFFER_MINS = 30

    # Sync Pro strategy flags for worker process to match Backtest tab
    if 'pro_flags' in fixed_params:
        flags = fixed_params['pro_flags']
        cfg.DOL_TP_ENABLED = flags.get('dol_tp', False)
        cfg.KILLZONE_ENABLED = flags.get('killzone', False)
        cfg.HTF_POI_ENABLED = flags.get('htf_poi', False)
        cfg.CONFLUENCE_REQUIRE_MANDATORY = flags.get('mandatory', False)
        cfg.REGIME_FILTER_ENABLED = flags.get('regime', False)
        cfg.ML_SIZING_ENABLED = flags.get('ml_sizing', False)
        cfg.ML_RANK_ENABLED = flags.get('ml_rank', False)
        cfg.MULTI_TF_CONFLUENCE_ENABLED = flags.get('multi_tf_conf', False)
        cfg.MULTI_TF_GATE_ENABLED = flags.get('multi_tf_gate', False)
        cfg.REGIME_ADAPTIVE_PARAMS_ENABLED = flags.get('regime_adaptive', False)

    logger.info("opt_worker: process initialised (PID %s)", __import__('os').getpid())




def get_braille_sparkline(balances, length=15):
    if not balances:
        return ' ' * length
    if len(balances) <= 1:
        return ' ' * length
    w = 2 * length
    n = len(balances)
    sampled = []
    for i in range(w):
        idx = int(round(i * (n - 1) / (w - 1))) if w > 1 else 0
        idx = max(0, min(n - 1, idx))
        sampled.append(balances[idx])
    min_val = min(sampled)
    max_val = max(sampled)
    val_range = max_val - min_val
    left_dots = {0: 64, 1: 4, 2: 2, 3: 1}
    right_dots = {0: 128, 1: 32, 2: 16, 3: 8}
    chars = []
    for i in range(length):
        v1 = sampled[2 * i]
        v2 = sampled[2 * i + 1]
        if val_range == 0:
            y1, y2 = 0, 0
        else:
            y1 = int(((v1 - min_val) / val_range) * 3)
            y2 = int(((v2 - min_val) / val_range) * 3)
        mask = 0
        mask |= left_dots[y1]
        mask |= right_dots[y2]
        chars.append(chr(0x2800 + mask))
    return ''.join(chars)


# ── Combo runner ─────────────────────────────────────────────────────────────

def run_combo(override_dict):
    """Execute a single grid-optimisation combo."""
    if isinstance(override_dict, (list, tuple)):
        (methods, sl_mode, trail_type, ote_flt, htf_flt, disp_flt, disc_flt, 
         sweep_flt, vp_v, smt_v, max_c, rrr_v, dl_v, sb_bias) = override_dict
        
        override_dict = {
            'ict_method': methods[0] if isinstance(methods, (list, tuple)) and len(methods) > 0 else methods,
            'fvg_sl_mode': sl_mode,
            'trail_type': trail_type,
            'use_ote_filter': ote_flt,
            'use_htf_filter': htf_flt,
            'fvg_displacement_only': disp_flt,
            'fvg_discount_premium_only': disc_flt,
            'fvg_recent_sweep_only': sweep_flt,
            'use_volume_profile': vp_v,
            'use_smt_divergence': smt_v,
            'max_concurrent_trades': max_c,
            'min_rrr': rrr_v,
            'max_rrr': rrr_v,
            'daily_loss_limit': dl_v,
            'sb_require_htf_bias': sb_bias
        }
    elif isinstance(override_dict, dict):
        override_dict = dict(override_dict)
        if 'methods' in override_dict:
            m_val = override_dict.pop('methods')
            override_dict['ict_method'] = m_val[0] if isinstance(m_val, (list, tuple)) and len(m_val) > 0 else m_val

    import config as cfg
    import sys

    fp = _worker_state['fixed_params']

    # Per-combo config globals
    dl_val = override_dict.get('daily_loss_limit', 1.0)
    vp_val = override_dict.get('use_volume_profile', False)
    smt_val = override_dict.get('use_smt_divergence', False)
    concurr = override_dict.get('max_concurrent_trades', 2)

    cfg.DAILY_LOSS_LIMIT_PCT = dl_val
    cfg.USE_VOLUME_PROFILE_CONFLUENCE = vp_val
    cfg.USE_SMT_DIVERGENCE = smt_val
    cfg.MAX_CONCURRENT_TRADES = concurr

    try:
        from backtester import combined_backtest

        kwargs = {
            'symbols': [fp['symbol']],
            'date_from': fp['date_from'],
            'date_to': fp['date_to'],
            'initial_balance': fp['initial_balance'],
            'risk_percent': fp['risk_percent'],
            'fixed_lot': fp['fixed_lot'],
            'risk_mode': fp['risk_mode'],
            'trailing_methods': fp['trailing_methods'],
            'ict_params': fp['ict_params'],
            'use_dynamic_rrr': fp['use_dynamic_rrr'],
            'trade_on_all_tfs': fp['trade_all_tfs'],
            'use_ultra_low_tf': fp['use_ultra_low_tf'],
            'spread_cost': fp['spread_cost'],
            'slippage_points': fp['slippage_pts'],
            'commission_per_lot': fp['commission'],
            'session_filter': fp['session_filter'],
            'session_start': fp['session_start'],
            'session_end': fp['session_end'],
            'bypass_htf_conf': fp['bypass_htf_conf'],
            'trail_params': {'trail_pct': fp['trail_pct']},
            'require_bos_fvg': fp['require_bos_fvg'],
            'enable_slippage_recovery': fp['slippage_recovery'],
            'anti_gap_enabled': fp['anti_gap_enabled'],
            'anti_gap_mult': fp['anti_gap_mult'],
            'use_symbol_profiles': False,
            'smt_correlated_pair': fp.get('smt_correlated_pair', 'DXY'),
            'min_confluence_score': fp['min_conf'],
            'min_fvg_size': fp['min_fvg_size'],
            'ml_filter': fp.get('ml_filter', False),
            'ml_min_confidence': fp.get('ml_min_confidence', 0.60),
            'progress_callback': None,
        }
        
        # Apply the overrides for this combo
        kwargs.update(override_dict)
        
        # Whitelist: only pass keys that combined_backtest() actually accepts
        _VALID_CB_KEYS = {
            'symbols', 'date_from', 'date_to', 'initial_balance', 'risk_percent',
            'fixed_lot', 'risk_mode', 'trailing_methods', 'ict_params',
            'ict_method', 'min_rrr', 'use_dynamic_rrr', 'trade_on_all_tfs',
            'use_ultra_low_tf', 'fvg_sl_mode', 'spread_cost', 'slippage_points',
            'commission_per_lot', 'session_filter', 'session_start', 'session_end',
            'progress_callback', 'use_htf_filter', 'use_ote_filter', 'bypass_htf_conf',
            'trail_type', 'trail_params', 'require_bos_fvg', 'enable_slippage_recovery',
            'anti_gap_enabled', 'anti_gap_mult', 'fvg_sl_spread_buffer', 'limit_touch_fill',
            'fvg_displacement_only', 'fvg_discount_premium_only', 'fvg_recent_sweep_only',
            'sb_require_htf_bias', 'use_symbol_profiles', 'clear_cache', 'ml_filter',
            'ml_min_confidence', 'use_smt_divergence', 'smt_correlated_pair',
            'use_volume_profile', 'use_institutional_orderflow', 'institutional_lookback',
            'institutional_threshold', 'max_concurrent_trades', 'daily_loss_limit',
            'min_confluence_score', 'min_fvg_size',
        }
        kwargs = {k: v for k, v in kwargs.items() if k in _VALID_CB_KEYS}

        trades, metrics = combined_backtest(**kwargs)

        trades_count = len(trades)
        is_waste = False
        if trades_count == 1:
            t = trades[0]
            entry_t = t.get("entry_time")
            exit_t = t.get("exit_time")
            if entry_t and exit_t:
                import pandas as pd
                if isinstance(entry_t, str):
                    entry_t = pd.to_datetime(entry_t)
                if isinstance(exit_t, str):
                    exit_t = pd.to_datetime(exit_t)
                try:
                    dur_days = (exit_t - entry_t).total_seconds() / 86400.0
                    if dur_days > 10.0:
                        is_waste = True
                except Exception:
                    pass

        if is_waste:
            roi = -999.0
            dd = 0.0
            wr = 0.0
            pf = 0.0
            calmar = -999.0
            balance_hist = []
        else:
            roi = metrics.get('roi_pct', 0.0)
            dd = metrics.get('max_drawdown_pct', 0.0)
            wr = metrics.get('overall_win_rate', metrics.get('win_rate', 0.0))
            pf = metrics.get('profit_factor', 0.0)
            calmar = roi / max(0.1, dd)
            balance_hist = metrics.get('balance_history', [])

        # Evaluate Out-of-Sample (OOS) validation if oos_date_to is provided
        oos_metrics_res = None
        if fp.get('oos_date_to') and fp.get('oos_date_to') > fp.get('date_to'):
            try:
                oos_kwargs = dict(kwargs)
                oos_kwargs['date_from'] = fp['date_to']
                oos_kwargs['date_to'] = fp['oos_date_to']
                oos_trades, oos_m = combined_backtest(**oos_kwargs)
                oos_roi = oos_m.get('roi_pct', 0.0)
                oos_dd = oos_m.get('max_drawdown_pct', 0.0)
                oos_calmar = oos_roi / max(0.1, oos_dd)
                oos_metrics_res = {
                    'roi': oos_roi,
                    'dd': oos_dd,
                    'calmar': oos_calmar,
                    'trades': len(oos_trades),
                    'win_rate': oos_m.get('overall_win_rate', oos_m.get('win_rate', 0.0)),
                    'pf': oos_m.get('profit_factor', 0.0)
                }
            except Exception as ex_oos:
                logger.error("OOS evaluation failed: %s", ex_oos)

        return {
            'methods': kwargs.get('ict_method', ["FVG Return"]),
            'fvg_sl_mode': kwargs.get('fvg_sl_mode', "Normal"),
            'trail_type': kwargs.get('trail_type', "None (Disabled)"),
            'use_ote_filter': kwargs.get('use_ote_filter', True),
            'use_htf_filter': kwargs.get('use_htf_filter', True),
            'use_vp': vp_val,
            'use_smt': smt_val,
            'fvg_displacement_only': kwargs.get('fvg_displacement_only', True),
            'fvg_discount_premium_only': kwargs.get('fvg_discount_premium_only', True),
            'fvg_recent_sweep_only': kwargs.get('fvg_recent_sweep_only', False),
            'max_concurrent': concurr,
            'rrr': kwargs.get('min_rrr', 1.5),
            'daily_loss': dl_val,
            'roi': roi,
            'dd': dd,
            'win_rate': wr,
            'pf': pf,
            'trades': trades_count,
            'calmar': calmar,
            'is_waste': is_waste,
            'sb_require_htf_bias': kwargs.get('sb_require_htf_bias', False),
            'balance_history': balance_hist,
            'oos_metrics': oos_metrics_res
        }

    except Exception as ex:
        import traceback
        logger.error("opt_worker combo failed: %s\n%s", ex, traceback.format_exc())
        return {
            'error': traceback.format_exc(),
            'trades': 0,
            'roi': 0.0,
            'calmar': 0.0,
            'is_waste': True
        }