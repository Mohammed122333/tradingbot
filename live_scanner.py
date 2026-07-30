import time
import datetime
import logging
import threading
import traceback
import pandas as pd
import numpy as np
try:
    import MetaTrader5 as mt5
except ImportError:
    import tests.MetaTrader5 as mt5

import config
from utils import (
    normalize_and_split_lots, get_contract_size, get_data, get_spread,
    is_crypto_symbol, is_in_trading_session, resolve_ict_methods, get_legacy_ob_retest_methods,
    check_rrr, wait_for_seconds, get_symbol_profile_overrides, resolve_method_param, is_in_news_embargo,
    calculate_dynamic_lot_size, _wilders_smooth_full_njit
)
from trade_memory import is_trade_executed, add_trade_to_memory
from recovery_engine import enforce_anti_gap_sl, update_slippage_recovery, get_recovery_adjustments
from indicators import (
    calculate_ote_zone, get_ict_model_parameters, detect_order_blocks,
    detect_liquidity_sweep, detect_all_liquidity_sweeps, detect_fair_value_gaps,
    get_unmitigated_fvgs, detect_swing_points, detect_market_structure,
    check_bos_after_mss, check_displacement_quality
)
from detectors import (
    detect_silver_bullet_signals, detect_judas_swing, detect_breaker_blocks,
    detect_iofr_signals, compute_fvg_sl_price, detect_mmxm_signals,
    detect_ict_model_signals
)
from simulation import calculate_tp_sl, get_structure_tp, calculate_confluence_score, compute_effective_entry, get_higher_tf_trend
from regime_manager import rm, classify_regime
from ml_engine import ml_score_signal, _ml_live_model
import ict_pro

_live_indicator_cache = {}

# Extracted modules (keeps live_scanner.py focused on the main trading loop)
from live_helpers import (
    get_broker_today_start, is_daily_drawdown_limit_hit,
    get_magic_number, get_method_from_magic, get_method_from_deal,
    get_method_from_position, count_open_positions_for_method,
    is_daily_loss_limit_hit_for_method, check_and_clear_market_closed_flag,
    log_broker_error, format_order_comment,
)
from live_orders import (
    cancel_pending_orders_for_method, place_limit_order,
    check_pending_limit_orders, scan_priority_setups,
    live_ict_trailing_stop, get_filling_mode,
)

logger = logging.getLogger()


def run_live_trading(initial_risk_percent=1.0, risk_mode="Risk", fixed_lot=0.1, symbols=None,
                     trailing_methods=None, use_dynamic_rrr=True, fvg_sl_mode=None,
                     ict_method="ICT Retest", ict_params=None, min_rrr=1.5,
                     trade_on_all_tfs=True, use_ultra_low_tf=False, session_filter=True,
                     session_start=7, session_end=21, use_htf_filter=True, use_ote_filter=True, bypass_htf_conf=False,
                     trail_type=None, trail_params=None, require_bos_fvg=False,
                     ml_enabled=False, ml_min_confidence=0.60,
                     fvg_displacement_only=True, fvg_discount_premium_only=True, fvg_recent_sweep_only=False,
                     sb_require_htf_bias=False,
                     use_symbol_profiles=False):
    if fvg_sl_mode is None:
        fvg_sl_mode = config.FVG_SL_NORMAL
    if trail_type is None:
        trail_type = config.TRAIL_TYPE_PARTIAL
    if symbols is None:
        symbols = [config.DEFAULT_SYMBOL]
    if ict_params is None:
        ict_params = get_ict_model_parameters("Default")
    if trail_params is None:
        trail_params = {}
    # Make available for the trailing stop threads
    trail_type_selected = trail_type
    trail_params_dict = trail_params

    # Save original values for symbol overrides
    orig_params = {
        'initial_risk_percent': initial_risk_percent,
        'risk_mode': risk_mode,
        'fixed_lot': fixed_lot,
        'use_dynamic_rrr': use_dynamic_rrr,
        'fvg_sl_mode': fvg_sl_mode,
        'ict_method': ict_method,
        'min_rrr': min_rrr,
        'trade_on_all_tfs': trade_on_all_tfs,
        'use_ultra_low_tf': use_ultra_low_tf,
        'session_filter': session_filter,
        'session_start': session_start,
        'session_end': session_end,
        'use_htf_filter': use_htf_filter,
        'use_ote_filter': use_ote_filter,
        'bypass_htf_conf': bypass_htf_conf,
        'trail_type': trail_type,
        'trail_pct': trail_params.get('trail_pct', 0.5) if 'trail_pct' in trail_params else 0.5,
        'require_bos_fvg': require_bos_fvg,
        'fvg_displacement_only': fvg_displacement_only,
        'fvg_discount_premium_only': fvg_discount_premium_only,
        'fvg_recent_sweep_only': fvg_recent_sweep_only,
        'sb_require_htf_bias': sb_require_htf_bias,
        'anti_gap_enabled': config.ANTI_GAP_SL_ENABLED,
        'anti_gap_mult': config.ANTI_GAP_ATR_MULTIPLIER,
        'max_concurrent_trades': config.MAX_CONCURRENT_TRADES,
        'daily_loss_limit': config.DAILY_LOSS_LIMIT_PCT,
        'min_fvg_size': config.MIN_FVG_SIZE_SPREADS,
        'min_confluence_score': config.MIN_CONFLUENCE_SCORE,
        'trail_methods': trailing_methods
    }

    methods_to_run = resolve_ict_methods(ict_method)

    if not mt5.initialize():
        logger.critical("Failed to connect/initialize MetaTrader 5 inside run_live_trading.")
        raise RuntimeError("Failed to connect/initialize MetaTrader 5 inside run_live_trading. Please check MT5 connection.")

    from broker_clock import validate_broker_clock
    ok, clock_msg = validate_broker_clock(symbols[0] if symbols else None)
    if not ok and getattr(config, 'PARITY_STRICT', True) and not getattr(config, 'OFFLINE_BACKTESTING', False):
        raise RuntimeError(
            "Broker clock unverified/mismatched - refusing to trade a "
            "time-based ICT strategy on a wrong clock. " + clock_msg)

    # Log broker swap settings and returned values on startup
    try:
        from cost_model import log_broker_swap_info
        log_broker_swap_info(symbols)
    except Exception as exc_swap:
        logger.debug("Failed to log startup broker swap info: %s", exc_swap)

    # ── Regime-adaptive parameter loading ───────────────────────────────
    if getattr(config, 'REGIME_ADAPTIVE_PARAMS_ENABLED', False):
        try:
            sample_sym = symbols[0] if symbols else config.DEFAULT_SYMBOL
            sample_df = get_data(sample_sym, mt5.TIMEFRAME_H1, 100, live=True)
            if sample_df is not None and len(sample_df) >= 20:
                sample_df = sample_df.iloc[:-1]  # drop forming bar
                regime_info = classify_regime(sample_df)
                regime_params = rm.params_for_regime(regime_info['label'])
                rm.apply_params(regime_params)
                logger.info("Live regime: %s (params: %s)", regime_info['label'],
                            {k: v for k, v in regime_params.items()
                             if k in ('max_concurrent_trades', 'min_confluence_score', 'min_fvg_size_spreads')})
        except Exception as exc:
            logger.debug("regime_manager: could not classify live regime — %s", exc)

    # === Recovery Logic for Bot Restart ===
    logger.info("Initializing bot: Recovering existing positions and pending orders...")
    
    # 1. Recover Pending Limit Orders
    try:
        orders = mt5.orders_get()
        if orders:
            for order in orders:
                if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
                    sym = order.symbol
                    dir_str = "Buy" if order.type == mt5.ORDER_TYPE_BUY_LIMIT else "Sell"
                    already_tracked = False
                    for setup_key, info in list(config.pending_limit_orders.items()):
                        tickets_chk = info.get('tickets') or ([info.get('ticket')] if info.get('ticket') else [])
                        if order.ticket in tickets_chk:
                            already_tracked = True
                            break
                    if not already_tracked:
                        setup_key = f"recover_{order.ticket}"
                        risk_pts = abs(order.price_open - order.sl) if order.sl > 0 else 0
                        if risk_pts == 0:
                            point = mt5.symbol_info(sym).point if mt5.symbol_info(sym) else 0.0001
                            risk_pts = 100 * point  # Approximation
                        pos_method = get_method_from_position(order)
                        use_ts = (trailing_methods is None) or (pos_method in trailing_methods)
                        if trail_type_selected is None or trail_type_selected == "None":
                            use_ts = False
                            
                        config.pending_limit_orders[setup_key] = {
                            'tickets': [order.ticket],
                            'symbol': sym,
                            'direction': dir_str,
                            'entry_price': order.price_open,
                            'sl_price': order.sl,
                            'tp_price': order.tp,
                            'lot_size': order.volume_initial,
                            'ict_method': pos_method,
                            'ob_timestamp': str(pd.Timestamp(order.time_setup, unit='s')) if hasattr(order, 'time_setup') and order.time_setup > 0 else str(pd.Timestamp.now()),
                            'ob_type': dir_str,
                            'tf_name': "M15",
                            'trailing_info': {
                                'use_ts': use_ts,
                                'risk_pts': risk_pts,
                                'tf_value': mt5.TIMEFRAME_M15,
                                'trail_type': trail_type_selected,
                                'trail_params': trail_params_dict,
                                'use_symbol_profiles': use_symbol_profiles,
                                'ict_method': pos_method
                            },
                            'placed_time': order.time_setup if hasattr(order, 'time_setup') and order.time_setup > 0 else time.time(),
                        }
                        logger.info(f"Recovered Pending Order: {order.ticket} {dir_str} {sym}")
        
        # 2. Recover Open Positions
        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                sym = pos.symbol
                dir_str = "Buy" if pos.type == mt5.POSITION_TYPE_BUY else "Sell"
                risk_pts = abs(pos.price_open - pos.sl) if pos.sl > 0 else 0
                if risk_pts == 0:
                     point = mt5.symbol_info(sym).point if mt5.symbol_info(sym) else 0.0001
                     risk_pts = 100 * point
                pos_method = get_method_from_position(pos)
                if pos_method == "Unknown" and trailing_methods:
                    pos_method = trailing_methods[0]

                use_ts = (trailing_methods is None) or (pos_method in trailing_methods)
                if trail_type_selected is None or trail_type_selected == "None":
                    use_ts = False
                
                if use_ts:
                    logger.info(f"Recovered Active Position: {pos.ticket} {dir_str} {sym} [Method: {pos_method}]. Attaching trailing stop.")
                    threading.Thread(target=live_ict_trailing_stop,
                        args=(pos.ticket, sym, risk_pts, dir_str, mt5.TIMEFRAME_M15,
                              trail_type_selected, trail_params_dict, use_symbol_profiles),
                        kwargs={'ict_method': pos_method},
                        daemon=True).start()
                else:
                    logger.info(f"Recovered Active Position: {pos.ticket} {dir_str} {sym} [Method: {pos_method}]. No trailing stop configured.")
    except Exception as e:
        logger.error(f"Failed to recover existing orders: {e}")
    # ====================================

    logged_live_states = {}
    last_heartbeat_time = time.time() - 60  # Force immediate first heartbeat

    while config.live_trading_running and not config.stop_event.is_set():
        # MT5 Connection Heartbeat & Reconnection Safety Check
        term_info = mt5.terminal_info()
        if term_info is None or (hasattr(term_info, 'connected') and not term_info.connected):
            config.current_activity = "Suspended: Waiting for MT5 connection..."
            if logged_live_states.get("MT5_DISCONNECTED") != "YES":
                logger.warning("[MT5 Connection] Terminal or broker connection is inactive. Waiting for connection...")
                logged_live_states["MT5_DISCONNECTED"] = "YES"
            time.sleep(2)
            continue
        else:
            if logged_live_states.get("MT5_DISCONNECTED") == "YES":
                logger.info("[MT5 Connection] Terminal and broker connection re-established. Resuming scanning.")
                logged_live_states["MT5_DISCONNECTED"] = "NO"

        # Check global account daily drawdown limit
        drawdown_hit, drawdown_msg = is_daily_drawdown_limit_hit()
        if drawdown_hit:
            config.current_activity = f"Suspended: {drawdown_msg}"
            if logged_live_states.get("GLOBAL_DRAWDOWN_HIT") != "YES":
                logger.warning(f"[GLOBAL DRAWDOWN] {drawdown_msg}. Suspending live trading.")
                logged_live_states["GLOBAL_DRAWDOWN_HIT"] = "YES"
            time.sleep(10)
            continue
        else:
            if logged_live_states.get("GLOBAL_DRAWDOWN_HIT") == "YES":
                logger.info("[GLOBAL DRAWDOWN] Account recovery or daily reset detected. Resuming live trading.")
                logged_live_states["GLOBAL_DRAWDOWN_HIT"] = "NO"

        # Check and cancel pending limit orders for any method whose daily drawdown limit is hit
        for symbol in symbols:
            sym_defaults = {'ict_method': orig_params['ict_method']}
            sym_params = get_symbol_profile_overrides(symbol, sym_defaults) if use_symbol_profiles else sym_defaults
            methods_to_run_check = resolve_ict_methods(sym_params['ict_method'])
            for method in methods_to_run_check:
                limit_hit, limit_msg = is_daily_loss_limit_hit_for_method(method, symbol, use_symbol_profiles)
                if limit_hit:
                    cancel_pending_orders_for_method(symbol, method)

        current_time_sec = time.time()
        if current_time_sec - last_heartbeat_time >= 60:
            n_pending = len(config.pending_limit_orders)
            n_priority = len(config.priority_scan_setups)
            active_m_list = []
            for s in symbols:
                if use_symbol_profiles:
                    m_str = get_symbol_profile_overrides(s, {'ict_method': orig_params['ict_method']}).get('ict_method', ict_method)
                else:
                    m_str = orig_params['ict_method']
                active_m_list.extend(resolve_ict_methods(m_str))
            unique_methods = sorted(list(set(active_m_list)))
            logger.info("Pulse [Live Tracker]: Active Scan for %s | Methods: [%s] | %d pending limits | %d priority setups...",
                       ", ".join(symbols), ", ".join(unique_methods), n_pending, n_priority)
            last_heartbeat_time = current_time_sec

        # === Check pending limit orders for fills (Option C monitor) ===
        if config.pending_limit_orders:
            check_pending_limit_orders()

        # === Rapid priority scan (Option B fallback) ===
        if config.priority_scan_setups:
            scan_priority_setups()

        for symbol in symbols:
            try:
                # Load profile overrides for this specific symbol
                sym_defaults = {
                    'risk_percent': orig_params['initial_risk_percent'],
                    'fixed_lot': orig_params['fixed_lot'],
                    'use_dynamic_rrr': orig_params['use_dynamic_rrr'],
                    'fvg_sl_mode': orig_params['fvg_sl_mode'],
                    'ict_method': orig_params['ict_method'],
                    'min_rrr': orig_params['min_rrr'],
                    'trade_on_all_tfs': orig_params['trade_on_all_tfs'],
                    'use_ultra_low_tf': orig_params['use_ultra_low_tf'],
                    'session_filter': orig_params['session_filter'],
                    'session_start': orig_params['session_start'],
                    'session_end': orig_params['session_end'],
                    'use_htf_filter': orig_params['use_htf_filter'],
                    'use_ote_filter': orig_params['use_ote_filter'],
                    'bypass_htf_conf': orig_params['bypass_htf_conf'],
                    'trail_type': orig_params['trail_type'],
                    'trail_pct': orig_params['trail_pct'],
                    'require_bos_fvg': orig_params['require_bos_fvg'],
                    'fvg_displacement_only': orig_params['fvg_displacement_only'],
                    'fvg_discount_premium_only': orig_params['fvg_discount_premium_only'],
                    'fvg_recent_sweep_only': orig_params['fvg_recent_sweep_only'],
                    'sb_require_htf_bias': orig_params['sb_require_htf_bias'],
                    'anti_gap_enabled': orig_params['anti_gap_enabled'],
                    'anti_gap_mult': orig_params['anti_gap_mult'],
                    'max_concurrent_trades': orig_params['max_concurrent_trades'],
                    'daily_loss_limit': orig_params['daily_loss_limit'],
                    'min_fvg_size': orig_params['min_fvg_size'],
                    'min_confluence_score': orig_params['min_confluence_score'],
                    'trail_methods': orig_params['trail_methods']
                }
                if use_symbol_profiles:
                    sym_params = get_symbol_profile_overrides(symbol, sym_defaults)
                else:
                    sym_params = sym_defaults
                
                # Overwrite local variables for this loop iteration
                initial_risk_percent = sym_params['risk_percent']
                fixed_lot = sym_params['fixed_lot']
                use_dynamic_rrr = sym_params['use_dynamic_rrr']
                fvg_sl_mode = sym_params['fvg_sl_mode']
                ict_method = sym_params['ict_method']
                trailing_methods = sym_params['trail_methods']
                min_rrr = sym_params['min_rrr']
                trade_on_all_tfs = sym_params['trade_on_all_tfs']
                use_ultra_low_tf = sym_params['use_ultra_low_tf']
                session_filter = sym_params['session_filter']
                session_start = sym_params['session_start']
                session_end = sym_params['session_end']
                use_htf_filter = sym_params['use_htf_filter']
                use_ote_filter = sym_params['use_ote_filter']
                bypass_htf_conf = sym_params['bypass_htf_conf']
                trail_type = sym_params['trail_type']
                trail_params = {'trail_pct': sym_params['trail_pct']}
                require_bos_fvg = sym_params['require_bos_fvg']
                fvg_displacement_only = sym_params['fvg_displacement_only']
                fvg_discount_premium_only = sym_params['fvg_discount_premium_only']
                fvg_recent_sweep_only = sym_params['fvg_recent_sweep_only']
                sb_require_htf_bias = sym_params['sb_require_htf_bias']
                
                config.ANTI_GAP_SL_ENABLED = sym_params['anti_gap_enabled']
                config.ANTI_GAP_ATR_MULTIPLIER = sym_params['anti_gap_mult']
                
                methods_to_run = resolve_ict_methods(ict_method)

                # Instrument-aware parameters (adapt OB thresholds per symbol class)
                ict_params = get_ict_model_parameters("Default", symbol)

                # Fix #14: Harmonize Live / Backtest Timezones (Broker Time)
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    continue
                # MT5 times are POSIX timestamps representing broker time
                broker_time = datetime.datetime.fromtimestamp(tick.time, datetime.timezone.utc).replace(tzinfo=None)
                
                if session_filter and not is_crypto_symbol(symbol) and not is_in_trading_session(broker_time, session_start, session_end):
                    if "Silver Bullet" in methods_to_run:
                        methods_to_run = ["Silver Bullet"]
                    else:
                        config.current_activity = f"Outside trading session ({session_start}:00-{session_end}:00 Broker Time)"
                        continue

                if getattr(config, 'NEWS_EMBARGO_ENABLED', False) and not is_crypto_symbol(symbol):
                    if is_in_news_embargo(broker_time):
                        config.current_activity = f"Institutional News Embargo active for {symbol}"
                        continue

                if getattr(config, 'NEWS_FILTER_ENABLED', False):
                    from news_filter import is_macro_news_embargo
                    buffer_mins = getattr(config, 'NEWS_FILTER_BUFFER_MINS', 30)
                    is_macro_embargo, news_reason = is_macro_news_embargo(symbol, broker_time, is_live=True, buffer_mins=buffer_mins)
                    if is_macro_embargo:
                        config.current_activity = f"Macro News Embargo: {news_reason}"
                        continue

                higher_tf_trend = "neutral"
                if use_htf_filter:
                    config.current_activity = f"Analyzing higher TFs for {symbol}..."
                    higher_tf_trend = get_higher_tf_trend(symbol, ict_params)

                if use_ultra_low_tf:
                    timeframes_to_scan = config.ULTRA_LOW_TFS[:]
                else:
                    timeframes_to_scan = config.LOWER_TFS[:] if trade_on_all_tfs else [("M15", mt5.TIMEFRAME_M15)]

                for tf_name, tf_value in timeframes_to_scan:
                    config.current_activity = f"Scanning {symbol} ({tf_name}) for [{', '.join(methods_to_run)}]..."
                    if logged_live_states.get(f"{symbol}_{tf_name}_SCAN_LOGGED") != "YES":
                        logger.info("[%s] %s Searching setups for active methods: [%s]", tf_name, symbol, ", ".join(methods_to_run))
                        logged_live_states[f"{symbol}_{tf_name}_SCAN_LOGGED"] = "YES"

                    # General concurrent trades check removed (checked per-method instead)

                    df = get_data(symbol, tf_value, 1500, live=True)  # Fetch 1500 bars for extended session context
                    if df is None or df.empty:
                        continue
                    df = df.sort_index()
                    
                    # Fetch Correlated pairs for SMT
                    corr_dfs_dict = {}
                    if getattr(config, 'USE_SMT_DIVERGENCE', False) and getattr(config, 'SMT_CORRELATED_PAIR', ''):
                        import re
                        suffix_match = re.search(r'([a-z.]+)$', symbol)
                        suffix = suffix_match.group(1) if suffix_match else ""
                        if config.SMT_CORRELATED_PAIR.upper() == "AUTO":
                            pairs = ["DXY"]
                            if "EURUSD" in symbol or "GBPUSD" in symbol or "AUDUSD" in symbol or "NZDUSD" in symbol:
                                pairs = ["DXY", "USDCAD", "USDCHF"] if "GBP" not in symbol else ["DXY", "EURUSD"]
                            elif "USD" in symbol:
                                pairs = ["EURUSD", "GBPUSD"]
                        else:
                            pairs = [p.strip() for p in config.SMT_CORRELATED_PAIR.split(',') if p.strip()]
                        for p in pairs:
                            pair_to_fetch = p + suffix if suffix else p
                            c_df = get_data(pair_to_fetch, tf_value, 1500, live=True)
                            if c_df is None or c_df.empty:
                                c_df = get_data(p, tf_value, 1500, live=True)
                            if c_df is not None and not c_df.empty:
                                corr_dfs_dict[p] = c_df.sort_index()
                    
                    # Fix #11: Exclude forming (last) bar from analysis
                    analysis_df = df.iloc[:-1]
                    if len(analysis_df) < 20:
                        continue
                    
                    latest_bar = analysis_df.iloc[-1]
                    
                    # Calculate current ATR matching backtester (Wilder's RMA formula)
                    _highs_atr = analysis_df['high'].values
                    _lows_atr = analysis_df['low'].values
                    _closes_atr = analysis_df['close'].values
                    tr_atr = np.zeros(len(analysis_df))
                    tr_atr[1:] = np.maximum(_highs_atr[1:] - _lows_atr[1:], np.maximum(np.abs(_highs_atr[1:] - _closes_atr[:-1]), np.abs(_lows_atr[1:] - _closes_atr[:-1])))
                    tr_atr[0] = _highs_atr[0] - _lows_atr[0]
                    period = 14
                    atr_series = _wilders_smooth_full_njit(tr_atr, period, tr_atr[0])
                    current_atr = atr_series[-1]

                    # === Fix 6: Regime gate (pure price-action; stand aside in chop) ===
                    _regime = ict_pro.regime_of(analysis_df, current_atr)
                    _regime_size_mult = float(_regime.get('size_mult', 1.0) or 1.0)
                    _regime_conf_bump = int(_regime.get('conf_bump', 0) or 0)
                    if not _regime.get('tradeable', True):
                        config.current_activity = (
                            f"{symbol} {tf_name}: regime '{_regime.get('label')}' "
                            f"({_regime.get('structure')}) - standing aside")
                        continue

                    ote_low, ote_high, swing_low, swing_high = calculate_ote_zone(
                        analysis_df, lookback=50, fib_low=ict_params['fib_low'], fib_high=ict_params['fib_high'])

                    current_spread = get_spread(symbol)

                    cache_key = (symbol, tf_name, latest_bar.name)
                    if cache_key in _live_indicator_cache:
                        cached_data = _live_indicator_cache[cache_key]
                        all_obs = cached_data['all_obs']
                        all_fvgs = cached_data['all_fvgs']
                        fvgs = cached_data['fvgs']
                        structure = cached_data['structure']
                        mss = cached_data['mss']
                        swing_highs = cached_data['swing_highs']
                        swing_lows = cached_data['swing_lows']
                        bullish_sweep = cached_data['bullish_sweep']
                        bearish_sweep = cached_data['bearish_sweep']
                    else:
                        all_obs = detect_order_blocks(analysis_df,
                            threshold_factor=ict_params['threshold_factor'],
                            reversal_threshold=ict_params['reversal_threshold'],
                            lookahead=ict_params['lookahead'],
                            spread=current_spread)

                        # === ICT Analysis (matches backtest) ===
                        all_fvgs = detect_fair_value_gaps(analysis_df)
                        fvgs = get_unmitigated_fvgs(analysis_df, all_fvgs)
                        structure, mss, swing_highs, swing_lows = detect_market_structure(analysis_df)
                        bullish_sweep, bearish_sweep, _, _ = detect_liquidity_sweep(analysis_df)
                        
                        # Parity with backtest: filter swing points, OBs, and FVGs to the active 199 closed bars window
                        # so that preceding history (first 100+ bars) is strictly used for indicator warmup only.
                        window_start_time = analysis_df.index[-min(199, len(analysis_df))]
                        swing_highs = [sh for sh in swing_highs if sh['time'] >= window_start_time]
                        swing_lows = [sl for sl in swing_lows if sl['time'] >= window_start_time]
                        if not all_obs.empty:
                            all_obs = all_obs[all_obs.index >= window_start_time]
                        fvgs = [f for f in fvgs if f['time'] >= window_start_time]

                        _live_indicator_cache[cache_key] = {
                            'all_obs': all_obs, 'all_fvgs': all_fvgs, 'fvgs': fvgs,
                            'structure': structure, 'mss': mss,
                            'swing_highs': swing_highs, 'swing_lows': swing_lows,
                            'bullish_sweep': bullish_sweep, 'bearish_sweep': bearish_sweep
                        }
                        
                        # Cleanup old caches to prevent memory leak
                        keys_to_delete = [k for k in _live_indicator_cache if k[0] == symbol and k[1] == tf_name and k[2] != latest_bar.name]
                        for k in keys_to_delete:
                            del _live_indicator_cache[k]

                    # Process OBs (if any exist)
                    if not all_obs.empty:
                        all_obs = all_obs.sort_index()
                        time_diffs = pd.Series(
                            np.abs(pd.to_datetime(latest_bar.name) - pd.to_datetime(all_obs.index)),
                            index=all_obs.index)
                        # Process top 3 nearest OBs (matches backtest)
                        sorted_ob_indices = time_diffs.argsort()[:3]
                        
                        for ob_rank_idx in sorted_ob_indices:
                            nearest_index = all_obs.index[ob_rank_idx]
                            nearest_ob = all_obs.iloc[ob_rank_idx]
                            ob_type = nearest_ob.get('order_block_type', 'bullish')

                            ob_timestamp = str(nearest_index)
                            base_key = f"{symbol}_{tf_name}_{ob_timestamp}"

                            # The overall OB logic is still checked, but is_trade_executed is moved per method
                            # === ICT Structure filter (matches backtest) ===
                            structure_ok = False
                            if structure == "neutral":
                                structure_ok = True
                            elif (ob_type == "bullish" and structure == "bullish") or \
                                 (ob_type == "bearish" and structure == "bearish"):
                                structure_ok = True
                            elif mss:
                                if (ob_type == "bullish" and mss == "bullish") or \
                                   (ob_type == "bearish" and mss == "bearish"):
                                    structure_ok = True

                            if not structure_ok:
                                continue  # Strict enforcement of market structure alignment
                                
                            # Displacement quality check (matches backtest)
                            has_displacement = False
                            if 'confirmation_bar' in nearest_ob and not pd.isna(nearest_ob.get('confirmation_bar', np.nan)):
                                conf_bar = int(nearest_ob['confirmation_bar'])
                                has_displacement = check_displacement_quality(analysis_df, conf_bar)

                            if not has_displacement:
                                continue  # Strict enforcement of displacement
                                
                            tick = mt5.symbol_info_tick(symbol)
                            if tick is None:
                                continue
                            if ob_type == "bullish":
                                direction = "Buy"
                                current_price = tick.ask
                            else:
                                direction = "Sell"
                                current_price = tick.bid

                            for method in get_legacy_ob_retest_methods(methods_to_run):
                                setup_key = f"{symbol}_{tf_name}_{ob_timestamp}_{method}"
                                if is_trade_executed(symbol, tf_name, ob_timestamp, ob_type, method):
                                    continue
                                    
                                last_state = logged_live_states.get(setup_key, "NEW")
                                
                                m_max_concurr = resolve_method_param('max_concurrent_trades', method, sym_params, config.MAX_CONCURRENT_TRADES)
                                if m_max_concurr > 0:
                                    open_count = count_open_positions_for_method(symbol, method)
                                    if open_count >= m_max_concurr:
                                        if last_state != "CONCUR_LIMIT":
                                            logger.info("[%s] %s %s blocked: Max concurrent trades reached (%d/%d)", tf_name, symbol, method, open_count, m_max_concurr)
                                            logged_live_states[setup_key] = "CONCUR_LIMIT"
                                        continue
                                        
                                limit_hit, limit_msg = is_daily_loss_limit_hit_for_method(method, symbol, use_symbol_profiles)
                                if limit_hit:
                                    if last_state != "DAILY_LIMIT":
                                        logger.warning("[%s] %s %s blocked: %s", tf_name, symbol, method, limit_msg)
                                        logged_live_states[setup_key] = "DAILY_LIMIT"
                                    continue
                                    
                                m_use_htf_filter = resolve_method_param('use_htf_filter', method, sym_params, use_htf_filter)
                                m_bypass_htf_conf = resolve_method_param('bypass_htf_conf', method, sym_params, bypass_htf_conf)
                                m_use_ote_filter = resolve_method_param('use_ote_filter', method, sym_params, use_ote_filter)
                                m_min_rrr = resolve_method_param('min_rrr', method, sym_params, min_rrr)
                                m_use_dynamic_rrr = resolve_method_param('use_dynamic_rrr', method, sym_params, use_dynamic_rrr)
                                m_risk_percent = resolve_method_param('risk_percent', method, sym_params, initial_risk_percent)
                                m_fixed_lot = resolve_method_param('fixed_lot', method, sym_params, fixed_lot)
                                m_use_fixed_lot = resolve_method_param('use_fixed_lot', method, sym_params, risk_mode == "Fixed")
                                m_anti_gap_enabled = resolve_method_param('anti_gap_enabled', method, sym_params, config.ANTI_GAP_SL_ENABLED)
                                m_anti_gap_mult = resolve_method_param('anti_gap_mult', method, sym_params, config.ANTI_GAP_ATR_MULTIPLIER)
                                m_trail_type = resolve_method_param('trail_type', method, sym_params, trail_type_selected)
                                m_trail_pct = resolve_method_param('trail_pct', method, sym_params, trail_params_dict.get('trail_pct', 0.5) if trail_params_dict else 0.5)
                                m_trail_params = {'trail_pct': m_trail_pct}
                                m_min_confluence_score = resolve_method_param('min_confluence_score', method, sym_params, config.MIN_CONFLUENCE_SCORE)
                                m_min_confluence_score += _regime_conf_bump  # Fix 6: require more confluence in weak regimes
                                
                                effective_entry = compute_effective_entry(current_price, nearest_ob, symbol, method, direction)

                                # === Confluence scoring (matches backtest) ===
                                confluence_score, confluence_details = calculate_confluence_score(
                                    nearest_ob, ob_type, fvgs, structure, mss,
                                    bullish_sweep, bearish_sweep,
                                    higher_tf_trend, ote_low, ote_high, effective_entry,
                                    analysis_df, has_displacement,
                                    entry_time=pd.Timestamp.now(),
                                    symbol=symbol)  # Fix 2+5: killzone points + mandatory gate

                                if confluence_score < m_min_confluence_score:
                                    if last_state != "LOW_CONF":
                                        logger.info("[%s] %s %s confluence too low (%d): %s", tf_name, symbol, method, confluence_score, ", ".join(confluence_details))
                                        logged_live_states[setup_key] = "LOW_CONF"
                                    continue

                                # --- SMT Divergence Filter ---
                                if getattr(config, 'USE_SMT_DIVERGENCE', False) and corr_dfs_dict:
                                    from indicators import check_advanced_smt
                                    # Use the latest closed bar as reference
                                    has_smt = check_advanced_smt(analysis_df, corr_dfs_dict, len(analysis_df)-1, direction, lookback=20)
                                    if not has_smt:
                                        if last_state != "NO_SMT":
                                            logger.info("[%s] %s %s skipped: No SMT divergence detected.", tf_name, symbol, method)
                                            logged_live_states[setup_key] = "NO_SMT"
                                        continue

                                # HTF filter (strict — unless bypassed by high conf)
                                if m_use_htf_filter and higher_tf_trend != "neutral":
                                    if not (m_bypass_htf_conf and confluence_score >= 2):
                                        if higher_tf_trend == "bullish" and ob_type != "bullish":
                                            if logged_live_states.get(base_key + "_HTF") != "BLOCKED":
                                                logger.info("[%s] %s Found %s OB, but HTF trend is %s. Skipping.", tf_name, symbol, ob_type, higher_tf_trend)
                                                logged_live_states[base_key + "_HTF"] = "BLOCKED"
                                            continue
                                        elif higher_tf_trend == "bearish" and ob_type != "bearish":
                                            if logged_live_states.get(base_key + "_HTF") != "BLOCKED":
                                                logger.info("[%s] %s Found %s OB, but HTF trend is %s. Skipping.", tf_name, symbol, ob_type, higher_tf_trend)
                                                logged_live_states[base_key + "_HTF"] = "BLOCKED"
                                            continue

                                # OTE filter: strict application when enabled
                                if m_use_ote_filter:
                                    if not (ote_low <= effective_entry <= ote_high):
                                        if last_state != "NOT_OTE":
                                            logger.info("[%s] %s %s %s target (%.5f) outside OTE zone. Conf=%d", tf_name, symbol, method, direction, effective_entry, confluence_score)
                                            logged_live_states[setup_key] = "NOT_OTE"
                                        continue

                                # === ML GATE: ML model is the final decision maker ===
                                if ml_enabled and _ml_live_model['loaded']:
                                    # Pre-compute a rough TP/SL for ML scoring
                                    _si = mt5.symbol_info(symbol)
                                    _pt = _si.point if _si else 0.00001
                                    _tp_est, _sl_est = calculate_tp_sl(symbol, analysis_df, nearest_ob, direction, effective_entry, _pt, min_rrr=m_min_rrr)
                                    _rrr_est = abs(effective_entry - _tp_est) / max(abs(effective_entry - _sl_est), _pt) if abs(effective_entry - _sl_est) > 0 else 0

                                    ml_proba, _ = ml_score_signal(
                                        trade_direction=direction,
                                        method=method,
                                        confluence_score=confluence_score,
                                        htf_trend=higher_tf_trend,
                                        ob_type=ob_type,
                                        entry_price=effective_entry,
                                        sl_price=_sl_est,
                                        tp_price=_tp_est,
                                        rrr=_rrr_est,
                                        entry_time=pd.Timestamp.now(),
                                        confluence_details=confluence_details,
                                    )
                                    if ml_proba < ml_min_confidence:
                                        if last_state != "ML_REJECT":
                                            logger.info("[ML] %s %s %s REJECTED by ML (prob=%.1f%% < %.0f%%)",
                                                       tf_name, symbol, method, ml_proba * 100, ml_min_confidence * 100)
                                            logged_live_states[setup_key] = "ML_REJECT"
                                        continue
                                    else:
                                        logger.info("[ML] %s %s %s APPROVED by ML (prob=%.1f%% >= %.0f%%)",
                                                   tf_name, symbol, method, ml_proba * 100, ml_min_confidence * 100)

                                # Fix 4: HTF POI nesting - only enter inside an aligned HTF POI
                                if not ict_pro.passes_htf_poi(symbol, direction, effective_entry):
                                    if last_state != "HTF_POI_MISS":
                                        logger.info("[%s] %s %s skipped: entry not inside an aligned HTF POI.", tf_name, symbol, method)
                                        logged_live_states[setup_key] = "HTF_POI_MISS"
                                    continue

                                # === OPTION C: Place limit order if price hasn't reached entry ===
                                if (direction == "Buy" and current_price > effective_entry) or \
                                   (direction == "Sell" and current_price < effective_entry):
                                    # Skip if already have a pending limit or priority setup for this
                                    if setup_key in config.pending_limit_orders or setup_key in config.priority_scan_setups:
                                        continue

                                    # Pre-compute TP/SL/lots for the limit order
                                    symbol_info = mt5.symbol_info(symbol)
                                    point = symbol_info.point if symbol_info else 0.00001
                                    TP_lim, SL_lim = calculate_tp_sl(symbol, analysis_df, nearest_ob, direction, effective_entry, point, min_rrr=m_min_rrr)

                                    if m_use_dynamic_rrr:
                                        struct_tp = get_structure_tp(effective_entry, direction, swing_highs, swing_lows, current_spread)
                                        if struct_tp is not None:
                                            if direction == "Buy":
                                                TP_lim = max(TP_lim, struct_tp)
                                            else:
                                                TP_lim = min(TP_lim, struct_tp)

                                    if symbol_info is not None:
                                        min_stop_distance = max(symbol_info.trade_stops_level * point, current_spread * 1.5, point * 10)
                                    else:
                                        min_stop_distance = current_spread * 3
                                    if direction == "Buy":
                                        if (effective_entry - SL_lim) < min_stop_distance:
                                            SL_lim = effective_entry - min_stop_distance - point
                                        adjusted_risk = effective_entry - SL_lim
                                        min_tp = effective_entry + adjusted_risk * m_min_rrr
                                        TP_lim = max(TP_lim, min_tp)
                                    else:
                                        if (SL_lim - effective_entry) < min_stop_distance:
                                            SL_lim = effective_entry + min_stop_distance + point
                                        adjusted_risk = SL_lim - effective_entry
                                        min_tp = effective_entry - adjusted_risk * m_min_rrr
                                        TP_lim = min(TP_lim, min_tp)

                                    # Fix 1+3: Draw-on-Liquidity TP (target real liquidity, keep >= min RRR)
                                    _dol = ict_pro.dol_target(symbol, direction, effective_entry, SL_lim, m_min_rrr)
                                    if _dol is not None:
                                        TP_lim = _dol

                                    rrr_ok, rrr_value = check_rrr(effective_entry, TP_lim, SL_lim, direction, m_min_rrr)
                                    if not rrr_ok:
                                        if last_state != "BAD_RRR":
                                            logger.info("[%s] %s setup rejected: RRR %.2f < %.2f", tf_name, symbol, rrr_value, m_min_rrr)
                                            logged_live_states[setup_key] = "BAD_RRR"
                                        continue

                                    account_info = mt5.account_info()
                                    if account_info is None:
                                        continue
                                    current_balance = account_info.balance
                                    risk_points = abs(effective_entry - SL_lim)
                                    if risk_points <= 0:
                                        continue
                                    if not m_use_fixed_lot:
                                        risk_amount = current_balance * (m_risk_percent / 100)
                                        lot_size = calculate_dynamic_lot_size(symbol, risk_amount, risk_points)
                                        if m_anti_gap_enabled:
                                            _contract_sz = get_contract_size(symbol)
                                            SL_lim, lot_size, _gap_widened, _, _ = enforce_anti_gap_sl(
                                                effective_entry, SL_lim, direction, analysis_df, lot_size,
                                                risk_amount, _contract_sz, atr_multiplier=m_anti_gap_mult, point=point, digits=digits)
                                            if _gap_widened:
                                                risk_points = abs(effective_entry - SL_lim)
                                                if direction == "Buy":
                                                    min_tp = effective_entry + risk_points * m_min_rrr
                                                    TP_lim = max(TP_lim, min_tp)
                                                else:
                                                    min_tp = effective_entry - risk_points * m_min_rrr
                                                    TP_lim = min(TP_lim, min_tp)
                                        # Fix 6+7: regime size-down + ML conviction sizing
                                        lot_size *= _regime_size_mult * ict_pro.ml_size_multiplier(locals().get('ml_proba', 0))
                                    else:
                                        lot_size = m_fixed_lot

                                    comment_str = format_order_comment(method, direction, tf_name, confluence_score)
                                    use_ts = (trailing_methods is None) or (method in trailing_methods)
                                    if m_trail_type is None or m_trail_type == "None":
                                        use_ts = False
                                    trail_info = {
                                        'use_ts': use_ts,
                                        'risk_pts': risk_points,
                                        'tf_value': tf_value,
                                        'trail_type': m_trail_type,
                                        'trail_params': m_trail_params,
                                        'use_symbol_profiles': use_symbol_profiles,
                                        'ict_method': method
                                    }

                                    # Try limit order first (Option C)
                                    limit_placed = place_limit_order(
                                        symbol, direction, effective_entry, SL_lim, TP_lim,
                                        lot_size, comment_str, setup_key, trailing_info=trail_info,
                                        ob_timestamp=ob_timestamp, ob_type=ob_type, tf_name=tf_name)

                                    if limit_placed:
                                        logger.info("[%s] %s %s Conf=%d [%s] LIMIT placed @ %.5f (Current: %.5f)",
                                                   tf_name, symbol, method, confluence_score,
                                                   ", ".join(confluence_details), effective_entry, current_price)
                                        logged_live_states[setup_key] = "LIMIT_PLACED"
                                    else:
                                        # Limit failed → Option B: add to priority scan queue
                                        config.priority_scan_setups[setup_key] = {
                                            'symbol': symbol,
                                            'direction': direction,
                                            'entry_price': effective_entry,
                                            'sl_price': SL_lim,
                                            'tp_price': TP_lim,
                                            'lot_size': lot_size,
                                            'comment': comment_str,
                                            'trailing_info': trail_info,
                                            'created_time': time.time(),
                                            'ob_timestamp': ob_timestamp,
                                            'ob_type': ob_type,
                                            'tf_name': tf_name,
                                        }
                                        logger.info("[%s] %s %s Conf=%d [%s] Limit failed — PRIORITY SCAN @ %.5f (Current: %.5f)",
                                                   tf_name, symbol, method, confluence_score,
                                                   ", ".join(confluence_details), effective_entry, current_price)
                                        logged_live_states[setup_key] = "PRIORITY_SCAN"
                                    continue

                                symbol_info = mt5.symbol_info(symbol)
                                point = symbol_info.point if symbol_info else 0.00001
                                digits = symbol_info.digits if symbol_info else 5
                                TP_default, SL = calculate_tp_sl(symbol, analysis_df, nearest_ob, direction, current_price, point, min_rrr=m_min_rrr)

                                # Structure-based dynamic TP vs Fixed RRR
                                if m_use_dynamic_rrr:
                                    struct_tp = get_structure_tp(current_price, direction, swing_highs, swing_lows, current_spread)
                                    if struct_tp is not None:
                                        if direction == "Buy":
                                            TP_default = max(TP_default, struct_tp)
                                        else:
                                            TP_default = min(TP_default, struct_tp)

                                if symbol_info is not None:
                                    min_stop_distance = max(symbol_info.trade_stops_level * point, current_spread * 1.5, point * 10)
                                else:
                                    min_stop_distance = current_spread * 3

                                # Adjust SL for min stop distance, then RECALCULATE TP to maintain min_rrr
                                if direction == "Buy":
                                    if (current_price - SL) < min_stop_distance:
                                        SL = current_price - min_stop_distance - point
                                    adjusted_risk = current_price - SL
                                    min_tp = current_price + adjusted_risk * m_min_rrr
                                    TP_default = max(TP_default, min_tp)
                                    if (TP_default - current_price) < min_stop_distance:
                                        TP_default = current_price + min_stop_distance + point
                                else:
                                    if (SL - current_price) < min_stop_distance:
                                        SL = current_price + min_stop_distance + point
                                    adjusted_risk = SL - current_price
                                    min_tp = current_price - adjusted_risk * m_min_rrr
                                    TP_default = min(TP_default, min_tp)
                                    if (current_price - TP_default) < min_stop_distance:
                                        TP_default = current_price - min_stop_distance - point

                                rrr_ok, rrr_value = check_rrr(current_price, TP_default, SL, direction, m_min_rrr)
                                if not rrr_ok:
                                    if last_state != "BAD_RRR":
                                        logger.info("[%s] %s setup rejected: RRR %.2f < %.2f", tf_name, symbol, rrr_value, m_min_rrr)
                                        logged_live_states[setup_key] = "BAD_RRR"
                                    continue

                                account_info = mt5.account_info()
                                if account_info is None:
                                    continue
                                current_balance = account_info.balance
                                if not m_use_fixed_lot:
                                    risk_amount = current_balance * (m_risk_percent / 100)
                                    risk_points = (current_price - SL) if direction == "Buy" else (SL - current_price)
                                    if risk_points <= 0:
                                        if last_state != "INVALID_RISK":
                                            logger.error("[%s] %s Invalid risk calculation.", tf_name, symbol)
                                            logged_live_states[setup_key] = "INVALID_RISK"
                                        continue
                                    lot_size = calculate_dynamic_lot_size(symbol, risk_amount, risk_points)
                                    # Anti-Gap SL Protection: widen SL if too tight, reduce lot to keep risk constant
                                    if m_anti_gap_enabled:
                                        _contract_sz = get_contract_size(symbol)
                                        SL, lot_size, _gap_widened, _, _ = enforce_anti_gap_sl(
                                            current_price, SL, direction, analysis_df, lot_size,
                                            risk_amount, _contract_sz, atr_multiplier=m_anti_gap_mult, point=point, digits=digits)
                                        if _gap_widened:
                                            # Recalculate TP to maintain min_rrr with the new wider SL
                                            adjusted_risk = abs(current_price - SL)
                                            if direction == "Buy":
                                                min_tp = current_price + adjusted_risk * m_min_rrr
                                                TP_default = max(TP_default, min_tp)
                                            else:
                                                min_tp = current_price - adjusted_risk * m_min_rrr
                                                TP_default = min(TP_default, min_tp)
                                else:
                                    lot_size = m_fixed_lot

                                # Apply slippage recovery adjustments
                                lot_size, min_rrr_adj, SL = get_recovery_adjustments(
                                    lot_size, m_min_rrr, SL, current_price, direction, is_fixed_lot=m_use_fixed_lot)
                                if min_rrr_adj != m_min_rrr:
                                    # Re-check RRR with adjusted values
                                    rrr_ok2, rrr_val2 = check_rrr(current_price, TP_default, SL, direction, min_rrr_adj)
                                    if not rrr_ok2:
                                        logger.info("[%s] %s Recovery RRR check failed: %.2f < %.2f", tf_name, symbol, rrr_val2, min_rrr_adj)
                                        continue

                                order_type = mt5.ORDER_TYPE_BUY if direction == "Buy" else mt5.ORDER_TYPE_SELL
                                order_price = tick.ask if direction == "Buy" else tick.bid
                                comment_str = format_order_comment(method, direction, tf_name, confluence_score)
                                
                                target_lots = normalize_and_split_lots(symbol, lot_size)
                                if not target_lots:
                                    if last_state != "INVALID_LOT":
                                        logger.error("[%s] %s Lot size too small after normalization.", tf_name, symbol)
                                        logged_live_states[setup_key] = "INVALID_LOT"
                                    continue
                                    
                                success_overall = False
                                for split_lot in target_lots:
                                    request = {
                                        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                                        "volume": split_lot, "type": order_type, "price": round(order_price, digits),
                                        "sl": round(SL, digits), "tp": round(TP_default, digits), "deviation": 10,
                                        "magic": get_magic_number(method), "comment": comment_str,
                                        "type_time": mt5.ORDER_TIME_GTC, "type_filling": get_filling_mode(symbol),
                                    }
                                    result = mt5.order_send(request)
                                    if result is None:
                                        continue
                                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                                        for fb in [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC]:
                                            request["type_filling"] = fb
                                            res_fb = mt5.order_send(request)
                                            if res_fb and res_fb.retcode == mt5.TRADE_RETCODE_DONE:
                                                result = res_fb
                                                break
                                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                        logger.info(">>> EXECUTED %s %s [%s] Lot: %s @ %.5f | SL: %.5f | TP: %.5f | Conf: %d [%s]", direction, symbol, tf_name, split_lot, order_price, SL, TP_default, confluence_score, ", ".join(confluence_details))
                                        success_overall = True
                                        use_ts = (trailing_methods is None) or (method in trailing_methods)
                                        if m_trail_type is None or m_trail_type == "None":
                                            use_ts = False
                                        if use_ts:
                                            risk_pts = abs(current_price - SL)
                                            threading.Thread(target=live_ict_trailing_stop,
                                                args=(result.order, symbol, risk_pts, direction, tf_value,
                                                      m_trail_type, m_trail_params, use_symbol_profiles),
                                                kwargs={'ict_method': method}, daemon=True).start()
                                    else:
                                        logger.error("[%s] %s order failed: %s", tf_name, symbol, result.comment)

                                if success_overall:
                                    add_trade_to_memory(symbol, tf_name, ob_timestamp, ob_type, method)
                                    logged_live_states[setup_key] = "EXECUTED"
                                elif last_state != "API_ERROR":
                                    logger.error("[%s] %s mt5.order_send failed for all chunks", tf_name, symbol)
                                    logged_live_states[setup_key] = "API_ERROR"

                    # === FVG-ONLY TRADES (new signal source — more trades) ===
                    if fvgs and "FVG Return" in methods_to_run:
                        for fvg in fvgs:
                            fvg_time_str = str(fvg['time'])
                            fvg_type = fvg['type']
                            base_key = f"{symbol}_{tf_name}_{fvg_time_str}_FVG"
                            
                            if is_trade_executed(symbol, tf_name, fvg_time_str, fvg_type, "FVG Return"):
                                if logged_live_states.get(base_key + "_EXEC") != "DONE":
                                    logger.info("[%s] %s %s FVG already traded previously. Skipping.", tf_name, symbol, fvg_type)
                                    logged_live_states[base_key + "_EXEC"] = "DONE"
                                continue

                            method_name = "FVG Return"
                            m_max_concurr = resolve_method_param('max_concurrent_trades', method_name, sym_params, config.MAX_CONCURRENT_TRADES)
                            if m_max_concurr > 0:
                                open_count = count_open_positions_for_method(symbol, method_name)
                                if open_count >= m_max_concurr:
                                    if logged_live_states.get(base_key + "_CONCUR") != "BLOCKED":
                                        logger.info("[%s] %s %s blocked: Max concurrent trades reached (%d/%d)", tf_name, symbol, method_name, open_count, m_max_concurr)
                                        logged_live_states[base_key + "_CONCUR"] = "BLOCKED"
                                    continue
                                    
                            limit_hit, limit_msg = is_daily_loss_limit_hit_for_method(method_name, symbol, use_symbol_profiles)
                            if limit_hit:
                                if logged_live_states.get(base_key + "_DL") != "BLOCKED":
                                    logger.warning("[%s] %s %s blocked: %s", tf_name, symbol, method_name, limit_msg)
                                    logged_live_states[base_key + "_DL"] = "BLOCKED"
                                continue

                            # FVG size filter (match backtest — parity fix)
                            m_min_fvg_size = resolve_method_param('min_fvg_size', method_name, sym_params, config.MIN_FVG_SIZE_SPREADS)
                            if m_min_fvg_size > 0 and fvg['size'] < current_spread * m_min_fvg_size:
                                continue

                            trade_dir = "Buy" if fvg_type == "bullish" else "Sell"
                            
                            # Resolve FVG Return settings per-method
                            method_name = "FVG Return"
                            m_require_bos_fvg = resolve_method_param('require_bos_fvg', method_name, sym_params, require_bos_fvg)
                            m_use_htf_filter = resolve_method_param('use_htf_filter', method_name, sym_params, use_htf_filter)
                            m_fvg_displacement_only = resolve_method_param('fvg_displacement_only', method_name, sym_params, fvg_displacement_only)
                            m_fvg_discount_premium_only = resolve_method_param('fvg_discount_premium_only', method_name, sym_params, fvg_discount_premium_only)
                            m_fvg_recent_sweep_only = resolve_method_param('fvg_recent_sweep_only', method_name, sym_params, fvg_recent_sweep_only)
                            m_fvg_sl_mode = resolve_method_param('fvg_sl_mode', method_name, sym_params, fvg_sl_mode)
                            m_min_rrr = resolve_method_param('min_rrr', method_name, sym_params, min_rrr)
                            m_use_dynamic_rrr = resolve_method_param('use_dynamic_rrr', method_name, sym_params, use_dynamic_rrr)
                            m_risk_percent = resolve_method_param('risk_percent', method_name, sym_params, initial_risk_percent)
                            m_fixed_lot = resolve_method_param('fixed_lot', method_name, sym_params, fixed_lot)
                            m_use_fixed_lot = resolve_method_param('use_fixed_lot', method_name, sym_params, risk_mode == "Fixed")
                            m_anti_gap_enabled = resolve_method_param('anti_gap_enabled', method_name, sym_params, config.ANTI_GAP_SL_ENABLED)
                            m_anti_gap_mult = resolve_method_param('anti_gap_mult', method_name, sym_params, config.ANTI_GAP_ATR_MULTIPLIER)
                            m_trail_type = resolve_method_param('trail_type', method_name, sym_params, trail_type_selected)
                            m_trail_pct = resolve_method_param('trail_pct', method_name, sym_params, trail_params_dict.get('trail_pct', 0.5) if trail_params_dict else 0.5)
                            m_trail_params = {'trail_pct': m_trail_pct}

                            # Structure filter for FVG trades
                            if structure != "neutral":
                                if (fvg_type == "bullish" and structure == "bearish") or \
                                   (fvg_type == "bearish" and structure == "bullish"):
                                    if not mss:
                                        if logged_live_states.get(base_key + "_STRUC") != "BLOCKED":
                                            logger.info("[%s] %s Found %s FVG, but Structure is %s. Skipping.", tf_name, symbol, fvg_type, structure)
                                            logged_live_states[base_key + "_STRUC"] = "BLOCKED"
                                        continue

                            # Require BoS after MSS
                            if m_require_bos_fvg:
                                if not check_bos_after_mss(swing_highs, swing_lows, trade_dir, df=analysis_df):
                                    if logged_live_states.get(base_key + "_BOS") != "BLOCKED":
                                        logger.info("[%s] %s %s FVG rejected: No BoS after MSS.", tf_name, symbol, fvg_type)
                                        logged_live_states[base_key + "_BOS"] = "BLOCKED"
                                    continue

                            m_use_ote_filter = resolve_method_param('use_ote_filter', method_name, sym_params, use_ote_filter)

                            # --- SMT Divergence Filter (FVG) ---
                            if getattr(config, 'USE_SMT_DIVERGENCE', False) and corr_dfs_dict:
                                from indicators import check_advanced_smt
                                has_smt = check_advanced_smt(analysis_df, corr_dfs_dict, len(analysis_df)-1, trade_dir.lower() + "ish", lookback=20)
                                if not has_smt:
                                    if logged_live_states.get(base_key + "_SMT") != "BLOCKED":
                                        logger.info("[%s] %s FVG rejected: No SMT divergence.", tf_name, symbol)
                                        logged_live_states[base_key + "_SMT"] = "BLOCKED"
                                    continue

                            # HTF filter for FVG trades
                            if m_use_htf_filter and higher_tf_trend != "neutral":
                                if (fvg_type == "bullish" and higher_tf_trend == "bearish") or \
                                   (fvg_type == "bearish" and higher_tf_trend == "bullish"):
                                    if logged_live_states.get(base_key + "_HTF") != "BLOCKED":
                                        logger.info("[%s] %s Found %s FVG, but HTF trend is %s. Skipping.", tf_name, symbol, fvg_type, higher_tf_trend)
                                        logged_live_states[base_key + "_HTF"] = "BLOCKED"
                                    continue

                            # OTE filter for FVG trades
                            if m_use_ote_filter:
                                if not (ote_low <= fvg['mid'] <= ote_high):
                                    if logged_live_states.get(base_key + "_OTE") != "BLOCKED":
                                        logger.info("[%s] %s %s FVG Return outside OTE zone. Skipping.", tf_name, symbol, fvg_type)
                                        logged_live_states[base_key + "_OTE"] = "BLOCKED"
                                    continue
                                    
                            # FVG Advanced SMC/ICT Filters
                            if m_fvg_displacement_only:
                                if fvg.get('body_ratio', 1.0) < 0.5 or fvg.get('atr_mult', 1.0) < 0.3:
                                    if logged_live_states.get(base_key + "_DISPLACE") != "BLOCKED":
                                        logger.info("[%s] %s Found %s FVG, but rejected by displacement filter (Body: %.2f, ATR_mult: %.2f). Skipping.",
                                                    tf_name, symbol, fvg_type, fvg.get('body_ratio', 1.0), fvg.get('atr_mult', 1.0))
                                        logged_live_states[base_key + "_DISPLACE"] = "BLOCKED"
                                    continue

                            if m_fvg_discount_premium_only:
                                recent_window = analysis_df.iloc[-50:]
                                if not recent_window.empty:
                                    swing_h = recent_window['high'].max()
                                    swing_l = recent_window['low'].min()
                                    midpoint = (swing_h + swing_l) / 2.0
                                    if trade_dir == "Buy" and fvg['mid'] >= midpoint:
                                        if logged_live_states.get(base_key + "_PD") != "BLOCKED":
                                            logger.info("[%s] %s Bullish FVG mid %.5f >= equilibrium %.5f (Premium zone). Skipping.", tf_name, symbol, fvg['mid'], midpoint)
                                            logged_live_states[base_key + "_PD"] = "BLOCKED"
                                        continue
                                    elif trade_dir == "Sell" and fvg['mid'] <= midpoint:
                                        if logged_live_states.get(base_key + "_PD") != "BLOCKED":
                                            logger.info("[%s] %s Bearish FVG mid %.5f <= equilibrium %.5f (Discount zone). Skipping.", tf_name, symbol, fvg['mid'], midpoint)
                                            logged_live_states[base_key + "_PD"] = "BLOCKED"
                                        continue

                            if m_fvg_recent_sweep_only:
                                recent_sweeps = detect_all_liquidity_sweeps(analysis_df)
                                has_recent_sweep = False
                                for sw in recent_sweeps:
                                    dist = len(analysis_df) - sw['idx']
                                    if trade_dir == "Buy" and sw['type'] == 'bullish' and dist <= 20:
                                        has_recent_sweep = True
                                        break
                                    elif trade_dir == "Sell" and sw['type'] == 'bearish' and dist <= 20:
                                        has_recent_sweep = True
                                        break
                                if not has_recent_sweep:
                                    if logged_live_states.get(base_key + "_SWEEP") != "BLOCKED":
                                        logger.info("[%s] %s %s FVG rejected: No recent liquidity sweep in last 20 bars.", tf_name, symbol, fvg_type)
                                        logged_live_states[base_key + "_SWEEP"] = "BLOCKED"
                                    continue

                            entry_price = fvg['mid']

                            # Fix 4: HTF POI nesting - only enter inside an aligned HTF POI
                            if not ict_pro.passes_htf_poi(symbol, trade_dir, entry_price):
                                if logged_live_states.get(base_key + "_HTFPOI") != "BLOCKED":
                                    logger.info("[%s] %s %s FVG skipped: entry not inside an aligned HTF POI.", tf_name, symbol, fvg_type)
                                    logged_live_states[base_key + "_HTFPOI"] = "BLOCKED"
                                continue
                            
                            # 1. Pre-calculate SL and TP at theoretical entry level
                            spread = get_spread(symbol)
                            buffer = spread * 2
                            SL_fvg = compute_fvg_sl_price(fvg, trade_dir, m_fvg_sl_mode, analysis_df, swing_highs, swing_lows, buffer, spread)
                            
                            if trade_dir == "Buy":
                                risk_pts = entry_price - SL_fvg
                            else:
                                risk_pts = SL_fvg - entry_price

                            if risk_pts <= 0:
                                continue

                            TP_fvg = entry_price + risk_pts * m_min_rrr if trade_dir == "Buy" else entry_price - risk_pts * m_min_rrr
                            if m_use_dynamic_rrr:
                                struct_tp = get_structure_tp(entry_price, trade_dir, swing_highs, swing_lows, spread)
                                if struct_tp is not None:
                                    TP_fvg = max(TP_fvg, struct_tp) if trade_dir == "Buy" else min(TP_fvg, struct_tp)

                            # Fix 1+3: Draw-on-Liquidity TP (target real liquidity, keep >= min RRR)
                            _dol = ict_pro.dol_target(symbol, trade_dir, entry_price, SL_fvg, m_min_rrr)
                            if _dol is not None:
                                TP_fvg = _dol

                            rrr_ok, rrr_value = check_rrr(entry_price, TP_fvg, SL_fvg, trade_dir, m_min_rrr)
                            if not rrr_ok:
                                continue

                            # 2. Extract FVG ML features
                            idx = fvg['bar_index']
                            disp_high = analysis_df['high'].iloc[idx]
                            disp_low = analysis_df['low'].iloc[idx]
                            disp_open = analysis_df['open'].iloc[idx]
                            disp_close = analysis_df['close'].iloc[idx]
                            disp_range = max(disp_high - disp_low, 1e-8)
                            
                            fvg_upper_wick_ratio = (disp_high - max(disp_open, disp_close)) / disp_range
                            fvg_lower_wick_ratio = (min(disp_open, disp_close) - disp_low) / disp_range
                            fvg_gap_ratio = fvg['size'] / disp_range
                            
                            from detectors import convert_time_to_ny_hour
                            fvg_hour_ny = convert_time_to_ny_hour(fvg['time'])
                            fvg_is_sb_window = 1.0 if (fvg_hour_ny == 10) or (fvg_hour_ny == 14) or (fvg_hour_ny == 3) else 0.0
                            
                            if idx + 1 < len(analysis_df):
                                c3_high = analysis_df['high'].iloc[idx + 1]
                                c3_low = analysis_df['low'].iloc[idx + 1]
                                c3_open = analysis_df['open'].iloc[idx + 1]
                                c3_close = analysis_df['close'].iloc[idx + 1]
                                c3_range = max(c3_high - c3_low, 1e-8)
                                fvg_c3_body_ratio = abs(c3_close - c3_open) / c3_range
                            else:
                                fvg_c3_body_ratio = 0.5

                            # Calculate dynamic confluence score for FVG Return
                            fvg_conf_score = 0
                            fvg_conf_details = []
                            fvg_sig_dir = "bullish" if trade_dir == "Buy" else "bearish"
                            
                            if structure == fvg_sig_dir:
                                fvg_conf_score += 2
                                fvg_conf_details.append("Structure")
                            if higher_tf_trend == fvg_sig_dir:
                                fvg_conf_score += 1
                                fvg_conf_details.append("HTF")
                            if ote_low <= entry_price <= ote_high:
                                fvg_conf_score += 1
                                fvg_conf_details.append("OTE")
                            if (trade_dir == "Buy" and bullish_sweep) or (trade_dir == "Sell" and bearish_sweep):
                                fvg_conf_score += 2
                                fvg_conf_details.append("Sweep")
                            if fvg.get('body_ratio', 1.0) >= 0.6:
                                fvg_conf_score += 1
                                fvg_conf_details.append("Displacement")
                            if fvg_conf_score == 0:
                                fvg_conf_score = 1
                                fvg_conf_details.append("FVG Imbalance")
                            fvg_conf_details_str = ", ".join(fvg_conf_details)

                            # === ML GATE ===
                            if ml_enabled and _ml_live_model['loaded']:
                                _si = mt5.symbol_info(symbol)
                                _pt = _si.point if _si else 0.00001
                                _rrr_est = abs(entry_price - TP_fvg) / max(abs(entry_price - SL_fvg), _pt) if abs(entry_price - SL_fvg) > 0 else 0

                                ml_proba, _ = ml_score_signal(
                                    trade_direction=trade_dir,
                                    method="FVG Return",
                                    confluence_score=fvg_conf_score,
                                    htf_trend=higher_tf_trend,
                                    ob_type=fvg_type,
                                    entry_price=entry_price,
                                    sl_price=SL_fvg,
                                    tp_price=TP_fvg,
                                    rrr=_rrr_est,
                                    entry_time=pd.Timestamp.now(),
                                    confluence_details=fvg_conf_details_str,
                                    fvg_size=fvg['size'],
                                    fvg_body_ratio=fvg.get('body_ratio', 1.0),
                                    fvg_upper_wick_ratio=fvg_upper_wick_ratio,
                                    fvg_lower_wick_ratio=fvg_lower_wick_ratio,
                                    fvg_gap_ratio=fvg_gap_ratio,
                                    fvg_is_sb_window=fvg_is_sb_window,
                                    fvg_c3_body_ratio=fvg_c3_body_ratio
                                )
                                if ml_proba < ml_min_confidence:
                                    if logged_live_states.get(base_key + "_ML_REJECT") != "YES":
                                        logger.info("[ML] [%s] %s FVG Return REJECTED by ML (prob=%.1f%% < %.0f%%)",
                                                   tf_name, symbol, ml_proba * 100, ml_min_confidence * 100)
                                        logged_live_states[base_key + "_ML_REJECT"] = "YES"
                                    continue
                                else:
                                    if logged_live_states.get(base_key + "_ML_APPROVE") != "YES":
                                        logger.info("[ML] [%s] %s FVG Return APPROVED by ML (prob=%.1f%% >= %.0f%%)",
                                                   tf_name, symbol, ml_proba * 100, ml_min_confidence * 100)
                                        logged_live_states[base_key + "_ML_APPROVE"] = "YES"

                            tick = mt5.symbol_info_tick(symbol)
                            if tick is None:
                                continue
                            current_price = tick.ask if trade_dir == "Buy" else tick.bid
                            
                            if (trade_dir == "Buy" and current_price > entry_price) or \
                               (trade_dir == "Sell" and current_price < entry_price):
                                fvg_setup_key = f"{base_key}_{trade_dir}"
                                # Skip if already have a pending limit or priority setup
                                if fvg_setup_key in config.pending_limit_orders or fvg_setup_key in config.priority_scan_setups:
                                    continue

                                account_info = mt5.account_info()
                                if account_info is None:
                                    continue
                                current_balance = account_info.balance
                                if not m_use_fixed_lot:
                                    risk_amount = current_balance * (m_risk_percent / 100)
                                    lot_size = calculate_dynamic_lot_size(symbol, risk_amount, risk_pts)
                                    # Anti-Gap SL Protection for FVG trades
                                    if m_anti_gap_enabled:
                                        _contract_sz = get_contract_size(symbol)
                                        SL_fvg, lot_size, _gap_widened, _, _ = enforce_anti_gap_sl(
                                            entry_price, SL_fvg, trade_dir, analysis_df, lot_size,
                                            risk_amount, _contract_sz, atr_multiplier=m_anti_gap_mult,
                                            point=mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else 0.00001)
                                        if _gap_widened:
                                            risk_pts = abs(entry_price - SL_fvg)
                                            if trade_dir == "Buy":
                                                TP_fvg = max(TP_fvg, entry_price + risk_pts * m_min_rrr)
                                            else:
                                                TP_fvg = min(TP_fvg, entry_price - risk_pts * m_min_rrr)
                                    # Fix 6+7: regime size-down + ML conviction sizing
                                    lot_size *= _regime_size_mult * ict_pro.ml_size_multiplier(locals().get('ml_proba', 0))
                                else:
                                    lot_size = m_fixed_lot

                                comment_str = format_order_comment("FVG Return", trade_dir, tf_name, fvg_conf_score)
                                use_ts = (trailing_methods is None) or ("FVG Return" in trailing_methods)
                                if m_trail_type is None or m_trail_type == "None":
                                    use_ts = False
                                trail_info = {
                                    'use_ts': use_ts,
                                    'risk_pts': risk_pts,
                                    'tf_value': tf_value,
                                    'trail_type': m_trail_type,
                                    'trail_params': m_trail_params,
                                    'use_symbol_profiles': use_symbol_profiles,
                                    'ict_method': "FVG Return"
                                }

                                # Try limit order first (Option C)
                                limit_placed = place_limit_order(
                                    symbol, trade_dir, entry_price, SL_fvg, TP_fvg,
                                    lot_size, comment_str, fvg_setup_key, trailing_info=trail_info,
                                    ob_timestamp=fvg_time_str, ob_type=fvg_type, tf_name=tf_name)

                                if limit_placed:
                                    logger.info("[%s] %s FVG LIMIT placed %s @ %.5f (Current: %.5f)",
                                               tf_name, symbol, trade_dir, entry_price, current_price)
                                    logged_live_states[base_key + "_EXEC"] = "LIMIT_PLACED"
                                else:
                                    # Limit failed → Option B: priority scan
                                    config.priority_scan_setups[fvg_setup_key] = {
                                        'symbol': symbol,
                                        'direction': trade_dir,
                                        'entry_price': entry_price,
                                        'sl_price': SL_fvg,
                                        'tp_price': TP_fvg,
                                        'lot_size': lot_size,
                                        'comment': comment_str,
                                        'trailing_info': trail_info,
                                        'created_time': time.time(),
                                        'ob_timestamp': fvg_time_str,
                                        'ob_type': fvg_type,
                                        'tf_name': tf_name,
                                    }
                                    logger.info("[%s] %s FVG Limit failed — PRIORITY SCAN %s @ %.5f (Current: %.5f)",
                                               tf_name, symbol, trade_dir, entry_price, current_price)
                                    logged_live_states[base_key + "_EXEC"] = "PRIORITY_SCAN"
                                continue

                            # Option A: Market order execution (price already past entry)
                            SL = SL_fvg
                            # Anti-Gap SL Protection for FVG market execution
                            if m_anti_gap_enabled and not m_use_fixed_lot:
                                account_info_fvg = mt5.account_info()
                                if account_info_fvg:
                                    _risk_amt_fvg = account_info_fvg.balance * (m_risk_percent / 100)
                                    _contract_sz_fvg = get_contract_size(symbol)
                                    _point_fvg = mt5.symbol_info(symbol).point if mt5.symbol_info(symbol) else 0.00001
                                    SL, _, _gap_w, _, _ = enforce_anti_gap_sl(
                                        current_price, SL, trade_dir, analysis_df, 0.01,
                                        _risk_amt_fvg, _contract_sz_fvg, atr_multiplier=m_anti_gap_mult, point=_point_fvg)

                            if trade_dir == "Buy":
                                risk_pts = current_price - SL
                            else:
                                risk_pts = SL - current_price

                            if risk_pts <= 0:
                                continue

                            struct_tp = get_structure_tp(current_price, trade_dir, swing_highs, swing_lows, spread)
                            TP_default = current_price + risk_pts * m_min_rrr if trade_dir == "Buy" else current_price - risk_pts * m_min_rrr
                            if m_use_dynamic_rrr and struct_tp is not None:
                                TP_default = max(TP_default, struct_tp) if trade_dir == "Buy" else min(TP_default, struct_tp)

                            # Fix 1+3: Draw-on-Liquidity TP
                            _dol = ict_pro.dol_target(symbol, trade_dir, current_price, SL, m_min_rrr)
                            if _dol is not None:
                                TP_default = _dol

                            rrr_ok, rrr_value = check_rrr(current_price, TP_default, SL, trade_dir, m_min_rrr)
                            if not rrr_ok:
                                if logged_live_states.get(base_key + "_BAD_RRR") != "YES":
                                    logger.info("[%s] %s FVG setup rejected: RRR %.2f < %.2f", tf_name, symbol, rrr_value, m_min_rrr)
                                    logged_live_states[base_key + "_BAD_RRR"] = "YES"
                                continue

                            account_info = mt5.account_info()
                            if account_info is None:
                                continue
                            current_balance = account_info.balance
                            current_equity = getattr(account_info, 'equity', current_balance)

                            # Dynamic Peak Equity & Drawdown Halt Check (Section 4.1 & 4.7)
                            from state import app_state
                            if current_equity > app_state.peak_equity:
                                app_state.peak_equity = current_equity

                            if app_state.use_equity_drawdown and app_state.peak_equity > 0:
                                drawdown_pct = (app_state.peak_equity - current_equity) / app_state.peak_equity * 100.0
                                max_dd = getattr(config, 'MAX_TOTAL_DRAWDOWN_PCT', 10.0)
                                if drawdown_pct >= max_dd:
                                    logger.warning("[RISK HALT] Current equity drawdown (%.2f%%) exceeds max allowed (%.2f%%). Trading paused.",
                                                   drawdown_pct, max_dd)
                                    continue

                            max_consec = getattr(config, 'MAX_CONSEC_LOSSES', 6)
                            if app_state.consec_losses >= max_consec:
                                logger.warning("[RISK HALT] Consecutive losses (%d) reached limit (%d). Trading paused.",
                                               app_state.consec_losses, max_consec)
                                continue

                            if not m_use_fixed_lot:
                                risk_amount = current_balance * (m_risk_percent / 100)
                                lot_size = calculate_dynamic_lot_size(symbol, risk_amount, risk_pts)
                            else:
                                lot_size = m_fixed_lot

                            # Apply slippage recovery adjustments. FIX: this is the MARKET
                            # order ("Option A") path, so recovery must operate on the variables
                            # the order actually uses -- current_price / SL / TP_default -- not
                            # the stale FVG-limit variables entry_price / SL_fvg / TP_fvg.
                            # Previously the widened recovery SL was written to SL_fvg and then
                            # discarded (the order was sent with the un-widened SL), and RRR was
                            # validated against prices that were never traded.
                            _pre_recovery_sl = SL
                            lot_size, min_rrr_adj, SL = get_recovery_adjustments(
                                lot_size, m_min_rrr, SL, current_price, trade_dir, is_fixed_lot=m_use_fixed_lot)
                            if min_rrr_adj != m_min_rrr or SL != _pre_recovery_sl:
                                # Recompute risk distance and TP from the (possibly widened) SL
                                risk_pts = (current_price - SL) if trade_dir == "Buy" else (SL - current_price)
                                if risk_pts <= 0:
                                    continue
                                TP_default = current_price + risk_pts * min_rrr_adj if trade_dir == "Buy" else current_price - risk_pts * min_rrr_adj
                                if m_use_dynamic_rrr and struct_tp is not None:
                                    TP_default = max(TP_default, struct_tp) if trade_dir == "Buy" else min(TP_default, struct_tp)
                                rrr_ok2, rrr_val2 = check_rrr(current_price, TP_default, SL, trade_dir, min_rrr_adj)
                                if not rrr_ok2:
                                    logger.info("[%s] %s FVG Recovery RRR check failed: %.2f < %.2f", tf_name, symbol, rrr_val2, min_rrr_adj)
                                    continue

                            # Fix 6+7: regime size-down + ML conviction sizing
                            if not m_use_fixed_lot:
                                lot_size *= _regime_size_mult * ict_pro.ml_size_multiplier(locals().get('ml_proba', 0))

                            order_type = mt5.ORDER_TYPE_BUY if trade_dir == "Buy" else mt5.ORDER_TYPE_SELL
                            order_price = tick.ask if trade_dir == "Buy" else tick.bid
                            comment_str = format_order_comment("FVG Return", trade_dir, tf_name, fvg_conf_score)
                            
                            target_lots = normalize_and_split_lots(symbol, lot_size)
                            if not target_lots:
                                if logged_live_states.get(base_key + "_INVALID_LOT") != "YES":
                                    logger.error("[%s] %s FVG Lot size too small after normalization.", tf_name, symbol)
                                    logged_live_states[base_key + "_INVALID_LOT"] = "YES"
                                continue
                                
                            success_overall = False
                            result_order = None
                            
                            for split_lot in target_lots:
                                symbol_info_fvg = mt5.symbol_info(symbol)
                                digits = symbol_info_fvg.digits if symbol_info_fvg else 5
                                request = {
                                    "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                                    "volume": split_lot, "type": order_type, "price": round(order_price, digits),
                                    "sl": round(SL, digits), "tp": round(TP_default, digits), "deviation": 10,
                                    "magic": get_magic_number("FVG Return"), "comment": comment_str,
                                    "type_time": mt5.ORDER_TIME_GTC, "type_filling": get_filling_mode(symbol),
                                }
                                result = mt5.order_send(request)
                                if result is None:
                                    continue
                                if result.retcode != mt5.TRADE_RETCODE_DONE:
                                    for fb in [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC]:
                                        request["type_filling"] = fb
                                        res_fb = mt5.order_send(request)
                                        if res_fb and res_fb.retcode == mt5.TRADE_RETCODE_DONE:
                                            result = res_fb
                                            break
                                if result.retcode == 10014:
                                    mt5.shutdown()
                                    time.sleep(1)
                                    mt5.initialize()
                                    continue
                                if result.retcode == mt5.TRADE_RETCODE_DONE:
                                    logger.info(">>> EXECUTED FVG %s %s [%s] Lot: %s @ %.5f | SL: %.5f | TP: %.5f | Conf: %d [%s]",
                                                trade_dir, symbol, tf_name, split_lot, order_price, SL, TP_default, fvg_conf_score, fvg_conf_details_str)
                                    success_overall = True
                                    use_ts = (trailing_methods is None) or ("FVG Return" in trailing_methods)
                                    if m_trail_type is None or m_trail_type == "None":
                                        use_ts = False
                                    if use_ts:
                                        risk_pts_trail = abs(current_price - SL)
                                        threading.Thread(target=live_ict_trailing_stop,
                                            args=(result.order, symbol, risk_pts_trail, trade_dir, tf_value,
                                                  m_trail_type, m_trail_params, use_symbol_profiles, "FVG Return"), daemon=True).start()
                                else:
                                    logger.error("[%s] %s FVG market order failed: %s", tf_name, symbol, result.comment)

                            if success_overall:
                                add_trade_to_memory(symbol, tf_name, fvg_time_str, fvg_type, "FVG Return")
                                logged_live_states[base_key + "_EXEC"] = "EXECUTED"
                            elif logged_live_states.get(base_key + "_API_ERROR") != "YES":
                                logger.error("[%s] %s FVG mt5.order_send failed for all chunks", tf_name, symbol)
                                logged_live_states[base_key + "_API_ERROR"] = "YES"

                    # === INDEPENDENT METHOD DETECTORS (LIVE) ===
                    real_signals = []
                    
                    # Always define these: the MMXM / ICT Model 2022 / 2025 detector
                    # blocks below reference them even when "Silver Bullet" is not among
                    # the selected methods (previously an UnboundLocalError).
                    sb_htf_bias_val = resolve_method_param('sb_require_htf_bias', 'Silver Bullet', sym_params, sb_require_htf_bias)
                    htf_bias_direction = "neutral"

                    if "Silver Bullet" in methods_to_run:
                        sb_fvg_sl_mode = resolve_method_param('fvg_sl_mode', 'Silver Bullet', sym_params, fvg_sl_mode)
                        sb_fvg_displacement_only = resolve_method_param('fvg_displacement_only', 'Silver Bullet', sym_params, fvg_displacement_only)
                        sb_fvg_discount_premium_only = resolve_method_param('fvg_discount_premium_only', 'Silver Bullet', sym_params, fvg_discount_premium_only)
                        sb_fvg_recent_sweep_only = resolve_method_param('fvg_recent_sweep_only', 'Silver Bullet', sym_params, fvg_recent_sweep_only)
                        sb_htf_bias_val = resolve_method_param('sb_require_htf_bias', 'Silver Bullet', sym_params, sb_require_htf_bias)
                        
                        htf_bias_direction = "neutral"
                        if sb_htf_bias_val:
                            try:
                                h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 50)
                                h4_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 50)
                                if h1_rates is not None and len(h1_rates) > 0 and h4_rates is not None and len(h4_rates) > 0:
                                    h1_df = pd.DataFrame(h1_rates)
                                    h4_df = pd.DataFrame(h4_rates)
                                    h1_df['time'] = pd.to_datetime(h1_df['time'], unit='s')
                                    h4_df['time'] = pd.to_datetime(h4_df['time'], unit='s')
                                    h1_df.set_index('time', inplace=True)
                                    h4_df.set_index('time', inplace=True)
                                    h1_struct, _, _, _ = detect_market_structure(h1_df, swing_lookback=5)
                                    h4_struct, _, _, _ = detect_market_structure(h4_df, swing_lookback=5)
                                    if h1_struct == h4_struct and h1_struct != "neutral":
                                        htf_bias_direction = h1_struct
                            except Exception as e:
                                logger.error(f"Failed to fetch HTF data live for {symbol}: {e}")

                        real_signals.extend(detect_silver_bullet_signals(
                            analysis_df, broker_time, fvgs, structure, mss, spread=current_spread, broker_tz_offset=0,
                            fvg_sl_mode=sb_fvg_sl_mode,
                            fvg_displacement_only=sb_fvg_displacement_only,
                            fvg_discount_premium_only=sb_fvg_discount_premium_only,
                            fvg_recent_sweep_only=sb_fvg_recent_sweep_only,
                            sb_require_htf_bias=sb_htf_bias_val,
                            swing_highs=swing_highs, swing_lows=swing_lows,
                            htf_bias_direction=htf_bias_direction
                        ))
                    if "Judas Swing" in methods_to_run:
                        real_signals.extend(detect_judas_swing(
                            analysis_df, broker_time, structure, spread=current_spread, window=15, broker_tz_offset=0,
                            swing_highs=swing_highs, swing_lows=swing_lows, atr_val=current_atr
                        ))
                    if "Breaker Block" in methods_to_run:
                        real_signals.extend(detect_breaker_blocks(analysis_df, all_obs, spread=current_spread, atr_val=current_atr))
                    if "IOFR" in methods_to_run:
                        real_signals.extend(detect_iofr_signals(
                            analysis_df, spread=current_spread,
                            swing_highs=swing_highs, swing_lows=swing_lows, atr_val=current_atr
                        ))
                    if "MMXM" in methods_to_run:
                        real_signals.extend(detect_mmxm_signals(
                            analysis_df, broker_time, fvgs, structure, mss, spread=current_spread,
                            swing_highs=swing_highs, swing_lows=swing_lows, atr_val=current_atr,
                            use_mtf_alignment=sb_htf_bias_val, htf_bias_direction=htf_bias_direction
                        ))
                    if "ICT Model 2022" in methods_to_run:
                        real_signals.extend(detect_ict_model_signals(
                            analysis_df, broker_time, fvgs, structure, mss, spread=current_spread, model_version="2022",
                            swing_highs=swing_highs, swing_lows=swing_lows, atr_val=current_atr,
                            use_mtf_alignment=sb_htf_bias_val, htf_bias_direction=htf_bias_direction
                        ))
                    if "ICT Model 2025" in methods_to_run:
                        real_signals.extend(detect_ict_model_signals(
                            analysis_df, broker_time, fvgs, structure, mss, spread=current_spread, model_version="2025",
                            swing_highs=swing_highs, swing_lows=swing_lows, atr_val=current_atr,
                            use_mtf_alignment=sb_htf_bias_val, htf_bias_direction=htf_bias_direction
                        ))

                    # Fix 7: rank candidate setups so the best ideas claim limited concurrency slots
                    for _s in real_signals:
                        try:
                            _sd = "bullish" if _s.get('trade_dir') == "Buy" else "bearish"
                            _s['_rank'] = (2 if structure == _sd else 0) + (1 if higher_tf_trend == _sd else 0)
                        except Exception:
                            _s['_rank'] = 0
                    real_signals = ict_pro.rank_setups(real_signals, prob_key='_rank', conf_key='_rank')

                    for sig in real_signals:
                        method = sig['method']
                        trade_dir = sig['trade_dir']
                        entry_price = sig['entry']
                        SL = sig['sl']

                        # Fix 4: HTF POI nesting - only take entries inside an aligned HTF POI
                        if not ict_pro.passes_htf_poi(symbol, trade_dir, entry_price):
                            continue

                        sig_time = sig.get('fvg', {}).get('time', broker_time) if isinstance(sig.get('fvg'), dict) else broker_time
                        setup_key = f"{symbol}_{tf_name}_{sig_time}_{method}"
                        if is_trade_executed(symbol, tf_name, str(sig_time), sig['type'], method):
                            continue

                        last_state = logged_live_states.get(setup_key, "NEW")

                        m_max_concurr = resolve_method_param('max_concurrent_trades', method, sym_params, config.MAX_CONCURRENT_TRADES)
                        if m_max_concurr > 0:
                            open_count = count_open_positions_for_method(symbol, method)
                            if open_count >= m_max_concurr:
                                if last_state != "CONCUR_LIMIT":
                                    logger.info("[%s] %s %s blocked: Max concurrent trades reached (%d/%d)", tf_name, symbol, method, open_count, m_max_concurr)
                                    logged_live_states[setup_key] = "CONCUR_LIMIT"
                                continue
                                
                        limit_hit, limit_msg = is_daily_loss_limit_hit_for_method(method, symbol, use_symbol_profiles)
                        if limit_hit:
                            if last_state != "DAILY_LIMIT":
                                logger.warning("[%s] %s %s blocked: %s", tf_name, symbol, method, limit_msg)
                                logged_live_states[setup_key] = "DAILY_LIMIT"
                            continue

                        m_use_htf_filter = resolve_method_param('use_htf_filter', method, sym_params, use_htf_filter)
                        m_use_ote_filter = resolve_method_param('use_ote_filter', method, sym_params, use_ote_filter)
                        m_min_rrr = resolve_method_param('min_rrr', method, sym_params, min_rrr)

                        if m_use_ote_filter:
                            if not (ote_low <= entry_price <= ote_high):
                                if last_state != "OTE_LIMIT":
                                    logger.info("[%s] %s %s blocked: Outside OTE zone.", tf_name, symbol, method)
                                    logged_live_states[setup_key] = "OTE_LIMIT"
                                continue
                        m_use_dynamic_rrr = resolve_method_param('use_dynamic_rrr', method, sym_params, use_dynamic_rrr)
                        m_risk_percent = resolve_method_param('risk_percent', method, sym_params, initial_risk_percent)
                        m_fixed_lot = resolve_method_param('fixed_lot', method, sym_params, fixed_lot)
                        m_use_fixed_lot = resolve_method_param('use_fixed_lot', method, sym_params, risk_mode == "Fixed")
                        m_trail_type = resolve_method_param('trail_type', method, sym_params, trail_type_selected)
                        m_trail_pct = resolve_method_param('trail_pct', method, sym_params, trail_params_dict.get('trail_pct', 0.5) if trail_params_dict else 0.5)
                        m_trail_params = {'trail_pct': m_trail_pct}

                        # === ML GATE: Live FVG and Independent Strategy filtering ===
                        if ml_enabled and _ml_live_model['loaded']:
                            fvg_dict = sig.get('fvg', {}) if isinstance(sig.get('fvg'), dict) else {}
                            fvg_upper_wick_ratio = 0.1
                            fvg_lower_wick_ratio = 0.1
                            fvg_gap_ratio = 0.3
                            fvg_is_sb_window = 0.0
                            fvg_c3_body_ratio = 0.5
                            
                            if fvg_dict and 'bar_index' in fvg_dict:
                                idx = fvg_dict['bar_index']
                                if idx < len(analysis_df):
                                    disp_high = analysis_df['high'].iloc[idx]
                                    disp_low = analysis_df['low'].iloc[idx]
                                    disp_open = analysis_df['open'].iloc[idx]
                                    disp_close = analysis_df['close'].iloc[idx]
                                    disp_range = max(disp_high - disp_low, 1e-8)
                                    
                                    fvg_upper_wick_ratio = (disp_high - max(disp_open, disp_close)) / disp_range
                                    fvg_lower_wick_ratio = (min(disp_open, disp_close) - disp_low) / disp_range
                                    fvg_gap_ratio = fvg_dict['size'] / disp_range
                                    
                                    from detectors import convert_time_to_ny_hour
                                    fvg_hour_ny = convert_time_to_ny_hour(fvg_dict['time'])
                                    fvg_is_sb_window = 1.0 if (fvg_hour_ny == 10) or (fvg_hour_ny == 14) or (fvg_hour_ny == 3) else 0.0
                                    
                                    if idx + 1 < len(analysis_df):
                                        c3_high = analysis_df['high'].iloc[idx + 1]
                                        c3_low = analysis_df['low'].iloc[idx + 1]
                                        c3_open = analysis_df['open'].iloc[idx + 1]
                                        c3_close = analysis_df['close'].iloc[idx + 1]
                                        c3_range = max(c3_high - c3_low, 1e-8)
                                        fvg_c3_body_ratio = abs(c3_close - c3_open) / c3_range
                                        
                            _si = mt5.symbol_info(symbol)
                            _pt = _si.point if _si else 0.00001
                            _tp_est = entry_price + abs(entry_price - SL) * m_min_rrr if trade_dir == "Buy" else entry_price - abs(entry_price - SL) * m_min_rrr
                            _sl_est = SL
                            _rrr_est = abs(entry_price - _tp_est) / max(abs(entry_price - _sl_est), _pt) if abs(entry_price - _sl_est) > 0 else 0
                            
                            fvg_conf_score = 0
                            sig_dir = "bullish" if trade_dir == "Buy" else "bearish"
                            if structure == sig_dir: fvg_conf_score += 2
                            if higher_tf_trend == sig_dir: fvg_conf_score += 1
                            if (trade_dir == "Buy" and bullish_sweep) or (trade_dir == "Sell" and bearish_sweep): fvg_conf_score += 2
                            if fvg_conf_score == 0: fvg_conf_score = 2
                            
                            ml_proba, _ = ml_score_signal(
                                trade_direction=trade_dir,
                                method=method,
                                confluence_score=fvg_conf_score,
                                htf_trend=higher_tf_trend,
                                ob_type=sig['type'],
                                entry_price=entry_price,
                                sl_price=_sl_est,
                                tp_price=_tp_est,
                                rrr=_rrr_est,
                                entry_time=pd.Timestamp.now(),
                                confluence_details=method,
                                fvg_size=fvg_dict.get('size', 0.0),
                                fvg_body_ratio=fvg_dict.get('body_ratio', 1.0),
                                fvg_upper_wick_ratio=fvg_upper_wick_ratio,
                                fvg_lower_wick_ratio=fvg_lower_wick_ratio,
                                fvg_gap_ratio=fvg_gap_ratio,
                                fvg_is_sb_window=fvg_is_sb_window,
                                fvg_c3_body_ratio=fvg_c3_body_ratio
                            )
                            if ml_proba < ml_min_confidence:
                                if last_state != "ML_REJECT":
                                    logger.info("[ML] %s %s %s REJECTED by ML (prob=%.1f%% < %.0f%%)",
                                               tf_name, symbol, method, ml_proba * 100, ml_min_confidence * 100)
                                    logged_live_states[setup_key] = "ML_REJECT"
                                continue
                            else:
                                if last_state == "NEW" or last_state == "ML_REJECT":
                                    logger.info("[ML] %s %s %s APPROVED by ML (prob=%.1f%% >= %.0f%%)",
                                               tf_name, symbol, method, ml_proba * 100, ml_min_confidence * 100)

                        tick = mt5.symbol_info_tick(symbol)
                        if tick is None:
                            continue
                        current_price = tick.ask if trade_dir == "Buy" else tick.bid
                        
                        if m_use_htf_filter and higher_tf_trend != "neutral":
                            if (trade_dir == "Buy" and higher_tf_trend == "bearish") or \
                               (trade_dir == "Sell" and higher_tf_trend == "bullish"):
                                if last_state != "HTF_BLOCKED":
                                    logger.info("[%s] %s %s blocked by HTF trend.", tf_name, symbol, method)
                                    logged_live_states[setup_key] = "HTF_BLOCKED"
                                continue

                        # Anti-Gap SL Protection (parity with backtester independent-signal path)
                        m_anti_gap_enabled = resolve_method_param('anti_gap_enabled', method, sym_params, config.ANTI_GAP_SL_ENABLED)
                        m_anti_gap_mult = resolve_method_param('anti_gap_mult', method, sym_params, config.ANTI_GAP_ATR_MULTIPLIER)
                        if m_anti_gap_enabled and current_atr > 0:
                            _req_dist = current_atr * m_anti_gap_mult
                            if abs(entry_price - SL) < _req_dist:
                                SL = (entry_price - _req_dist) if trade_dir == "Buy" else (entry_price + _req_dist)
                        risk_pts = abs(entry_price - SL)
                        if risk_pts <= 0:
                            continue
                        
                        struct_tp = get_structure_tp(entry_price, trade_dir, swing_highs, swing_lows, current_spread)
                        TP_default = entry_price + risk_pts * m_min_rrr if trade_dir == "Buy" else entry_price - risk_pts * m_min_rrr
                        if m_use_dynamic_rrr and struct_tp is not None:
                            TP_default = max(TP_default, struct_tp) if trade_dir == "Buy" else min(TP_default, struct_tp)
                        # Fix 1+3: Draw-on-Liquidity TP
                        _dol = ict_pro.dol_target(symbol, trade_dir, entry_price, SL, m_min_rrr)
                        if _dol is not None:
                            TP_default = _dol
                        rrr_ok, rrr_value = check_rrr(entry_price, TP_default, SL, trade_dir, m_min_rrr)
                        if not rrr_ok:
                            if last_state != "BAD_RRR":
                                logger.info("[%s] %s %s rejected: RRR %.2f < %.2f", tf_name, symbol, method, rrr_value, m_min_rrr)
                                logged_live_states[setup_key] = "BAD_RRR"
                            continue

                        # Check if limit order is needed (hasn't reached entry)
                        if (trade_dir == "Buy" and current_price > entry_price) or \
                           (trade_dir == "Sell" and current_price < entry_price):
                            if setup_key in config.pending_limit_orders or setup_key in config.priority_scan_setups:
                                continue
                            
                            account_info = mt5.account_info()
                            if not account_info:
                                continue
                            
                            if not m_use_fixed_lot:
                                risk_amt = account_info.balance * (m_risk_percent / 100)
                                lot_size = calculate_dynamic_lot_size(symbol, risk_amt, risk_pts)
                            else:
                                lot_size = m_fixed_lot

                            # Apply slippage recovery adjustments
                            lot_size, min_rrr_adj, SL = get_recovery_adjustments(
                                lot_size, m_min_rrr, SL, entry_price, trade_dir, is_fixed_lot=m_use_fixed_lot)
                            if min_rrr_adj != m_min_rrr:
                                # Re-check RRR with adjusted values
                                rrr_ok2, rrr_val2 = check_rrr(entry_price, TP_default, SL, trade_dir, min_rrr_adj)
                                if not rrr_ok2:
                                    logger.info("[%s] %s %s Recovery RRR check failed (Limit): %.2f < %.2f", tf_name, symbol, method, rrr_val2, min_rrr_adj)
                                    continue

                            # Fix 6+7: regime size-down + ML conviction sizing
                            if not m_use_fixed_lot:
                                lot_size *= _regime_size_mult * ict_pro.ml_size_multiplier(locals().get('ml_proba', 0))

                            _use_ts_ind = (trailing_methods is None) or (method in trailing_methods)
                            if m_trail_type is None or m_trail_type == "None":
                                _use_ts_ind = False
                            trail_info = {
                                'use_ts': _use_ts_ind,
                                'risk_pts': risk_pts,
                                'trail_type': m_trail_type,
                                'trail_params': m_trail_params,
                                'tf_value': tf_value,
                                'use_symbol_profiles': use_symbol_profiles,
                                'ict_method': method
                            }
                            ind_comment_str = format_order_comment(method, trade_dir, tf_name)
                            limit_placed = place_limit_order(
                                symbol, trade_dir, entry_price, SL, TP_default,
                                lot_size, ind_comment_str, setup_key, trailing_info=trail_info,
                                ob_timestamp=str(sig_time), ob_type=sig['type'], tf_name=tf_name)
                            
                            if limit_placed:
                                logger.info("[%s] %s %s LIMIT placed %s @ %.5f", tf_name, symbol, method, trade_dir, entry_price)
                                logged_live_states[setup_key] = "LIMIT_PLACED"
                            else:
                                config.priority_scan_setups[setup_key] = {
                                    'symbol': symbol, 'direction': trade_dir,
                                    'entry_price': entry_price, 'sl_price': SL, 'tp_price': TP_default,
                                    'lot_size': lot_size, 'comment': ind_comment_str,
                                    'trailing_info': trail_info, 'created_time': time.time(),
                                    'ob_timestamp': str(sig_time), 'ob_type': sig['type'], 'tf_name': tf_name,
                                }
                                logger.info("[%s] %s %s PRIORITY SCAN queued", tf_name, symbol, method)
                                logged_live_states[setup_key] = "PRIORITY_SCAN"
                            continue

                        # Execute market order
                        market_entry = current_price
                        risk_pts = abs(market_entry - SL)
                        if risk_pts <= 0:
                            continue
                        struct_tp = get_structure_tp(market_entry, trade_dir, swing_highs, swing_lows, current_spread)
                        TP_default = market_entry + risk_pts * m_min_rrr if trade_dir == "Buy" else market_entry - risk_pts * m_min_rrr
                        if m_use_dynamic_rrr and struct_tp is not None:
                            TP_default = max(TP_default, struct_tp) if trade_dir == "Buy" else min(TP_default, struct_tp)
                        # Fix 1+3: Draw-on-Liquidity TP
                        _dol = ict_pro.dol_target(symbol, trade_dir, market_entry, SL, m_min_rrr)
                        if _dol is not None:
                            TP_default = _dol
                        rrr_ok, rrr_value = check_rrr(market_entry, TP_default, SL, trade_dir, m_min_rrr)
                        if not rrr_ok:
                            if last_state != "BAD_RRR_MARKET":
                                logger.info("[%s] %s %s market entry rejected: RRR %.2f < %.2f", tf_name, symbol, method, rrr_value, m_min_rrr)
                                logged_live_states[setup_key] = "BAD_RRR_MARKET"
                            continue

                        account_info = mt5.account_info()
                        if not account_info:
                            continue
                        if not m_use_fixed_lot:
                            risk_amt = account_info.balance * (m_risk_percent / 100)
                            lot_size = calculate_dynamic_lot_size(symbol, risk_amt, risk_pts)
                        else:
                            lot_size = m_fixed_lot

                        # Apply slippage recovery adjustments
                        lot_size, min_rrr_adj, SL = get_recovery_adjustments(
                            lot_size, m_min_rrr, SL, market_entry, trade_dir, is_fixed_lot=m_use_fixed_lot)
                        risk_pts = abs(market_entry - SL)  # Recalculate with adjusted SL
                        if min_rrr_adj != m_min_rrr:
                            # Re-check RRR with adjusted values
                            rrr_ok2, rrr_val2 = check_rrr(market_entry, TP_default, SL, trade_dir, min_rrr_adj)
                            if not rrr_ok2:
                                logger.info("[%s] %s %s Recovery RRR check failed: %.2f < %.2f", tf_name, symbol, method, rrr_val2, min_rrr_adj)
                                continue

                        # Fix 6+7: regime size-down + ML conviction sizing
                        if not m_use_fixed_lot:
                            lot_size *= _regime_size_mult * ict_pro.ml_size_multiplier(locals().get('ml_proba', 0))

                        target_lots = normalize_and_split_lots(symbol, lot_size)
                        if not target_lots:
                            continue

                        success_overall = False
                        result_order = None
                        
                        tick = mt5.symbol_info_tick(symbol)
                        if tick is None:
                            continue
                        order_price = tick.ask if trade_dir == "Buy" else tick.bid
                        
                        symbol_info_sb = mt5.symbol_info(symbol)
                        digits = symbol_info_sb.digits if symbol_info_sb else 5
                        order_type = mt5.ORDER_TYPE_BUY if trade_dir == "Buy" else mt5.ORDER_TYPE_SELL

                        for split_lot in target_lots:
                            request = {
                                "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                                "volume": split_lot, "type": order_type, "price": round(order_price, digits),
                                "sl": round(SL, digits), "tp": round(TP_default, digits), "deviation": 10,
                                "magic": get_magic_number(method), "comment": format_order_comment(method, trade_dir, tf_name),
                                "type_time": mt5.ORDER_TIME_GTC, "type_filling": get_filling_mode(symbol),
                            }
                            res = mt5.order_send(request)
                            if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                                for fb in [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC]:
                                    request["type_filling"] = fb
                                    res_fb = mt5.order_send(request)
                                    if res_fb and res_fb.retcode == mt5.TRADE_RETCODE_DONE:
                                        res = res_fb
                                        break
                            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                success_overall = True
                                logger.info(">>> EXECUTED %s %s %s [%s] Lot: %s @ %.5f", method, trade_dir, symbol, tf_name, split_lot, order_price)
                                use_ts = (trailing_methods is None) or (method in trailing_methods)
                                if m_trail_type is None or m_trail_type == "None":
                                    use_ts = False
                                if use_ts:
                                    threading.Thread(target=live_ict_trailing_stop,
                                        args=(res.order, symbol, risk_pts, trade_dir, tf_value,
                                              m_trail_type, m_trail_params, use_symbol_profiles),
                                        kwargs={'ict_method': method}, daemon=True).start()

                        if success_overall:
                            add_trade_to_memory(symbol, tf_name, str(sig_time), sig['type'], method)
                            logged_live_states[setup_key] = "EXECUTED"
                        else:
                            if logged_live_states.get(setup_key) != "API_ERROR":
                                logger.error("[%s] %s %s market order failed for all chunks", tf_name, symbol, method)
                                logged_live_states[setup_key] = "API_ERROR"

            except Exception as e:
                logger.error("Exception in live trading for %s: %s", symbol, e)
                logger.error(traceback.format_exc())
        config.current_activity = "Waiting for next check..."
        # Reduced from 5s to 3s for faster priority scan responsiveness
        for i in range(3):
            if config.stop_event.is_set():
                break
            # Quick priority scan between main scans (Option B responsiveness)
            if config.priority_scan_setups:
                scan_priority_setups()
            if config.pending_limit_orders:
                check_pending_limit_orders()
            config.current_activity = f"Idle - Next scan in {3-i}s | {len(config.pending_limit_orders)} limits | {len(config.priority_scan_setups)} priority"
            time.sleep(1)
