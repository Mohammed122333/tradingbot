"""Order management — pending orders, limit placement, trailing stop.

Extracted from live_scanner.py to reduce file size.
"""
import logging
import time
import datetime
import threading

try:
    import MetaTrader5 as mt5
except ImportError:
    import tests.MetaTrader5 as mt5
import pandas as pd

import config
from utils import normalize_and_split_lots, get_symbol_profile_overrides
from live_helpers import check_and_clear_market_closed_flag, get_magic_number, log_broker_error
from trade_memory import add_trade_to_memory
from error_boundary import error_boundary
import humanize

logger = logging.getLogger()


def cancel_pending_orders_for_method(symbol, method):
    for setup_key, info in list(config.pending_limit_orders.items()):
        if info.get('symbol') == symbol:
            order_method = info.get('ict_method') or info.get('trailing_info', {}).get('ict_method')
            if order_method == method:
                if check_and_clear_market_closed_flag(symbol, info):
                    logger.info("Market closed for %s, deferring cancellation of %s", symbol, setup_key)
                    continue
                try:
                    tickets = info.get('tickets') or ([info.get('ticket')] if info.get('ticket') else [])
                    for ticket in tickets:
                        result = mt5.order_send({
                            "action": mt5.TRADE_ACTION_REMOVE,
                            "order": ticket,
                        })
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"Cancelled pending order {ticket} for {symbol} {method}")
                except Exception as e:
                    logger.error(f"Error cancelling order: {e}")
                config.pending_limit_orders.pop(setup_key, None)


def get_filling_mode(symbol):
    info = mt5.symbol_info(symbol)
    if info is not None and hasattr(info, 'filling_mode'):
        mode = info.filling_mode
        if mode & 4:  # SYMBOL_FILLING_RETURN
            return mt5.ORDER_FILLING_RETURN
        if mode & 1:  # SYMBOL_FILLING_FOK
            return mt5.ORDER_FILLING_FOK
        if mode & 2:  # SYMBOL_FILLING_IOC
            return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def place_limit_order(symbol, direction, entry_price, sl_price, tp_price, lot_size, comment_str,
                      arg8=None, arg9=None, order_type=None, trailing_info=None, ict_method=None, setup_key=None,
                      ob_timestamp=None, ob_type=None, tf_name=None):
    if setup_key is None:
        if arg9 is not None:
            setup_key = arg9
            if ict_method is None:
                ict_method = arg8
        elif arg8 is not None:
            setup_key = arg8

    if trailing_info is None and isinstance(ict_method, dict):
        trailing_info = ict_method
        ict_method = trailing_info.get('ict_method')

    if ict_method is None and trailing_info:
        ict_method = trailing_info.get('ict_method')
    if ict_method is None:
        ict_method = "FVG Return"

    if setup_key is None:
        setup_key = f"{symbol}_{direction}_{entry_price}"

    if order_type is None:
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "Buy" else mt5.ORDER_TYPE_SELL_LIMIT

    sym_info = mt5.symbol_info(symbol)
    digits = sym_info.digits if sym_info and hasattr(sym_info, 'digits') else 5
    point = sym_info.point if sym_info and hasattr(sym_info, 'point') else 0.00001
    entry_price = round(entry_price, digits)
    if sl_price > 0:
        sl_price = round(sl_price, digits)
    if tp_price > 0:
        tp_price = round(tp_price, digits)

    # Prevent duplicate limit order placement on bot restart / re-scan
    want_magic = get_magic_number(ict_method)
    dup_tol = max(point * 10, entry_price * 1e-5)
    existing_mt5_orders = mt5.orders_get(symbol=symbol)
    if existing_mt5_orders:
        for ex_ord in existing_mt5_orders:
            ex_dir = "Buy" if ex_ord.type == mt5.ORDER_TYPE_BUY_LIMIT else ("Sell" if ex_ord.type == mt5.ORDER_TYPE_SELL_LIMIT else None)
            if (ex_dir == direction
                and getattr(ex_ord, 'magic', None) == want_magic
                and abs(ex_ord.price_open - entry_price) <= dup_tol):
                logger.info(f"Pending limit order already active on MT5 (Ticket {ex_ord.ticket} {direction} @ {ex_ord.price_open:.5f}). Tracking {setup_key}.")
                if setup_key not in config.pending_limit_orders:
                    t_info = trailing_info if trailing_info else {
                        'ict_method': ict_method,
                        'entry_price': entry_price,
                        'initial_risk': abs(entry_price - sl_price),
                        'breakeven_activated': False,
                        'breakeven_set': False,
                    }
                    config.pending_limit_orders[setup_key] = {
                        'ticket': ex_ord.ticket,
                        'tickets': [ex_ord.ticket],
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': ex_ord.price_open,
                        'sl_price': ex_ord.sl,
                        'tp_price': ex_ord.tp,
                        'lot_size': ex_ord.volume_initial,
                        'ict_method': ict_method,
                        'trailing_info': t_info,
                        'placed_time': time.time(),
                        'ob_timestamp': ob_timestamp,
                        'ob_type': ob_type,
                        'tf_name': tf_name,
                    }
                return True
    target_lots = normalize_and_split_lots(symbol, lot_size)
    if not target_lots:
        logger.warning(f"Lot size {lot_size:.4f} below min for {symbol}, skipping limit order")
        return None

    filling_type = get_filling_mode(symbol)
    placed_orders = []
    last_result = None

    humanize.entry_delay()

    for chunk_lot in target_lots:
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": chunk_lot,
            "type": order_type,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10,
            "magic": get_magic_number(ict_method),
            "comment": comment_str,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        result = mt5.order_send(request)
        last_result = result
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            placed_orders.append(result.order)
        else:
            fallback_fillings = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC]
            for fb in fallback_fillings:
                if fb == filling_type:
                    continue
                request["type_filling"] = fb
                result = mt5.order_send(request)
                last_result = result
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    placed_orders.append(result.order)
                    break

    if placed_orders:
        t_info = trailing_info if trailing_info else {
            'ict_method': ict_method,
            'entry_price': entry_price,
            'initial_risk': abs(entry_price - sl_price),
            'breakeven_activated': False,
            'breakeven_set': False,
        }
        total_placed_lot = round(sum(target_lots[:len(placed_orders)]), 8)
        config.pending_limit_orders[setup_key] = {
            'ticket': placed_orders[0],
            'tickets': placed_orders,
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'lot_size': total_placed_lot,
            'ict_method': ict_method,
            'trailing_info': t_info,
            'placed_time': time.time(),
            'ob_timestamp': ob_timestamp,
            'ob_type': ob_type,
            'tf_name': tf_name,
        }
        logger.info(f"Limit order placed: {symbol} {direction} at {entry_price} lot={total_placed_lot:.4f} ({len(placed_orders)} order(s))")
        return last_result
    else:
        error_comment = last_result.comment if last_result else "None"
        retcode = last_result.retcode if last_result else None
        log_broker_error(symbol, "Limit Order Failed", error_comment, retcode=retcode,
                         details=f"{direction} @ {entry_price}, SL: {sl_price}, TP: {tp_price}, Lot: {lot_size}")
        return None


def check_pending_limit_orders():
    if not config.pending_limit_orders:
        return

    expiry_hours = float(getattr(config, 'PENDING_LIMIT_EXPIRY_HOURS', 2.0))
    expiry_sec = expiry_hours * 3600.0 if expiry_hours > 0 else 0
    now = time.time()

    for setup_key, info in list(config.pending_limit_orders.items()):
        try:
            symbol = info.get('symbol', '')
            placed_time = info.get('placed_time', now)
            tickets = info.get('tickets') or ([info.get('ticket')] if info.get('ticket') else [])

            # 1. Expiration check: cancel limit order in MT5 if older than expiry_hours
            if expiry_sec > 0 and (now - placed_time) >= expiry_sec:
                age_hours = (now - placed_time) / 3600.0
                for ticket in tickets:
                    try:
                        res = mt5.order_send({
                            "action": mt5.TRADE_ACTION_REMOVE,
                            "order": ticket,
                        })
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"Cancelled expired limit order {ticket} for {symbol} after {age_hours:.1f}h")
                        else:
                            comment = res.comment if res else "unknown error"
                            logger.warning(f"Removing expired tracking for order {ticket} ({comment})")
                    except Exception as e_rem:
                        logger.error(f"Error sending cancellation for limit order {ticket}: {e_rem}")

                ob_ts = info.get('ob_timestamp')
                ob_type = info.get('ob_type')
                tf_n = info.get('tf_name')
                ict_m = info.get('ict_method', 'FVG Return')
                if not (ob_ts and ob_type and tf_n):
                    parts = setup_key.split('_')
                    if len(parts) >= 4:
                        symbol = symbol or parts[0]
                        tf_n = tf_n or parts[1]
                        ob_ts = ob_ts or parts[2]
                        ob_type = ob_type or (parts[3] if len(parts) >= 5 else info.get('direction', 'Buy'))

                if ob_ts and ob_type and tf_n and symbol:
                    add_trade_to_memory(symbol, tf_n, ob_ts, ob_type, ict_m)
                    logger.info(f"Marked expired setup {setup_key} in trade memory so it will not be re-entered.")

                config.pending_limit_orders.pop(setup_key, None)
                continue
            # 2. Status check: check if filled or removed in MT5
            orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
            if orders is not None:
                active_ids = {o.ticket for o in orders}
                if tickets and not any(t in active_ids for t in tickets):
                    # Check if the limit order was filled into active positions
                    # Retry up to 3 times with short delay to handle MT5 API propagation lag
                    filled_positions = []
                    for _retry in range(3):
                        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
                        if positions:
                            for p in positions:
                                if p.ticket in tickets or (hasattr(p, 'order') and p.order in tickets):
                                    filled_positions.append(p)
                        if filled_positions:
                            break
                        time.sleep(0.5)

                    t_info = info.get('trailing_info') or {}
                    ict_method = info.get('ict_method') or t_info.get('ict_method') or "FVG Return"
                    trail_type = t_info.get('trail_type')
                    trail_params = t_info.get('trail_params') or {}
                    use_symbol_profiles = t_info.get('use_symbol_profiles', False)
                    entry_price = info.get('entry_price', 0.0)
                    sl_price = info.get('sl_price', 0.0)
                    risk_pts = abs(entry_price - sl_price)

                    if filled_positions:
                        ob_timestamp = info.get('ob_timestamp')
                        ob_type = info.get('ob_type')
                        tf_name = info.get('tf_name')
                        if ob_timestamp and ob_type and tf_name:
                            add_trade_to_memory(symbol, tf_name, ob_timestamp, ob_type, ict_method)

                        if trail_type and trail_type != "None":
                            for filled_pos in filled_positions:
                                pos_ticket = filled_pos.ticket
                                logger.info(f"Limit order filled for {setup_key}! Spawning trailing stop thread for position {pos_ticket} ({symbol}).")
                                threading.Thread(
                                    target=live_ict_trailing_stop,
                                    args=(pos_ticket, symbol, risk_pts, info.get('direction', 'Buy'), mt5.TIMEFRAME_M15,
                                          trail_type, dict(trail_params), use_symbol_profiles),
                                    kwargs={'ict_method': ict_method},
                                    daemon=True
                                ).start()
                        else:
                            logger.info(f"Limit order {tickets} filled for {setup_key}. No trailing stop configured.")
                    else:
                        logger.info(f"Limit order {tickets} removed for {setup_key}")
                        ob_ts = info.get('ob_timestamp')
                        ob_type = info.get('ob_type')
                        tf_n = info.get('tf_name')
                        if not (ob_ts and ob_type and tf_n):
                            parts = setup_key.split('_')
                            if len(parts) >= 4:
                                symbol = symbol or parts[0]
                                tf_n = tf_n or parts[1]
                                ob_ts = ob_ts or parts[2]
                                ob_type = ob_type or (parts[3] if len(parts) >= 5 else info.get('direction', 'Buy'))
                        if ob_ts and ob_type and tf_n and symbol:
                            add_trade_to_memory(symbol, tf_n, ob_ts, ob_type, ict_method)

                    config.pending_limit_orders.pop(setup_key, None)
        except Exception as e:
            logger.error(f"Error checking pending orders: {e}")


def scan_priority_setups():
    if not config.priority_scan_setups:
        return

    expiry_hours = float(getattr(config, 'PENDING_LIMIT_EXPIRY_HOURS', 2.0))
    expiry_sec = expiry_hours * 3600.0 if expiry_hours > 0 else 0
    now = time.time()

    for key in list(config.priority_scan_setups.keys()):
        try:
            setup = config.priority_scan_setups.get(key, {})
            if not setup:
                continue

            created_time = setup.get('created_time') or setup.get('placed_time') or now
            if expiry_sec > 0 and (now - created_time) >= expiry_sec:
                age_hours = (now - created_time) / 3600.0
                logger.info(f"Priority setup {key} expired after {age_hours:.1f}h. Removed.")
                ob_ts = setup.get('ob_timestamp')
                ob_type = setup.get('ob_type')
                tf_n = setup.get('tf_name')
                ict_m = setup.get('ict_method', 'FVG Return')
                sym = setup.get('symbol', '')
                if not (ob_ts and ob_type and tf_n):
                    parts = key.split('_')
                    if len(parts) >= 4:
                        sym = sym or parts[0]
                        tf_n = tf_n or parts[1]
                        ob_ts = ob_ts or parts[2]
                        ob_type = ob_type or (parts[3] if len(parts) >= 5 else setup.get('direction', 'Buy'))
                if ob_ts and ob_type and tf_n and sym:
                    add_trade_to_memory(sym, tf_n, ob_ts, ob_type, ict_m)
                config.priority_scan_setups.pop(key, None)
                continue

            current_price = None
            tick = mt5.symbol_info_tick(setup.get('symbol', ''))
            if tick:
                current_price = tick.bid if setup.get('direction') == 'Buy' else tick.ask

            if current_price is None:
                continue

            entry = setup.get('entry_price', 0)
            direction = setup.get('direction', 'Buy')
            max_slip = setup.get('max_slippage_pips', 0)
            if not max_slip:
                max_slip = config.SLIPPAGE_MAX_POINTS

            if direction == 'Buy':
                if current_price <= entry + max_slip * setup.get('point', 0.0001):
                    place_limit_order(
                        setup['symbol'], direction, entry,
                        setup['sl_price'], setup['tp_price'],
                        setup['lot_size'], setup.get('comment', ''),
                        setup_key=key,
                        ict_method=setup.get('ict_method', 'FVG Return'),
                        trailing_info=setup.get('trailing_info'),
                        ob_timestamp=setup.get('ob_timestamp'),
                        ob_type=setup.get('ob_type'),
                        tf_name=setup.get('tf_name'),
                    )
                    config.priority_scan_setups.pop(key, None)
            else:
                if current_price >= entry - max_slip * setup.get('point', 0.0001):
                    place_limit_order(
                        setup['symbol'], direction, entry,
                        setup['sl_price'], setup['tp_price'],
                        setup['lot_size'], setup.get('comment', ''),
                        setup_key=key,
                        ict_method=setup.get('ict_method', 'FVG Return'),
                        trailing_info=setup.get('trailing_info'),
                        ob_timestamp=setup.get('ob_timestamp'),
                        ob_type=setup.get('ob_type'),
                        tf_name=setup.get('tf_name'),
                    )
                    config.priority_scan_setups.pop(key, None)
        except Exception as e:
            logger.error(f"Error scanning priority setup {key}: {e}")
            config.priority_scan_setups.pop(key, None)


_active_trailing_tickets = set()


@error_boundary(name="TrailingStop", auto_restart=False)
def live_ict_trailing_stop(position_ticket, symbol, initial_risk, direction="Buy", tf_value=None,
                            trail_type=None, trail_params=None, use_symbol_profiles=False,
                            ict_method=None, breakeven_after_r=0.5, trail_activation_r=1.0):
    if position_ticket in _active_trailing_tickets:
        logger.info(f"Trailing stop thread already active for ticket {position_ticket}, skipping duplicate spawn.")
        return
    _active_trailing_tickets.add(position_ticket)

    try:
        _live_ict_trailing_stop_body(position_ticket, symbol, initial_risk, direction, tf_value,
                                     trail_type, trail_params, use_symbol_profiles,
                                     ict_method, breakeven_after_r, trail_activation_r)
    finally:
        _active_trailing_tickets.discard(position_ticket)


def _live_ict_trailing_stop_body(position_ticket, symbol, initial_risk, direction="Buy", tf_value=None,
                                  trail_type=None, trail_params=None, use_symbol_profiles=False,
                                  ict_method=None, breakeven_after_r=0.5, trail_activation_r=1.0):
    if use_symbol_profiles and ict_method:
        try:
            sym_defaults = {
                'trail_type': trail_type,
                'trail_pct': trail_params.get('trail_pct', 0.5) if isinstance(trail_params, dict) else 0.5
            }
            sym_params = get_symbol_profile_overrides(symbol, sym_defaults)
            trail_type = resolve_method_param('trail_type', ict_method, sym_params, trail_type)
            if isinstance(trail_params, dict):
                trail_params = dict(trail_params)  # shallow copy to avoid mutating shared state
                trail_params['trail_pct'] = resolve_method_param('trail_pct', ict_method, sym_params, trail_params.get('trail_pct', 0.5))
        except Exception as e:
            logger.debug(f"Could not load symbol profile overrides for trailing stop: {e}")

    try:
        from broker import get_broker
        broker = get_broker()
    except Exception as e:
        logger.error(f"Trailing stop thread failed to initialize broker for {symbol}: {e}")
        return

    sym_info = mt5.symbol_info(symbol)
    digits = sym_info.digits if sym_info and hasattr(sym_info, 'digits') else 5
    point = sym_info.point if sym_info and hasattr(sym_info, 'point') else 0.00001

    max_wait = datetime.timedelta(hours=72)
    start_time = datetime.datetime.now()
    last_log_time = 0
    last_error_time = 0
    breakeven_activated = False
    breakeven_set = False
    tp_adjusted = False

    while datetime.datetime.now() - start_time < max_wait:
        if config.stop_event.is_set() or not config.live_trading_running:
            break

        try:
            positions = broker.positions_get(symbol=symbol)
            if positions is None:
                time.sleep(1)
                continue

            pos = None
            for p in positions:
                if p.ticket == position_ticket:
                    pos = p
                    break

            if pos is None:
                logger.info(f"Position {position_ticket} ({symbol}) closed, ending trailing thread")
                break

            current_price = pos.price_current
            open_price = pos.price_open
            current_pnl = pos.profit

            # Ensure risk_pips is valid
            risk_pips = initial_risk
            if risk_pips <= 0 and pos.sl > 0:
                risk_pips = abs(open_price - pos.sl)
            if risk_pips <= 0:
                risk_pips = 100 * point

            if direction == "Buy":
                pips_moved = (current_price - open_price)
            else:
                pips_moved = (open_price - current_price)

            r_multiple = pips_moved / risk_pips

            now_ts = time.time()
            can_send = (now_ts - last_error_time) > 5.0

            # 1. Breakeven Adjustment
            effective_be_r = 1.0 if trail_type == "True Partial (50% at 1R)" else breakeven_after_r
            if not breakeven_activated and r_multiple >= effective_be_r:
                breakeven_activated = True

            if breakeven_activated and not breakeven_set and r_multiple >= (effective_be_r * 0.9) and can_send:
                sl_adjust = round(open_price + (risk_pips * 0.1) if direction == "Buy" else open_price - (risk_pips * 0.1), digits)
                if abs(sl_adjust - pos.sl) > (point * 0.5):
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position_ticket,
                        "sl": sl_adjust,
                        "tp": pos.tp,
                    }
                    result = broker.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        breakeven_set = True
                        logger.info(f"Breakeven set to {sl_adjust} for {symbol} position {position_ticket}")
                    else:
                        last_error_time = now_ts
                        ret_code = getattr(result, 'retcode', 'UNKNOWN')
                        logger.warning(f"Failed to set Breakeven SL for {symbol} ticket {position_ticket} (code: {ret_code}). Will retry on next poll.")

            # 2. True Partial TP (50% close at 1R)
            if trail_type == "True Partial (50% at 1R)" and r_multiple >= 1.0 and not tp_adjusted and can_send:
                raw_close_lots = pos.volume * 0.5
                close_lots_list = normalize_and_split_lots(symbol, raw_close_lots)
                if close_lots_list:
                    partial_close_lots = close_lots_list[0]
                    # Guard: if volume is minimum lot size (e.g., 0.01), lock profit via SL move to 0.5R
                    if partial_close_lots >= pos.volume:
                        tp_adjusted = True
                        be_05r = round(open_price + (risk_pips * 0.5) if direction == "Buy" else open_price - (risk_pips * 0.5), digits)
                        sl_req = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position_ticket,
                            "sl": be_05r,
                            "tp": pos.tp,
                        }
                        broker.order_send(sl_req)
                        logger.info(f"Position volume {pos.volume:.2f} at min lot — locked 50% profit by moving SL to 0.5R ({be_05r}) for ticket {position_ticket}")
                    else:
                        close_request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": partial_close_lots,
                            "type": mt5.ORDER_TYPE_SELL if direction == "Buy" else mt5.ORDER_TYPE_BUY,
                            "position": position_ticket,
                            "price": current_price,
                            "deviation": 10,
                            "magic": pos.magic,  # inherit the position's own magic, don't re-derive from the comment
                            "comment": "Partial TP",
                            "type_filling": get_filling_mode(symbol),
                        }
                        result = broker.order_send(close_request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            tp_adjusted = True
                            logger.info(f"Partial TP executed ({partial_close_lots} lot) for {symbol} position {position_ticket}")
                        else:
                            last_error_time = now_ts
                else:
                    tp_adjusted = True
                    logger.info(f"Position volume {pos.volume:.2f} lot too small for 50% partial close on {symbol}. Keeping full trade active with Breakeven SL.")

            # 3. Profit Lock (50% trailing above 1R)
            if trail_type in ["Profit Lock (50%)", "True Partial (50% at 1R)"] and r_multiple >= 1.0 and breakeven_set and can_send:
                locked_profit_pts = max(risk_pips * 0.1, pips_moved / 2.0)
                lock_price = round(open_price + locked_profit_pts if direction == "Buy" else open_price - locked_profit_pts, digits)
                should_lock = (direction == "Buy" and (pos.sl == 0 or lock_price > pos.sl + point * 0.5)) or \
                              (direction == "Sell" and (pos.sl == 0 or lock_price < pos.sl - point * 0.5))
                if should_lock:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position_ticket,
                        "sl": lock_price,
                        "tp": pos.tp,
                    }
                    result = broker.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Profit Lock SL updated to {lock_price} for {symbol} position {position_ticket}")
                    else:
                        last_error_time = now_ts

            # 4. Structure-Based Trailing (Phase 3 / 5-Bar Swings)
            if (trail_type == "Structure (5-Bar Swings)" or (trail_type == "True Partial (50% at 1R)" and r_multiple >= 2.0)) and can_send:
                try:
                    df_recent = get_data(symbol, mt5.TIMEFRAME_M5, 50, live=True)
                    if df_recent is not None and len(df_recent) >= 15:
                        sh, sl_sw = detect_swing_points(df_recent, lookback=5)
                        if direction == "Buy" and sl_sw:
                            last_sl_price = df_recent['low'].iloc[sl_sw[-1]]
                            new_sl = round(last_sl_price - point * 2, digits)
                            if new_sl > pos.sl + point * 0.5:
                                request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": position_ticket,
                                    "sl": new_sl,
                                    "tp": pos.tp,
                                }
                                result = broker.order_send(request)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    logger.info(f"Structure SL trailed to {new_sl} for {symbol} ticket {position_ticket}")
                        elif direction == "Sell" and sh:
                            last_sh_price = df_recent['high'].iloc[sh[-1]]
                            new_sl = round(last_sh_price + point * 2, digits)
                            if pos.sl == 0 or new_sl < pos.sl - point * 0.5:
                                request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": position_ticket,
                                    "sl": new_sl,
                                    "tp": pos.tp,
                                }
                                result = broker.order_send(request)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    logger.info(f"Structure SL trailed to {new_sl} for {symbol} ticket {position_ticket}")
                except Exception as exc_struct:
                    logger.debug(f"Structure trail error: {exc_struct}")

            # 4. Percentage Trail
            if trail_type == "Percentage Trail" and r_multiple >= trail_activation_r and can_send:
                trail_pct = trail_params.get('trail_pct', 0.5) if isinstance(trail_params, dict) else 0.5
                if direction == "Buy":
                    new_sl = round(current_price * (1.0 - trail_pct / 100.0), digits)
                    if pos.sl == 0 or new_sl > pos.sl + point * 0.5:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position_ticket,
                            "sl": new_sl,
                            "tp": pos.tp,
                        }
                        result = broker.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"Percentage trail SL moved to {new_sl:.{digits}f} for {symbol} position {position_ticket}")
                        else:
                            last_error_time = now_ts
                else:
                    new_sl = round(current_price * (1.0 + trail_pct / 100.0), digits)
                    if pos.sl == 0 or new_sl < pos.sl - point * 0.5:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position_ticket,
                            "sl": new_sl,
                            "tp": pos.tp,
                        }
                        result = broker.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"Percentage trail SL moved to {new_sl:.{digits}f} for {symbol} position {position_ticket}")
                        else:
                            last_error_time = now_ts

            now_ts = time.time()
            if now_ts - last_log_time > 60:
                logger.debug(f"Trailing {symbol} pos={position_ticket} r={r_multiple:.2f} pnl={current_pnl:.2f}")
                last_log_time = now_ts

            time.sleep(1)

        except Exception as e:
            logger.error(f"Trailing stop error for {symbol}: {e}")
            time.sleep(5)
