"""Helper functions for the live scanner — positions, drawdown, magic numbers.

Extracted from live_scanner.py to reduce file size.
"""
import datetime
import logging
import os
import json

try:
    import MetaTrader5 as mt5
except ImportError:
    import tests.MetaTrader5 as mt5

import config
from utils import get_symbol_profile_overrides

logger = logging.getLogger()


def log_broker_error(symbol, action, error_msg, retcode=None, details=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    retcode_str = f" [RetCode: {retcode}]" if retcode is not None else ""
    details_str = f" | Details: {details}" if details else ""
    line = f"{timestamp} [{symbol}] {action}: {error_msg}{retcode_str}{details_str}\n"
    logger.warning(f"[BROKER ERROR] [{symbol}] {action}: {error_msg}{retcode_str}{details_str}")
    try:
        with open("broker_errors.log", "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        logger.error(f"Failed to write to broker_errors.log: {e}")


def get_broker_today_start(fallback_symbol="XAUUSDm"):
    """Calculate the starting datetime (00:00:00) of the broker's day in local computer time."""
    now = datetime.datetime.now()
    tick = mt5.symbol_info_tick(fallback_symbol)
    if tick is None:
        symbols = mt5.symbols_get()
        if symbols:
            for s in symbols:
                if s.visible:
                    tick = mt5.symbol_info_tick(s.name)
                    if tick is not None:
                        break
    if tick is not None:
        broker_time = datetime.datetime.fromtimestamp(tick.time, datetime.timezone.utc).replace(tzinfo=None)
        broker_today_start = datetime.datetime(broker_time.year, broker_time.month, broker_time.day, 0, 0, 0)
        offset_sec = (now - broker_time).total_seconds()
        
        if abs(offset_sec) > 300 or (now.timestamp() - tick.time) > 60:
            return datetime.datetime(now.year, now.month, now.day, 0, 0, 0)
            
        return broker_today_start + datetime.timedelta(seconds=offset_sec)
    else:
        return datetime.datetime(now.year, now.month, now.day, 0, 0, 0)


def is_daily_drawdown_limit_hit():
    """Check if the daily drawdown limit has been hit on the MT5 account.
    Returns (True, message) if hit, (False, '') otherwise."""
    if config.DAILY_LOSS_LIMIT_PCT <= 0:
        return False, ""

    account_info = mt5.account_info()
    if account_info is None:
        logger.warning("[DAILY DRAWDOWN] Failed to get account info from MT5.")
        return False, ""

    balance = account_info.balance
    today_start = get_broker_today_start()
    now = datetime.datetime.now()
    deals = mt5.history_deals_get(today_start, now + datetime.timedelta(hours=2))

    daily_loss_today = 0.0
    daily_net_pnl = 0.0

    if deals is None:
        logger.warning("[DAILY DRAWDOWN] Deals is None, returning False.")
        return False, ""

    if deals:
        for deal in deals:
            if getattr(deal, 'type', None) not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL):
                continue
            
            profit = getattr(deal, 'profit', 0.0) or 0.0
            comm = getattr(deal, 'commission', 0.0) or 0.0
            swap = getattr(deal, 'swap', 0.0) or 0.0
            
            daily_net_pnl += profit + comm + swap
            if getattr(deal, 'entry', None) in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT):
                pnl = profit + comm + swap
                if pnl < 0:
                    daily_loss_today += abs(pnl)

    day_start_balance = balance - daily_net_pnl
    daily_loss_limit_currency = day_start_balance * (config.DAILY_LOSS_LIMIT_PCT / 100)

    if daily_loss_today >= daily_loss_limit_currency:
        msg = (f"Daily loss limit hit! Realized loss today: ${daily_loss_today:.2f} "
               f">= Limit: ${daily_loss_limit_currency:.2f} "
               f"(Start Balance: ${day_start_balance:.2f})")
        return True, msg

    return False, ""


# ── Single source of truth: method <-> magic <-> comment token ────────────
METHOD_MAGIC = {
    "FVG Return": 123457,
    "Silver Bullet": 123458,
    "MMXM": 123459,
    "ICT Model 2022": 123460,
    "ICT Model 2025": 123461,
    "ICT Retest": 123462,
    "OB Retest": 123463,
    "Order Block Retest": 123464,
    "Judas Swing": 123465,
    "Breaker Block": 123466,
    "IOFR": 123467,
    "Inversion FVG": 123468,
    "Unicorn": 123469,
}
DEFAULT_MAGIC = 123456

# Reverse map is DERIVED, so it can never drift out of sync again.
MAGIC_METHOD = {}
for _m, _mg in METHOD_MAGIC.items():
    MAGIC_METHOD.setdefault(_mg, []).append(_m)
# Keep the legacy OB-retest aliasing under one magic.
MAGIC_METHOD[123462] = ["ICT Retest", "OB Retest", "Order Block Retest"]

# Comment fallback, checked in order (longest/most specific first).
COMMENT_TOKENS = [
    ("IFVG", "Inversion FVG"), ("UNICORN", "Unicorn"),
    ("MODEL2022", "ICT Model 2022"), ("MODEL2025", "ICT Model 2025"),
    ("2022", "ICT Model 2022"), ("2025", "ICT Model 2025"),
    ("MMXM", "MMXM"), ("SILVER", "Silver Bullet"), ("SB", "Silver Bullet"),
    ("JUDAS", "Judas Swing"), ("BREAKER", "Breaker Block"),
    ("IOFR", "IOFR"), ("FVG", "FVG Return"),
    ("RETEST", "ICT Retest"), ("OB", "ICT Retest"),
]

def get_magic_number(method):
    return METHOD_MAGIC.get(method, DEFAULT_MAGIC)

def get_method_from_magic(magic, target_method=None):
    methods = MAGIC_METHOD.get(magic)
    if not methods:
        return None
    if target_method and target_method in methods:
        return target_method
    return methods[0]

def _method_from_comment(comment, target_method=None):
    c = (comment or "").upper()
    for token, method in COMMENT_TOKENS:
        if token in c:
            return method
    return "Unknown"

def _resolve_method(obj, target_method=None):
    method = get_method_from_magic(getattr(obj, 'magic', DEFAULT_MAGIC), target_method)
    if method:
        return method
    return _method_from_comment(getattr(obj, 'comment', ''), target_method)

def get_method_from_deal(deal, target_method=None):
    return _resolve_method(deal, target_method)

def get_method_from_position(pos, target_method=None):
    return _resolve_method(pos, target_method)

METHOD_SHORT = {
    "Silver Bullet": "SB", "ICT Retest": "OB", "OB Retest": "OB",
    "Order Block Retest": "OB", "ICT Model 2022": "Model2022",
    "ICT Model 2025": "Model2025", "FVG Return": "FVG",
    "Judas Swing": "Judas", "Breaker Block": "Breaker",
    "IOFR": "IOFR", "MMXM": "MMXM",
    "Inversion FVG": "IFVG", "Unicorn": "Unicorn",
}

def format_order_comment(method, direction, tf_name, confluence_score=None):
    clean_method = METHOD_SHORT.get(method, method)
    if confluence_score is not None:
        comment = f"ICT {clean_method} {direction} ({tf_name}) C{confluence_score}"
    else:
        comment = f"ICT {clean_method} {direction} ({tf_name})"
    return comment[:31]


def count_open_positions_for_method(symbol, method):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return 0
    count = 0
    for pos in positions:
        if get_method_from_position(pos, method) == method:
            count += 1
    return count


def is_daily_loss_limit_hit_for_method(method, symbol, use_symbol_profiles=True):
    daily_loss_limit_pct = 0.0

    if use_symbol_profiles:
        sym_defaults = {'daily_loss_limit': config.DAILY_LOSS_LIMIT_PCT}
        sym_params = get_symbol_profile_overrides(symbol, sym_defaults)
        method_override_found = False
        method_override_val = 0.0
        if sym_params and "methods" in sym_params and isinstance(sym_params["methods"], dict):
            method_overrides = sym_params["methods"].get(method)
            if method_overrides and isinstance(method_overrides, dict) and 'daily_loss_limit' in method_overrides:
                method_override_found = True
                method_override_val = float(method_overrides['daily_loss_limit'])

        if method_override_found:
            daily_loss_limit_pct = method_override_val
        else:
            profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbol_profiles.json")
            has_symbol_limit = False
            symbol_limit_val = 0.0
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, 'r') as f:
                        profiles = json.load(f)
                        prof = profiles.get(symbol)
                        if not prof:
                            symbol_upper = symbol.upper()
                            for k, v in profiles.items():
                                k_upper = k.upper()
                                if k_upper in symbol_upper or symbol_upper in k_upper or (k_upper == "GOLD" and "XAU" in symbol_upper):
                                    prof = v
                                    break
                        if prof and 'daily_loss_limit' in prof:
                            has_symbol_limit = True
                            symbol_limit_val = float(prof['daily_loss_limit'])
                except Exception:
                    pass

            if has_symbol_limit:
                daily_loss_limit_pct = symbol_limit_val
            else:
                daily_loss_limit_pct = config.DAILY_LOSS_LIMIT_PCT
    else:
        daily_loss_limit_pct = config.DAILY_LOSS_LIMIT_PCT

    if daily_loss_limit_pct <= 0.0:
        return False, ""

    account_info = mt5.account_info()
    if account_info is None:
        logger.warning(f"[DAILY DRAWDOWN] Failed to get account info from MT5 for {method}.")
        return False, ""

    balance = account_info.balance
    today_start = get_broker_today_start(symbol)
    now = datetime.datetime.now()
    deals = mt5.history_deals_get(today_start, now + datetime.timedelta(hours=2))

    daily_loss_today = 0.0
    daily_net_pnl = 0.0

    if deals is None:
        logger.warning(f"[DAILY DRAWDOWN] Deals is None for {method}, returning False.")
        return False, ""

    if deals:
        for deal in deals:
            if getattr(deal, 'type', None) not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL):
                continue
                
            profit = getattr(deal, 'profit', 0.0) or 0.0
            comm = getattr(deal, 'commission', 0.0) or 0.0
            swap = getattr(deal, 'swap', 0.0) or 0.0
            
            daily_net_pnl += profit + comm + swap
            if getattr(deal, 'symbol', '') == symbol and get_method_from_deal(deal, method) == method:
                if getattr(deal, 'entry', None) in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT):
                    pnl = profit + comm + swap
                    if pnl < 0:
                        daily_loss_today += abs(pnl)

    day_start_balance = balance - daily_net_pnl
    if day_start_balance <= 0:
        day_start_balance = balance if balance > 0 else 10000.0

    daily_loss_limit_currency = day_start_balance * (daily_loss_limit_pct / 100.0)

    if daily_loss_today >= daily_loss_limit_currency:
        msg = (f"{method} Daily loss limit hit! Realized loss today: ${daily_loss_today:.2f} "
               f">= Limit: ${daily_loss_limit_currency:.2f} "
               f"(Start Balance: ${day_start_balance:.2f})")
        return True, msg

    return False, ""


def check_and_clear_market_closed_flag(symbol, info):
    if 'market_closed_tick_time_at_fail' in info:
        current_tick = mt5.symbol_info_tick(symbol)
        if current_tick and current_tick.time <= info['market_closed_tick_time_at_fail']:
            return True
        else:
            del info['market_closed_tick_time_at_fail']
    return False
