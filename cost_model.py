"""Unified transaction-cost model.

SINGLE SOURCE OF TRUTH for spread (and, by extension, transaction cost) shared
by the backtester, the trade simulator and the live scanner. Centralising this
is what keeps backtest and live in parity: everyone resolves the spread the
same way, so the same setup is gated/filled/costed identically.

IMPORTANT CONVENTIONS
---------------------
* Every spread returned by this module is expressed in PRICE units (already
  multiplied by `point`), never in raw points.
* `spread_cost` argument semantics (used across the whole codebase):
    - spread_cost >  0  -> explicit FIXED spread measured in POINTS.
    - spread_cost == 0  -> AUTO: use the most realistic spread available
                            (dataset 'spread' column, else live broker tick,
                            else a per-symbol default). This matches the
                            long-standing intent that "0 == real spread".
    - spread_cost <  0  -> AUTO (same as 0). Kept for backwards-compat callers
                            that used a negative sentinel.
* We NEVER silently return 0.0 for an AUTO request (the old behaviour that made
  every offline backtest frictionless and therefore falsely profitable).
"""

import logging

import pandas as pd

logger = logging.getLogger()


def point_for_symbol(symbol, info=None):
    """Best-effort point size for a symbol.

    Order matters: JPY and metals must be matched BEFORE any USD-suffix
    heuristic, because "USDJPYm"[-4:] == "JPYM" contains no "USD" and used to
    fall through to the 0.01 crypto/index branch (10x wrong).
    """
    # 1) Always prefer the broker's own answer.
    if info is not None:
        p = getattr(info, "point", None)
        if p:
            return float(p)
        d = getattr(info, "digits", None)
        if d is not None:
            return 10.0 ** (-int(d))

    sym_u = (symbol or "").upper()

    # 2) Explicit class table, most specific first.
    if "XAU" in sym_u or "GOLD" in sym_u:
        return 0.01  # gold quotes to 2 dp on most feeds
    if "XAG" in sym_u or "SILVER" in sym_u:
        return 0.001
    if "JPY" in sym_u:
        return 0.001  # 3-digit JPY pairs
    if any(c in sym_u for c in ("BTC", "ETH", "SOL", "BNB", "LTC", "XRP", "DOGE", "ADA", "DOT")):
        return 0.01
    if any(c in sym_u for c in ("US30", "US500", "USTEC", "NAS", "SPX", "DAX", "NDX", "DJI", "UK100", "WS30")):
        return 0.01
    if "XTI" in sym_u or "XBR" in sym_u or "OIL" in sym_u or "WTI" in sym_u:
        return 0.001

    # 3) Default: 5-digit FX.
    return 0.00001


def default_spread_points(symbol):
    """Realistic per-symbol default spread IN POINTS.

    Read from ``config.DEFAULT_SPREAD_POINTS`` (a dict keyed by a substring of
    the symbol, with a ``'DEFAULT'`` entry), or a scalar, falling back to a
    sane constant. This guarantees an AUTO spread is never 0 even with no data.
    """
    import config
    table = getattr(config, "DEFAULT_SPREAD_POINTS", None)
    if isinstance(table, dict):
        sym_u = (symbol or "").upper()
        for key, val in table.items():
            if key != "DEFAULT" and key.upper() in sym_u:
                return float(val)
        return float(table.get("DEFAULT", 12.0))
    if table is not None:
        try:
            return float(table)
        except (TypeError, ValueError):
            pass
    return 12.0


def resolve_spread(symbol, df=None, point=None, spread_cost=0):
    """Return the spread for ``symbol`` in PRICE units.

    See the module docstring for ``spread_cost`` semantics.
    """
    if point is None:
        point = point_for_symbol(symbol)

    # Explicit, positive fixed spread in points.
    if spread_cost is not None and spread_cost > 0.0:
        return float(spread_cost) * point

    # AUTO (spread_cost == 0, None, or negative sentinel).
    # 1) Dataset 'spread' column (MT5 stores spread in POINTS).
    if df is not None and hasattr(df, "columns") and "spread" in df.columns and len(df) > 0:
        med_sp = df["spread"].median()
        if pd.notna(med_sp) and med_sp > 0:
            return float(med_sp) * point

    # 2) Live broker tick (ask-bid, already in PRICE units) when actually online.
    import config
    if not getattr(config, "OFFLINE_BACKTESTING", False):
        try:
            from utils import get_broker
            live_sp = get_broker().spread_from_tick(symbol)
            if live_sp and live_sp > 0:
                return float(live_sp)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("cost_model: live spread lookup failed for %s: %s", symbol, exc)

    # 3) Realistic default - never a silent zero.
    return default_spread_points(symbol) * point


def spread_multiplier():
    """Configured safety multiplier applied consistently to modelled cost.

    Set ``config.ESTIMATED_SPREAD_MULTIPLIER`` to 1.0 for maximum backtest/live
    realism; values > 1 make the backtest deliberately pessimistic.
    """
    import config
    try:
        return float(getattr(config, "ESTIMATED_SPREAD_MULTIPLIER", 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0


def cost_spread(symbol, df=None, point=None, spread_cost=0):
    """Spread in PRICE units including the safety multiplier.

    This is the value that should be used for BOTH entry-fill modelling and
    exit modelling so the two never diverge.
    """
    return resolve_spread(symbol, df=df, point=point, spread_cost=spread_cost) * spread_multiplier()


def estimate_swap_cost(symbol, trade_direction, hold_days, lot_size=1.0):
    """Estimate overnight financing/swap cost for holding trades multi-day.
    
    Supports Swap-Free / Islamic accounts (config.SWAP_FREE_ACCOUNT) and fetches
    exact live broker swap rates from MT5 symbol_info when available.
    """
    import config
    if getattr(config, 'SWAP_FREE_ACCOUNT', True) or not getattr(config, 'ENABLE_SWAP_COST', False):
        return 0.0

    if hold_days < 0.5:
        return 0.0

    # Try fetching live broker swap rates from MT5 when online
    if not getattr(config, "OFFLINE_BACKTESTING", False):
        try:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(symbol)
            if info:
                # Mode 0 = Disabled / Swap Free
                if getattr(info, 'swap_mode', 1) == 0 or (getattr(info, 'swap_long', 0) == 0 and getattr(info, 'swap_short', 0) == 0):
                    return 0.0
                
                swap_rate = info.swap_long if trade_direction == "Buy" else info.swap_short
                point = info.point if info.point else 0.00001
                contract_size = info.trade_contract_size if hasattr(info, 'trade_contract_size') and info.trade_contract_size else 100000.0
                
                days_to_bill = int(hold_days)
                # If swap is given in points
                if getattr(info, 'swap_mode', 1) == 1:
                    swap_cost_usd = abs(swap_rate * point * contract_size * lot_size * days_to_bill)
                    return swap_cost_usd
        except Exception as exc:
            logger.debug("cost_model: live broker swap lookup failed for %s: %s", symbol, exc)

    # Offline fallback rate table — MUST cover every symbol class.
    sym_u = (symbol or "").upper()
    if "XAU" in sym_u or "GOLD" in sym_u:
        daily_swap_per_lot = 12.0 if trade_direction == "Buy" else 4.0
    elif "XAG" in sym_u or "SILVER" in sym_u:
        daily_swap_per_lot = 6.0 if trade_direction == "Buy" else 2.0
    elif any(c in sym_u for c in ("BTC", "ETH", "SOL", "XRP", "DOGE", "ADA")):
        daily_swap_per_lot = 15.0
    elif any(c in sym_u for c in ("US30", "US500", "USTEC", "NAS", "SPX", "DAX", "DJI", "UK100")):
        daily_swap_per_lot = 8.0
    elif "JPY" in sym_u:
        daily_swap_per_lot = 3.0
    else:
        daily_swap_per_lot = 5.0  # generic FX

    # Triple swap on the configured rollover weekday (MT5 default: Wednesday).
    days_to_bill = int(hold_days)
    if days_to_bill >= 7:
        days_to_bill += (days_to_bill // 7) * 2  # approximate 3x Wednesdays
    return max(0.0, daily_swap_per_lot * lot_size * days_to_bill)


def log_broker_swap_info(symbols=None):
    """Log broker swap settings & returned values for symbols during bot startup."""
    import config
    try:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            logger.info("[STARTUP SWAP CHECK] MetaTrader5 module not available (offline/linux) - skipping live broker swap check.")
            return

        if not mt5.initialize():
            logger.info("[STARTUP SWAP CHECK] MT5 terminal not initialized - skipping live broker swap check.")
            return

        if symbols is None:
            symbols = [getattr(config, 'DEFAULT_SYMBOL', 'XAUUSDm')]
        elif isinstance(symbols, str):
            symbols = [symbols]

        swap_free_cfg = getattr(config, 'SWAP_FREE_ACCOUNT', True)
        logger.info("=== [STARTUP SWAP CHECK] Config SWAP_FREE_ACCOUNT=%s ===", swap_free_cfg)

        swap_mode_names = {
            0: "DISABLED (Swap-Free)",
            1: "POINTS",
            2: "CURRENCY_SYMBOL",
            3: "CURRENCY_MARGIN",
            4: "PERCENT_INTEREST",
            5: "REOPEN_CURRENT",
            6: "REOPEN_BID"
        }

        for sym in symbols:
            info = mt5.symbol_info(sym)
            if info is None:
                logger.warning("[STARTUP SWAP CHECK] Symbol '%s': mt5.symbol_info returned None", sym)
                continue

            sm_mode = getattr(info, 'swap_mode', 0)
            sm_mode_str = swap_mode_names.get(sm_mode, f"UNKNOWN({sm_mode})")
            swap_long = getattr(info, 'swap_long', 0.0)
            swap_short = getattr(info, 'swap_short', 0.0)
            roll_3day = getattr(info, 'swap_rollover3days', 3)

            is_swap_free = (sm_mode == 0) or (swap_long == 0.0 and swap_short == 0.0) or swap_free_cfg

            logger.info(
                "[STARTUP SWAP CHECK] Broker Swap Info for '%s': "
                "swap_mode=%s (%s) | "
                "swap_long=%s | swap_short=%s | "
                "rollover3days=%s | Effective Swap-Free=%s",
                sym, sm_mode, sm_mode_str, swap_long, swap_short, roll_3day, is_swap_free
            )
    except Exception as e:
        logger.error("[STARTUP SWAP CHECK] Exception querying broker swaps: %s", e)
