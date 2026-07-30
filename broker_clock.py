"""Broker clock detection + validation.
MT5 returns tick.time / bar time as a POSIX timestamp of the SERVER's local
clock reinterpreted as UTC. Declaring BROKER_TIMEZONE='UTC' when the server is
EET therefore shifts every NY-hour conversion by 2-3 hours -- fatal for a
time-based methodology like ICT.

Call validate_broker_clock() once at startup. It compares the broker's reported
time to real UTC and tells you the actual offset.
"""
import datetime
import logging
try:
    import pytz
except ImportError:
    pytz = None
import config

logger = logging.getLogger(__name__)

_OFFSET_TO_TZ = {
    0: 'Etc/UTC',
    2: 'Europe/Helsinki',  # EET (winter)
    3: 'Europe/Helsinki',  # EEST (summer)
    1: 'Europe/London',
    -4: 'America/New_York',
    -5: 'America/New_York',
}


def detect_broker_offset_hours(symbol=None):
    """Return the broker server clock's offset from real UTC, in hours."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None

    symbol = symbol or getattr(config, 'DEFAULT_SYMBOL', 'XAUUSDm')
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        syms = mt5.symbols_get() or ()
        for s in syms:
            if getattr(s, 'visible', False):
                tick = mt5.symbol_info_tick(s.name)
                if tick is not None:
                    break
    if tick is None:
        return None

    # What the server CLAIMS the time is (its local clock, labelled UTC).
    server_as_utc = datetime.datetime.fromtimestamp(tick.time, datetime.timezone.utc)
    real_utc = datetime.datetime.now(datetime.timezone.utc)
    delta_h = (server_as_utc - real_utc).total_seconds() / 3600.0
    return round(delta_h * 2) / 2.0  # snap to nearest half hour


def validate_broker_clock(symbol=None, autofix=True):
    """Log + optionally correct config.BROKER_TIMEZONE. Returns (ok, message)."""
    offset = detect_broker_offset_hours(symbol)
    if offset is None:
        msg = ("Could not read a broker tick; BROKER_TIMEZONE=%s UNVERIFIED. "
               "Every killzone / Silver Bullet window depends on this."
               % getattr(config, 'BROKER_TIMEZONE', '?'))
        logger.warning("[BROKER CLOCK] %s", msg)
        return False, msg

    configured = getattr(config, 'BROKER_TIMEZONE', 'Etc/UTC')
    if pytz is not None:
        try:
            tz = pytz.timezone(configured)
            cfg_offset = tz.utcoffset(datetime.datetime.utcnow()).total_seconds() / 3600.0
        except Exception:
            cfg_offset = 0.0
    else:
        cfg_offset = 0.0

    logger.info("[BROKER CLOCK] server offset from real UTC: %+.1fh | "
                "BROKER_TIMEZONE=%s implies %+.1fh", offset, configured, cfg_offset)

    if abs(offset - cfg_offset) < 0.25:
        return True, f"Broker clock verified ({offset:+.1f}h, {configured})"

    suggested = _OFFSET_TO_TZ.get(int(offset), 'Etc/UTC')
    msg = (f"BROKER_TIMEZONE MISMATCH: server is {offset:+.1f}h from UTC but "
           f"config says {configured} ({cfg_offset:+.1f}h). "
           f"Silver Bullet / killzones / news embargo are shifted by "
           f"{offset - cfg_offset:+.1f}h. Suggested: '{suggested}'.")
    logger.error("[BROKER CLOCK] %s", msg)

    if autofix:
        config.BROKER_TIMEZONE = suggested
        clear_time_caches()
        logger.warning("[BROKER CLOCK] auto-set BROKER_TIMEZONE='%s'", suggested)

    return False, msg


def clear_time_caches():
    """convert_time_to_ny_hour/_dt are lru_cached but read the MUTABLE
    config.BROKER_TIMEZONE. Changing tz without clearing = stale forever."""
    try:
        from detectors import convert_time_to_ny_hour, convert_time_to_ny_dt
        convert_time_to_ny_hour.cache_clear()
        convert_time_to_ny_dt.cache_clear()
    except Exception:
        pass
