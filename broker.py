"""
Broker-agnostic interface for market data and order execution.

Usage:
    from broker import get_broker
    broker = get_broker()
    rates = broker.copy_rates_from_pos("XAUUSDm", 15, 0, 100)
    info = broker.symbol_info("XAUUSDm")
    broker.order_send(request)
"""
import abc
import datetime
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Timeframe constants (broker-agnostic, matches MT5 values)
# ---------------------------------------------------------------------------
TIMEFRAME_M1  = 1
TIMEFRAME_M5  = 5
TIMEFRAME_M15 = 15
TIMEFRAME_H1  = 16385
TIMEFRAME_H4  = 16388
TIMEFRAME_D1  = 16408
TIMEFRAME_W1  = 32769

TIMEFRAME_NAMES = {
    TIMEFRAME_M1: "M1", TIMEFRAME_M5: "M5", TIMEFRAME_M15: "M15",
    TIMEFRAME_H1: "H1", TIMEFRAME_H4: "H4", TIMEFRAME_D1: "D1", TIMEFRAME_W1: "W1",
}
TIMEFRAME_MINUTES = {
    TIMEFRAME_M1: 1, TIMEFRAME_M5: 5, TIMEFRAME_M15: 15,
    TIMEFRAME_H1: 60, TIMEFRAME_H4: 240, TIMEFRAME_D1: 1440, TIMEFRAME_W1: 10080,
}

# Trade constants
ORDER_TYPE_BUY  = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT  = 2
ORDER_TYPE_SELL_LIMIT = 3

TRADE_ACTION_DEAL    = 1
TRADE_ACTION_PENDING = 5
TRADE_ACTION_SLTP    = 6
TRADE_ACTION_REMOVE  = 7

ORDER_TIME_GTC     = 0
ORDER_FILLING_FOK  = 0
ORDER_FILLING_IOC  = 1
ORDER_FILLING_RETURN = 2

TRADE_RETCODE_DONE = 10009

POSITION_TYPE_BUY  = 0

DEAL_ENTRY_IN    = 0
DEAL_ENTRY_OUT   = 1
DEAL_ENTRY_INOUT = 2


# ---------------------------------------------------------------------------
# Abstract Broker Interface
# ---------------------------------------------------------------------------
class Broker(abc.ABC):
    """Abstract base for all broker implementations."""

    # ── Lifecycle ────────────────────────────────────────────────────
    @abc.abstractmethod
    def initialize(self, path: str = "", login: int = 0,
                   password: str = "", server: str = "",
                   **kwargs) -> bool: ...

    @abc.abstractmethod
    def shutdown(self) -> None: ...

    @property
    @abc.abstractmethod
    def connected(self) -> bool: ...

    # ── Market data ──────────────────────────────────────────────────
    @abc.abstractmethod
    def symbol_info(self, symbol: str) -> Any: ...

    @abc.abstractmethod
    def symbol_info_tick(self, symbol: str) -> Any: ...

    @abc.abstractmethod
    def symbol_select(self, symbol: str, enable: bool = True) -> bool: ...

    @abc.abstractmethod
    def symbols_get(self, group: str = "") -> tuple: ...

    @abc.abstractmethod
    def terminal_info(self) -> Any: ...

    @abc.abstractmethod
    def account_info(self) -> Any: ...

    @abc.abstractmethod
    def last_error(self) -> tuple: ...

    # ── Historical rates ─────────────────────────────────────────────
    @abc.abstractmethod
    def copy_rates_from_pos(self, symbol: str, tf: int, start: int,
                            count: int) -> Any: ...

    @abc.abstractmethod
    def copy_rates_range(self, symbol: str, tf: int,
                         date_from: datetime.datetime,
                         date_to: datetime.datetime) -> Any: ...

    # ── Trading ──────────────────────────────────────────────────────
    @abc.abstractmethod
    def order_send(self, request: dict) -> Any: ...

    @abc.abstractmethod
    def orders_get(self, symbol: str = "", group: str = "") -> tuple: ...

    @abc.abstractmethod
    def positions_get(self, symbol: str = "") -> tuple: ...

    @abc.abstractmethod
    def history_deals_get(self, date_from: datetime.datetime,
                          date_to: datetime.datetime,
                          group: str = "") -> tuple: ...

    # ── Calendar ─────────────────────────────────────────────────────
    @abc.abstractmethod
    def calendar_events(self, from_date: datetime.datetime = None,
                        to_date: datetime.datetime = None) -> tuple: ...

    # ── Convenience ──────────────────────────────────────────────────
    def rates_to_df(self, rates, tz: str = None) -> pd.DataFrame:
        """Convert raw rates to a DataFrame with datetime index."""
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        if tz:
            df.index = df.index.tz_localize('UTC').tz_convert(tz)
        return df

    def spread_from_tick(self, symbol: str) -> float:
        """Get current spread in price units."""
        tick = self.symbol_info_tick(symbol)
        if tick is not None and tick.ask > 0 and tick.bid > 0 and tick.ask > tick.bid:
            return tick.ask - tick.bid
        info = self.symbol_info(symbol)
        if info:
            if getattr(info, 'spread', 0) > 0:
                return info.spread * info.point
            return info.point * 20
        return 0.0


# ---------------------------------------------------------------------------
# MT5 Broker Implementation
# ---------------------------------------------------------------------------
class MT5Broker(Broker):
    """MetaTrader 5 broker implementation."""

    def __init__(self):
        self._mt5 = None
        self._initialized = False

    def _ensure_import(self):
        if self._mt5 is None:
            import MetaTrader5 as mt5
            self._mt5 = mt5

    # ── Lifecycle ────────────────────────────────────────────────────
    def initialize(self, path: str = "", login: int = 0,
                   password: str = "", server: str = "",
                   **kwargs) -> bool:
        self._ensure_import()
        try:
            if login and password and server:
                result = self._mt5.initialize(path=path, login=login,
                                              password=password, server=server)
            else:
                result = self._mt5.initialize(path=path)
            self._initialized = bool(result)
            return result
        except Exception:
            self._initialized = False
            return False

    def shutdown(self) -> None:
        self._ensure_import()
        try:
            self._mt5.shutdown()
        except Exception:
            pass
        self._initialized = False

    @property
    def connected(self) -> bool:
        try:
            self._ensure_import()
            info = self._mt5.terminal_info()
            return info is not None and info.connected
        except Exception:
            return False

    def symbol_info(self, symbol: str):
        self._ensure_import()
        if hasattr(self._mt5, 'symbol_info'):
            return self._mt5.symbol_info(symbol)
        return None

    def symbol_info_tick(self, symbol: str):
        self._ensure_import()
        if hasattr(self._mt5, 'symbol_info_tick'):
            return self._mt5.symbol_info_tick(symbol)
        return None

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        self._ensure_import()
        return self._mt5.symbol_select(symbol, enable)

    def symbols_get(self, group: str = ""):
        self._ensure_import()
        return self._mt5.symbols_get(group) if group else self._mt5.symbols_get()

    def terminal_info(self):
        self._ensure_import()
        return self._mt5.terminal_info()

    def account_info(self):
        self._ensure_import()
        return self._mt5.account_info()

    def last_error(self):
        self._ensure_import()
        return self._mt5.last_error()

    # ── Historical rates ─────────────────────────────────────────────
    def copy_rates_from_pos(self, symbol: str, tf: int, start: int, count: int):
        self._ensure_import()
        return self._mt5.copy_rates_from_pos(symbol, tf, start, count)

    def copy_rates_range(self, symbol: str, tf: int,
                         date_from: datetime.datetime,
                         date_to: datetime.datetime):
        self._ensure_import()
        return self._mt5.copy_rates_range(symbol, tf, date_from, date_to)

    # ── Trading ──────────────────────────────────────────────────────
    def order_send(self, request: dict):
        self._ensure_import()
        return self._mt5.order_send(request)

    def orders_get(self, symbol: str = "", group: str = ""):
        self._ensure_import()
        if symbol:
            return self._mt5.orders_get(symbol=symbol)
        if group:
            return self._mt5.orders_get(group=group)
        return self._mt5.orders_get()

    def positions_get(self, symbol: str = ""):
        self._ensure_import()
        if symbol:
            return self._mt5.positions_get(symbol=symbol)
        return self._mt5.positions_get()

    def history_deals_get(self, date_from: datetime.datetime,
                          date_to: datetime.datetime,
                          group: str = ""):
        self._ensure_import()
        if group:
            return self._mt5.history_deals_get(date_from, date_to, group=group)
        return self._mt5.history_deals_get(date_from, date_to)

    # ── Calendar ─────────────────────────────────────────────────────
    def calendar_events(self, from_date=None, to_date=None):
        self._ensure_import()
        if from_date:
            return self._mt5.calendar_event_get(from_date) if to_date else ()
        return self._mt5.calendar_event_get()


# ---------------------------------------------------------------------------
# Global broker instance
# ---------------------------------------------------------------------------
_broker_instance: Broker = None


def set_broker(broker: Broker) -> None:
    """Swap the active broker (e.g. for testing with a mock)."""
    global _broker_instance
    _broker_instance = broker


def get_broker() -> Broker:
    """Return the singleton broker, defaulting to MT5Broker."""
    global _broker_instance
    if _broker_instance is None:
        _broker_instance = MT5Broker()
    return _broker_instance


# Convenience: set up a default MT5Broker on import
_broker_instance = MT5Broker()
