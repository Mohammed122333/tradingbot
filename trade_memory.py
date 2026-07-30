"""Thin facade over state.app_state — the ONLY owner of trade_memory.json.
Kept so existing imports keep working.
"""
import logging
from state import app_state

logger = logging.getLogger()


def _key(symbol, timeframe, timestamp, direction, method=""):
    return f"{symbol}|{timeframe}|{timestamp}|{direction}|{method}"


def save_trade_memory():
    app_state.save_trade_memory()


def add_trade_to_memory(symbol, timeframe, timestamp, direction, method=""):
    app_state.mutate_executed_ob(_key(symbol, timeframe, timestamp, direction, method))


def is_trade_executed(symbol, timeframe, timestamp, direction, method=""):
    with app_state.dict_access('_executed_ob'):
        return _key(symbol, timeframe, timestamp, direction, method) in app_state.executed_ob


def clear_trade_memory():
    app_state.clear_executed_ob()
