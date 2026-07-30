"""Offline smoke test for the patched bot.

Proves:
  1. Every patched module imports cleanly (with a mocked MetaTrader5).
  2. cost_model resolves a realistic, NON-ZERO spread for every mode.
  3. get_spread no longer returns 0.0 offline.
  4. The trailing-stop JIT path runs for trail types 2 AND 3, both directions,
     without the old `trail_pct_val` NameError.
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
sys.path.insert(0, BOT)   # bot modules (config, cost_model, ...)
sys.path.insert(0, HERE)  # MetaTrader5 stub

import numpy as np
import pandas as pd

failures = []


def check(label, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        failures.append(label)
    print(f"  [{status}] {label}{(' -> ' + extra) if extra else ''}")


print("=== 1. Import patched modules ===")
import config
config.OFFLINE_BACKTESTING = True
for mod in [
    "cost_model", "broker", "utils", "backtest_metrics", "indicators",
    "detectors", "simulation", "simulation_scoring", "regime",
    "regime_manager", "backtester", "live_scanner",
]:
    try:
        __import__(mod)
        print(f"  [PASS] import {mod}")
    except Exception as exc:
        failures.append(f"import {mod}")
        print(f"  [FAIL] import {mod} -> {exc}")
        traceback.print_exc()

import cost_model
import utils
import simulation

print("\n=== 2. cost_model spread resolution (must all be > 0) ===")
# Fixed spread in POINTS: 20 pts * 0.001 point = 0.02
fx = cost_model.resolve_spread("XAUUSD", spread_cost=20)
check("fixed spread XAU 20pts == 0.02", abs(fx - 0.02) < 1e-9, f"{fx}")

# AUTO with a dataset 'spread' column (median 30 pts): 30 * 0.001 = 0.03
df = pd.DataFrame({"spread": [28, 30, 32], "close": [2000, 2001, 2002]})
au_df = cost_model.resolve_spread("XAUUSD", df=df, spread_cost=0)
check("AUTO w/ dataset median 30pts == 0.03", abs(au_df - 0.03) < 1e-9, f"{au_df}")

# AUTO with no data -> per-symbol default (XAU 20 pts * 0.001 = 0.02), NEVER 0
au_def = cost_model.resolve_spread("XAUUSD", spread_cost=0)
check("AUTO default XAU non-zero", au_def > 0, f"{au_def}")

# AUTO for FX default (12 pts * 0.00001 = 0.00012)
au_fx = cost_model.resolve_spread("EURUSD", spread_cost=0)
check("AUTO default EURUSD non-zero", au_fx > 0, f"{au_fx}")

# The exact bug the user hit: offline get_spread used to return 0.0
gs = utils.get_spread("XAUUSD")
check("utils.get_spread offline is non-zero", gs > 0, f"{gs}")

print("\n=== 3. Trailing-stop JIT: no NameError for trail types 2 & 3 ===")


def make_path(direction):
    """Build a synthetic M1 path that fills then runs strongly in-profit so the
    breakeven + trailing block is exercised (without instantly hitting TP)."""
    n = 40
    base = pd.Timestamp("2024-01-01 00:00:00")
    times = np.array([np.datetime64(base + pd.Timedelta(minutes=i)) for i in range(n)])
    if direction == 1:  # Buy: rise from 100 toward (but not reaching) TP 110
        closes = np.linspace(100.0, 109.0, n)
    else:               # Sell: fall from 100 toward (but not reaching) TP 90
        closes = np.linspace(100.0, 91.0, n)
    opens = closes.copy()
    highs = closes + 0.3
    lows = closes - 0.3
    return times, opens.astype(float), highs.astype(float), lows.astype(float), closes.astype(float)


def run_jit(direction, trail_type_val):
    times, opens, highs, lows, closes = make_path(direction)
    t_int64 = times.view(np.int64)
    entry_price = 100.0
    if direction == 1:
        tp_price, sl_price = 110.0, 95.0
    else:
        tp_price, sl_price = 90.0, 105.0
    return simulation._run_trade_simulation_jit(
        times, opens, highs, lows, closes,
        direction, entry_price, tp_price, sl_price,
        True, 5.0, 0.02, 7200,
        trail_type_val, 0.5, 0.0,
        highs, lows, t_int64,
        int(t_int64[0]), int(t_int64[0]), False,
    )


for direction, name in [(1, "Buy"), (-1, "Sell")]:
    for ttv in (2, 3):
        try:
            res = run_jit(direction, ttv)
            check(f"JIT {name} trail_type={ttv} runs", res is not None and len(res) == 4, f"outcome={res[3]}")
        except Exception as exc:
            check(f"JIT {name} trail_type={ttv} runs", False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

print("\n=== RESULT ===")
if failures:
    print(f"SMOKE TEST FAILED ({len(failures)} failure(s)): {failures}")
    sys.exit(1)
print("SMOKE TEST PASSED - all checks green.")
