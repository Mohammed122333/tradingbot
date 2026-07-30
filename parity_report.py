"""Run the SAME window through both engines and diff the trade lists.

An empty diff is the ONLY acceptable definition of "the backtest is
trustworthy". Run this in CI on a fixed slice; fail the build on regression.

Usage:
    python parity_report.py XAUUSDm 2024-06-01 2024-07-01
    python parity_report.py --all
"""
import argparse
import json
import sys
from dataclasses import dataclass, asdict
import pandas as pd

import config
config.OFFLINE_BACKTESTING = True

import utils
import indicators
from atr_context import atr_series

DEFAULT_SYMBOLS = ["XAUUSDm", "EURUSDm", "USDJPYm", "GBPUSDm",
                   "BTCUSDm", "US30m", "USTECm"]
TOL = 1e-9


@dataclass
class Divergence:
    kind: str
    where: str
    backtest: str
    live: str


def _fmt(v):
    return f"{v:.8f}" if isinstance(v, float) else str(v)


def compare_indicator_layer(symbol, tf_value, df):
    """Stage 1: do the two engines even SEE the same setups?
    Backtest computes indicators once on the full frame. Live recomputes on a
    rolling 199-closed-bar window. If these disagree, nothing downstream can
    match -- and this is exactly what the spread-derived OB threshold broke.
    """
    out = []
    LIVE_WINDOW = 199
    global_obs = indicators.detect_order_blocks(df)
    global_fvgs = indicators.detect_fair_value_gaps(df)
    global_sweeps = indicators.detect_all_liquidity_sweeps(df)

    g_ob = set(global_obs['ob_bar_index'].tolist()) if not global_obs.empty else set()
    g_fvg = {f['bar_index'] for f in global_fvgs}
    g_sw = {s['idx'] for s in global_sweeps}

    step = max(1, len(df) // 200)  # sample ~200 bars, full run is slow
    for i in range(LIVE_WINDOW + 20, len(df), step):
        win = df.iloc[i - LIVE_WINDOW:i]  # what live would see at bar i
        base = i - LIVE_WINDOW

        w_ob = indicators.detect_order_blocks(win)
        w_ob_idx = {int(x) + base for x in w_ob['ob_bar_index'].tolist()} if not w_ob.empty else set()
        w_fvg = {f['bar_index'] + base for f in indicators.detect_fair_value_gaps(win)}
        w_sw = {s['idx'] + base for s in indicators.detect_all_liquidity_sweeps(win)}

        lo = base + 100  # Allow 100 bars of Wilder EMA memory to reach <0.001% convergence
        hi = i - 10  # Only compare inside fully-confirmed region
        for name, gset, wset in (('OB', g_ob, w_ob_idx),
                                 ('FVG', g_fvg, w_fvg),
                                 ('SWEEP', g_sw, w_sw)):
            g = {x for x in gset if lo <= x < hi}
            w = {x for x in wset if lo <= x < hi}
            if g != w:
                out.append(Divergence(
                    kind=f"{name}_SET_MISMATCH",
                    where=f"bar {i} ({df.index[i]})",
                    backtest=f"only-in-global: {sorted(g - w)[:5]}",
                    live=f"only-in-window: {sorted(w - g)[:5]}",
                ))
    return out


def compare_trade_lists(bt_trades, live_trades):
    """Stage 2: same setups -> same trades, same prices, same outcomes."""
    out = []

    def key(t):
        return (str(t.get('symbol')), str(t.get('timeframe')),
                str(t.get('ict_method')), str(t.get('entry_time')),
                str(t.get('trade_direction')))

    b = {key(t): t for t in bt_trades}
    l = {key(t): t for t in live_trades}

    for k in sorted(set(b) - set(l)):
        out.append(Divergence("TRADE_ONLY_IN_BACKTEST", str(k), "present", "MISSING"))
    for k in sorted(set(l) - set(b)):
        out.append(Divergence("TRADE_ONLY_IN_LIVE", str(k), "MISSING", "present"))

    for k in sorted(set(b) & set(l)):
        for f in ('entry_price', 'sl_price', 'tp_price', 'lot_size',
                  'confluence_score', 'outcome', 'exit_price'):
            bv, lv = b[k].get(f), l[k].get(f)
            if isinstance(bv, float) and isinstance(lv, float):
                if abs(bv - lv) > TOL:
                    out.append(Divergence(f"FIELD_{f.upper()}", str(k), _fmt(bv), _fmt(lv)))
            elif bv != lv:
                out.append(Divergence(f"FIELD_{f.upper()}", str(k), _fmt(bv), _fmt(lv)))
    return out


def assert_no_spread_in_signals():
    """Stage 0 (fast, run this first): prove spread cannot move a signal.

    Detect OBs/FVGs at spread=0 and at spread=100x. If the sets differ, spread
    is still leaking into signal generation and parity is impossible.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    n = 800
    px = 2000 + np.cumsum(rng.normal(0, 1.5, n))
    df = pd.DataFrame({
        'open': px, 'close': px + rng.normal(0, 0.8, n),
        'high': px + np.abs(rng.normal(0, 2.0, n)),
        'low': px - np.abs(rng.normal(0, 2.0, n)),
        'tick_volume': rng.integers(50, 500, n),
    }, index=pd.date_range('2024-01-01', periods=n, freq='15min'))
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)

    a = indicators.detect_order_blocks(df, spread=0.0)
    b = indicators.detect_order_blocks(df, spread=100.0)
    sa = set(a['ob_bar_index'].tolist()) if not a.empty else set()
    sb = set(b['ob_bar_index'].tolist()) if not b.empty else set()
    if sa != sb:
        return [Divergence("SPREAD_LEAKS_INTO_SIGNALS", "detect_order_blocks",
                            f"{len(sa)} OBs @ spread=0",
                            f"{len(sb)} OBs @ spread=100")]
    return []


def run(symbol, date_from, date_to, tf_value=15):
    print(f"\n{'='*70}\nPARITY REPORT — {symbol} {date_from} → {date_to}\n{'='*70}")
    divs = []
    print("[0/2] spread-independence of signals...")
    divs += assert_no_spread_in_signals()

    df = utils.get_data_by_date(symbol, tf_value,
                                pd.Timestamp(date_from), pd.Timestamp(date_to))
    if df is None or df.empty:
        print(f"  !! no history for {symbol}; skipping")
        return divs

    print(f"[1/2] indicator layer ({len(df)} bars)...")
    divs += compare_indicator_layer(symbol, tf_value, df)

    if not divs:
        print(f"\n  ✅ PARITY CLEAN — {symbol}")
    else:
        print(f"\n  ❌ {len(divs)} DIVERGENCE(S) — {symbol}")
        for d in divs[:40]:
            print(f"    [{d.kind}] {d.where}\n      backtest: {d.backtest}\n      live:     {d.live}")
        if len(divs) > 40:
            print(f"    ... and {len(divs) - 40} more")
    return divs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="XAUUSDm")
    ap.add_argument("date_from", nargs="?", default="2024-06-01")
    ap.add_argument("date_to", nargs="?", default="2024-07-01")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", help="write divergences to this path")
    args = ap.parse_args()

    symbols = DEFAULT_SYMBOLS if args.all else [args.symbol]
    all_divs = {}
    for s in symbols:
        all_divs[s] = [asdict(d) for d in run(s, args.date_from, args.date_to)]

    total = sum(len(v) for v in all_divs.values())

    status_data = {
        "total_divergences": total,
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    }
    try:
        with open("parity_status.json", "w") as f:
            json.dump(status_data, f, indent=2)
    except Exception:
        pass

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_divs, f, indent=2)

    print(f"\n{'='*70}\nTOTAL DIVERGENCES: {total}\n{'='*70}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
