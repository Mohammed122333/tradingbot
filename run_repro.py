import os
import sys
import json
import tkinter as tk
from tkinter import ttk
import threading

import config

# Mock messagebox so it doesn't block
import tkinter.messagebox
tkinter.messagebox.askyesnocancel = lambda *a, **k: False
tkinter.messagebox.showinfo = lambda *a, **k: None

# Disable mt5 plotting or anything that blocks
import matplotlib
matplotlib.use('Agg')

def main():
    import ui
    root = tk.Tk()
    app = ui.GoodBotUI(root)

    # Load newtest.json
    print("Loading newtest.json...")
    with open("newtest.json", "r") as f:
        settings = json.load(f)
        
    SPECIAL_KEYS = [
        "bt_symbol_overrides", "live_symbol_overrides",
        "bt_base_cached_settings", "live_base_cached_settings",
        "bt_currently_selected_override_symbol", "live_currently_selected_override_symbol",
        "bt_currently_selected_override_method", "live_currently_selected_override_method"
    ]

    app.bt_symbol_overrides = app._sanitize_overrides(settings.get("bt_symbol_overrides", {}))
    app.live_symbol_overrides = app._sanitize_overrides(settings.get("live_symbol_overrides", {}))
    app.bt_base_cached_settings = settings.get("bt_base_cached_settings", {})
    app.live_base_cached_settings = settings.get("live_base_cached_settings", {})
    app.bt_currently_selected_override_symbol = settings.get("bt_currently_selected_override_symbol", None)
    app.live_currently_selected_override_symbol = settings.get("live_currently_selected_override_symbol", None)
    app.bt_currently_selected_override_method = settings.get("bt_currently_selected_override_method", None)
    app.live_currently_selected_override_method = settings.get("live_currently_selected_override_method", None)

    for name, value in settings.items():
        if name in ["bt_methods_listbox", "live_methods_listbox", "bt_trail_methods_listbox", "live_trail_methods_listbox"] + SPECIAL_KEYS:
            continue
        if hasattr(app, name):
            widget = getattr(app, name)
            if isinstance(widget, ttk.Combobox):
                widget.set(str(value))
            elif isinstance(widget, ttk.Entry):
                widget.delete(0, tk.END)
                widget.insert(0, str(value))
            elif isinstance(widget, (tk.BooleanVar, tk.StringVar, tk.IntVar, tk.DoubleVar)):
                widget.set(value)

    app.bt_methods_listbox.selection_clear(0, tk.END)
    for idx, method in enumerate(app.bt_methods_listbox.get(0, tk.END)):
        if method in settings.get("bt_methods_listbox", []):
            app.bt_methods_listbox.selection_set(idx)

    app.bt_trail_methods_listbox.selection_clear(0, tk.END)
    for idx, method in enumerate(app.bt_trail_methods_listbox.get(0, tk.END)):
        if method in settings.get("bt_trail_methods_listbox", []):
            app.bt_trail_methods_listbox.selection_set(idx)
            
    # Hook the methods that actually do the runs to just capture the arguments
    bt_args = {}
    opt_args = {}

    import backtester
    original_bt = backtester.combined_backtest
    def mock_bt(symbol, *args, **kwargs):
        print(f"Captured Backtest arguments for {symbol}")
        with open("dump_bt_args.json", "w") as f:
            json.dump(kwargs, f, indent=4, default=str)
        return {"XAUUSDm": {"overall": {}, "methods": []}}
    backtester.combined_backtest = mock_bt

    import multiprocessing as mp
    original_pool = mp.Pool
    class MockPool:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def imap_unordered(self, func, tasks):
            print(f"Captured {len(tasks)} Opt tasks.")
            if tasks:
                with open("dump_opt_args.json", "w") as f:
                    json.dump(tasks[0], f, indent=4, default=str)
            return []
    mp.Pool = MockPool
    
    import utils
    # Prevent data loading which takes time
    utils.get_data_by_date = lambda *args, **kwargs: None
    utils.get_ml_dataset = lambda *args, **kwargs: (None, None)

    print("Triggering Backtest...")
    app.run_backtest()
    # It starts a thread, we wait for it
    for t in threading.enumerate():
        if "Backtest" in t.name or t.name != "MainThread":
            t.join()

    print("Triggering Opt...")
    app.run_grid_optimization()
    for t in threading.enumerate():
        if "GridOpt" in t.name or t.name != "MainThread":
            t.join()

    print("Done generating debug dumps.")

if __name__ == "__main__":
    main()
