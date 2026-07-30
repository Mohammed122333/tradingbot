import json
import tkinter as tk
from tkinter import ttk
import sys
import os

# mock messagebox
import tkinter.messagebox
tkinter.messagebox.askyesnocancel = lambda *a, **k: False
tkinter.messagebox.showinfo = lambda *a, **k: None

# Disable matplot and mt5 init
import matplotlib
matplotlib.use('Agg')
try:
    import MetaTrader5 as mt5
except ImportError:
    import tests.MetaTrader5 as mt5
mt5.initialize = lambda *a, **k: True

def main():
    import ui
    root = tk.Tk()
    app = ui.GoodBotUI(root)

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

    # Dump Backtest Settings
    # We call run_backtest to see what it eventually resolves to.
    import backtester
    
    dump_bt = {}
    def mock_bt_run(symbol, *args, **kwargs):
        dump_bt.update(kwargs)
        return {"XAUUSDm": {"overall": {}, "methods": []}}
    backtester.combined_backtest = mock_bt_run
    
    app.run_backtest()
    
    # Dump Grid Opt settings
    # We need to extract the fixed_params from run_grid_optimization
    # We can patch state to intercept it
    dump_opt = {}
    
    # We have to execute app.run_grid_optimization
    # It builds self.grid_opt_state
    app.run_grid_optimization()
    dump_opt = app.grid_opt_state['fixed_params']
    
    with open("bt_resolved.json", "w") as f:
        json.dump(dump_bt, f, indent=4, default=str)
        
    with open("opt_resolved.json", "w") as f:
        json.dump(dump_opt, f, indent=4, default=str)
        
    print("Dumps created: bt_resolved.json, opt_resolved.json")
    
if __name__ == '__main__':
    main()
