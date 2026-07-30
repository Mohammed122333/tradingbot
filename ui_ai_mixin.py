"""AI / ML Analyst tab for the trading UI."""
import logging
import os
import pickle
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import datetime
try:
    import MetaTrader5 as mt5
except ImportError:
    import tests.MetaTrader5 as mt5

from utils import get_data, CACHE_DIR
from indicators import (
    get_ict_model_parameters,
    detect_order_blocks,
    detect_market_structure,
)
from backtester import generate_trade_signals
from ml_engine import (
    ml_load_model,
    ML_AVAILABLE,
    ml_train_and_analyze,
    ml_generate_ict_report,
    ml_save_model,
    walk_forward_ml_analysis,
)

logger = logging.getLogger()


class AIAnalystMixin:
    """Mixin providing the ML ICT Analyst tab for TradingUI."""

    def _create_ai_tab(self):
        self.ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ai_frame, text="ML ICT Analyst")

        control_frame = ttk.Frame(self.ai_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(control_frame, text="Saved Backtest Run:").pack(side=tk.LEFT, padx=5)
        self.ai_run_selector = ttk.Combobox(control_frame, width=30, state="readonly")
        self.ai_run_selector.pack(side=tk.LEFT, padx=5)

        self.ai_refresh_btn = ttk.Button(control_frame, text="Refresh", command=self._refresh_ai_runs)
        self.ai_refresh_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text=" | Live Chart:").pack(side=tk.LEFT, padx=5)
        self.ai_symbol = ttk.Entry(control_frame, width=12)
        self.ai_symbol.insert(0, "BTCUSDm")
        self.ai_symbol.pack(side=tk.LEFT, padx=5)

        self.ai_tf = ttk.Combobox(control_frame, values=["M1", "M5", "M15", "H1", "H4", "D1"], width=4)
        self.ai_tf.set("M5")
        self.ai_tf.pack(side=tk.LEFT, padx=5)

        self.ai_btn = ttk.Button(control_frame, text="Train ML on Selected Backtest", command=self.run_ai_analyst)
        self.ai_btn.pack(side=tk.LEFT, padx=15)

        self.ai_status = ttk.Label(control_frame, text="Status: Idle", foreground="blue")
        self.ai_status.pack(side=tk.LEFT, padx=10)

        self.ai_text = scrolledtext.ScrolledText(self.ai_frame, wrap=tk.WORD, font=('Consolas', 10), bg="#1e1e1e", fg="#d4d4d4")
        self.ai_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        welcome_msg = (
            "========================================================================\n"
            "  \U0001f9e0 ICT MACHINE LEARNING ANALYST (Local)\n"
            "========================================================================\n\n"
            "This tool uses Scikit-Learn (Random Forest & Neural Networks) to learn\n"
            "which ICT methods work best based on your past Backtests.\n\n"
            "1. Run a Backtest or Optimization (this automatically saves a raw .pkl file).\n"
            "2. Select the saved run from the dropdown above.\n"
            "3. The ML models will learn the exact winning patterns from that run.\n"
            "4. It will save the model (e.g. ml_model_XAUUSDm.pkl) for future filtering.\n"
            "5. It will then 'read' the current live chart and predict the probability\n"
            "   of success for any active setups based on the trained model.\n"
        )
        self.ai_text.insert(tk.END, welcome_msg)
        self._refresh_ai_runs()

    def _refresh_ai_runs(self):
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BackTestReports")
        if not os.path.exists(report_dir):
            self.ai_run_selector['values'] = []
            self.ai_run_selector.set("No backtests found")
            return

        import glob
        pkl_files = glob.glob(os.path.join(report_dir, "raw_backtest_*.pkl"))
        pkl_files.sort(key=os.path.getmtime, reverse=True)

        display_names = []
        self._ai_run_paths = {}
        for f in pkl_files:
            basename = os.path.basename(f)
            parts = basename.replace("raw_backtest_", "").replace(".pkl", "").split("_")
            if len(parts) >= 2:
                try:
                    dt_str = f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]} {parts[1][:2]}:{parts[1][2:4]}:{parts[1][4:]}"
                    display_name = dt_str
                except Exception:
                    display_name = basename
            else:
                display_name = basename
            display_names.append(display_name)
            self._ai_run_paths[display_name] = f

        if display_names:
            self.ai_run_selector['values'] = display_names
            self.ai_run_selector.current(0)
        else:
            self.ai_run_selector['values'] = []
            self.ai_run_selector.set("No backtests found")

    def run_ai_analyst(self):
        if not ML_AVAILABLE:
            messagebox.showerror("Error", "scikit-learn is not installed.\nPlease run: pip install scikit-learn")
            return

        selected_run = self.ai_run_selector.get()
        if not selected_run or selected_run == "No backtests found" or not hasattr(self, '_ai_run_paths') or selected_run not in self._ai_run_paths:
            messagebox.showerror("Error", "Please select a valid backtest run from the dropdown.\nIf empty, run a backtest first and it will appear here.")
            return

        pkl_path = self._ai_run_paths[selected_run]

        symbol = self.ai_symbol.get().strip()
        tf_str = self.ai_tf.get().strip()

        tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                  "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
        tf_val = tf_map.get(tf_str, mt5.TIMEFRAME_M5)

        self.ai_btn.config(state=tk.DISABLED)
        self.ai_status.config(text=f"Status: Loading {selected_run}...", foreground="orange")
        self.ai_text.delete(1.0, tk.END)
        self.ai_text.insert(tk.END, f"Loading raw trade data from {selected_run}...\n")
        self.update()

        import threading
        threading.Thread(target=self._ai_analyst_thread, args=(pkl_path, symbol, tf_str, tf_val), daemon=True).start()

    def _ai_analyst_thread(self, pkl_path, symbol, tf_str, tf_val):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            all_results = data.get('results', [])
            symbols = data.get('symbols', [])

            if not all_results:
                self.after(0, lambda: messagebox.showerror("Error", "The selected backtest run is empty."))
                return

            best_run = max(all_results, key=lambda x: x[2].get('net_profit', 0))
            trades = best_run[1]
            train_symbol = symbols[0] if symbols else symbol

            self.after(0, lambda: self.ai_text.insert(tk.END, f"Extracted best run (RRR/Config: {best_run[0]}).\n"))
            self.after(0, lambda: self.ai_text.insert(tk.END, f"Found {len(trades)} historical trades. Training ML models...\n"))
            self.after(0, lambda: self.ai_status.config(text="Status: Training Machine Learning models...", foreground="orange"))

            ml_results = ml_train_and_analyze(trades, train_symbol, tf_val)

            if 'error' not in ml_results and len(trades) >= 40:
                self.after(0, lambda: self.ai_status.config(text="Status: Running Walk-Forward validation...", foreground="orange"))
                self.after(0, lambda: self.ai_text.insert(tk.END, "Running Walk-Forward Out-Of-Sample validation...\n"))
                wf_results = walk_forward_ml_analysis(trades, train_symbol, tf_val)
                if 'error' not in wf_results:
                    ml_results['walk_forward'] = wf_results
                    self.after(0, lambda: self.ai_text.insert(tk.END, f"\u2705 Walk-Forward complete: {wf_results['n_folds']} folds, Avg OOS Accuracy: {wf_results['avg_oos_accuracy']:.1f}%\n"))
                else:
                    self.after(0, lambda wf=wf_results: self.ai_text.insert(tk.END, f"\u26a0\ufe0f Walk-Forward skipped: {wf.get('error', 'Unknown')}\n"))

            if 'error' not in ml_results:
                model_path = os.path.join(CACHE_DIR, f"ml_model_{train_symbol}.pkl")
                ml_save_model(ml_results, filepath=model_path, extra_meta={'symbol': train_symbol})
                ml_load_model(model_path)
                best_name = ml_results.get('best_model_name', 'rf')
                _NAMES = {'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'nn': 'Neural Network',
                          'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'cat': 'CatBoost'}
                self.after(0, lambda: self.ai_text.insert(tk.END, f"\n\u2705 ML Model successfully trained on {len(trades)} trades and SAVED for {train_symbol}!\n"))
                self.after(0, lambda bn=best_name: self.ai_text.insert(tk.END, f"\U0001f3c6 Best model auto-selected: {_NAMES.get(bn, bn)} ({ml_results.get('best_model_accuracy', 0):.1f}%)\nYou can now enable 'ML Filter' in the Backtest or Live Trading tabs.\n"))

            report = ml_generate_ict_report(ml_results)

            self.after(0, lambda: self.ai_text.insert(tk.END, "\n" + report + "\n"))

            self.after(0, lambda: self.ai_status.config(text="Status: Reading current live chart...", foreground="orange"))
            self.after(0, lambda: self.ai_text.insert(tk.END, f"\n\n{'='*80}\n  \U0001f441\ufe0f CURRENT LIVE CHART READING ({symbol} {tf_str})\n{'='*80}\n\n"))

            df = get_data(symbol, tf_val, 200, live=True)
            if df is None or len(df) < 50:
                self.after(0, lambda: self.ai_text.insert(tk.END, "\u274c Could not fetch live chart data.\n"))
            else:
                analysis_df = df.iloc[:-1]

                _, _, swing_highs, swing_lows = detect_market_structure(analysis_df)
                ms_val = analysis_df.iloc[-1]['market_structure'] if 'market_structure' in analysis_df.columns else "Unknown"
                self.after(0, lambda: self.ai_text.insert(tk.END, f"Market Structure: {ms_val}\n"))

                obs_df = detect_order_blocks(analysis_df)
                total_obs = len(obs_df) if (obs_df is not None and not obs_df.empty) else 0
                self.after(0, lambda: self.ai_text.insert(tk.END, f"Order Blocks found: {total_obs}\n"))

                signals = generate_trade_signals(symbol, tf_str, tf_val,
                                                 datetime.datetime.now() - datetime.timedelta(days=2),
                                                 datetime.datetime.now(),
                                                 get_ict_model_parameters("Default"),
                                                 trailing_methods=[],
                                                 min_rrr=1.5,
                                                 require_bos_fvg=False)

                if not isinstance(signals, list):
                    signals = []

                active_signals = [s for s in signals if isinstance(s, dict) and s.get('entry_time') and s['entry_time'] > (datetime.datetime.now() - datetime.timedelta(hours=4))]

                if active_signals:
                    self.after(0, lambda: self.ai_text.insert(tk.END, f"\n\U0001f3af FOUND {len(active_signals)} RECENT SIGNALS:\n"))
                    for sig in active_signals:
                        self.after(0, lambda s=sig: self.ai_text.insert(tk.END, f"- {s.get('trade_direction', 'UNK')} via {s.get('ict_method', 'UNK')} @ {s.get('entry_price', 0.0):.5f}\n"))
                else:
                    self.after(0, lambda: self.ai_text.insert(tk.END, "\nNo active ICT signals currently forming.\n"))

            self.after(0, lambda: self.ai_status.config(text="Status: Idle", foreground="blue"))
        except Exception as e:
            import traceback
            logger.error("ML Tab Error: %s\n%s", e, traceback.format_exc())
            self.after(0, lambda msg=str(e): messagebox.showerror("AI Error", f"An error occurred: {msg}\nCheck console log for details."))
            self.after(0, lambda: self.ai_status.config(text="Status: Error", foreground="red"))
        finally:
            self.after(0, lambda: self.ai_btn.config(state=tk.NORMAL))
