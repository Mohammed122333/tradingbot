"""Chart Analysis tab for the trading UI — candlestick chart with ICT annotations."""
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    import MetaTrader5 as mt5
except ImportError:
    import tests.MetaTrader5 as mt5

from utils import get_spread, get_data
from indicators import (
    get_ict_model_parameters,
    detect_order_blocks,
    detect_fair_value_gaps,
    get_unmitigated_fvgs,
    detect_swing_points,
    detect_all_liquidity_sweeps,
    calculate_ote_zone,
    calculate_market_structure_chronologically,
)

logger = logging.getLogger()


class ChartMixin:
    """Mixin providing chart analysis tab for TradingUI."""

    def _create_chart_tab(self):
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="Chart Analysis")

        control_frame = ttk.Frame(self.chart_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(control_frame, text="Symbol:").pack(side=tk.LEFT, padx=3)
        self.chart_symbol = ttk.Entry(control_frame, width=15)
        self.chart_symbol.insert(0, "EURUSDm")
        self.chart_symbol.pack(side=tk.LEFT, padx=3)

        ttk.Label(control_frame, text="Timeframe:").pack(side=tk.LEFT, padx=3)
        self.chart_tf = ttk.Combobox(control_frame, values=["M1", "M5", "M15", "H1", "H4", "D1"], width=5)
        self.chart_tf.set("M15")
        self.chart_tf.pack(side=tk.LEFT, padx=3)

        ttk.Label(control_frame, text="Bars:").pack(side=tk.LEFT, padx=3)
        self.chart_bars = ttk.Entry(control_frame, width=5)
        self.chart_bars.insert(0, "200")
        self.chart_bars.pack(side=tk.LEFT, padx=3)

        ttk.Button(control_frame, text="Load Chart", command=self._plot_chart).pack(side=tk.LEFT, padx=10)

        toggles_frame = ttk.Frame(self.chart_frame)
        toggles_frame.pack(fill=tk.X, padx=5, pady=5)

        self.chart_show_fvg = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggles_frame, text="FVG", variable=self.chart_show_fvg, command=self._plot_chart).pack(side=tk.LEFT, padx=5)

        self.chart_show_ob = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggles_frame, text="Order Blocks", variable=self.chart_show_ob, command=self._plot_chart).pack(side=tk.LEFT, padx=5)

        self.chart_show_mss = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggles_frame, text="Market Structure", variable=self.chart_show_mss, command=self._plot_chart).pack(side=tk.LEFT, padx=5)

        self.chart_show_ote = tk.BooleanVar(value=False)
        ttk.Checkbutton(toggles_frame, text="OTE Zone", variable=self.chart_show_ote, command=self._plot_chart).pack(side=tk.LEFT, padx=5)

        self.chart_show_sweeps = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggles_frame, text="Liquidity Sweeps", variable=self.chart_show_sweeps, command=self._plot_chart).pack(side=tk.LEFT, padx=5)

        self.chart_show_bos = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggles_frame, text="BoS", variable=self.chart_show_bos, command=self._plot_chart).pack(side=tk.LEFT, padx=5)

        self.chart_plot_frame = ttk.Frame(self.chart_frame)
        self.chart_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chart_canvas = None

    def _plot_chart(self):
        symbol = self.chart_symbol.get().strip()
        tf_name = self.chart_tf.get().strip()
        try:
            bars = int(self.chart_bars.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number of bars.")
            return

        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1
        }
        if tf_name not in tf_map:
            messagebox.showerror("Error", "Invalid timeframe.")
            return

        if not mt5.initialize():
            messagebox.showerror("Error", "Failed to connect to MetaTrader 5. Make sure the terminal is open.")
            return

        mt5.symbol_select(symbol, True)

        df = get_data(symbol, tf_map[tf_name], bars, live=True)
        if df is None or df.empty:
            messagebox.showerror("Error", f"Failed to fetch data for {symbol}. Make sure the symbol is valid and selected in Market Watch.")
            return

        fig, ax = plt.subplots(figsize=(14, 7), dpi=120)
        MAX_ZONES = 6
        ZONE_EXTEND = 18
        MAX_STRUCT = 8

        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')

        ax.grid(color='#1e293b', linestyle=':', linewidth=0.5, alpha=0.5)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#1e293b')
            ax.spines[spine].set_linewidth(1.2)

        ax.tick_params(colors='#94a3b8', labelsize=8)

        x = np.arange(len(df))
        up = df.close >= df.open
        down = df.close < df.open

        ax.vlines(x[up], df.low[up], df.high[up], color='#10b981', linewidth=1.2)
        ax.vlines(x[down], df.low[down], df.high[down], color='#f43f5e', linewidth=1.2)
        ax.bar(x[up], df.close[up] - df.open[up], 0.6, bottom=df.open[up], color='#10b981', edgecolor='#10b981', linewidth=0.5)
        ax.bar(x[down], df.open[down] - df.close[down], 0.6, bottom=df.close[down], color='#f43f5e', edgecolor='#f43f5e', linewidth=0.5)

        if self.chart_show_fvg.get():
            all_fvgs = detect_fair_value_gaps(df)
            unmit = get_unmitigated_fvgs(df, all_fvgs)
            for fvg in unmit[-MAX_ZONES:]:
                try:
                    idx = df.index.get_loc(fvg['time'])
                    edge_color = '#10b981' if fvg['type'] == 'bullish' else '#f43f5e'
                    face_color = '#10b981' if fvg['type'] == 'bullish' else '#f43f5e'
                    ax.add_patch(patches.Rectangle(
                        (idx, fvg['bottom']),
                        min(len(df)-idx, ZONE_EXTEND),
                        fvg['top']-fvg['bottom'],
                        edgecolor=edge_color, facecolor=face_color,
                        linestyle='--', linewidth=0.8, alpha=0.12
                    ))
                except KeyError:
                    pass

        if self.chart_show_ob.get():
            ict_p = get_ict_model_parameters("Default", symbol)
            spread = get_spread(symbol)
            obs = detect_order_blocks(df, ict_p['threshold_factor'], ict_p['reversal_threshold'], ict_p['lookahead'], spread)
            for i, row in obs.tail(MAX_ZONES).iterrows():
                try:
                    idx = df.index.get_loc(i)
                    edge_color = '#6366f1' if row['order_block_type'] == 'bullish' else '#a855f7'
                    face_color = '#6366f1' if row['order_block_type'] == 'bullish' else '#a855f7'
                    ax.add_patch(patches.Rectangle(
                        (idx, row['bottom']),
                        min(len(df)-idx, ZONE_EXTEND),
                        row['top']-row['bottom'],
                        edgecolor=edge_color, facecolor=face_color,
                        linestyle='--', linewidth=0.8, alpha=0.12
                    ))
                except KeyError:
                    pass

        if self.chart_show_mss.get() or self.chart_show_bos.get():
            swing_highs, swing_lows = detect_swing_points(df, lookback=5)

            for sh in swing_highs:
                try:
                    idx = df.index.get_loc(sh['time'])
                    ax.plot(idx, sh['price'], 'o', color='#475569', markersize=3, alpha=0.5, zorder=5)
                except KeyError:
                    pass
            for sl_pt in swing_lows:
                try:
                    idx = df.index.get_loc(sl_pt['time'])
                    ax.plot(idx, sl_pt['price'], 'o', color='#475569', markersize=3, alpha=0.5, zorder=5)
                except KeyError:
                    pass

            states, mss_events, bos_events = calculate_market_structure_chronologically(df, lookback=5)

            if self.chart_show_mss.get():
                for _k, ev in enumerate(mss_events[-MAX_STRUCT:]):
                    try:
                        color = '#10b981' if ev['type'] == 'bullish' else '#f43f5e'
                        label = 'MSS \u2191' if ev['type'] == 'bullish' else 'MSS \u2193'
                        base = -12 if ev['type'] == 'bullish' else 6
                        ytext = base + (-8 if _k % 2 else 0)

                        ax.hlines(y=ev['price'], xmin=ev['idx'], xmax=ev['break_idx'],
                                  color=color, linestyle='--', linewidth=1.2, alpha=0.8)
                        ax.annotate(label, (ev['break_idx'], ev['price']), textcoords='offset points',
                                    xytext=(5, ytext), fontsize=7, fontweight='bold', color='#ffffff',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor=color, lw=1, alpha=0.9))
                    except Exception:
                        pass

            if self.chart_show_bos.get():
                for _k, ev in enumerate(bos_events[-MAX_STRUCT:]):
                    try:
                        color = '#10b981' if ev['type'] == 'bullish' else '#f43f5e'
                        label = 'BoS \u2191' if ev['type'] == 'bullish' else 'BoS \u2193'
                        base = -12 if ev['type'] == 'bullish' else 6
                        ytext = base + (-8 if _k % 2 else 0)

                        ax.hlines(y=ev['price'], xmin=ev['idx'], xmax=ev['break_idx'],
                                  color=color, linestyle='-', linewidth=1.0, alpha=0.8)
                        ax.annotate(label, (ev['break_idx'], ev['price']), textcoords='offset points',
                                    xytext=(5, ytext), fontsize=7, fontweight='bold', color='#ffffff',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f172a', edgecolor=color, lw=0.8, alpha=0.9))
                    except Exception:
                        pass

        if self.chart_show_sweeps.get():
            sweeps = detect_all_liquidity_sweeps(df)
            for sw in sweeps:
                try:
                    idx = sw['idx']
                    if sw['type'] == 'bullish':
                        ax.plot(idx, sw['price'], '^', color='#eab308', markersize=5, zorder=4)
                        ax.annotate('\u26a1', (idx, sw['price']), textcoords='offset points',
                                    xytext=(0, -10), fontsize=7, color='#eab308', ha='center')
                    else:
                        ax.plot(idx, sw['price'], 'v', color='#c084fc', markersize=5, zorder=4)
                        ax.annotate('\u26a1', (idx, sw['price']), textcoords='offset points',
                                    xytext=(0, 4), fontsize=7, color='#c084fc', ha='center')
                except Exception:
                    pass

        if self.chart_show_ote.get():
            ict_p = get_ict_model_parameters("Default")
            ote_low, ote_high, sl, sh = calculate_ote_zone(df, fib_low=ict_p['fib_low'], fib_high=ict_p['fib_high'])
            if ote_low != 0 and ote_high != 0:
                ax.axhspan(ote_low, ote_high, color='#06b6d4', alpha=0.08, linestyle=':', edgecolor='#06b6d4', linewidth=0.8)

        ax.set_title(f"{symbol} {tf_name} \u2014 ICT Institutional Structure", fontsize=10, fontweight='bold', color='#f1f5f9', pad=15)
        ax.set_xlim(-1, len(df))

        tick_indices = np.linspace(0, len(df)-1, 5, dtype=int)
        valid_indices = [i for i in tick_indices if i < len(df)]
        ax.set_xticks(valid_indices)
        ax.set_xticklabels([df.index[i].strftime('%m-%d %H:%M') for i in valid_indices])

        try:
            fig.tight_layout()
        except Exception:
            pass

        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()

        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_plot_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)
