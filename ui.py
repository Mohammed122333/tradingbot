import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from PIL import Image, ImageDraw, ImageTk
import numpy as np
from opt_worker import get_braille_sparkline
import datetime

def generate_sparkline_image(balances, width=100, height=28):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    if not balances or len(balances) <= 1:
        draw.line([(0, height // 2), (width - 1, height // 2)], fill=(128, 128, 128, 150), width=2)
        return img
        
    n = len(balances)
    # Sample balances to width points
    indices = (np.arange(width) * (n - 1) / (width - 1)).astype(int)
    points = np.array([balances[i] for i in indices], dtype=np.float64)
        
    min_val = points.min()
    max_val = points.max()
    val_range = max_val - min_val
    
    if val_range == 0:
        y_coords = np.full(width, height // 2, dtype=np.int32)
    else:
        norm = (points - min_val) / val_range
        y_coords = (2 + (1 - norm) * (height - 5)).astype(np.int32)

    is_positive = balances[-1] >= balances[0]

    # Build gradient fill using NumPy array operations (replaces per-pixel draw.point)
    import numpy as _np
    pixels = _np.zeros((height, width, 4), dtype=_np.uint8)
    
    # Create row indices grid
    row_indices = _np.arange(height).reshape(-1, 1)  # (height, 1)
    y_row = y_coords.reshape(1, -1)  # (1, width)
    
    # Mask: fill from y_coords[x] down to bottom for each column
    fill_mask = row_indices >= y_row  # (height, width)
    
    # Compute alpha gradient for each filled pixel
    # factor = (height - 1 - cy) / (height - 1 - y) where cy >= y
    denominator = _np.maximum(height - 1 - y_coords, 1).reshape(1, -1)  # avoid div by zero
    factor = (height - 1 - row_indices) / denominator  # (height, width)
    factor = _np.clip(factor, 0.0, 1.0)
    alpha = (5 + factor * 50).astype(_np.uint8)
    
    if is_positive:
        pixels[fill_mask, 0] = 46
        pixels[fill_mask, 1] = 204
        pixels[fill_mask, 2] = 113
    else:
        pixels[fill_mask, 0] = 231
        pixels[fill_mask, 1] = 76
        pixels[fill_mask, 2] = 60
    
    # Apply alpha only to filled pixels
    alpha_flat = _np.where(fill_mask, alpha, 0).astype(_np.uint8)
    pixels[:, :, 3] = alpha_flat
    
    # Convert NumPy array to PIL Image and composite
    fill_img = Image.fromarray(pixels, "RGBA")
    img = Image.alpha_composite(img, fill_img)
    draw = ImageDraw.Draw(img)

    coords = list(zip(range(width), y_coords.tolist()))
    if is_positive:
        line_color = (46, 204, 113, 255)
    else:
        line_color = (231, 76, 60, 255)
        
    draw.line(coords, fill=line_color, width=2, joint="round")
    return img
import logging
import time
import threading
from queue import Queue, Empty
import traceback
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os
import random
import numpy as np
import pandas as pd
try:
    import MetaTrader5 as mt5
except ImportError:
    import tests.MetaTrader5 as mt5
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")

import config
from utils import get_spread, get_data
from ui_chart_mixin import ChartMixin
from ui_ai_mixin import AIAnalystMixin
from trade_memory import clear_trade_memory, save_trade_memory
from indicators import (
    get_ict_model_parameters,
    detect_order_blocks,
    detect_fair_value_gaps,
    get_unmitigated_fvgs,
    detect_swing_points,
    detect_market_structure,
    detect_all_liquidity_sweeps,
    detect_liquidity_sweep,
    calculate_ote_zone
)
from simulation import calculate_confluence_score, get_higher_tf_trend
from live_scanner import run_live_trading
from backtester import combined_backtest, generate_trade_signals
from ml_engine import (
    ml_load_model,
    _ml_live_model,
    ML_AVAILABLE,
    ML_MODEL_FILE,
    ml_train_and_analyze,
    ml_generate_ict_report,
    ml_save_model,
    walk_forward_ml_analysis
)

logger = logging.getLogger()

class TradingUI(ChartMixin, AIAnalystMixin, tk.Tk):
    def __init__(self):
        super().__init__()
        self.after_ids = []
        self.ui_update_queue = Queue()
        self.title(f"ICT Strategy Framework v{config.BOT_VERSION} - Fixed (All 13 Issues Resolved)")
        self.geometry("1100x800")
        self.opt_images = {}
        # Pre-generate a single shared empty PhotoImage for all empty/waste balance curves
        empty_img = generate_sparkline_image([], width=100, height=28)
        self.empty_sparkline_photo = ImageTk.PhotoImage(empty_img)
        self.bt_base_cached_settings = {}
        self.bt_symbol_overrides = {}
        self.bt_currently_selected_override_symbol = None
        self.bt_currently_selected_override_method = None
        self.live_base_cached_settings = {}
        self.live_symbol_overrides = {}
        self.live_currently_selected_override_symbol = None
        self.live_currently_selected_override_method = None
        self.is_running = True
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.start_mt5_heartbeat()
        self.update_account_info()
        self.process_log_queue()
        self.process_ui_update_queue()
        self.update_activity()

    def start_mt5_heartbeat(self):
        def heartbeat_thread():
            import time
            import MetaTrader5 as mt5
            while getattr(self, 'is_running', False):
                try:
                    term_info = mt5.terminal_info()
                    if term_info is None:
                        # Attempt to initialize
                        initialized = mt5.initialize()
                        if not initialized:
                            self.ui_update_queue.put(lambda: self.connection_label.config(text="MT5: DISCONNECTED", foreground="red"))
                        else:
                            self.ui_update_queue.put(lambda: self.connection_label.config(text="MT5: CONNECTED", foreground="green"))
                    else:
                        if hasattr(term_info, 'connected') and not term_info.connected:
                            # Try to initialize again to trigger reconnect
                            mt5.initialize()
                            self.ui_update_queue.put(lambda: self.connection_label.config(text="MT5: NO BROKER CONNECTION", foreground="orange"))
                        else:
                            self.ui_update_queue.put(lambda: self.connection_label.config(text="MT5: CONNECTED", foreground="green"))
                except Exception as e:
                    logger.error("MT5 heartbeat error: %s", e)
                    self.ui_update_queue.put(lambda: self.connection_label.config(text="MT5: ERROR", foreground="red"))
                
                # Check every 5 seconds
                time.sleep(5)
                
        import threading
        threading.Thread(target=heartbeat_thread, daemon=True).start()

    def create_widgets(self):
        # Settings Top Frame
        settings_top = ttk.Frame(self)
        settings_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(settings_top, text="Save Settings", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(settings_top, text="Load Settings", command=self.load_config).pack(side=tk.LEFT, padx=5)
        
        # Real Parity Badge
        try:
            with open("parity_status.json") as f:
                _p = json.load(f)
            _n = int(_p.get("total_divergences", -1))
            _when = _p.get("generated_at", "?")
            if _n == 0:
                _txt, _fg = f"✅ PARITY CLEAN ({_when})", "#008000"
            else:
                _txt, _fg = f"⚠️ {_n} PARITY DIVERGENCES ({_when})", "#c00000"
        except Exception:
            _txt, _fg = "⚠️ PARITY UNVERIFIED — run parity_report.py", "#c07000"
        tk.Label(settings_top, text=_txt, fg=_fg, font=("Helvetica", 9, "bold")).pack(side=tk.RIGHT, padx=10)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self._create_chart_tab()
        self._create_ai_tab()

        # === LIVE TRADING TAB ===
        self.control_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.control_frame, text="Trading Controls")

        control_paned = ttk.PanedWindow(self.control_frame, orient=tk.VERTICAL)
        control_paned.pack(fill=tk.BOTH, expand=True)

        # Top pane: Scrollable controls container
        controls_container = ttk.Frame(control_paned)
        controls_canvas = tk.Canvas(controls_container, highlightthickness=0)
        controls_scrollbar = ttk.Scrollbar(controls_container, orient=tk.VERTICAL, command=controls_canvas.yview)
        controls_inner_frame = ttk.Frame(controls_canvas)

        def _on_controls_configure(e):
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        def _on_canvas_configure(e):
            controls_canvas.itemconfig(controls_canvas_window, width=e.width)

        controls_inner_frame.bind("<Configure>", _on_controls_configure)
        controls_canvas_window = controls_canvas.create_window((0, 0), window=controls_inner_frame, anchor="nw")
        controls_canvas.bind("<Configure>", _on_canvas_configure)
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        def _on_mousewheel(event):
            if event.num == 4:
                controls_canvas.yview_scroll(-2, "units")
            elif event.num == 5:
                controls_canvas.yview_scroll(2, "units")
            elif hasattr(event, 'delta') and event.delta:
                controls_canvas.yview_scroll(int(-1 * (event.delta / 60)), "units")

        controls_canvas.bind("<Enter>", lambda e: controls_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        controls_canvas.bind("<Leave>", lambda e: controls_canvas.unbind_all("<MouseWheel>"))
        controls_canvas.bind("<Button-4>", lambda e: controls_canvas.yview_scroll(-2, "units"))
        controls_canvas.bind("<Button-5>", lambda e: controls_canvas.yview_scroll(2, "units"))

        controls_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control_paned.add(controls_container, weight=3)

        # Bottom pane: Dedicated Log and Status container
        self.log_container = ttk.Frame(control_paned)
        control_paned.add(self.log_container, weight=2)

        live_frame = ttk.LabelFrame(controls_inner_frame, text="Live Trading")
        live_frame.pack(fill=tk.X, padx=5, pady=5)

        live_control_frame = ttk.Frame(live_frame)
        live_control_frame.pack(fill=tk.X, padx=5, pady=5)
        self.start_btn = ttk.Button(live_control_frame, text="Start Live Trading", command=self.start_live_trading)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(live_control_frame, text="Stop Live Trading", command=self.stop_live_trading, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.clear_memory_btn = ttk.Button(live_control_frame, text="Clear Trade Memory", command=self.clear_trade_memory)
        self.clear_memory_btn.pack(side=tk.LEFT, padx=5)
        self.launch_mt5_btn = ttk.Button(live_control_frame, text="Launch MT5 Terminal", command=self.launch_mt5_terminal)
        self.launch_mt5_btn.pack(side=tk.LEFT, padx=5)

        live_settings_frame = ttk.Frame(live_frame)
        live_settings_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(live_settings_frame, text="Risk (%):").pack(side=tk.LEFT, padx=3)
        self.risk_entry = ttk.Entry(live_settings_frame, width=5)
        self.risk_entry.insert(0, "1.0")
        self.risk_entry.pack(side=tk.LEFT, padx=3)
        ttk.Label(live_settings_frame, text="Symbols:").pack(side=tk.LEFT, padx=3)
        self.live_symbols_entry = ttk.Entry(live_settings_frame, width=30)
        self.live_symbols_entry.insert(0, "BTCUSDm, ETHUSDm, XAUUSDm, EURUSDm, US30m, USTECm, GBPUSDm, USDJPYm")
        self.live_symbols_entry.pack(side=tk.LEFT, padx=3)
        
        # Checkbox for symbol-specific overrides - Live
        self.live_use_overrides_var = tk.BooleanVar(value=False)
        self.live_use_overrides_cb = ttk.Checkbutton(
            live_settings_frame, text="Use Symbol Profiles", 
            variable=self.live_use_overrides_var, 
            command=self.on_toggle_live_overrides,
            state=tk.DISABLED
        )
        self.live_use_overrides_cb.pack(side=tk.LEFT, padx=(15, 3))
        
        ttk.Label(live_settings_frame, text="Profile:").pack(side=tk.LEFT, padx=3)
        self.live_override_symbol_dropdown = ttk.Combobox(live_settings_frame, width=12, state="disabled")
        self.live_override_symbol_dropdown.pack(side=tk.LEFT, padx=3)
        self.live_override_symbol_dropdown.bind("<<ComboboxSelected>>", self.on_select_live_override_symbol)
        
        # Checkbox for method-specific overrides - Live
        self.live_use_method_overrides_var = tk.BooleanVar(value=False)
        self.live_use_method_overrides_cb = ttk.Checkbutton(
            live_settings_frame, text="Method Overrides",
            variable=self.live_use_method_overrides_var,
            command=self.on_toggle_live_method_overrides,
            state=tk.DISABLED
        )
        self.live_use_method_overrides_cb.pack(side=tk.LEFT, padx=(15, 3))
        
        ttk.Label(live_settings_frame, text="Method:").pack(side=tk.LEFT, padx=3)
        self.live_override_method_dropdown = ttk.Combobox(live_settings_frame, width=15, state="disabled", values=config.ICT_METHODS)
        self.live_override_method_dropdown.pack(side=tk.LEFT, padx=3)
        self.live_override_method_dropdown.bind("<<ComboboxSelected>>", self.on_select_live_override_method)
        
        self.live_symbols_entry.bind("<KeyRelease>", lambda e: self.check_live_symbol_count_for_overrides())
        
        
        live_settings2 = ttk.Frame(live_frame)
        live_settings2.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(live_settings2, text="Min RRR:").pack(side=tk.LEFT, padx=3)
        self.min_rrr_entry = ttk.Entry(live_settings2, width=5)
        self.min_rrr_entry.insert(0, "1.5")
        self.min_rrr_entry.pack(side=tk.LEFT, padx=3)
        
        self.use_dynamic_rrr_live = tk.BooleanVar(value=True)
        ttk.Checkbutton(live_settings2, text="Dynamic RRR (Structure TP)", variable=self.use_dynamic_rrr_live).pack(side=tk.LEFT, padx=3)

        live_settings3 = ttk.Frame(live_frame)
        live_settings3.pack(fill=tk.X, padx=5, pady=2)
        self.trade_all_tfs_live = tk.BooleanVar(value=True)
        ttk.Checkbutton(live_settings3, text="Multi-TF", variable=self.trade_all_tfs_live).pack(side=tk.LEFT, padx=3)
        self.use_ultra_low_tf_live = tk.BooleanVar(value=False)
        ttk.Checkbutton(live_settings3, text="Ultra Low TF", variable=self.use_ultra_low_tf_live).pack(side=tk.LEFT, padx=3)
        self.session_filter_live = tk.BooleanVar(value=True)
        self.session_filter_live_cb = ttk.Checkbutton(live_settings3, text="Session Filter", variable=self.session_filter_live)
        self.session_filter_live_cb.pack(side=tk.LEFT, padx=3)
        ttk.Label(live_settings3, text="Session (Broker):").pack(side=tk.LEFT, padx=3)
        self.session_start_live = ttk.Entry(live_settings3, width=3)
        self.session_start_live.insert(0, "7")
        self.session_start_live.pack(side=tk.LEFT)
        ttk.Label(live_settings3, text="-").pack(side=tk.LEFT)
        self.session_end_live = ttk.Entry(live_settings3, width=3)
        self.session_end_live.insert(0, "21")
        self.session_end_live.pack(side=tk.LEFT)
        self.htf_filter_live = tk.BooleanVar(value=True)
        ttk.Checkbutton(live_settings3, text="HTF Filter", variable=self.htf_filter_live).pack(side=tk.LEFT, padx=3)
        self.bypass_htf_live = tk.BooleanVar(value=False)
        ttk.Checkbutton(live_settings3, text="Bypass HTF if Conf>=2", variable=self.bypass_htf_live).pack(side=tk.LEFT, padx=3)
        self.ote_filter_live = tk.BooleanVar(value=True)
        ttk.Checkbutton(live_settings3, text="OTE Filter", variable=self.ote_filter_live).pack(side=tk.LEFT, padx=3)

        risk_mode_frame = ttk.Frame(live_frame)
        risk_mode_frame.pack(fill=tk.X, padx=5, pady=2)
        self.risk_mode_var = tk.StringVar(value="Risk")
        ttk.Label(risk_mode_frame, text="Risk Mode:").pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(risk_mode_frame, text="Risk %", variable=self.risk_mode_var, value="Risk").pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(risk_mode_frame, text="Fixed Lot", variable=self.risk_mode_var, value="Fixed").pack(side=tk.LEFT, padx=3)
        ttk.Label(risk_mode_frame, text="Fixed Lot:").pack(side=tk.LEFT, padx=3)
        self.fixed_lot_entry = ttk.Entry(risk_mode_frame, width=5)
        self.fixed_lot_entry.insert(0, "0.1")
        self.fixed_lot_entry.pack(side=tk.LEFT, padx=3)

        live_risk_mgr_frame = ttk.Frame(live_frame)
        live_risk_mgr_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(live_risk_mgr_frame, text="Daily Loss %:").pack(side=tk.LEFT, padx=3)
        self.live_daily_loss = ttk.Entry(live_risk_mgr_frame, width=4)
        self.live_daily_loss.insert(0, str(config.DAILY_LOSS_LIMIT_PCT))
        self.live_daily_loss.pack(side=tk.LEFT, padx=3)
        ttk.Label(live_risk_mgr_frame, text="Max Concurr:").pack(side=tk.LEFT, padx=3)
        self.live_max_concurr = ttk.Entry(live_risk_mgr_frame, width=4)
        self.live_max_concurr.insert(0, str(config.MAX_CONCURRENT_TRADES))
        self.live_max_concurr.pack(side=tk.LEFT, padx=3)
        ttk.Label(live_risk_mgr_frame, text="Min FVG Size:").pack(side=tk.LEFT, padx=3)
        self.live_min_fvg = ttk.Entry(live_risk_mgr_frame, width=4)
        self.live_min_fvg.insert(0, str(config.MIN_FVG_SIZE_SPREADS))
        self.live_min_fvg.pack(side=tk.LEFT, padx=3)
        ttk.Label(live_risk_mgr_frame, text="Min Conf Score:").pack(side=tk.LEFT, padx=3)
        self.live_min_conf = ttk.Entry(live_risk_mgr_frame, width=4)
        self.live_min_conf.insert(0, str(config.MIN_CONFLUENCE_SCORE))
        self.live_min_conf.pack(side=tk.LEFT, padx=3)
        ttk.Label(live_risk_mgr_frame, text="Limit Expiry (hrs):").pack(side=tk.LEFT, padx=3)
        self.live_limit_expiry = ttk.Entry(live_risk_mgr_frame, width=4)
        self.live_limit_expiry.insert(0, str(getattr(config, 'PENDING_LIMIT_EXPIRY_HOURS', 2.0)))
        self.live_limit_expiry.pack(side=tk.LEFT, padx=3)

        self.live_swap_free = tk.BooleanVar(value=bool(getattr(config, 'SWAP_FREE_ACCOUNT', True)))
        ttk.Checkbutton(live_risk_mgr_frame, text="Swap-Free Account (0 Swap)", variable=self.live_swap_free, command=lambda: setattr(config, 'SWAP_FREE_ACCOUNT', self.live_swap_free.get())).pack(side=tk.LEFT, padx=5)

        method_frame = ttk.Frame(live_frame)
        method_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(method_frame, text="Methods:").pack(side=tk.LEFT, padx=3)
        self.live_methods_listbox = tk.Listbox(method_frame, selectmode=tk.MULTIPLE, height=6, exportselection=False)
        for idx, m in enumerate(config.ICT_METHODS):
            self.live_methods_listbox.insert(tk.END, m)
            if m in config.CORE_METHODS:
                self.live_methods_listbox.select_set(idx)
        scroll1 = ttk.Scrollbar(method_frame, orient=tk.VERTICAL, command=self.live_methods_listbox.yview)
        self.live_methods_listbox.config(yscrollcommand=scroll1.set)
        self.live_methods_listbox.pack(side=tk.LEFT, padx=3)
        scroll1.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # FVG SL Mode (radio buttons)
        self.live_fvg_sl_frame = ttk.LabelFrame(method_frame, text="FVG SL Mode")
        self.live_fvg_sl_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        self.live_fvg_sl_mode = tk.StringVar(value=config.FVG_SL_NORMAL)
        for mode in config.FVG_SL_MODES:
            ttk.Radiobutton(self.live_fvg_sl_frame, text=mode, variable=self.live_fvg_sl_mode, value=mode).pack(anchor=tk.W)
        
        self.live_require_bos_fvg = tk.BooleanVar(value=False)
        self.live_require_bos_cb = ttk.Checkbutton(method_frame, text="Require BoS after MSS (FVG)", variable=self.live_require_bos_fvg, state=tk.NORMAL)
        self.live_require_bos_cb.pack(side=tk.LEFT, padx=5)

        # FVG Advanced Filters frame
        self.live_fvg_filters_frame = ttk.LabelFrame(method_frame, text="FVG Advanced Filters")
        self.live_fvg_filters_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        self.live_fvg_displacement_only = tk.BooleanVar(value=True)
        self.live_fvg_displacement_cb = ttk.Checkbutton(self.live_fvg_filters_frame, text="Displacement Filter", variable=self.live_fvg_displacement_only)
        self.live_fvg_displacement_cb.pack(anchor=tk.W)
        
        self.live_fvg_discount_premium_only = tk.BooleanVar(value=True)
        self.live_fvg_discount_premium_cb = ttk.Checkbutton(self.live_fvg_filters_frame, text="Discount/Premium", variable=self.live_fvg_discount_premium_only)
        self.live_fvg_discount_premium_cb.pack(anchor=tk.W)
        
        self.live_fvg_recent_sweep_only = tk.BooleanVar(value=False)
        self.live_fvg_recent_sweep_cb = ttk.Checkbutton(self.live_fvg_filters_frame, text="Recent Sweep Only", variable=self.live_fvg_recent_sweep_only)
        self.live_fvg_recent_sweep_cb.pack(anchor=tk.W)
        
        self.live_sb_require_htf_bias = tk.BooleanVar(value=False)
        self.live_sb_require_htf_bias_cb = ttk.Checkbutton(self.live_fvg_filters_frame, text="Strict MTF Alignment (H1/H4)", variable=self.live_sb_require_htf_bias)
        self.live_sb_require_htf_bias_cb.pack(anchor=tk.W)
        
        self.live_methods_listbox.bind('<<ListboxSelect>>', self.on_live_methods_select)
        
        ttk.Label(method_frame, text="Trail Methods:").pack(side=tk.LEFT, padx=(10, 3))
        self.live_trail_methods_listbox = tk.Listbox(method_frame, selectmode=tk.MULTIPLE, height=6, exportselection=False)
        for idx, m in enumerate(config.ICT_METHODS):
            self.live_trail_methods_listbox.insert(tk.END, m)
            if m in config.CORE_METHODS:
                self.live_trail_methods_listbox.select_set(idx)
        scroll_ts1 = ttk.Scrollbar(method_frame, orient=tk.VERTICAL, command=self.live_trail_methods_listbox.yview)
        self.live_trail_methods_listbox.config(yscrollcommand=scroll_ts1.set)
        self.live_trail_methods_listbox.pack(side=tk.LEFT, padx=3)
        scroll_ts1.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        self.on_live_methods_select(None)

        # Trail Stop Type (radio buttons) - Live
        trail_type_frame_live = ttk.LabelFrame(live_frame, text="Trailing Stop Type")
        trail_type_frame_live.pack(fill=tk.X, padx=5, pady=2)
        self.trail_type_live = tk.StringVar(value=config.TRAIL_TYPE_PARTIAL)
        ttk.Radiobutton(trail_type_frame_live, text="True Partial 50%", variable=self.trail_type_live, value=config.TRAIL_TYPE_PARTIAL).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(trail_type_frame_live, text="Profit Lock (50%)", variable=self.trail_type_live, value=config.TRAIL_TYPE_ATR).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(trail_type_frame_live, text="% Trail", variable=self.trail_type_live, value=config.TRAIL_TYPE_PERCENT).pack(side=tk.LEFT, padx=3)
        ttk.Label(trail_type_frame_live, text="  Trail %:").pack(side=tk.LEFT, padx=2)
        self.trail_pct_live = ttk.Entry(trail_type_frame_live, width=4)
        self.trail_pct_live.insert(0, str(config.DEFAULT_TRAIL_PERCENT))
        self.trail_pct_live.pack(side=tk.LEFT, padx=1)

        # Anti-Gap SL Protection - Live
        antigap_frame_live = ttk.LabelFrame(live_frame, text="Anti-Gap SL Protection (prevents flash crash slippage)")
        antigap_frame_live.pack(fill=tk.X, padx=5, pady=2)
        self.anti_gap_enabled_live = tk.BooleanVar(value=True)
        ttk.Checkbutton(antigap_frame_live, text="Enable", variable=self.anti_gap_enabled_live,
                        command=self._toggle_anti_gap).pack(side=tk.LEFT, padx=3)
        ttk.Label(antigap_frame_live, text="ATR Multiplier:").pack(side=tk.LEFT, padx=3)
        self.anti_gap_atr_mult_live = ttk.Entry(antigap_frame_live, width=4)
        self.anti_gap_atr_mult_live.insert(0, str(config.ANTI_GAP_ATR_MULTIPLIER))
        self.anti_gap_atr_mult_live.pack(side=tk.LEFT, padx=3)
        ttk.Label(antigap_frame_live, text="(Min SL = ATR × mult. Lot auto-reduced. Higher = safer from gaps)").pack(side=tk.LEFT, padx=3)

        # Slippage Recovery toggle - Live
        recovery_frame_live = ttk.Frame(live_frame)
        recovery_frame_live.pack(fill=tk.X, padx=5, pady=2)
        self.slippage_recovery_live = tk.BooleanVar(value=False)
        ttk.Checkbutton(recovery_frame_live, text="Enable Slippage Recovery", variable=self.slippage_recovery_live, command=self._toggle_slippage_recovery).pack(side=tk.LEFT, padx=3)

        # Elite Institutional Features - Live
        inst_live_frame = ttk.LabelFrame(live_frame, text="🏛 Elite Institutional Features")
        inst_live_frame.pack(fill=tk.X, padx=5, pady=5)
        
        inst_live_row1 = ttk.Frame(inst_live_frame); inst_live_row1.pack(fill=tk.X, pady=2)
        inst_live_row2 = ttk.Frame(inst_live_frame); inst_live_row2.pack(fill=tk.X, pady=2)
        
        # SMT Divergence
        self.live_smt_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(inst_live_row1, text="Require SMT Divergence", variable=self.live_smt_enabled).pack(side=tk.LEFT, padx=3)
        ttk.Label(inst_live_row1, text="Correlated Pair(s):").pack(side=tk.LEFT, padx=3)
        self.live_smt_pair = ttk.Combobox(inst_live_row1, width=15, values=["DXY,GBPUSD", "DXY", "EURUSD", "GBPUSD", "XAGUSD", "SPX500", "US30", "NQ100"])
        self.live_smt_pair.pack(side=tk.LEFT, padx=3)
        
        # Volume Profile
        self.live_vp_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(inst_live_row1, text="Use Volume Profile (POC/HVN)", variable=self.live_vp_enabled).pack(side=tk.LEFT, padx=(15, 3))
        
        # LLM Sentiment
        self.live_llm_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(inst_live_row1, text="LLM News Sentiment", variable=self.live_llm_enabled).pack(side=tk.LEFT, padx=(15, 3))
        
        # Macro News Filter
        self.live_news_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(inst_live_row2, text="Macro News Filter (Red Folder)", variable=self.live_news_enabled).pack(side=tk.LEFT, padx=3)
        ttk.Label(inst_live_row2, text="Buffer (mins):").pack(side=tk.LEFT, padx=3)
        self.live_news_buffer = ttk.Entry(inst_live_row2, width=5)
        self.live_news_buffer.insert(0, "30")
        self.live_news_buffer.pack(side=tk.LEFT, padx=3)

        # === Pro Strategy Upgrades (v17) ===
        pro_frame = ttk.LabelFrame(live_frame, text="🏆 Pro Strategy Upgrades (v17) — Live")
        pro_frame.pack(fill=tk.X, padx=5, pady=2)
        pro_row1 = ttk.Frame(pro_frame); pro_row1.pack(fill=tk.X, padx=3, pady=1)
        pro_row2 = ttk.Frame(pro_frame); pro_row2.pack(fill=tk.X, padx=3, pady=1)

        self.pro_dol_tp = tk.BooleanVar(value=bool(getattr(config, 'DOL_TP_ENABLED', True)))
        ttk.Checkbutton(pro_row1, text="Draw-on-Liquidity TP", variable=self.pro_dol_tp, command=self._apply_pro_flags).pack(side=tk.LEFT, padx=3)
        self.pro_killzone = tk.BooleanVar(value=bool(getattr(config, 'KILLZONE_ENABLED', True)))
        ttk.Checkbutton(pro_row1, text="Killzone Weighting", variable=self.pro_killzone, command=self._apply_pro_flags).pack(side=tk.LEFT, padx=3)
        self.pro_htf_poi = tk.BooleanVar(value=bool(getattr(config, 'HTF_POI_ENABLED', True)))
        ttk.Checkbutton(pro_row1, text="HTF POI Nesting", variable=self.pro_htf_poi, command=self._apply_pro_flags).pack(side=tk.LEFT, padx=3)
        self.pro_mandatory = tk.BooleanVar(value=bool(getattr(config, 'CONFLUENCE_REQUIRE_MANDATORY', True)))
        ttk.Checkbutton(pro_row1, text="Mandatory Sweep+Displacement", variable=self.pro_mandatory, command=self._apply_pro_flags).pack(side=tk.LEFT, padx=3)

        self.pro_regime = tk.BooleanVar(value=bool(getattr(config, 'REGIME_FILTER_ENABLED', True)))
        ttk.Checkbutton(pro_row2, text="Regime Filter (stand aside in chop)", variable=self.pro_regime, command=self._apply_pro_flags).pack(side=tk.LEFT, padx=3)
        self.pro_ml_sizing = tk.BooleanVar(value=bool(getattr(config, 'ML_SIZING_ENABLED', True)))
        ttk.Checkbutton(pro_row2, text="ML Conviction Sizing", variable=self.pro_ml_sizing, command=self._apply_pro_flags).pack(side=tk.LEFT, padx=3)
        self.pro_ml_rank = tk.BooleanVar(value=bool(getattr(config, 'ML_RANK_ENABLED', True)))
        ttk.Checkbutton(pro_row2, text="ML Setup Ranking", variable=self.pro_ml_rank, command=self._apply_pro_flags).pack(side=tk.LEFT, padx=3)
        self.pro_multi_tf_conf = tk.BooleanVar(value=bool(getattr(config, 'MULTI_TF_CONFLUENCE_ENABLED', False)))
        ttk.Checkbutton(pro_row2, text="Multi-TF Confluence", variable=self.pro_multi_tf_conf, command=self._apply_pro_flags).pack(side=tk.LEFT, padx=3)
        self.pro_multi_tf_gate = tk.BooleanVar(value=bool(getattr(config, 'MULTI_TF_GATE_ENABLED', False)))
        ttk.Checkbutton(pro_row2, text="Multi-TF Gate", variable=self.pro_multi_tf_gate, command=self._apply_pro_flags).pack(side=tk.LEFT, padx=3)
        self.pro_regime_adaptive = tk.BooleanVar(value=bool(getattr(config, 'REGIME_ADAPTIVE_PARAMS_ENABLED', False)))
        ttk.Checkbutton(pro_row2, text="Regime Adaptive Params", variable=self.pro_regime_adaptive, command=self._apply_pro_flags).pack(side=tk.LEFT, padx=3)
        ttk.Button(pro_row2, text="Train Regime...", command=self.train_regime_csv).pack(side=tk.LEFT, padx=5)
        self._apply_pro_flags()

        # === ML Live Autonomous Trading ===
        ml_live_frame = ttk.LabelFrame(live_frame, text="🧠 ML Autonomous Trading (Requires Trained Model)")
        ml_live_frame.pack(fill=tk.X, padx=5, pady=2)
        self.ml_live_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(ml_live_frame, text="Enable ML Filter", variable=self.ml_live_enabled).pack(side=tk.LEFT, padx=3)
        ttk.Label(ml_live_frame, text="Min Conf%:").pack(side=tk.LEFT, padx=3)
        self.ml_live_min_conf = ttk.Entry(ml_live_frame, width=4)
        self.ml_live_min_conf.insert(0, "60")
        self.ml_live_min_conf.pack(side=tk.LEFT, padx=3)
        self.ml_live_load_btn = ttk.Button(ml_live_frame, text="Load Saved Model", command=self.load_ml_model)
        self.ml_live_load_btn.pack(side=tk.LEFT, padx=10)
        self.ml_live_status = ttk.Label(ml_live_frame, text="Status: No Model Loaded", foreground="red")
        self.ml_live_status.pack(side=tk.LEFT, padx=3)

        activity_frame = ttk.Frame(live_frame)
        activity_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(activity_frame, text="Activity:").pack(side=tk.LEFT, padx=3)
        self.activity_label = ttk.Label(activity_frame, text="Idle")
        self.activity_label.pack(side=tk.LEFT, padx=3)

        # === BACKTEST TAB ===
        self.backtest_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.backtest_frame, text="Backtest")

        # Sub Notebook inside Backtest Frame
        self.bt_sub_notebook = ttk.Notebook(self.backtest_frame)
        self.bt_sub_notebook.pack(fill=tk.BOTH, expand=True)

        self.single_bt_tab = ttk.Frame(self.bt_sub_notebook)
        self.bt_sub_notebook.add(self.single_bt_tab, text="Single Run / Smart Optimize")

        self.optimization_tab = ttk.Frame(self.bt_sub_notebook)
        self.bt_sub_notebook.add(self.optimization_tab, text="Grid Optimization")

        self.oos_test_tab = ttk.Frame(self.bt_sub_notebook)
        self.bt_sub_notebook.add(self.oos_test_tab, text="Out of Sample Test")

        # Row 1: Symbols and Dates
        bt_row1 = ttk.Frame(self.single_bt_tab)
        bt_row1.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(bt_row1, text="Symbols:").pack(side=tk.LEFT, padx=3)
        self.backtest_symbol_entry = ttk.Entry(bt_row1, width=25)
        self.backtest_symbol_entry.insert(0, "BTCUSDm, ETHUSDm")
        self.backtest_symbol_entry.pack(side=tk.LEFT, padx=3)
        ttk.Label(bt_row1, text="From (YYYY-MM-DD):").pack(side=tk.LEFT, padx=3)
        self.backtest_date_from = ttk.Entry(bt_row1, width=12)
        self.backtest_date_from.insert(0, "2026-02-13")
        self.backtest_date_from.pack(side=tk.LEFT, padx=3)
        ttk.Label(bt_row1, text="To (YYYY-MM-DD):").pack(side=tk.LEFT, padx=3)
        self.backtest_date_to = ttk.Entry(bt_row1, width=12)
        self.backtest_date_to.insert(0, "2026-03-13")
        self.backtest_date_to.pack(side=tk.LEFT, padx=3)

        # Checkbox for symbol-specific overrides
        self.bt_use_overrides_var = tk.BooleanVar(value=False)
        self.bt_use_overrides_cb = ttk.Checkbutton(
            bt_row1, text="Use Symbol Profiles", 
            variable=self.bt_use_overrides_var, 
            command=self.on_toggle_bt_overrides,
            state=tk.DISABLED
        )
        self.bt_use_overrides_cb.pack(side=tk.LEFT, padx=(15, 3))
        
        ttk.Label(bt_row1, text="Profile:").pack(side=tk.LEFT, padx=3)
        self.bt_override_symbol_dropdown = ttk.Combobox(bt_row1, width=12, state="disabled")
        self.bt_override_symbol_dropdown.pack(side=tk.LEFT, padx=3)
        self.bt_override_symbol_dropdown.bind("<<ComboboxSelected>>", self.on_select_override_symbol)
        
        # Checkbox for method-specific overrides
        self.bt_use_method_overrides_var = tk.BooleanVar(value=False)
        self.bt_use_method_overrides_cb = ttk.Checkbutton(
            bt_row1, text="Method Overrides",
            variable=self.bt_use_method_overrides_var,
            command=self.on_toggle_bt_method_overrides,
            state=tk.DISABLED
        )
        self.bt_use_method_overrides_cb.pack(side=tk.LEFT, padx=(15, 3))
        
        ttk.Label(bt_row1, text="Method:").pack(side=tk.LEFT, padx=3)
        self.bt_override_method_dropdown = ttk.Combobox(bt_row1, width=15, state="disabled", values=config.ICT_METHODS)
        self.bt_override_method_dropdown.pack(side=tk.LEFT, padx=3)
        self.bt_override_method_dropdown.bind("<<ComboboxSelected>>", self.on_select_override_method)
        
        self.backtest_symbol_entry.bind("<KeyRelease>", lambda e: self.check_symbol_count_for_overrides())

        # Row 2: Balance, Risk, Lot
        bt_row2 = ttk.Frame(self.single_bt_tab)
        bt_row2.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(bt_row2, text="Balance:").pack(side=tk.LEFT, padx=3)
        self.backtest_balance_entry = ttk.Entry(bt_row2, width=10)
        self.backtest_balance_entry.insert(0, "10000")
        self.backtest_balance_entry.pack(side=tk.LEFT, padx=3)
        ttk.Label(bt_row2, text="Risk %:").pack(side=tk.LEFT, padx=3)
        self.backtest_risk_entry = ttk.Entry(bt_row2, width=5)
        self.backtest_risk_entry.insert(0, "1.0")
        self.backtest_risk_entry.pack(side=tk.LEFT, padx=3)
        ttk.Label(bt_row2, text="Fixed Lot:").pack(side=tk.LEFT, padx=3)
        self.backtest_fixed_lot_entry = ttk.Entry(bt_row2, width=5)
        self.backtest_fixed_lot_entry.insert(0, "0.1")
        self.backtest_fixed_lot_entry.pack(side=tk.LEFT, padx=3)
        self.backtest_use_fixed_lot = tk.BooleanVar(value=False)
        ttk.Checkbutton(bt_row2, text="Use Fixed Lot", variable=self.backtest_use_fixed_lot).pack(side=tk.LEFT, padx=3)
        ttk.Label(bt_row2, text="RRR Min:").pack(side=tk.LEFT, padx=2)
        self.backtest_min_rrr_entry = ttk.Entry(bt_row2, width=4)
        self.backtest_min_rrr_entry.insert(0, "1.5")
        self.backtest_min_rrr_entry.pack(side=tk.LEFT, padx=1)
        ttk.Label(bt_row2, text="Max:").pack(side=tk.LEFT, padx=2)
        self.backtest_max_rrr_entry = ttk.Entry(bt_row2, width=4)
        self.backtest_max_rrr_entry.insert(0, "1.5")
        self.backtest_max_rrr_entry.pack(side=tk.LEFT, padx=1)
        ttk.Label(bt_row2, text="Step:").pack(side=tk.LEFT, padx=2)
        self.backtest_incr_rrr_entry = ttk.Entry(bt_row2, width=4)
        self.backtest_incr_rrr_entry.insert(0, "0.5")
        self.backtest_incr_rrr_entry.pack(side=tk.LEFT, padx=1)

        # Row 3: Method, TF, Trailing
        bt_row3 = ttk.Frame(self.single_bt_tab)
        bt_row3.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(bt_row3, text="Methods:").pack(side=tk.LEFT, padx=3)
        self.bt_methods_listbox = tk.Listbox(bt_row3, selectmode=tk.MULTIPLE, height=6, exportselection=False)
        for idx, m in enumerate(config.ICT_METHODS):
            self.bt_methods_listbox.insert(tk.END, m)
            if m in config.CORE_METHODS:
                self.bt_methods_listbox.select_set(idx)
        scroll2 = ttk.Scrollbar(bt_row3, orient=tk.VERTICAL, command=self.bt_methods_listbox.yview)
        self.bt_methods_listbox.config(yscrollcommand=scroll2.set)
        self.bt_methods_listbox.pack(side=tk.LEFT, padx=3)
        scroll2.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # FVG SL Mode (radio buttons)
        self.bt_fvg_sl_frame = ttk.LabelFrame(bt_row3, text="FVG SL Mode")
        self.bt_fvg_sl_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        self.bt_fvg_sl_mode = tk.StringVar(value=config.FVG_SL_NORMAL)
        for mode in config.FVG_SL_MODES:
            ttk.Radiobutton(self.bt_fvg_sl_frame, text=mode, variable=self.bt_fvg_sl_mode, value=mode).pack(anchor=tk.W)
        
        self.bt_require_bos_fvg = tk.BooleanVar(value=False)
        self.bt_require_bos_cb = ttk.Checkbutton(bt_row3, text="Require BoS after MSS (FVG)", variable=self.bt_require_bos_fvg, state=tk.NORMAL)
        self.bt_require_bos_cb.pack(side=tk.LEFT, padx=5)

        # FVG Advanced Filters frame
        self.bt_fvg_filters_frame = ttk.LabelFrame(bt_row3, text="FVG Advanced Filters")
        self.bt_fvg_filters_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        self.bt_fvg_displacement_only = tk.BooleanVar(value=True)
        self.bt_fvg_displacement_cb = ttk.Checkbutton(self.bt_fvg_filters_frame, text="Displacement Filter", variable=self.bt_fvg_displacement_only)
        self.bt_fvg_displacement_cb.pack(anchor=tk.W)
        
        self.bt_fvg_discount_premium_only = tk.BooleanVar(value=True)
        self.bt_fvg_discount_premium_cb = ttk.Checkbutton(self.bt_fvg_filters_frame, text="Discount/Premium", variable=self.bt_fvg_discount_premium_only)
        self.bt_fvg_discount_premium_cb.pack(anchor=tk.W)
        
        self.bt_fvg_recent_sweep_only = tk.BooleanVar(value=False)
        self.bt_fvg_recent_sweep_cb = ttk.Checkbutton(self.bt_fvg_filters_frame, text="Recent Sweep Only", variable=self.bt_fvg_recent_sweep_only)
        self.bt_fvg_recent_sweep_cb.pack(anchor=tk.W)
        
        self.bt_sb_require_htf_bias = tk.BooleanVar(value=False)
        self.bt_sb_require_htf_bias_cb = ttk.Checkbutton(self.bt_fvg_filters_frame, text="Strict MTF Alignment (H1/H4)", variable=self.bt_sb_require_htf_bias)
        self.bt_sb_require_htf_bias_cb.pack(anchor=tk.W)
        
        self.bt_methods_listbox.bind('<<ListboxSelect>>', self.on_bt_methods_select)
        
        self.trade_all_tfs_backtest = tk.BooleanVar(value=True)
        ttk.Checkbutton(bt_row3, text="Multi-TF", variable=self.trade_all_tfs_backtest).pack(side=tk.LEFT, padx=3)
        self.use_ultra_low_tf_backtest = tk.BooleanVar(value=False)
        ttk.Checkbutton(bt_row3, text="Ultra Low TF", variable=self.use_ultra_low_tf_backtest).pack(side=tk.LEFT, padx=3)
        self.use_dynamic_rrr_backtest = tk.BooleanVar(value=True)
        ttk.Checkbutton(bt_row3, text="Dynamic RRR (Structure TP)", variable=self.use_dynamic_rrr_backtest).pack(side=tk.LEFT, padx=3)
        
        ttk.Label(bt_row3, text="Trail Methods:").pack(side=tk.LEFT, padx=(10, 3))
        self.bt_trail_methods_listbox = tk.Listbox(bt_row3, selectmode=tk.MULTIPLE, height=6, exportselection=False)
        for idx, m in enumerate(config.ICT_METHODS):
            self.bt_trail_methods_listbox.insert(tk.END, m)
            if m in config.CORE_METHODS:
                self.bt_trail_methods_listbox.select_set(idx)
        scroll_ts2 = ttk.Scrollbar(bt_row3, orient=tk.VERTICAL, command=self.bt_trail_methods_listbox.yview)
        self.bt_trail_methods_listbox.config(yscrollcommand=scroll_ts2.set)
        self.bt_trail_methods_listbox.pack(side=tk.LEFT, padx=3)
        scroll_ts2.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        self.on_bt_methods_select(None)

        # Trail Stop Type (radio buttons) - Backtest
        trail_type_frame_bt = ttk.LabelFrame(self.single_bt_tab, text="Trailing Stop Type")
        trail_type_frame_bt.pack(fill=tk.X, padx=5, pady=2)
        self.trail_type_bt = tk.StringVar(value=config.TRAIL_TYPE_PARTIAL)
        ttk.Radiobutton(trail_type_frame_bt, text="True Partial 50%", variable=self.trail_type_bt, value=config.TRAIL_TYPE_PARTIAL).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(trail_type_frame_bt, text="Profit Lock (50%)", variable=self.trail_type_bt, value=config.TRAIL_TYPE_ATR).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(trail_type_frame_bt, text="% Trail", variable=self.trail_type_bt, value=config.TRAIL_TYPE_PERCENT).pack(side=tk.LEFT, padx=3)
        ttk.Label(trail_type_frame_bt, text="  Trail %:").pack(side=tk.LEFT, padx=2)
        self.trail_pct_bt = ttk.Entry(trail_type_frame_bt, width=4)
        self.trail_pct_bt.insert(0, str(config.DEFAULT_TRAIL_PERCENT))
        self.trail_pct_bt.pack(side=tk.LEFT, padx=1)

        # Row 4: Realism Settings
        realism_frame = ttk.LabelFrame(self.single_bt_tab, text="Backtest Realism Settings")
        realism_frame.pack(fill=tk.X, padx=5, pady=3)
        bt_row4 = ttk.Frame(realism_frame)
        bt_row4.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(bt_row4, text="Spread Cost:").pack(side=tk.LEFT, padx=3)
        self.bt_spread_cost = ttk.Entry(bt_row4, width=8)
        self.bt_spread_cost.insert(0, "0.0")
        self.bt_spread_cost.pack(side=tk.LEFT, padx=3)
        ttk.Button(bt_row4, text="Auto-Fill Spread", command=self.auto_fill_spread).pack(side=tk.LEFT, padx=3)
        ttk.Label(bt_row4, text="Slippage (pts):").pack(side=tk.LEFT, padx=3)
        self.bt_slippage = ttk.Entry(bt_row4, width=5)
        self.bt_slippage.insert(0, "5")
        self.bt_slippage.pack(side=tk.LEFT, padx=3)
        ttk.Label(bt_row4, text="Commission/Lot ($):").pack(side=tk.LEFT, padx=3)
        self.bt_commission = ttk.Entry(bt_row4, width=5)
        self.bt_commission.insert(0, "7.0")
        self.bt_commission.pack(side=tk.LEFT, padx=3)
        self.bt_session_filter = tk.BooleanVar(value=True)
        self.bt_session_filter_cb = ttk.Checkbutton(bt_row4, text="Session Filter", variable=self.bt_session_filter)
        self.bt_session_filter_cb.pack(side=tk.LEFT, padx=3)
        ttk.Label(bt_row4, text="Session(Broker):").pack(side=tk.LEFT, padx=3)
        self.bt_session_start = ttk.Entry(bt_row4, width=3)
        self.bt_session_start.insert(0, "7")
        self.bt_session_start.pack(side=tk.LEFT)
        ttk.Label(bt_row4, text="-").pack(side=tk.LEFT)
        self.bt_session_end = ttk.Entry(bt_row4, width=3)
        self.bt_session_end.insert(0, "21")
        self.bt_session_end.pack(side=tk.LEFT)
        self.htf_filter_backtest = tk.BooleanVar(value=True)
        ttk.Checkbutton(bt_row4, text="HTF Filter", variable=self.htf_filter_backtest).pack(side=tk.LEFT, padx=3)
        self.bypass_htf_backtest = tk.BooleanVar(value=False)
        ttk.Checkbutton(bt_row4, text="Bypass HTF if Conf>=2", variable=self.bypass_htf_backtest).pack(side=tk.LEFT, padx=3)
        self.ote_filter_backtest = tk.BooleanVar(value=True)
        ttk.Checkbutton(bt_row4, text="OTE Filter", variable=self.ote_filter_backtest).pack(side=tk.LEFT, padx=3)

        bt_filter_frame = ttk.Frame(realism_frame)
        bt_filter_frame.pack(fill=tk.X, padx=5, pady=3)
        self.bt_slippage_recovery_bt = tk.BooleanVar(value=True)
        ttk.Checkbutton(bt_filter_frame, text="Enable Slippage Recovery", variable=self.bt_slippage_recovery_bt).pack(side=tk.LEFT, padx=3)
        ttk.Label(bt_filter_frame, text="Daily Loss %:").pack(side=tk.LEFT, padx=3)
        self.bt_daily_loss = ttk.Entry(bt_filter_frame, width=4)
        self.bt_daily_loss.insert(0, str(config.DAILY_LOSS_LIMIT_PCT))
        self.bt_daily_loss.pack(side=tk.LEFT, padx=3)
        ttk.Label(bt_filter_frame, text="Max Concurr:").pack(side=tk.LEFT, padx=3)
        self.bt_max_concurr = ttk.Entry(bt_filter_frame, width=4)
        self.bt_max_concurr.insert(0, str(config.MAX_CONCURRENT_TRADES))
        self.bt_max_concurr.pack(side=tk.LEFT, padx=3)
        ttk.Label(bt_filter_frame, text="Min FVG Size:").pack(side=tk.LEFT, padx=3)
        self.bt_min_fvg = ttk.Entry(bt_filter_frame, width=4)
        self.bt_min_fvg.insert(0, str(config.MIN_FVG_SIZE_SPREADS))
        self.bt_min_fvg.pack(side=tk.LEFT, padx=3)
        ttk.Label(bt_filter_frame, text="Min Conf:").pack(side=tk.LEFT, padx=3)
        self.bt_min_conf = ttk.Entry(bt_filter_frame, width=3)
        self.bt_min_conf.insert(0, str(config.MIN_CONFLUENCE_SCORE))
        self.bt_min_conf.pack(side=tk.LEFT, padx=3)
        
        self.bt_anti_gap = tk.BooleanVar(value=True)
        ttk.Checkbutton(bt_filter_frame, text="Anti-Gap SL", variable=self.bt_anti_gap).pack(side=tk.LEFT, padx=3)
        self.bt_anti_gap_mult = ttk.Entry(bt_filter_frame, width=3)
        self.bt_anti_gap_mult.insert(0, str(config.ANTI_GAP_ATR_MULTIPLIER))
        self.bt_anti_gap_mult.pack(side=tk.LEFT, padx=3)
        
        ttk.Label(bt_filter_frame, text="SL Spread Buf:").pack(side=tk.LEFT, padx=3)
        self.bt_sl_spread_buffer = ttk.Entry(bt_filter_frame, width=3)
        self.bt_sl_spread_buffer.insert(0, "2.0")
        self.bt_sl_spread_buffer.pack(side=tk.LEFT, padx=3)
        
        self.bt_limit_touch_fill = tk.BooleanVar(value=False)
        ttk.Checkbutton(bt_filter_frame, text="Touch Fills", variable=self.bt_limit_touch_fill).pack(side=tk.LEFT, padx=3)

        self.bt_use_mt5_data = tk.BooleanVar(value=not getattr(config, 'OFFLINE_BACKTESTING', True))
        self.bt_use_mt5_data_cb = ttk.Checkbutton(
            bt_filter_frame, text="Use MT5 data", 
            variable=self.bt_use_mt5_data, 
            command=self.on_toggle_use_mt5_data
        )
        self.bt_use_mt5_data_cb.pack(side=tk.LEFT, padx=3)

        # Elite Institutional Features - Backtest
        inst_bt_frame = ttk.LabelFrame(self.single_bt_tab, text="🏛 Elite Institutional Features")
        inst_bt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        inst_bt_row1 = ttk.Frame(inst_bt_frame); inst_bt_row1.pack(fill=tk.X, pady=2)
        inst_bt_row2 = ttk.Frame(inst_bt_frame); inst_bt_row2.pack(fill=tk.X, pady=2)
        
        # SMT Divergence
        self.bt_smt_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(inst_bt_row1, text="Require SMT Divergence", variable=self.bt_smt_enabled).pack(side=tk.LEFT, padx=3)
        ttk.Label(inst_bt_row1, text="Correlated Pair(s):").pack(side=tk.LEFT, padx=3)
        self.bt_smt_pair = ttk.Combobox(inst_bt_row1, width=15, values=["DXY,GBPUSD", "DXY", "EURUSD", "GBPUSD", "XAGUSD", "SPX500", "US30", "NQ100"])
        self.bt_smt_pair.pack(side=tk.LEFT, padx=3)
        
        # Volume Profile
        self.bt_vp_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(inst_bt_row1, text="Use Volume Profile", variable=self.bt_vp_enabled).pack(side=tk.LEFT, padx=(15, 3))
        
        # LLM Sentiment (Disabled in Backtest)
        self.bt_llm_enabled = tk.BooleanVar(value=False)
        llm_cb = ttk.Checkbutton(inst_bt_row1, text="LLM News Sentiment (Live Only)", variable=self.bt_llm_enabled)
        llm_cb.pack(side=tk.LEFT, padx=(15, 3))
        llm_cb.config(state=tk.DISABLED)
        
        # Macro News Filter
        self.bt_news_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(inst_bt_row2, text="Macro News Filter (Red Folder)", variable=self.bt_news_enabled).pack(side=tk.LEFT, padx=3)
        ttk.Label(inst_bt_row2, text="Buffer (mins):").pack(side=tk.LEFT, padx=3)
        self.bt_news_buffer = ttk.Entry(inst_bt_row2, width=5)
        self.bt_news_buffer.insert(0, "30")
        self.bt_news_buffer.pack(side=tk.LEFT, padx=3)

        # === Pro Strategy Upgrades (v17) - Backtest (independent of the Live tab) ===
        bt_pro_frame = ttk.LabelFrame(self.single_bt_tab, text="🏆 Pro Strategy Upgrades (v17) — Backtest")
        bt_pro_frame.pack(fill=tk.X, padx=5, pady=5)
        bt_pro_row1 = ttk.Frame(bt_pro_frame); bt_pro_row1.pack(fill=tk.X, padx=3, pady=1)
        bt_pro_row2 = ttk.Frame(bt_pro_frame); bt_pro_row2.pack(fill=tk.X, padx=3, pady=1)

        self.bt_pro_dol_tp = tk.BooleanVar(value=bool(getattr(config, 'DOL_TP_ENABLED', True)))
        ttk.Checkbutton(bt_pro_row1, text="Draw-on-Liquidity TP", variable=self.bt_pro_dol_tp, command=lambda: self._apply_pro_flags('bt')).pack(side=tk.LEFT, padx=3)
        self.bt_pro_killzone = tk.BooleanVar(value=bool(getattr(config, 'KILLZONE_ENABLED', True)))
        ttk.Checkbutton(bt_pro_row1, text="Killzone Weighting", variable=self.bt_pro_killzone, command=lambda: self._apply_pro_flags('bt')).pack(side=tk.LEFT, padx=3)
        self.bt_pro_htf_poi = tk.BooleanVar(value=bool(getattr(config, 'HTF_POI_ENABLED', True)))
        ttk.Checkbutton(bt_pro_row1, text="HTF POI Nesting", variable=self.bt_pro_htf_poi, command=lambda: self._apply_pro_flags('bt')).pack(side=tk.LEFT, padx=3)
        self.bt_pro_mandatory = tk.BooleanVar(value=bool(getattr(config, 'CONFLUENCE_REQUIRE_MANDATORY', True)))
        ttk.Checkbutton(bt_pro_row1, text="Mandatory Sweep+Displacement", variable=self.bt_pro_mandatory, command=lambda: self._apply_pro_flags('bt')).pack(side=tk.LEFT, padx=3)

        self.bt_pro_regime = tk.BooleanVar(value=bool(getattr(config, 'REGIME_FILTER_ENABLED', True)))
        ttk.Checkbutton(bt_pro_row2, text="Regime Filter (stand aside in chop)", variable=self.bt_pro_regime, command=lambda: self._apply_pro_flags('bt')).pack(side=tk.LEFT, padx=3)
        self.bt_pro_ml_sizing = tk.BooleanVar(value=bool(getattr(config, 'ML_SIZING_ENABLED', True)))
        ttk.Checkbutton(bt_pro_row2, text="ML Conviction Sizing", variable=self.bt_pro_ml_sizing, command=lambda: self._apply_pro_flags('bt')).pack(side=tk.LEFT, padx=3)
        self.bt_pro_ml_rank = tk.BooleanVar(value=bool(getattr(config, 'ML_RANK_ENABLED', True)))
        ttk.Checkbutton(bt_pro_row2, text="ML Setup Ranking", variable=self.bt_pro_ml_rank, command=lambda: self._apply_pro_flags('bt')).pack(side=tk.LEFT, padx=3)
        self.bt_pro_multi_tf_conf = tk.BooleanVar(value=bool(getattr(config, 'MULTI_TF_CONFLUENCE_ENABLED', False)))
        ttk.Checkbutton(bt_pro_row2, text="Multi-TF Confluence", variable=self.bt_pro_multi_tf_conf, command=lambda: self._apply_pro_flags('bt')).pack(side=tk.LEFT, padx=3)
        self.bt_pro_multi_tf_gate = tk.BooleanVar(value=bool(getattr(config, 'MULTI_TF_GATE_ENABLED', False)))
        ttk.Checkbutton(bt_pro_row2, text="Multi-TF Gate", variable=self.bt_pro_multi_tf_gate, command=lambda: self._apply_pro_flags('bt')).pack(side=tk.LEFT, padx=3)
        self.bt_pro_regime_adaptive = tk.BooleanVar(value=bool(getattr(config, 'REGIME_ADAPTIVE_PARAMS_ENABLED', False)))
        ttk.Checkbutton(bt_pro_row2, text="Regime Adaptive Params", variable=self.bt_pro_regime_adaptive, command=lambda: self._apply_pro_flags('bt')).pack(side=tk.LEFT, padx=3)
        ttk.Button(bt_pro_row2, text="Train Regime...", command=self.train_regime_csv).pack(side=tk.LEFT, padx=5)

        # Run button + Progress bar
        bt_run = ttk.Frame(self.single_bt_tab)
        bt_run.pack(fill=tk.X, padx=5, pady=5)
        self.run_backtest_btn = ttk.Button(bt_run, text="Run Backtest", command=self.run_backtest)
        self.run_backtest_btn.pack(side=tk.LEFT, padx=5)
        self.stop_backtest_btn = ttk.Button(bt_run, text="Stop Backtest", command=self.stop_backtest, state=tk.DISABLED)
        self.stop_backtest_btn.pack(side=tk.LEFT, padx=5)
        self.copy_settings_btn = ttk.Button(bt_run, text="Copy Settings to Live", command=self.copy_settings_to_live)
        self.copy_settings_btn.pack(side=tk.LEFT, padx=5)
        
        self.bt_smart_optimize = tk.BooleanVar(value=False)
        ttk.Checkbutton(bt_run, text="Smart Optimize (Grid Search)", variable=self.bt_smart_optimize).pack(side=tk.LEFT, padx=15)
        
        # ML Filter controls
        self.bt_ml_filter = tk.BooleanVar(value=False)
        ttk.Checkbutton(bt_run, text="ML Filter", variable=self.bt_ml_filter).pack(side=tk.LEFT, padx=5)
        ttk.Label(bt_run, text="Min Conf%:").pack(side=tk.LEFT, padx=2)
        self.bt_ml_min_confidence = ttk.Entry(bt_run, width=4)
        self.bt_ml_min_confidence.insert(0, "60")
        self.bt_ml_min_confidence.pack(side=tk.LEFT, padx=2)

        progress_frame = ttk.LabelFrame(self.single_bt_tab, text="Backtest Progress")
        progress_frame.pack(fill=tk.X, padx=5, pady=3)
        self.bt_progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.bt_progress_bar.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        self.bt_progress_label = ttk.Label(progress_frame, text="Idle")
        self.bt_progress_label.pack(side=tk.LEFT, padx=5)
        self.bt_elapsed_label = ttk.Label(progress_frame, text="")
        self.bt_elapsed_label.pack(side=tk.LEFT, padx=5)
        self.bt_start_time = None

        # === GRID OPTIMIZATION TAB WIDGETS ===
        self.opt_paned = ttk.PanedWindow(self.optimization_tab, orient=tk.VERTICAL)
        self.opt_paned.pack(fill=tk.BOTH, expand=True)

        self.opt_top_container = ttk.Frame(self.opt_paned)
        self.opt_paned.add(self.opt_top_container, weight=1)

        self.opt_canvas = tk.Canvas(self.opt_top_container, highlightthickness=0)
        self.opt_scrollbar = ttk.Scrollbar(self.opt_top_container, orient="vertical", command=self.opt_canvas.yview)
        self.opt_top_pane = ttk.Frame(self.opt_canvas)

        self.opt_top_pane.bind("<Configure>", lambda e: self.opt_canvas.configure(scrollregion=self.opt_canvas.bbox("all")))
        self.opt_canvas_window = self.opt_canvas.create_window((0, 0), window=self.opt_top_pane, anchor="nw")
        
        # Ensure frame expands to canvas width
        self.opt_canvas.bind("<Configure>", lambda e: self.opt_canvas.itemconfig(self.opt_canvas_window, width=e.width))
        self.opt_canvas.configure(yscrollcommand=self.opt_scrollbar.set)

        self.opt_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.opt_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            if str(self.opt_canvas.winfo_toplevel().focus_get()) == str(self.opt_canvas) or True: # basic catch-all for mousewheel
                self.opt_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.opt_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.opt_bottom_pane = ttk.Frame(self.opt_paned)
        self.opt_paned.add(self.opt_bottom_pane, weight=0)

        # 1. Sync & Basic Settings Frame
        opt_controls = ttk.Frame(self.opt_top_pane)
        opt_controls.pack(fill=tk.X, padx=5, pady=3)
        
        ttk.Button(opt_controls, text="Sync Settings from Backtest", command=self.sync_opt_settings_from_backtest).pack(side=tk.LEFT, padx=3)
        
        ttk.Label(opt_controls, text="Symbol:").pack(side=tk.LEFT, padx=3)
        self.opt_symbol_entry = ttk.Entry(opt_controls, width=10)
        self.opt_symbol_entry.insert(0, "BTCUSDm")
        self.opt_symbol_entry.pack(side=tk.LEFT, padx=3)
        
        ttk.Label(opt_controls, text="From:").pack(side=tk.LEFT, padx=3)
        self.opt_date_from = ttk.Entry(opt_controls, width=11)
        self.opt_date_from.insert(0, "2026-02-13")
        self.opt_date_from.pack(side=tk.LEFT, padx=3)
        
        ttk.Label(opt_controls, text="To:").pack(side=tk.LEFT, padx=3)
        self.opt_date_to = ttk.Entry(opt_controls, width=11)
        self.opt_date_to.insert(0, "2026-03-13")
        self.opt_date_to.pack(side=tk.LEFT, padx=3)
        
        ttk.Label(opt_controls, text="Balance:").pack(side=tk.LEFT, padx=3)
        self.opt_balance_entry = ttk.Entry(opt_controls, width=8)
        self.opt_balance_entry.insert(0, "10000")
        self.opt_balance_entry.pack(side=tk.LEFT, padx=3)
        
        ttk.Label(opt_controls, text="Risk %:").pack(side=tk.LEFT, padx=3)
        self.opt_risk_entry = ttk.Entry(opt_controls, width=4)
        self.opt_risk_entry.insert(0, "0.25")
        self.opt_risk_entry.pack(side=tk.LEFT, padx=3)
        
        self.opt_use_fixed_lot = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_controls, text="Fixed Lot", variable=self.opt_use_fixed_lot).pack(side=tk.LEFT, padx=3)
        self.opt_fixed_lot_entry = ttk.Entry(opt_controls, width=4)
        self.opt_fixed_lot_entry.insert(0, "0.1")
        self.opt_fixed_lot_entry.pack(side=tk.LEFT, padx=3)

        # Basic settings row 2
        opt_controls_r2 = ttk.Frame(self.opt_top_pane)
        opt_controls_r2.pack(fill=tk.X, padx=5, pady=3)
        
        ttk.Label(opt_controls_r2, text="Spread Cost:").pack(side=tk.LEFT, padx=3)
        self.opt_spread_cost = ttk.Entry(opt_controls_r2, width=6)
        self.opt_spread_cost.insert(0, "0.0")
        self.opt_spread_cost.pack(side=tk.LEFT, padx=3)
        
        ttk.Label(opt_controls_r2, text="Slippage (pts):").pack(side=tk.LEFT, padx=3)
        self.opt_slippage = ttk.Entry(opt_controls_r2, width=4)
        self.opt_slippage.insert(0, "5")
        self.opt_slippage.pack(side=tk.LEFT, padx=3)
        
        ttk.Label(opt_controls_r2, text="Comm/Lot ($):").pack(side=tk.LEFT, padx=3)
        self.opt_commission = ttk.Entry(opt_controls_r2, width=5)
        self.opt_commission.insert(0, "7.0")
        self.opt_commission.pack(side=tk.LEFT, padx=3)
        
        self.opt_session_filter = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_controls_r2, text="Session Filter", variable=self.opt_session_filter).pack(side=tk.LEFT, padx=3)
        self.opt_session_start = ttk.Entry(opt_controls_r2, width=3)
        self.opt_session_start.insert(0, "7")
        self.opt_session_start.pack(side=tk.LEFT, padx=1)
        ttk.Label(opt_controls_r2, text="-").pack(side=tk.LEFT)
        self.opt_session_end = ttk.Entry(opt_controls_r2, width=3)
        self.opt_session_end.insert(0, "21")
        self.opt_session_end.pack(side=tk.LEFT, padx=1)
        
        self.opt_anti_gap = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_controls_r2, text="Anti-Gap SL", variable=self.opt_anti_gap).pack(side=tk.LEFT, padx=3)
        self.opt_anti_gap_mult = ttk.Entry(opt_controls_r2, width=3)
        self.opt_anti_gap_mult.insert(0, "2.0")
        self.opt_anti_gap_mult.pack(side=tk.LEFT, padx=3)
        
        self.opt_min_fvg_lbl = ttk.Label(opt_controls_r2, text="Min FVG Size:")
        self.opt_min_fvg_lbl.pack(side=tk.LEFT, padx=3)
        self.opt_min_fvg = ttk.Entry(opt_controls_r2, width=4)
        self.opt_min_fvg.insert(0, "0.0")
        self.opt_min_fvg.pack(side=tk.LEFT, padx=3)

        self.opt_use_mt5_data = tk.BooleanVar(value=not getattr(config, 'OFFLINE_BACKTESTING', True))
        self.opt_use_mt5_data_cb = ttk.Checkbutton(
            opt_controls_r2, text="Use MT5 data",
            variable=self.opt_use_mt5_data,
            command=self.on_toggle_opt_use_mt5_data
        )
        self.opt_use_mt5_data_cb.pack(side=tk.LEFT, padx=3)

        # Basic settings row 3 (Realism and other defaults)
        opt_controls_r3 = ttk.Frame(self.opt_top_pane)
        opt_controls_r3.pack(fill=tk.X, padx=5, pady=3)
        
        self.opt_trade_all_tfs = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_controls_r3, text="Multi-TF", variable=self.opt_trade_all_tfs).pack(side=tk.LEFT, padx=3)
        
        self.opt_use_ultra_low_tf = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_controls_r3, text="Ultra Low TF", variable=self.opt_use_ultra_low_tf).pack(side=tk.LEFT, padx=3)
        
        self.opt_bypass_htf_conf = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_controls_r3, text="Bypass HTF if Conf>=2", variable=self.opt_bypass_htf_conf).pack(side=tk.LEFT, padx=3)
        
        self.opt_require_bos_fvg = tk.BooleanVar(value=False)
        self.opt_require_bos_fvg_btn = ttk.Checkbutton(opt_controls_r3, text="Require BoS (FVG)", variable=self.opt_require_bos_fvg)
        self.opt_require_bos_fvg_btn.pack(side=tk.LEFT, padx=3)
        
        self.opt_slippage_recovery = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_controls_r3, text="Slippage Recovery", variable=self.opt_slippage_recovery).pack(side=tk.LEFT, padx=3)
        
        self.opt_use_dynamic_rrr = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_controls_r3, text="Dynamic RRR", variable=self.opt_use_dynamic_rrr).pack(side=tk.LEFT, padx=3)
        
        ttk.Label(opt_controls_r3, text="Min Conf:").pack(side=tk.LEFT, padx=3)
        self.opt_min_conf = ttk.Entry(opt_controls_r3, width=3)
        self.opt_min_conf.insert(0, "2")
        self.opt_min_conf.pack(side=tk.LEFT, padx=3)
        
        ttk.Label(opt_controls_r3, text="Daily Loss %:").pack(side=tk.LEFT, padx=3)
        self.opt_daily_loss = ttk.Entry(opt_controls_r3, width=4)
        self.opt_daily_loss.insert(0, "2.0")
        self.opt_daily_loss.pack(side=tk.LEFT, padx=3)
        
        ttk.Label(opt_controls_r3, text="Trail stop %:").pack(side=tk.LEFT, padx=3)
        self.opt_trail_pct = ttk.Entry(opt_controls_r3, width=4)
        self.opt_trail_pct.insert(0, "1.0")
        self.opt_trail_pct.pack(side=tk.LEFT, padx=3)

        # 2. Parameters Grid Frame
        opt_params_frame = ttk.LabelFrame(self.opt_top_pane, text="Grid Options to Sweep (Select Multiple)")
        opt_params_frame.pack(fill=tk.X, padx=5, pady=3)
        
        # Grid layout for sweep choices
        for col_idx in range(4):
            opt_params_frame.columnconfigure(col_idx, weight=1, uniform="group1")
        # Column 0: Methods
        col0 = ttk.Frame(opt_params_frame)
        col0.grid(row=0, column=0, sticky=tk.N+tk.S+tk.W+tk.E, padx=5, pady=2)
        ttk.Label(col0, text="1. Methods Sweep", font=('', 9, 'bold')).pack(anchor=tk.W, pady=2)
        
        self.opt_sweep_fvg = tk.BooleanVar(value=True)
        self.opt_sweep_fvg.trace_add("write", lambda *args: self.update_opt_fvg_widgets_state())
        self.opt_sweep_fvg_btn = ttk.Checkbutton(col0, text="FVG Return", variable=self.opt_sweep_fvg)
        self.opt_sweep_fvg_btn.pack(anchor=tk.W)
        self.opt_sweep_sb = tk.BooleanVar(value=True)
        self.opt_sweep_sb.trace_add("write", lambda *args: self.update_opt_fvg_widgets_state())
        self.opt_sweep_sb_btn = ttk.Checkbutton(col0, text="Silver Bullet", variable=self.opt_sweep_sb)
        self.opt_sweep_sb_btn.pack(anchor=tk.W)
        self.opt_sweep_mmxm = tk.BooleanVar(value=False)
        self.opt_sweep_mmxm_btn = ttk.Checkbutton(col0, text="MMXM", variable=self.opt_sweep_mmxm)
        self.opt_sweep_mmxm_btn.pack(anchor=tk.W)
        self.opt_sweep_2022 = tk.BooleanVar(value=False)
        self.opt_sweep_2022_btn = ttk.Checkbutton(col0, text="ICT Model 2022", variable=self.opt_sweep_2022)
        self.opt_sweep_2022_btn.pack(anchor=tk.W)
        self.opt_sweep_2025 = tk.BooleanVar(value=False)
        self.opt_sweep_2025_btn = ttk.Checkbutton(col0, text="ICT Model 2025", variable=self.opt_sweep_2025)
        self.opt_sweep_2025_btn.pack(anchor=tk.W)
        
        self.opt_sweep_combined_profitable = tk.BooleanVar(value=True)
        self.opt_sweep_combined_profitable_btn = ttk.Checkbutton(col0, text="Test Combined Profitable", variable=self.opt_sweep_combined_profitable)
        self.opt_sweep_combined_profitable_btn.pack(anchor=tk.W, pady=(5, 0))

        # Column 1: SL & Trailing Types
        col1 = ttk.Frame(opt_params_frame)
        col1.grid(row=0, column=1, sticky=tk.N+tk.S+tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(col1, text="2. SL Modes Sweep", font=('', 9, 'bold')).pack(anchor=tk.W, pady=2)
        self.opt_sweep_sl_normal = tk.BooleanVar(value=True)
        self.opt_sweep_sl_normal_btn = ttk.Checkbutton(col1, text="Normal SL", variable=self.opt_sweep_sl_normal)
        self.opt_sweep_sl_normal_btn.pack(anchor=tk.W)
        self.opt_sweep_sl_last_sweep = tk.BooleanVar(value=True)
        self.opt_sweep_sl_last_sweep_btn = ttk.Checkbutton(col1, text="Last Sweep SL", variable=self.opt_sweep_sl_last_sweep)
        self.opt_sweep_sl_last_sweep_btn.pack(anchor=tk.W)
        self.opt_sweep_sl_candle_ext = tk.BooleanVar(value=False)
        self.opt_sweep_sl_candle_ext_btn = ttk.Checkbutton(col1, text="Candle Extreme SL", variable=self.opt_sweep_sl_candle_ext)
        self.opt_sweep_sl_candle_ext_btn.pack(anchor=tk.W)
        
        ttk.Label(col1, text="3. Trailing Stop Sweep", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(5, 2))
        self.opt_sweep_trail_none = tk.BooleanVar(value=True)
        self.opt_sweep_trail_none_btn = ttk.Checkbutton(col1, text="No Trailing Stop", variable=self.opt_sweep_trail_none)
        self.opt_sweep_trail_none_btn.pack(anchor=tk.W)
        self.opt_sweep_trail_partial = tk.BooleanVar(value=True)
        self.opt_sweep_trail_partial_btn = ttk.Checkbutton(col1, text="True Partial (50% at 1R)", variable=self.opt_sweep_trail_partial)
        self.opt_sweep_trail_partial_btn.pack(anchor=tk.W)
        self.opt_sweep_trail_pct = tk.BooleanVar(value=False)
        self.opt_sweep_trail_pct_btn = ttk.Checkbutton(col1, text="Percentage Trail", variable=self.opt_sweep_trail_pct)
        self.opt_sweep_trail_pct_btn.pack(anchor=tk.W)
        self.opt_sweep_trail_atr = tk.BooleanVar(value=False)
        self.opt_sweep_trail_atr_btn = ttk.Checkbutton(col1, text="Profit Lock (50%)", variable=self.opt_sweep_trail_atr)
        self.opt_sweep_trail_atr_btn.pack(anchor=tk.W)

        ttk.Label(col1, text="3b. Apply Trailing to:", font=('', 9, 'bold')).pack(anchor=tk.W, pady=(5, 2))
        self.opt_trail_fvg = tk.BooleanVar(value=True)
        self.opt_trail_fvg_btn = ttk.Checkbutton(col1, text="FVG Return", variable=self.opt_trail_fvg)
        self.opt_trail_fvg_btn.pack(anchor=tk.W)
        self.opt_trail_sb = tk.BooleanVar(value=True)
        self.opt_trail_sb_btn = ttk.Checkbutton(col1, text="Silver Bullet", variable=self.opt_trail_sb)
        self.opt_trail_sb_btn.pack(anchor=tk.W)
        self.opt_trail_mmxm = tk.BooleanVar(value=True)
        self.opt_trail_mmxm_btn = ttk.Checkbutton(col1, text="MMXM", variable=self.opt_trail_mmxm)
        self.opt_trail_mmxm_btn.pack(anchor=tk.W)
        self.opt_trail_2022 = tk.BooleanVar(value=True)
        self.opt_trail_2022_btn = ttk.Checkbutton(col1, text="ICT Model 2022", variable=self.opt_trail_2022)
        self.opt_trail_2022_btn.pack(anchor=tk.W)
        self.opt_trail_2025 = tk.BooleanVar(value=True)
        self.opt_trail_2025_btn = ttk.Checkbutton(col1, text="ICT Model 2025", variable=self.opt_trail_2025)
        self.opt_trail_2025_btn.pack(anchor=tk.W)

        # Column 2: Filters Sweep
        col2 = ttk.Frame(opt_params_frame)
        col2.grid(row=0, column=2, sticky=tk.N+tk.S+tk.W+tk.E, padx=5, pady=2)
        ttk.Label(col2, text="4. Sweep Filter state?", font=('', 9, 'bold')).pack(anchor=tk.W, pady=2)
        
        self.opt_sweep_htf = tk.BooleanVar(value=False)
        self.opt_sweep_htf_btn = ttk.Checkbutton(col2, text="Sweep HTF Filter (T/F)", variable=self.opt_sweep_htf)
        self.opt_sweep_htf_btn.pack(anchor=tk.W)
        self.opt_sweep_ote = tk.BooleanVar(value=False)
        self.opt_sweep_ote_btn = ttk.Checkbutton(col2, text="Sweep OTE Filter (T/F)", variable=self.opt_sweep_ote)
        self.opt_sweep_ote_btn.pack(anchor=tk.W)
        self.opt_sweep_recent_sweep = tk.BooleanVar(value=False)
        self.opt_sweep_recent_sweep_btn = ttk.Checkbutton(col2, text="Sweep Recent Sweep (T/F)", variable=self.opt_sweep_recent_sweep)
        self.opt_sweep_recent_sweep_btn.pack(anchor=tk.W)
        self.opt_sweep_displacement = tk.BooleanVar(value=False)
        self.opt_sweep_displacement_btn = ttk.Checkbutton(col2, text="Sweep Displacement (T/F)", variable=self.opt_sweep_displacement)
        self.opt_sweep_displacement_btn.pack(anchor=tk.W)
        self.opt_sweep_disc_prem = tk.BooleanVar(value=False)
        self.opt_sweep_disc_prem_btn = ttk.Checkbutton(col2, text="Sweep Disc/Prem (T/F)", variable=self.opt_sweep_disc_prem)
        self.opt_sweep_disc_prem_btn.pack(anchor=tk.W)
        self.opt_sweep_sb_htf_bias = tk.BooleanVar(value=False)
        self.opt_sweep_sb_htf_bias_btn = ttk.Checkbutton(col2, text="Sweep Strict MTF (T/F)", variable=self.opt_sweep_sb_htf_bias)
        self.opt_sweep_sb_htf_bias_btn.pack(anchor=tk.W)
        self.opt_sweep_daily_loss = tk.BooleanVar(value=False)
        self.opt_sweep_daily_loss_btn = ttk.Checkbutton(col2, text="Sweep Daily Loss %", variable=self.opt_sweep_daily_loss)
        self.opt_sweep_daily_loss_btn.pack(anchor=tk.W)
        
        self.opt_sweep_vp = tk.BooleanVar(value=False)
        self.opt_sweep_vp_btn = ttk.Checkbutton(col2, text="Sweep Vol. Profile (T/F)", variable=self.opt_sweep_vp)
        self.opt_sweep_vp_btn.pack(anchor=tk.W)
        
        self.opt_sweep_smt = tk.BooleanVar(value=False)
        self.opt_sweep_smt_btn = ttk.Checkbutton(col2, text="Sweep SMT Div (T/F)", variable=self.opt_sweep_smt)
        self.opt_sweep_smt_btn.pack(anchor=tk.W)

        # Column 3: Defaults & Ranges
        col3 = ttk.Frame(opt_params_frame)
        col3.grid(row=0, column=3, sticky=tk.N+tk.S+tk.W+tk.E, padx=5, pady=2)
        
        ttk.Label(col3, text="5. Default Values (If not swept)", font=('', 8, 'italic')).pack(anchor=tk.W)
        self.opt_default_htf = tk.BooleanVar(value=True)
        self.opt_default_htf_btn = ttk.Checkbutton(col3, text="HTF Filter: ON", variable=self.opt_default_htf)
        self.opt_default_htf_btn.pack(anchor=tk.W)
        self.opt_default_ote = tk.BooleanVar(value=True)
        self.opt_default_ote_btn = ttk.Checkbutton(col3, text="OTE Filter: ON", variable=self.opt_default_ote)
        self.opt_default_ote_btn.pack(anchor=tk.W)
        self.opt_default_recent_sweep = tk.BooleanVar(value=False)
        self.opt_default_recent_sweep_btn = ttk.Checkbutton(col3, text="Recent Sweep: ON", variable=self.opt_default_recent_sweep)
        self.opt_default_recent_sweep_btn.pack(anchor=tk.W)
        self.opt_default_displacement = tk.BooleanVar(value=True)
        self.opt_default_displacement_btn = ttk.Checkbutton(col3, text="Displacement: ON", variable=self.opt_default_displacement)
        self.opt_default_displacement_btn.pack(anchor=tk.W)
        self.opt_default_disc_prem = tk.BooleanVar(value=True)
        self.opt_default_disc_prem_btn = ttk.Checkbutton(col3, text="Disc/Prem: ON", variable=self.opt_default_disc_prem)
        self.opt_default_disc_prem_btn.pack(anchor=tk.W)
        self.opt_default_sb_htf_bias = tk.BooleanVar(value=False)
        self.opt_default_sb_htf_bias_btn = ttk.Checkbutton(col3, text="Strict MTF Alignment: ON", variable=self.opt_default_sb_htf_bias)
        self.opt_default_sb_htf_bias_btn.pack(anchor=tk.W)
        
        self.opt_default_vp = tk.BooleanVar(value=False)
        self.opt_default_vp_btn = ttk.Checkbutton(col3, text="Vol. Profile: ON", variable=self.opt_default_vp)
        self.opt_default_vp_btn.pack(anchor=tk.W)
        
        self.opt_default_smt = tk.BooleanVar(value=False)
        self.opt_default_smt_btn = ttk.Checkbutton(col3, text="SMT Div: ON", variable=self.opt_default_smt)
        self.opt_default_smt_btn.pack(anchor=tk.W)
        
        self.opt_news_enabled = tk.BooleanVar(value=False)
        self.opt_news_btn = ttk.Checkbutton(col3, text="News Filter: ON", variable=self.opt_news_enabled)
        self.opt_news_btn.pack(anchor=tk.W)
        
        ttk.Label(col3, text="SMT Pair(s) / News Buffer:", font=('', 8, 'italic')).pack(anchor=tk.W, pady=(5, 0))
        params_f = ttk.Frame(col3)
        params_f.pack(anchor=tk.W)
        self.opt_smt_pair = ttk.Combobox(params_f, width=10, values=["AUTO", "DXY,GBPUSD", "DXY", "EURUSD"])
        self.opt_smt_pair.insert(0, "AUTO")
        self.opt_smt_pair.pack(side=tk.LEFT, padx=1)
        self.opt_news_buffer = ttk.Entry(params_f, width=4)
        self.opt_news_buffer.insert(0, "30")
        self.opt_news_buffer.pack(side=tk.LEFT, padx=1)
        
        # Concurrency & RRR Ranges
        ranges_f = ttk.Frame(col3)
        ranges_f.pack(anchor=tk.W, pady=5)
        ttk.Label(ranges_f, text="6. Concurrency Range:").grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.opt_concurr_min = ttk.Entry(ranges_f, width=3)
        self.opt_concurr_min.insert(0, "1")
        self.opt_concurr_min.grid(row=1, column=0, sticky=tk.W, padx=1)
        ttk.Label(ranges_f, text="to").grid(row=1, column=1, padx=1)
        self.opt_concurr_max = ttk.Entry(ranges_f, width=3)
        self.opt_concurr_max.insert(0, "3")
        self.opt_concurr_max.grid(row=1, column=2, sticky=tk.W, padx=1)
        
        ttk.Label(ranges_f, text="7. RRR Sweep:").grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(3, 0))
        self.opt_rrr_min = ttk.Entry(ranges_f, width=3)
        self.opt_rrr_min.insert(0, "1.5")
        self.opt_rrr_min.grid(row=3, column=0, sticky=tk.W, padx=1)
        self.opt_rrr_max = ttk.Entry(ranges_f, width=3)
        self.opt_rrr_max.insert(0, "3.0")
        self.opt_rrr_max.grid(row=3, column=1, sticky=tk.W, padx=1)
        self.opt_rrr_step = ttk.Entry(ranges_f, width=3)
        self.opt_rrr_step.insert(0, "0.5")
        self.opt_rrr_step.grid(row=3, column=2, sticky=tk.W, padx=1)
        
        ttk.Label(ranges_f, text="8. Daily Loss % Sweep:").grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(3, 0))
        self.opt_dl_min = ttk.Entry(ranges_f, width=4)
        self.opt_dl_min.insert(0, "1.0")
        self.opt_dl_min.grid(row=5, column=0, sticky=tk.W, padx=1)
        self.opt_dl_max = ttk.Entry(ranges_f, width=4)
        self.opt_dl_max.insert(0, "3.0")
        self.opt_dl_max.grid(row=5, column=1, sticky=tk.W, padx=1)
        self.opt_dl_step = ttk.Entry(ranges_f, width=4)
        self.opt_dl_step.insert(0, "0.5")
        self.opt_dl_step.grid(row=5, column=2, sticky=tk.W, padx=1)

        # Multiprocessing CPU cores control
        ttk.Label(ranges_f, text="9. CPU Cores:").grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(3, 0))
        self.opt_cpu_cores = ttk.Entry(ranges_f, width=4)
        import os
        self.opt_cpu_cores.insert(0, str(max(1, os.cpu_count() // 2)))
        self.opt_cpu_cores.grid(row=7, column=0, sticky=tk.W, padx=1)

        # 2.5 Pro Strategy Upgrades & ML Filter
        opt_pro_frame = ttk.LabelFrame(self.opt_top_pane, text="🏆 Pro Strategy Upgrades (v17) & ML Filter — Grid Opt")
        opt_pro_frame.pack(fill=tk.X, padx=5, pady=2)
        opt_pro_row1 = ttk.Frame(opt_pro_frame); opt_pro_row1.pack(fill=tk.X, padx=3, pady=1)
        opt_pro_row2 = ttk.Frame(opt_pro_frame); opt_pro_row2.pack(fill=tk.X, padx=3, pady=1)

        self.opt_pro_dol_tp = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_pro_row1, text="Draw-on-Liquidity TP", variable=self.opt_pro_dol_tp).pack(side=tk.LEFT, padx=3)
        self.opt_pro_killzone = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_pro_row1, text="Killzone Weighting", variable=self.opt_pro_killzone).pack(side=tk.LEFT, padx=3)
        self.opt_pro_htf_poi = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_pro_row1, text="HTF POI Nesting", variable=self.opt_pro_htf_poi).pack(side=tk.LEFT, padx=3)
        self.opt_pro_mandatory = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_pro_row1, text="Mandatory Sweep+Displacement", variable=self.opt_pro_mandatory).pack(side=tk.LEFT, padx=3)

        self.opt_pro_regime = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_pro_row2, text="Regime Filter", variable=self.opt_pro_regime).pack(side=tk.LEFT, padx=3)
        self.opt_pro_ml_sizing = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_pro_row2, text="ML Conviction Sizing", variable=self.opt_pro_ml_sizing).pack(side=tk.LEFT, padx=3)
        self.opt_pro_ml_rank = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_pro_row2, text="ML Setup Ranking", variable=self.opt_pro_ml_rank).pack(side=tk.LEFT, padx=3)
        self.opt_pro_multi_tf_conf = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_pro_row2, text="Multi-TF Confluence", variable=self.opt_pro_multi_tf_conf).pack(side=tk.LEFT, padx=3)
        self.opt_pro_multi_tf_gate = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_pro_row2, text="Multi-TF Gate", variable=self.opt_pro_multi_tf_gate).pack(side=tk.LEFT, padx=3)
        self.opt_pro_regime_adaptive = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_pro_row2, text="Regime Adaptive Params", variable=self.opt_pro_regime_adaptive).pack(side=tk.LEFT, padx=3)

        opt_ml_row = ttk.Frame(opt_pro_frame); opt_ml_row.pack(fill=tk.X, padx=3, pady=1)
        self.opt_ml_filter = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_ml_row, text="ML Setup Filter", variable=self.opt_ml_filter).pack(side=tk.LEFT, padx=3)
        ttk.Label(opt_ml_row, text="Min Conf %:").pack(side=tk.LEFT, padx=(10,2))
        self.opt_ml_min_confidence = ttk.Entry(opt_ml_row, width=5)
        self.opt_ml_min_confidence.insert(0, "60.0")
        self.opt_ml_min_confidence.pack(side=tk.LEFT)

        # 3. Execution Control Frame
        opt_exec = ttk.Frame(self.opt_top_pane)
        opt_exec.pack(fill=tk.X, padx=5, pady=5)
        
        self.opt_run_btn = ttk.Button(opt_exec, text="Start Grid Optimization", command=self.run_grid_optimization)
        self.opt_run_btn.pack(side=tk.LEFT, padx=3)
        self.opt_stop_btn = ttk.Button(opt_exec, text="Stop", command=self.stop_optimization, state=tk.DISABLED)
        self.opt_stop_btn.pack(side=tk.LEFT, padx=3)
        self.opt_pause_btn = ttk.Button(opt_exec, text="Pause", command=self.pause_optimization, state=tk.DISABLED)
        self.opt_pause_btn.pack(side=tk.LEFT, padx=3)
        self.opt_resume_btn = ttk.Button(opt_exec, text="Resume", command=self.resume_optimization, state=tk.DISABLED)
        self.opt_resume_btn.pack(side=tk.LEFT, padx=3)
        self.opt_save_btn = ttk.Button(opt_exec, text="Save Stage", command=self.save_opt_stage, state=tk.DISABLED)
        self.opt_save_btn.pack(side=tk.LEFT, padx=3)
        self.opt_load_btn = ttk.Button(opt_exec, text="Load Stage", command=self.load_opt_stage, state=tk.NORMAL)
        self.opt_load_btn.pack(side=tk.LEFT, padx=3)
        
        self.opt_progress_bar = ttk.Progressbar(opt_exec, mode='determinate', length=300)
        self.opt_progress_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.opt_progress_label = ttk.Label(opt_exec, text="Idle")
        self.opt_progress_label.pack(side=tk.LEFT, padx=5)
        self.opt_elapsed_label = ttk.Label(opt_exec, text="")
        self.opt_elapsed_label.pack(side=tk.LEFT, padx=5)

        # 4. Results Treeview Table Frame
        opt_results = ttk.LabelFrame(self.opt_bottom_pane, text="Optimization Results (Sorted by Calmar - Click Headers to Sort)")
        opt_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)
        
        # Action Buttons row
        opt_actions = ttk.Frame(opt_results)
        opt_actions.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=3)
        ttk.Button(opt_actions, text="Apply Selected to Backtest (Single Run)", command=self.apply_opt_to_backtest).pack(side=tk.LEFT, padx=5)
        ttk.Button(opt_actions, text="Apply Selected to Live Trading", command=self.apply_opt_to_live).pack(side=tk.LEFT, padx=5)
        ttk.Button(opt_actions, text="Save Optimization Report", command=self.save_opt_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(opt_actions, text="Export CSV (For Regime Training)", command=self.export_opt_csv).pack(side=tk.LEFT, padx=5)
        
        # Treeview setup
        style = ttk.Style()
        style.configure("Opt.Treeview", rowheight=35)
        
        cols = ("rank", "roi", "max_dd", "calmar", "trades", "win_rate", "pf", "methods", "sl_mode", "trail_type", "ote", "htf", "vp", "smt", "recent_sweep", "displacement", "disc_prem", "max_concurr", "rrr", "daily_loss", "sb_htf")
        self.opt_tree = ttk.Treeview(opt_results, columns=cols, show="tree headings", style="Opt.Treeview", height=10)
        
        self.opt_tree.heading("#0", text="Curve", anchor=tk.CENTER)
        self.opt_tree.column("#0", width=110, stretch=tk.NO, anchor=tk.CENTER)
        
        self.opt_tree.heading("rank", text="Rank", command=lambda: self.sort_treeview(self.opt_tree, "rank", False))
        self.opt_tree.heading("roi", text="ROI %", command=lambda: self.sort_treeview(self.opt_tree, "roi", False))
        self.opt_tree.heading("max_dd", text="Max DD %", command=lambda: self.sort_treeview(self.opt_tree, "max_dd", False))
        self.opt_tree.heading("calmar", text="Calmar", command=lambda: self.sort_treeview(self.opt_tree, "calmar", False))
        self.opt_tree.heading("trades", text="Trades", command=lambda: self.sort_treeview(self.opt_tree, "trades", False))
        self.opt_tree.heading("win_rate", text="Win Rate %", command=lambda: self.sort_treeview(self.opt_tree, "win_rate", False))
        self.opt_tree.heading("pf", text="PF", command=lambda: self.sort_treeview(self.opt_tree, "pf", False))
        self.opt_tree.heading("methods", text="Methods", command=lambda: self.sort_treeview(self.opt_tree, "methods", False))
        self.opt_tree.heading("sl_mode", text="SL Mode", command=lambda: self.sort_treeview(self.opt_tree, "sl_mode", False))
        self.opt_tree.heading("trail_type", text="Trail Type", command=lambda: self.sort_treeview(self.opt_tree, "trail_type", False))
        self.opt_tree.heading("ote", text="OTE", command=lambda: self.sort_treeview(self.opt_tree, "ote", False))
        self.opt_tree.heading("htf", text="HTF", command=lambda: self.sort_treeview(self.opt_tree, "htf", False))
        self.opt_tree.heading("vp", text="VP", command=lambda: self.sort_treeview(self.opt_tree, "vp", False))
        self.opt_tree.heading("smt", text="SMT", command=lambda: self.sort_treeview(self.opt_tree, "smt", False))
        self.opt_tree.heading("recent_sweep", text="Sweep", command=lambda: self.sort_treeview(self.opt_tree, "recent_sweep", False))
        self.opt_tree.heading("displacement", text="Disp", command=lambda: self.sort_treeview(self.opt_tree, "displacement", False))
        self.opt_tree.heading("disc_prem", text="Disc/Prem", command=lambda: self.sort_treeview(self.opt_tree, "disc_prem", False))
        self.opt_tree.heading("max_concurr", text="Concurr", command=lambda: self.sort_treeview(self.opt_tree, "max_concurr", False))
        self.opt_tree.heading("rrr", text="RRR", command=lambda: self.sort_treeview(self.opt_tree, "rrr", False))
        self.opt_tree.heading("daily_loss", text="DL %", command=lambda: self.sort_treeview(self.opt_tree, "daily_loss", False))
        self.opt_tree.heading("sb_htf", text="MTF Align", command=lambda: self.sort_treeview(self.opt_tree, "sb_htf", False))
        
        self.opt_tree.column("rank", width=40, anchor=tk.CENTER)
        self.opt_tree.column("roi", width=65, anchor=tk.CENTER)
        self.opt_tree.column("max_dd", width=65, anchor=tk.CENTER)
        self.opt_tree.column("calmar", width=55, anchor=tk.CENTER)
        self.opt_tree.column("trades", width=50, anchor=tk.CENTER)
        self.opt_tree.column("win_rate", width=75, anchor=tk.CENTER)
        self.opt_tree.column("pf", width=45, anchor=tk.CENTER)
        self.opt_tree.column("methods", width=110, anchor=tk.W)
        self.opt_tree.column("sl_mode", width=80, anchor=tk.CENTER)
        self.opt_tree.column("trail_type", width=95, anchor=tk.CENTER)
        self.opt_tree.column("ote", width=45, anchor=tk.CENTER)
        self.opt_tree.column("htf", width=45, anchor=tk.CENTER)
        self.opt_tree.column("vp", width=45, anchor=tk.CENTER)
        self.opt_tree.column("smt", width=45, anchor=tk.CENTER)
        self.opt_tree.column("recent_sweep", width=45, anchor=tk.CENTER)
        self.opt_tree.column("displacement", width=40, anchor=tk.CENTER)
        self.opt_tree.column("disc_prem", width=65, anchor=tk.CENTER)
        self.opt_tree.column("max_concurr", width=55, anchor=tk.CENTER)
        self.opt_tree.column("rrr", width=45, anchor=tk.CENTER)
        self.opt_tree.column("daily_loss", width=50, anchor=tk.CENTER)
        self.opt_tree.column("sb_htf", width=55, anchor=tk.CENTER)

        scrollbar_v = ttk.Scrollbar(opt_results, orient=tk.VERTICAL, command=self.opt_tree.yview)
        scrollbar_h = ttk.Scrollbar(opt_results, orient=tk.HORIZONTAL, command=self.opt_tree.xview)
        self.opt_tree.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        self.opt_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        self._create_oos_test_tab()

        # Log
        log_frame = ttk.LabelFrame(self.log_container, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 2))
        self.log_text = scrolledtext.ScrolledText(log_frame, state='normal', height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        status_frame = ttk.Frame(self.log_container)
        status_frame.pack(fill=tk.X, padx=5, pady=(2, 5))
        self.status_label = ttk.Label(status_frame, text="Status: Idle")
        self.status_label.pack(side=tk.LEFT, padx=5)
        self.connection_label = ttk.Label(status_frame, text="MT5: CONNECTED", foreground="green", font=('', 9, 'bold'))
        self.connection_label.pack(side=tk.RIGHT, padx=5)
        
        # Initialize overrides state
        self.check_symbol_count_for_overrides()
        self.check_live_symbol_count_for_overrides()
        self.update_opt_fvg_widgets_state()

    def update_opt_fvg_widgets_state(self):
        state = tk.NORMAL if (self.opt_sweep_fvg.get() or self.opt_sweep_sb.get()) else tk.DISABLED
        
        # SL modes checkbuttons
        if hasattr(self, 'opt_sweep_sl_normal_btn'):
            self.opt_sweep_sl_normal_btn.config(state=state)
        if hasattr(self, 'opt_sweep_sl_last_sweep_btn'):
            self.opt_sweep_sl_last_sweep_btn.config(state=state)
        if hasattr(self, 'opt_sweep_sl_candle_ext_btn'):
            self.opt_sweep_sl_candle_ext_btn.config(state=state)
            
        # Filters checkbuttons
        if hasattr(self, 'opt_sweep_recent_sweep_btn'):
            self.opt_sweep_recent_sweep_btn.config(state=state)
        if hasattr(self, 'opt_sweep_displacement_btn'):
            self.opt_sweep_displacement_btn.config(state=state)
        if hasattr(self, 'opt_sweep_disc_prem_btn'):
            self.opt_sweep_disc_prem_btn.config(state=state)
        if hasattr(self, 'opt_sweep_sb_htf_bias_btn'):
            self.opt_sweep_sb_htf_bias_btn.config(state=state)
            
        # Default filters checkbuttons
        if hasattr(self, 'opt_default_recent_sweep_btn'):
            self.opt_default_recent_sweep_btn.config(state=state)
        if hasattr(self, 'opt_default_displacement_btn'):
            self.opt_default_displacement_btn.config(state=state)
        if hasattr(self, 'opt_default_disc_prem_btn'):
            self.opt_default_disc_prem_btn.config(state=state)
        if hasattr(self, 'opt_default_sb_htf_bias_btn'):
            self.opt_default_sb_htf_bias_btn.config(state=state)
            
        # Require BoS and Min FVG size
        if hasattr(self, 'opt_require_bos_fvg_btn'):
            self.opt_require_bos_fvg_btn.config(state=state)
        if hasattr(self, 'opt_min_fvg_lbl'):
            self.opt_min_fvg_lbl.config(state=state)
        if hasattr(self, 'opt_min_fvg'):
            self.opt_min_fvg.config(state=state)

    def _create_oos_test_tab(self):
        # Frame for OOS Test inputs
        oos_input_frame = ttk.Frame(self.oos_test_tab)
        oos_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(oos_input_frame, text="Out of Sample Start Date (YYYY-MM-DD):").pack(side=tk.LEFT, padx=3)
        self.oos_date_from = ttk.Entry(oos_input_frame, width=12)
        import datetime
        self.oos_date_from.insert(0, datetime.datetime.now().strftime("%Y-%m-%d"))
        self.oos_date_from.pack(side=tk.LEFT, padx=3)
        
        ttk.Label(oos_input_frame, text="Min Calmar to test:").pack(side=tk.LEFT, padx=(15, 3))
        self.oos_min_calmar = ttk.Entry(oos_input_frame, width=5)
        self.oos_min_calmar.insert(0, "6.0")
        self.oos_min_calmar.pack(side=tk.LEFT, padx=3)
        
        self.oos_run_btn = ttk.Button(oos_input_frame, text="Run OOS Test", command=self.start_oos_test)
        self.oos_run_btn.pack(side=tk.LEFT, padx=15)
        
        self.oos_progress_label = ttk.Label(oos_input_frame, text="")
        self.oos_progress_label.pack(side=tk.LEFT, padx=5)

        # OOS Results treeview
        oos_results_frame = ttk.LabelFrame(self.oos_test_tab, text="Out of Sample Results (Sorted by OOS Calmar)")
        oos_results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        cols = ("orig_rank", "oos_calmar", "oos_roi", "oos_max_dd", "oos_win_rate", "oos_trades", 
                "methods", "sl_mode", "trail_type", "max_concurr", "rrr", "daily_loss")
        self.oos_tree = ttk.Treeview(oos_results_frame, columns=cols, show="headings", height=10)
        
        self.oos_tree.heading("orig_rank", text="Orig Rank", command=lambda: self.sort_treeview(self.oos_tree, "orig_rank", False))
        self.oos_tree.heading("oos_calmar", text="OOS Calmar", command=lambda: self.sort_treeview(self.oos_tree, "oos_calmar", False))
        self.oos_tree.heading("oos_roi", text="OOS ROI %", command=lambda: self.sort_treeview(self.oos_tree, "oos_roi", False))
        self.oos_tree.heading("oos_max_dd", text="OOS Max DD %", command=lambda: self.sort_treeview(self.oos_tree, "oos_max_dd", False))
        self.oos_tree.heading("oos_win_rate", text="OOS Win %", command=lambda: self.sort_treeview(self.oos_tree, "oos_win_rate", False))
        self.oos_tree.heading("oos_trades", text="OOS Trades", command=lambda: self.sort_treeview(self.oos_tree, "oos_trades", False))
        self.oos_tree.heading("methods", text="Methods", command=lambda: self.sort_treeview(self.oos_tree, "methods", False))
        self.oos_tree.heading("sl_mode", text="SL Mode", command=lambda: self.sort_treeview(self.oos_tree, "sl_mode", False))
        self.oos_tree.heading("trail_type", text="Trail Type", command=lambda: self.sort_treeview(self.oos_tree, "trail_type", False))
        self.oos_tree.heading("max_concurr", text="Concurr", command=lambda: self.sort_treeview(self.oos_tree, "max_concurr", False))
        self.oos_tree.heading("rrr", text="RRR", command=lambda: self.sort_treeview(self.oos_tree, "rrr", False))
        self.oos_tree.heading("daily_loss", text="DL %", command=lambda: self.sort_treeview(self.oos_tree, "daily_loss", False))
        
        self.oos_tree.column("orig_rank", width=65, anchor=tk.CENTER)
        self.oos_tree.column("oos_calmar", width=75, anchor=tk.CENTER)
        self.oos_tree.column("oos_roi", width=75, anchor=tk.CENTER)
        self.oos_tree.column("oos_max_dd", width=85, anchor=tk.CENTER)
        self.oos_tree.column("oos_win_rate", width=75, anchor=tk.CENTER)
        self.oos_tree.column("oos_trades", width=75, anchor=tk.CENTER)
        self.oos_tree.column("methods", width=120, anchor=tk.W)
        self.oos_tree.column("sl_mode", width=80, anchor=tk.CENTER)
        self.oos_tree.column("trail_type", width=95, anchor=tk.CENTER)
        self.oos_tree.column("max_concurr", width=55, anchor=tk.CENTER)
        self.oos_tree.column("rrr", width=45, anchor=tk.CENTER)
        self.oos_tree.column("daily_loss", width=50, anchor=tk.CENTER)
        
        scrollbar_v = ttk.Scrollbar(oos_results_frame, orient=tk.VERTICAL, command=self.oos_tree.yview)
        scrollbar_h = ttk.Scrollbar(oos_results_frame, orient=tk.HORIZONTAL, command=self.oos_tree.xview)
        self.oos_tree.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        self.oos_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

    def auto_fill_spread(self):
        """Show per-symbol spreads and set to AUTO mode (0 = auto-detect per symbol)."""
        symbols_str = self.backtest_symbol_entry.get()
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        if symbols:
            spread_info = []
            for sym in symbols:
                spread = get_spread(sym)
                estimated = spread * config.ESTIMATED_SPREAD_MULTIPLIER
                spread_info.append(f"{sym}: {spread:.6f} (est: {estimated:.6f})")
            self.bt_spread_cost.delete(0, tk.END)
            self.bt_spread_cost.insert(0, "0")
            info_msg = "Spread set to AUTO (0 = per-symbol auto-detect)\n\nCurrent spreads (x" + f"{config.ESTIMATED_SPREAD_MULTIPLIER}):\n" + "\n".join(spread_info)
            messagebox.showinfo("Per-Symbol Spreads", info_msg)
            logger.info("Spread set to AUTO mode. Per-symbol spreads: %s", ", ".join(spread_info))

    def clear_trade_memory(self):
        clear_trade_memory()
        messagebox.showinfo("Trade Memory", "All trade memory has been cleared")

    def _sanitize_overrides(self, overrides):
        if not isinstance(overrides, dict):
            return {}
        sanitized = {}
        for sym, val in overrides.items():
            if isinstance(val, dict):
                sym_val = val.copy()
                sym_val.pop('ict_method', None)
                sym_val.pop('trail_methods', None)
                if 'methods' in sym_val and isinstance(sym_val['methods'], dict):
                    methods_dict = {}
                    for m, mval in sym_val['methods'].items():
                        if isinstance(mval, dict):
                            m_val = mval.copy()
                            m_val.pop('ict_method', None)
                            m_val.pop('trail_methods', None)
                            methods_dict[m] = m_val
                    sym_val['methods'] = methods_dict
                sanitized[sym] = sym_val
        return sanitized

    def _apply_pro_flags(self, scope="live"):
        """Push the v17 Pro Strategy toggles into the config module. The Live tab
        and the Backtest tab each have their OWN independent set of toggles;
        whichever engine is about to run pushes ITS set into config first (the
        engines read these flags at call time). scope='live' uses self.pro_*,
        scope='bt' uses self.bt_pro_*."""
        p = "bt_pro_" if scope == "bt" else "pro_"
        try:
            config.DOL_TP_ENABLED = bool(getattr(self, p + "dol_tp").get())
            config.KILLZONE_ENABLED = bool(getattr(self, p + "killzone").get())
            config.HTF_POI_ENABLED = bool(getattr(self, p + "htf_poi").get())
            config.CONFLUENCE_REQUIRE_MANDATORY = bool(getattr(self, p + "mandatory").get())
            config.REGIME_FILTER_ENABLED = bool(getattr(self, p + "regime").get())
            config.ML_SIZING_ENABLED = bool(getattr(self, p + "ml_sizing").get())
            config.ML_RANK_ENABLED = bool(getattr(self, p + "ml_rank").get())
            config.MULTI_TF_CONFLUENCE_ENABLED = bool(getattr(self, p + "multi_tf_conf").get())
            config.MULTI_TF_GATE_ENABLED = bool(getattr(self, p + "multi_tf_gate").get())
            config.REGIME_ADAPTIVE_PARAMS_ENABLED = bool(getattr(self, p + "regime_adaptive").get())
        except Exception as e:
            logger.error("Failed to apply Pro Strategy flags (%s): %s", scope, e)

    def save_config(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Save Settings")
        if not filepath: return

        # Flush currently selected backtest overrides from UI to self.bt_symbol_overrides
        if getattr(self, 'bt_use_overrides_var', None) and self.bt_use_overrides_var.get() and getattr(self, 'bt_currently_selected_override_symbol', None):
            if getattr(self, 'bt_use_method_overrides_var', None) and self.bt_use_method_overrides_var.get() and getattr(self, 'bt_currently_selected_override_method', None):
                if getattr(self, 'save_bt_method_settings', None):
                    self.save_bt_method_settings(self.bt_currently_selected_override_symbol, self.bt_currently_selected_override_method)
            else:
                if getattr(self, 'get_current_ui_backtest_settings', None):
                    self.bt_symbol_overrides[self.bt_currently_selected_override_symbol] = self.get_current_ui_backtest_settings()

        # Flush currently selected live overrides from UI to self.live_symbol_overrides
        if getattr(self, 'live_use_overrides_var', None) and self.live_use_overrides_var.get() and getattr(self, 'live_currently_selected_override_symbol', None):
            if getattr(self, 'live_use_method_overrides_var', None) and self.live_use_method_overrides_var.get() and getattr(self, 'live_currently_selected_override_method', None):
                if getattr(self, 'save_live_method_settings', None):
                    self.save_live_method_settings(self.live_currently_selected_override_symbol, self.live_currently_selected_override_method)
            else:
                if getattr(self, 'get_current_ui_live_settings', None):
                    self.live_symbol_overrides[self.live_currently_selected_override_symbol] = self.get_current_ui_live_settings()

        settings = {}
        for name, widget in vars(self).items():
            if isinstance(widget, ttk.Entry):
                settings[name] = widget.get()
            elif isinstance(widget, (tk.BooleanVar, tk.StringVar, tk.IntVar, tk.DoubleVar)):
                settings[name] = widget.get()
            elif isinstance(widget, tk.Listbox):
                settings[name] = [widget.get(i) for i in widget.curselection()]
                
        # Save symbol/method overrides and associated states
        settings["bt_symbol_overrides"] = self._sanitize_overrides(self.bt_symbol_overrides)
        settings["live_symbol_overrides"] = self._sanitize_overrides(self.live_symbol_overrides)
        settings["bt_base_cached_settings"] = self.bt_base_cached_settings
        settings["live_base_cached_settings"] = self.live_base_cached_settings
        settings["bt_currently_selected_override_symbol"] = self.bt_currently_selected_override_symbol
        settings["live_currently_selected_override_symbol"] = self.live_currently_selected_override_symbol
        settings["bt_currently_selected_override_method"] = self.bt_currently_selected_override_method
        settings["live_currently_selected_override_method"] = self.live_currently_selected_override_method

        try:
            with open(filepath, "w") as f:
                json.dump(settings, f, indent=4)
            logger.info("Configuration saved to %s", filepath)
            messagebox.showinfo("Settings Saved", f"Configuration successfully saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def load_config(self):
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")], title="Load Settings")
        if not filepath: return
        try:
            with open(filepath, "r") as f:
                settings = json.load(f)
                
            SPECIAL_KEYS = [
                "bt_symbol_overrides", "live_symbol_overrides",
                "bt_base_cached_settings", "live_base_cached_settings",
                "bt_currently_selected_override_symbol", "live_currently_selected_override_symbol",
                "bt_currently_selected_override_method", "live_currently_selected_override_method"
            ]

            # 1. Load overrides and state variables first
            self.bt_symbol_overrides = self._sanitize_overrides(settings.get("bt_symbol_overrides", {}))
            self.live_symbol_overrides = self._sanitize_overrides(settings.get("live_symbol_overrides", {}))
            self.bt_base_cached_settings = settings.get("bt_base_cached_settings", {})
            self.live_base_cached_settings = settings.get("live_base_cached_settings", {})
            self.bt_currently_selected_override_symbol = settings.get("bt_currently_selected_override_symbol", None)
            self.live_currently_selected_override_symbol = settings.get("live_currently_selected_override_symbol", None)
            self.bt_currently_selected_override_method = settings.get("bt_currently_selected_override_method", None)
            self.live_currently_selected_override_method = settings.get("live_currently_selected_override_method", None)

            # 2. Load non-listbox widgets, excluding special keys
            for name, value in settings.items():
                if name in ["bt_methods_listbox", "live_methods_listbox", "bt_trail_methods_listbox", "live_trail_methods_listbox"] + SPECIAL_KEYS:
                    continue
                if hasattr(self, name):
                    widget = getattr(self, name)
                    if isinstance(widget, ttk.Combobox):
                        widget.set(str(value))
                    elif isinstance(widget, ttk.Entry):
                        widget.delete(0, tk.END)
                        widget.insert(0, str(value))
                    elif isinstance(widget, (tk.BooleanVar, tk.StringVar, tk.IntVar, tk.DoubleVar)):
                        widget.set(value)
                    elif isinstance(widget, tk.Listbox):
                        widget.selection_clear(0, tk.END)
                        for val in value:
                            if isinstance(val, int):
                                widget.selection_set(val)
                            else:
                                for i in range(widget.size()):
                                    if widget.get(i) == val:
                                        widget.selection_set(i)

            # 3. Load main method listboxes
            for name in ["bt_methods_listbox", "live_methods_listbox"]:
                if name in settings and hasattr(self, name):
                    widget = getattr(self, name)
                    value = settings[name]
                    widget.selection_clear(0, tk.END)
                    for val in value:
                        if isinstance(val, int):
                            widget.selection_set(val)
                        else:
                            for i in range(widget.size()):
                                if widget.get(i) == val:
                                    widget.selection_set(i)

            # 4. Synchronously trigger selection handlers to populate trailing listboxes
            if hasattr(self, "on_bt_methods_select"):
                self.on_bt_methods_select(None)
            if hasattr(self, "on_live_methods_select"):
                self.on_live_methods_select(None)

            # 5. Load trailing method listboxes
            for name in ["bt_trail_methods_listbox", "live_trail_methods_listbox"]:
                if name in settings and hasattr(self, name):
                    widget = getattr(self, name)
                    value = settings[name]
                    widget.selection_clear(0, tk.END)
                    for val in value:
                        if isinstance(val, int):
                            widget.selection_set(val)
                        else:
                            for i in range(widget.size()):
                                if widget.get(i) == val:
                                    widget.selection_set(i)

            # 6. Re-trigger symbol check handlers to configure overrides UI state
            self.check_symbol_count_for_overrides()
            self.check_live_symbol_count_for_overrides()

            # 7. Configure Backtest overrides UI state
            if self.bt_use_overrides_var.get():
                symbols_str = self.backtest_symbol_entry.get()
                symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
                self.bt_override_symbol_dropdown["values"] = symbols
                self.bt_override_symbol_dropdown.configure(state="readonly")
                self.bt_use_method_overrides_cb.config(state=tk.NORMAL)
                
                if self.bt_currently_selected_override_symbol and self.bt_currently_selected_override_symbol in symbols:
                    self.bt_override_symbol_dropdown.set(self.bt_currently_selected_override_symbol)
                    
                    if self.bt_use_method_overrides_var.get():
                        self.bt_override_method_dropdown.configure(state="readonly")
                        selected_methods = [self.bt_methods_listbox.get(i) for i in self.bt_methods_listbox.curselection()]
                        if not selected_methods:
                            selected_methods = config.CORE_METHODS[:]
                        self.bt_override_method_dropdown["values"] = selected_methods
                        
                        if self.bt_currently_selected_override_method and self.bt_currently_selected_override_method in selected_methods:
                            self.bt_override_method_dropdown.set(self.bt_currently_selected_override_method)
                            self._update_bt_fvg_widgets_for_override(self.bt_currently_selected_override_method)
                        else:
                            self.bt_override_method_dropdown.set("")
                    else:
                        self.bt_override_method_dropdown.set("")
                        self.bt_override_method_dropdown.configure(state="disabled")
                else:
                    self.bt_override_symbol_dropdown.set("")
                    self.bt_override_symbol_dropdown.configure(state="disabled")
            else:
                self.bt_override_symbol_dropdown.set("")
                self.bt_override_symbol_dropdown.configure(state="disabled")
                self.bt_use_method_overrides_var.set(False)
                self.bt_use_method_overrides_cb.config(state=tk.DISABLED)
                self.bt_override_method_dropdown.set("")
                self.bt_override_method_dropdown.configure(state="disabled")

            # 8. Configure Live overrides UI state
            if self.live_use_overrides_var.get():
                symbols_str = self.live_symbols_entry.get()
                symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
                self.live_override_symbol_dropdown["values"] = symbols
                self.live_override_symbol_dropdown.configure(state="readonly")
                self.live_use_method_overrides_cb.config(state=tk.NORMAL)
                
                if self.live_currently_selected_override_symbol and self.live_currently_selected_override_symbol in symbols:
                    self.live_override_symbol_dropdown.set(self.live_currently_selected_override_symbol)
                    
                    if self.live_use_method_overrides_var.get():
                        self.live_override_method_dropdown.configure(state="readonly")
                        selected_methods = [self.live_methods_listbox.get(i) for i in self.live_methods_listbox.curselection()]
                        if not selected_methods:
                            selected_methods = config.CORE_METHODS[:]
                        self.live_override_method_dropdown["values"] = selected_methods
                        
                        if self.live_currently_selected_override_method and self.live_currently_selected_override_method in selected_methods:
                            self.live_override_method_dropdown.set(self.live_currently_selected_override_method)
                            self._update_live_fvg_widgets_for_override(self.live_currently_selected_override_method)
                        else:
                            self.live_override_method_dropdown.set("")
                    else:
                        self.live_override_method_dropdown.set("")
                        self.live_override_method_dropdown.configure(state="disabled")
                else:
                    self.live_override_symbol_dropdown.set("")
                    self.live_override_symbol_dropdown.configure(state="disabled")
            else:
                self.live_override_symbol_dropdown.set("")
                self.live_override_symbol_dropdown.configure(state="disabled")
                self.live_use_method_overrides_var.set(False)
                self.live_use_method_overrides_cb.config(state=tk.DISABLED)
                self.live_override_method_dropdown.set("")
                self.live_override_method_dropdown.configure(state="disabled")

            # 9. Apply active symbol/method settings to the UI to ensure perfect sync
            if self.bt_use_overrides_var.get() and self.bt_currently_selected_override_symbol:
                if self.bt_use_method_overrides_var.get() and self.bt_currently_selected_override_method:
                    self.apply_backtest_settings_to_ui(self.get_merged_bt_settings(self.bt_currently_selected_override_symbol, self.bt_currently_selected_override_method))
                else:
                    self.apply_backtest_settings_to_ui(self.bt_symbol_overrides.get(self.bt_currently_selected_override_symbol, {}))
                    
            if self.live_use_overrides_var.get() and self.live_currently_selected_override_symbol:
                if self.live_use_method_overrides_var.get() and self.live_currently_selected_override_method:
                    self.apply_live_settings_to_ui(self.get_merged_live_settings(self.live_currently_selected_override_symbol, self.live_currently_selected_override_method))
                else:
                    self.apply_live_settings_to_ui(self.live_symbol_overrides.get(self.live_currently_selected_override_symbol, {}))

            logger.info("Configuration loaded from %s", filepath)
            messagebox.showinfo("Settings Loaded", f"Configuration successfully loaded from:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def on_toggle_use_mt5_data(self):
        use_mt5 = self.bt_use_mt5_data.get()
        config.OFFLINE_BACKTESTING = not use_mt5
        if hasattr(self, 'opt_use_mt5_data'):
            self.opt_use_mt5_data.set(use_mt5)
        logger.info(f"Backtest data source updated: {'MT5 API (Live MT5 Data)' if use_mt5 else 'Offline Data Cache'}")

    def on_toggle_opt_use_mt5_data(self):
        use_mt5 = self.opt_use_mt5_data.get()
        config.OFFLINE_BACKTESTING = not use_mt5
        if hasattr(self, 'bt_use_mt5_data'):
            self.bt_use_mt5_data.set(use_mt5)
        logger.info(f"Optimization data source updated: {'MT5 API (Live MT5 Data)' if use_mt5 else 'Offline Data Cache'}")

    def process_ui_update_queue(self):
        _batch_limit = 100  # Process up to 100 UI updates per cycle to prevent backlog lag
        count = 0
        while count < _batch_limit and not self.ui_update_queue.empty():
            try:
                task = self.ui_update_queue.get_nowait()
                task()
                count += 1
            except Exception as e:
                logger.error("Error in GUI update task: %s", e)
                break
        if self.winfo_exists():
            # If queue still has pending items, poll again quickly (10ms); otherwise standard 50ms interval
            interval = 10 if not self.ui_update_queue.empty() else 50
            self.after_ids.append(self.after(interval, self.process_ui_update_queue))

    def process_log_queue(self):
        messages = []
        _batch_limit = 200  # Collect up to 200 messages per cycle
        for _ in range(_batch_limit):
            if config.log_queue.empty():
                break
            try:
                record = config.log_queue.get_nowait()
                if isinstance(record, str):
                    messages.append(record)
                else:
                    messages.append(config.formatter.format(record))
            except Empty:
                break
            except Exception:
                break
        if messages:
            self.log_text.insert(tk.END, '\n'.join(messages) + '\n')
            # Cap the log widget to 5000 lines to prevent unbounded growth & O(n) slowdown
            line_count = int(self.log_text.index('end-1c').split('.')[0])
            if line_count > 5000:
                self.log_text.delete('1.0', f'{line_count - 5000}.0')
            self.log_text.see(tk.END)
        if self.winfo_exists():
            self.after_ids.append(self.after(100, self.process_log_queue))

    def load_ml_model(self):
        """Loads the trained ML model from disk so it can be used for live filtering."""
        success = ml_load_model()
        if success:
            acc = _ml_live_model['metadata'].get('xgb_accuracy', 0)
            if acc <= 0:
                acc = _ml_live_model['metadata'].get('rf_accuracy', 0)
            trades = _ml_live_model['metadata'].get('total_trades', 0)
            self.ml_live_status.config(text=f"Status: Loaded (Acc: {acc:.1f}%, {trades} trades)", foreground="green")
            self.ml_live_enabled.set(True)
            messagebox.showinfo("ML Model Loaded", f"Model successfully loaded!\n\nAccuracy: {acc:.1f}%\nTrained on {trades} trades.")
        else:
            self.ml_live_status.config(text="Status: Failed to load", foreground="red")
            self.ml_live_enabled.set(False)
            messagebox.showerror("Error", f"Failed to load ML model.\n\nMake sure you have run an ML-Filtered backtest first to train and save the model ({ML_MODEL_FILE}).")

    def update_account_info(self):
        def fetch_info():
            try:
                mt5.account_info()
            except Exception:
                pass
        threading.Thread(target=fetch_info, daemon=True).start()
        if self.winfo_exists():
            self.after_ids.append(self.after(5000, self.update_account_info))

    def update_activity(self):
        try:
            self.activity_label.config(text=config.current_activity)
        except Exception:
            pass
        if self.winfo_exists():
            self.after_ids.append(self.after(1000, self.update_activity))

    def stop_backtest(self):
        config.bt_stop_event.set()
        try:
            self.bt_progress_label.config(text="Stopping...")
            self.stop_backtest_btn.config(state=tk.DISABLED)
        except Exception:
            pass

    def run_backtest(self):
        import random
        random.seed(42)  # Enforce determinism for slippage/randomness across runs
        config.bt_stop_event.clear()
        self._apply_pro_flags('bt')  # sync Backtest-tab v17 Pro toggles into config
        symbols_str = self.backtest_symbol_entry.get()
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        if not symbols:
            messagebox.showerror("Error", "Enter at least one symbol.")
            return

        # ── Guard: profiles silently override the typed params (OPT vs single mismatch) ──
        # When "Use Symbol Profiles" is ON, combined_backtest replaces EVERY parameter
        # shown on this tab with the saved profile for the symbol (and its per-method
        # overrides). The OPT grid ALWAYS runs with profiles OFF, so results cannot match
        # while this is enabled. Make the choice explicit instead of silent.
        if self.bt_use_overrides_var.get():
            choice = messagebox.askyesnocancel(
                "Symbol Profiles Are ON",
                "'Use Symbol Profiles' is checked.\n\n"
                "The saved profile for this symbol (plus any per-method overrides) will "
                "REPLACE the parameters shown on this tab. Because the OPT grid never uses "
                "profiles, your results will NOT match the optimizer.\n\n"
                "• Yes  — Run WITH profiles (results reflect the profile, not the typed settings)\n"
                "• No   — Turn profiles OFF and run with exactly the parameters shown (matches OPT)\n"
                "• Cancel — Don't run"
            )
            if choice is None:
                return
            if choice is False:
                self.bt_use_overrides_var.set(False)
                try:
                    self.on_toggle_bt_overrides()
                except Exception:
                    pass

        # Auto-save overrides if profiles checkbox is checked
        if self.bt_use_overrides_var.get():
            if self.bt_currently_selected_override_symbol:
                if self.bt_use_method_overrides_var.get() and self.bt_currently_selected_override_method:
                    self.save_bt_method_settings(self.bt_currently_selected_override_symbol, self.bt_currently_selected_override_method)
                else:
                    self.bt_symbol_overrides[self.bt_currently_selected_override_symbol] = self.get_current_ui_backtest_settings()
            
            import json, os
            profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbol_profiles.json")
            existing_profiles = {}
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, "r", encoding="utf-8") as f:
                        existing_profiles = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading existing profiles on save: {e}")
            
            # Merge self.bt_symbol_overrides into existing_profiles
            for sym, settings in self.bt_symbol_overrides.items():
                # Strip UI-only session keys that should not persist to disk profiles
                save_settings = {k: v for k, v in settings.items() if k not in ('ict_method', 'trail_methods')}
                if "methods" not in save_settings and sym in existing_profiles and "methods" in existing_profiles[sym]:
                    save_settings["methods"] = existing_profiles[sym]["methods"]
                existing_profiles[sym] = save_settings
                
            try:
                with open(profile_path, "w", encoding="utf-8") as f:
                    json.dump(existing_profiles, f, indent=2)
                logger.info("Auto-saved active symbol profiles overrides to symbol_profiles.json")
            except Exception as e:
                logger.error(f"Failed to auto-save symbol profiles: {e}")

        # Resolve base (global default) settings to launch with
        # Always use the current UI settings as the base so user changes take immediate effect.
        # Symbol profiles (if enabled) will override these on a per-symbol basis.
        base_settings = self.get_current_ui_backtest_settings()
            
        base_settings['use_smt_divergence'] = self.bt_smt_enabled.get()
        base_settings['smt_correlated_pair'] = self.bt_smt_pair.get()
        base_settings['use_volume_profile'] = self.bt_vp_enabled.get()
        
        # Override global config for news immediately for Backtest
        config.NEWS_FILTER_ENABLED = self.bt_news_enabled.get()
        try:
            config.NEWS_FILTER_BUFFER_MINS = int(self.bt_news_buffer.get())
        except ValueError:
            config.NEWS_FILTER_BUFFER_MINS = 30

        use_mt5_data = self.bt_use_mt5_data.get() if hasattr(self, 'bt_use_mt5_data') else False
        config.OFFLINE_BACKTESTING = not use_mt5_data

        if not mt5.initialize():
            if use_mt5_data:
                messagebox.showerror("MT5 Connection Error", "Failed to connect to MetaTrader 5 while 'Use MT5 data' is checked. Please make sure the MT5 terminal is open and logged in.")
                return
            else:
                messagebox.showerror("Error", "Failed to connect to MetaTrader 5. Make sure the terminal is open.")
                return

        try:
            date_from = datetime.datetime.strptime(self.backtest_date_from.get().strip(), "%Y-%m-%d")
            date_to = datetime.datetime.strptime(self.backtest_date_to.get().strip(), "%Y-%m-%d")
            # Include the entire day of date_to (matches Grid Opt)
            date_to = date_to.replace(hour=23, minute=59, second=59)
        except ValueError:
            messagebox.showerror("Error", "Date format must be YYYY-MM-DD")
            return

        try:
            initial_balance = float(self.backtest_balance_entry.get())
            risk_percent = float(base_settings.get('risk_percent', 1.0))
            fixed_lot = float(base_settings.get('fixed_lot', 0.1))
            min_rrr_val = float(base_settings.get('min_rrr', 1.5))
            max_rrr_val = float(base_settings.get('max_rrr', 1.5))
            step_rrr_val = float(base_settings.get('step_rrr', 0.5))
            spread_cost = float(base_settings.get('spread_cost', 0.0))
            slippage_pts = int(base_settings.get('slippage_points', 5))
            commission = float(base_settings.get('commission_per_lot', 7.0))
            session_start = int(base_settings.get('session_start', 7))
            session_end = int(base_settings.get('session_end', 21))
            
            # Daily Loss / Max Concurrent always come from the panel here. When symbol
            # profiles are ON, per-symbol/per-method profile values take priority inside
            # the backtest and these panel values are the fallback (0 = no limit).
            # Empty fields must be filled in explicitly.
            if self.bt_daily_loss.get().strip() == "" or self.bt_max_concurr.get().strip() == "":
                messagebox.showerror("Missing Input", "Please enter Daily Loss % and Max Concurrent Trades before running (0 = no limit).")
                return
            config.DAILY_LOSS_LIMIT_PCT = float(base_settings.get('daily_loss_limit', 0.0))
            config.MAX_CONCURRENT_TRADES = int(base_settings.get('max_concurrent_trades', 3))
            config.MIN_FVG_SIZE_SPREADS = float(base_settings.get('min_fvg_size', 0.5))
            config.MIN_CONFLUENCE_SCORE = int(base_settings.get('min_confluence_score', 1))
            
        except (ValueError, TypeError, KeyError) as e:
            messagebox.showerror("Error", f"Please enter valid numbers for all fields: {e}")
            return

        risk_mode = "Fixed" if base_settings.get('use_fixed_lot', False) else "Risk"
        symbols_str = self.backtest_symbol_entry.get()
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        ict_params = get_ict_model_parameters("Default", symbols[0] if symbols else None)

        self.run_backtest_btn.config(state=tk.DISABLED)
        self.bt_progress_bar['value'] = 0
        self.bt_progress_label.config(text="Starting...")
        self.bt_elapsed_label.config(text="")
        self.bt_start_time = time.time()
        
        self.run_backtest_btn.config(state=tk.DISABLED)
        self.stop_backtest_btn.config(state=tk.NORMAL)
        
        logger.info("Starting backtest from %s to %s...", date_from.strftime('%Y-%m-%d'), date_to.strftime('%Y-%m-%d'))

        def update_progress(current, total, symbol, tf_name, phase=None):
            if config.bt_stop_event.is_set():
                return
            if total > 0:
                pct = min(current / total * 100, 100)
            else:
                pct = 0
            elapsed = time.time() - self.bt_start_time if self.bt_start_time else 0
            elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
            if phase:
                label_text = phase
            else:
                label_text = f"{symbol} {tf_name}: {current}/{total} bars"
            try:
                self.after(0, lambda: self.bt_progress_bar.configure(value=pct))
                self.after(0, lambda t=label_text: self.bt_progress_label.configure(text=t))
                self.after(0, lambda t=elapsed_str: self.bt_elapsed_label.configure(text=f"Elapsed: {t}"))
            except Exception:
                pass

        sel_in = self.bt_methods_listbox.curselection()
        selected_bt_methods = [self.bt_methods_listbox.get(i) for i in sel_in]
        if not selected_bt_methods:
            selected_bt_methods = config.CORE_METHODS[:]

        # Use base_settings instead of UI widgets reads for backtest parameters
        bt_ml_filter = self.bt_ml_filter.get()
        try:
            bt_ml_min_confidence = float(self.bt_ml_min_confidence.get()) / 100.0
        except ValueError:
            bt_ml_min_confidence = 0.60

        bt_trade_all_tfs = bool(base_settings.get('trade_on_all_tfs', True))
        bt_use_ultra_low_tf = bool(base_settings.get('use_ultra_low_tf', False))
        bt_fvg_sl_mode = base_settings.get('fvg_sl_mode', 'Normal')
        bt_session_filter = bool(base_settings.get('session_filter', True))
        bt_min_conf = int(base_settings.get('min_confluence_score', 1))
        bt_htf_filter = bool(base_settings.get('use_htf_filter', True))
        bt_ote_filter = bool(base_settings.get('use_ote_filter', True))
        bt_bypass_htf_conf = bool(base_settings.get('bypass_htf_conf', False))
        bt_trail_type = base_settings.get('trail_type', 'Swing')
        bt_trail_params = {'trail_pct': float(base_settings.get('trail_pct', 0.5))}
        bt_require_bos_fvg = bool(base_settings.get('require_bos_fvg', False))
        bt_anti_gap = bool(base_settings.get('anti_gap_enabled', False))
        try:
            bt_anti_gap_mult = float(base_settings.get('anti_gap_mult', 2.0))
        except ValueError:
            bt_anti_gap_mult = 2.0
        bt_fvg_displacement_only = bool(base_settings.get('fvg_displacement_only', True))
        bt_fvg_discount_premium_only = bool(base_settings.get('fvg_discount_premium_only', True))
        bt_fvg_recent_sweep_only = bool(base_settings.get('fvg_recent_sweep_only', False))
        bt_sb_require_htf_bias = bool(base_settings.get('sb_require_htf_bias', False))
        bt_slippage_recovery = bool(base_settings.get('slippage_recovery', False))
        bt_use_symbol_profiles = self.bt_use_overrides_var.get()
        bt_smart_optimize = bool(base_settings.get('smart_optimize', False))
        bt_use_dynamic_rrr = bool(base_settings.get('use_dynamic_rrr', True))

        sel_ts = self.bt_trail_methods_listbox.curselection()
        bt_ts_methods = [self.bt_trail_methods_listbox.get(i) for i in sel_ts]

        def run_bt():
            try:
                rrr_values = []
                current_r = min_rrr_val
                if step_rrr_val > 0:
                    while current_r <= max_rrr_val:
                        rrr_values.append(current_r)
                        current_r += step_rrr_val
                        current_r = round(current_r, 2)
                if not rrr_values:
                    rrr_values = [min_rrr_val]

                
                # Define parameter grid
                if bt_smart_optimize:
                    base_params = {
                        'rrr': 2.0 if len(rrr_values) == 0 else rrr_values[0],
                        'session': True,
                        'conf': 4,
                        'htf': True,
                        'ote': False,
                        'bos': True,
                        'fvg_sl_mode': config.FVG_SL_NORMAL,
                        'bypass': False,
                        'anti_gap': bt_anti_gap,
                        'anti_gap_mult': bt_anti_gap_mult
                    }
                    
                    mutation_spaces = {
                        'rrr': rrr_values if len(rrr_values) > 1 else [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                        'session': [True, False],
                        'conf': [3, 4, 5],
                        'htf': [True, False],
                        'ote': [True, False],
                        'bos': [True, False],
                        'fvg_sl_mode': config.FVG_SL_MODES,
                        'bypass': [True, False],
                        'anti_gap': [True, False],
                        'anti_gap_mult': [bt_anti_gap_mult]
                    }
                else:
                    runs = [{'rrr': r, 'session': bt_session_filter, 
                             'conf': bt_min_conf, 
                             'htf': bt_htf_filter, 
                             'ote': bt_ote_filter,
                             'bos': bt_require_bos_fvg,
                             'fvg_sl_mode': bt_fvg_sl_mode,
                             'bypass': bt_bypass_htf_conf,
                             'anti_gap': bt_anti_gap,
                             'anti_gap_mult': bt_anti_gap_mult} for r in rrr_values]

                all_results = []
                best_score = -float('inf')

                if bt_smart_optimize:
                    current_best_params = base_params.copy()
                    
                    for param_name, values in mutation_spaces.items():
                        if config.bt_stop_event.is_set(): break
                        for val in values:
                            if config.bt_stop_event.is_set(): break
                            
                            test_params = current_best_params.copy()
                            test_params[param_name] = val
                            
                            rrr = test_params['rrr']
                            session_fil = test_params['session']
                            conf = test_params['conf']
                            htf = test_params['htf']
                            ote = test_params['ote']
                            bos = test_params['bos']
                            fvg_sl_mode_val = test_params['fvg_sl_mode']
                            bypass = test_params['bypass']
                            ag_enabled = test_params['anti_gap']
                            ag_mult = test_params['anti_gap_mult']
                            
                            config.MIN_CONFLUENCE_SCORE = conf
                            
                            def decorated_progress(current, total, symbol, tf_name, phase=None):
                                phase_prefix = f"[(Opt) Mutating {param_name}={val}] "
                                if phase:
                                    update_progress(current, total, symbol, tf_name, phase_prefix + phase)
                                else:
                                    update_progress(current, total, symbol, tf_name, phase_prefix + f"{symbol} {tf_name}: {current}/{total}")
                            
                            trades, metrics = combined_backtest(
                                symbols, date_from, date_to, initial_balance, risk_percent, fixed_lot, risk_mode,
                                bt_ts_methods, ict_params,
                                ict_method=selected_bt_methods, min_rrr=rrr,
                                use_dynamic_rrr=bt_use_dynamic_rrr,
                                trade_on_all_tfs=bt_trade_all_tfs,
                                use_ultra_low_tf=bt_use_ultra_low_tf,
                                fvg_sl_mode=fvg_sl_mode_val,
                                spread_cost=spread_cost, slippage_points=slippage_pts,
                                commission_per_lot=commission,
                                session_filter=session_fil,
                                session_start=session_start, session_end=session_end,
                                progress_callback=decorated_progress,
                                use_htf_filter=htf,
                                use_ote_filter=ote,
                                bypass_htf_conf=bypass,
                                trail_type=bt_trail_type,
                                trail_params=bt_trail_params,
                                require_bos_fvg=bos,
                                enable_slippage_recovery=bt_slippage_recovery,
                                anti_gap_enabled=ag_enabled,
                                anti_gap_mult=ag_mult,
                                fvg_displacement_only=bt_fvg_displacement_only,
                                fvg_discount_premium_only=bt_fvg_discount_premium_only,
                                fvg_recent_sweep_only=bt_fvg_recent_sweep_only,
                                sb_require_htf_bias=bt_sb_require_htf_bias,
                                use_symbol_profiles=False, ml_filter=bt_ml_filter, ml_min_confidence=bt_ml_min_confidence,
                                use_smt_divergence=base_settings.get('use_smt_divergence', False),
                                smt_correlated_pair=base_settings.get('smt_correlated_pair', 'DXY'),
                                use_volume_profile=base_settings.get('use_volume_profile', False),
                                max_concurrent_trades=int(base_settings.get('max_concurrent_trades', 3)),
                                daily_loss_limit=float(base_settings.get('daily_loss_limit', 0.0)),
                                min_confluence_score=conf,
                                min_fvg_size=float(base_settings.get('min_fvg_size', 0.5))
                            )
                            # Fetch proper metric keys
                            pnl = metrics.get('net_profit', 0)
                            dd = metrics.get('max_drawdown_pct', 100.0)
                            score = pnl / max(0.1, dd) if pnl > 0.0 else pnl
                            
                            desc = f"RRR:{rrr:.1f}|Sess:{session_fil}|Conf:{conf}|HTF:{htf}|OTE:{ote}|ReqBoS:{bos}|FVG_SL:{fvg_sl_mode_val} (Score: {score:.1f})"
                            all_results.append((desc, trades, metrics))
                            
                            if score > best_score:
                                best_score = score
                                current_best_params = test_params.copy()
                                break
                    total_runs = len(all_results)
                else:
                    total_runs = len(runs)
                    for run_idx, params in enumerate(runs):
                        if config.bt_stop_event.is_set(): break
                        rrr = params['rrr']
                        session_fil = params['session']
                        conf = params['conf']
                        htf = params['htf']
                        ote = params['ote']
                        bos = params['bos']
                        fvg_sl_mode_val = params['fvg_sl_mode']
                        bypass = params['bypass']
                        ag_enabled = params['anti_gap']
                        ag_mult = params['anti_gap_mult']
                        
                        config.MIN_CONFLUENCE_SCORE = conf

                        def decorated_progress(current, total, symbol, tf_name, phase=None):
                            phase_prefix = f"[(Run {run_idx+1}/{total_runs}) R={rrr} S={session_fil} C={conf} H={htf} O={ote}] "
                            if phase:
                                update_progress(current, total, symbol, tf_name, phase_prefix + phase)
                            else:
                                update_progress(current, total, symbol, tf_name, phase_prefix + f"{symbol} {tf_name}: {current}/{total}")
                        
                        trades, metrics = combined_backtest(
                            symbols, date_from, date_to, initial_balance, risk_percent, fixed_lot, risk_mode,
                            bt_ts_methods, ict_params,
                            ict_method=selected_bt_methods, min_rrr=rrr,
                            use_dynamic_rrr=bt_use_dynamic_rrr,
                            trade_on_all_tfs=bt_trade_all_tfs,
                            use_ultra_low_tf=bt_use_ultra_low_tf,
                            fvg_sl_mode=fvg_sl_mode_val,
                            spread_cost=spread_cost, slippage_points=slippage_pts,
                            commission_per_lot=commission,
                            session_filter=session_fil,
                            session_start=session_start, session_end=session_end,
                            progress_callback=decorated_progress,
                            use_htf_filter=htf,
                            use_ote_filter=ote,
                            bypass_htf_conf=bypass,
                            trail_type=bt_trail_type,
                            trail_params=bt_trail_params,
                            require_bos_fvg=bos,
                            enable_slippage_recovery=bt_slippage_recovery,
                            anti_gap_enabled=ag_enabled,
                            anti_gap_mult=ag_mult,
                            fvg_displacement_only=bt_fvg_displacement_only,
                            fvg_discount_premium_only=bt_fvg_discount_premium_only,
                            fvg_recent_sweep_only=bt_fvg_recent_sweep_only,
                            sb_require_htf_bias=bt_sb_require_htf_bias,
                            use_symbol_profiles=bt_use_symbol_profiles, ml_filter=bt_ml_filter, ml_min_confidence=bt_ml_min_confidence,
                            use_smt_divergence=base_settings.get('use_smt_divergence', False),
                            smt_correlated_pair=base_settings.get('smt_correlated_pair', 'DXY'),
                            use_volume_profile=base_settings.get('use_volume_profile', False),
                            use_institutional_orderflow=base_settings.get('use_institutional_orderflow', False),
                            institutional_lookback=int(base_settings.get('institutional_lookback', 20)),
                            institutional_threshold=float(base_settings.get('institutional_threshold', 0.5)),
                            max_concurrent_trades=int(base_settings.get('max_concurrent_trades', 3)),
                            daily_loss_limit=float(base_settings.get('daily_loss_limit', 0.0)),
                            min_confluence_score=conf,
                            min_fvg_size=float(base_settings.get('min_fvg_size', 0.5))
                        )
                        
                        desc = rrr
                        all_results.append((desc, trades, metrics))
                
                if config.bt_stop_event.is_set():
                    self.after(0, lambda: self.bt_progress_label.configure(text="Stopped by user"))
                    return

                elapsed = time.time() - self.bt_start_time if self.bt_start_time else 0
                elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
                self.after(0, lambda: self.bt_progress_bar.configure(value=100))
                self.after(0, lambda: self.bt_progress_label.configure(text=f"Done! ({total_runs} runs)"))
                self.after(0, lambda t=elapsed_str: self.bt_elapsed_label.configure(text=f"Total: {t}"))
                if all_results:
                    try:
                        best_run = max(all_results, key=lambda x: x[2].get('net_profit', 0))
                        self.last_backtest_trades = best_run[1]
                        self.last_backtest_symbol = symbols[0]
                    except Exception as e:
                        logger.error("Error saving last backtest trades for ML: %s", e)
                self.after(0, lambda: self.show_multi_backtest_results(all_results, symbols, initial_balance, date_from, date_to))
            except Exception as e:
                error_msg = str(e)
                logger.error("Backtest error: %s\n%s", error_msg, traceback.format_exc())
                self.after(0, lambda msg=error_msg: messagebox.showerror("Backtest Error", msg))
                self.after(0, lambda: self.bt_progress_label.configure(text="Error!"))
            finally:
                self.after(0, lambda: self.run_backtest_btn.config(state=tk.NORMAL))
                self.after(0, lambda: self.stop_backtest_btn.config(state=tk.DISABLED))

        threading.Thread(target=run_bt, daemon=True).start()

    def stop_optimization(self):
        config.bt_stop_event.set()
        self.opt_progress_label.config(text="Stopping...")
        if hasattr(self, 'grid_opt_state') and self.grid_opt_state:
            self.grid_opt_state['is_paused'] = False # release any pause lock

    def pause_optimization(self):
        if hasattr(self, 'grid_opt_state') and self.grid_opt_state:
            self.grid_opt_state['is_paused'] = True
            self.opt_progress_label.config(text="Pausing...")
            self.opt_pause_btn.config(state=tk.DISABLED)
            self.opt_resume_btn.config(state=tk.NORMAL)
            self.opt_save_btn.config(state=tk.NORMAL)

    def resume_optimization(self):
        if hasattr(self, 'grid_opt_state') and self.grid_opt_state:
            self.grid_opt_state['is_paused'] = False
            self.opt_progress_label.config(text="Resuming...")
            self.opt_pause_btn.config(state=tk.NORMAL)
            self.opt_resume_btn.config(state=tk.DISABLED)
            self.opt_save_btn.config(state=tk.DISABLED)

    def save_opt_stage(self):
        if not hasattr(self, 'grid_opt_state') or not self.grid_opt_state:
            messagebox.showerror("Error", "No active optimization stage to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Optimization Stage"
        )
        if not file_path:
            return
            
        try:
            state_to_save = dict(self.grid_opt_state)
            
            # Convert Timestamp/datetime in fixed_params to strings
            fp_copy = dict(state_to_save.get('fixed_params', {}))
            for k in ['date_from', 'date_to']:
                if k in fp_copy and hasattr(fp_copy[k], 'strftime'):
                    fp_copy[k] = fp_copy[k].strftime("%Y-%m-%d %H:%M:%S")
            state_to_save['fixed_params'] = fp_copy
            
            with open(file_path, 'w') as f:
                json.dump(state_to_save, f, indent=4)
                
            messagebox.showinfo("Success", f"Optimization stage saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save stage: {e}")

    def load_opt_stage(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load Optimization Stage"
        )
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                loaded_state = json.load(f)
                
            required_keys = ['fixed_params', 'grid', 'current_index', 'opt_results', 'phase']
            if not all(k in loaded_state for k in required_keys):
                raise ValueError("JSON file is missing required optimization state fields.")
                
            self.grid_opt_state = loaded_state
            
            # Clear and populate Treeview
            for item in self.opt_tree.get_children():
                self.opt_tree.delete(item)
            self.opt_images.clear()
            
            results_to_display = list(self.grid_opt_state.get('opt_results', []))
            results_to_display.sort(key=lambda x: x.get('calmar', -999.0), reverse=True)
            self.display_sorted_opt_results(results_to_display)
            
            # Configure UI buttons
            self.opt_run_btn.config(state=tk.DISABLED)
            self.opt_stop_btn.config(state=tk.NORMAL)
            self.opt_pause_btn.config(state=tk.NORMAL)
            self.opt_resume_btn.config(state=tk.DISABLED)
            self.opt_save_btn.config(state=tk.DISABLED)
            self.opt_load_btn.config(state=tk.DISABLED)
            
            # Reset stop event
            config.bt_stop_event.clear()
            
            # Re-run optimization thread from loaded stage!
            threading.Thread(
                target=self.run_grid_opt_thread,
                daemon=True
            ).start()
            
            messagebox.showinfo("Success", f"Loaded stage with {len(results_to_display)} results. Resuming optimization...")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load stage: {e}")

    def sync_opt_settings_from_backtest(self):
        self.opt_symbol_entry.delete(0, tk.END)
        self.opt_symbol_entry.insert(0, self.backtest_symbol_entry.get())
        
        self.opt_date_from.delete(0, tk.END)
        self.opt_date_from.insert(0, self.backtest_date_from.get())
        
        self.opt_date_to.delete(0, tk.END)
        self.opt_date_to.insert(0, self.backtest_date_to.get())
        
        self.opt_balance_entry.delete(0, tk.END)
        self.opt_balance_entry.insert(0, self.backtest_balance_entry.get())
        
        self.opt_risk_entry.delete(0, tk.END)
        self.opt_risk_entry.insert(0, self.backtest_risk_entry.get())
        
        self.opt_fixed_lot_entry.delete(0, tk.END)
        self.opt_fixed_lot_entry.insert(0, self.backtest_fixed_lot_entry.get())
        self.opt_use_fixed_lot.set(self.backtest_use_fixed_lot.get())
        
        self.opt_spread_cost.delete(0, tk.END)
        self.opt_spread_cost.insert(0, self.bt_spread_cost.get())
        
        self.opt_slippage.delete(0, tk.END)
        self.opt_slippage.insert(0, self.bt_slippage.get())
        
        self.opt_commission.delete(0, tk.END)
        self.opt_commission.insert(0, self.bt_commission.get())
        
        self.opt_session_filter.set(self.bt_session_filter.get())
        self.opt_session_start.delete(0, tk.END)
        self.opt_session_start.insert(0, self.bt_session_start.get())
        self.opt_session_end.delete(0, tk.END)
        self.opt_session_end.insert(0, self.bt_session_end.get())
        
        self.opt_anti_gap.set(self.bt_anti_gap.get())
        self.opt_anti_gap_mult.delete(0, tk.END)
        self.opt_anti_gap_mult.insert(0, self.bt_anti_gap_mult.get())
        
        self.opt_min_fvg.delete(0, tk.END)
        self.opt_min_fvg.insert(0, self.bt_min_fvg.get())

        self.opt_trade_all_tfs.set(self.trade_all_tfs_backtest.get())
        self.opt_use_ultra_low_tf.set(self.use_ultra_low_tf_backtest.get())
        self.opt_bypass_htf_conf.set(self.bypass_htf_backtest.get())
        self.opt_require_bos_fvg.set(self.bt_require_bos_fvg.get())
        self.opt_slippage_recovery.set(self.bt_slippage_recovery_bt.get())
        self.opt_use_dynamic_rrr.set(self.use_dynamic_rrr_backtest.get())
        
        self.opt_min_conf.delete(0, tk.END)
        self.opt_min_conf.insert(0, self.bt_min_conf.get())
        
        self.opt_daily_loss.delete(0, tk.END)
        self.opt_daily_loss.insert(0, self.bt_daily_loss.get())
        
        self.opt_trail_pct.delete(0, tk.END)
        self.opt_trail_pct.insert(0, self.trail_pct_bt.get())
        
        # Sync trailing stop methods selection checkboxes
        sel_ts = self.bt_trail_methods_listbox.curselection()
        bt_ts_methods = [self.bt_trail_methods_listbox.get(i) for i in sel_ts]
        self.opt_trail_fvg.set("FVG Return" in bt_ts_methods)
        self.opt_trail_sb.set("Silver Bullet" in bt_ts_methods)
        self.opt_trail_mmxm.set("MMXM" in bt_ts_methods)
        self.opt_trail_2022.set("ICT Model 2022" in bt_ts_methods)
        self.opt_trail_2025.set("ICT Model 2025" in bt_ts_methods)

    def update_opt_ui(self, pct, lbl_text, elapsed_str):
        self.opt_progress_bar['value'] = pct
        self.opt_progress_label.config(text=lbl_text)
        self.opt_elapsed_label.config(text=f"Elapsed: {elapsed_str}")

    def add_opt_result_row(self, r):
        idx = len(self.opt_tree.get_children()) + 1
        
        shorthands = {
            "FVG Return": "FVG",
            "Silver Bullet": "Silver",
            "MMXM": "MMXM",
            "ICT Model 2022": "ICT2022",
            "ICT Model 2025": "ICT2025"
        }
        methods_str = "+".join([shorthands.get(m, m) for m in r['methods']])
        
        sl_int = r['fvg_sl_mode']
        if sl_int == config.FVG_SL_NORMAL: sl_str = config.FVG_SL_NORMAL
        elif sl_int == config.FVG_SL_SWEEP: sl_str = config.FVG_SL_SWEEP
        elif sl_int == config.FVG_SL_CANDLE: sl_str = config.FVG_SL_CANDLE
        elif sl_int == config.FVG_SL_BOS: sl_str = config.FVG_SL_BOS
        elif sl_int == config.FVG_SL_MSS: sl_str = config.FVG_SL_MSS
        else: sl_str = config.FVG_SL_NORMAL
        
        trail_str = str(r['trail_type'])
        ote_str = str(r['use_ote_filter'])
        htf_str = str(r['use_htf_filter'])
        vp_str = str(r.get('use_vp', False))
        smt_str = str(r.get('use_smt', False))
        sweep_str = str(r['fvg_recent_sweep_only'])
        disp_str = str(r['fvg_displacement_only'])
        disc_str = str(r['fvg_discount_premium_only'])
        concurr = r['max_concurrent']
        rrr = r['rrr']
        
        is_waste = r.get('is_waste', False)
        if is_waste:
            roi = "Waste of time"
            dd = "N/A"
            calmar = "N/A"
            trades = str(r.get('trades', 1))
            wr = "N/A"
            pf = "N/A"
        else:
            roi = f"{r['roi']:+.2f}%"
            dd = f"{r['dd']:.2f}%"
            calmar = f"{r['calmar']:.2f}"
            trades = r['trades']
            wr = f"{r['win_rate']:.1f}%"
            pf = f"{r['pf']:.2f}"
            
        dl = f"{r['daily_loss']:.1f}%"
        
        photo = self.empty_sparkline_photo
        
        sb_htf_str = str(r.get('sb_require_htf_bias', False))
        
        row_id = self.opt_tree.insert(
            "", tk.END,
            image=photo,
            values=(idx, roi, dd, calmar, trades, wr, pf, methods_str, sl_str, trail_str, ote_str, htf_str, vp_str, smt_str, sweep_str, disp_str, disc_str, concurr, rrr, dl, sb_htf_str)
        )

    def display_sorted_opt_results(self, sorted_results):
        for item in self.opt_tree.get_children():
            self.opt_tree.delete(item)
        self.opt_images.clear()
            
        for rank, r in enumerate(sorted_results, 1):
            methods_str = "+".join([m.split()[0] for m in r['methods']])
            sl_str = r['fvg_sl_mode']
            trail_str = str(r['trail_type'])
            ote_str = str(r['use_ote_filter'])
            htf_str = str(r['use_htf_filter'])
            vp_str = str(r.get('use_vp', False))
            smt_str = str(r.get('use_smt', False))
            sweep_str = str(r['fvg_recent_sweep_only'])
            disp_str = str(r['fvg_displacement_only'])
            disc_str = str(r['fvg_discount_premium_only'])
            concurr = r['max_concurrent']
            rrr = r['rrr']
            dl = f"{r['daily_loss']:.1f}%"
            
            is_waste = r.get('is_waste', False)
            if is_waste:
                roi = "Waste of time"
                dd = "N/A"
                calmar = "N/A"
                trades = str(r.get('trades', 1))
                wr = "N/A"
                pf = "N/A"
            else:
                roi = f"{r['roi']:+.2f}%"
                dd = f"{r['dd']:.2f}%"
                calmar = f"{r['calmar']:.2f}"
                trades = r['trades']
                wr = f"{r['win_rate']:.1f}%"
                pf = f"{r['pf']:.2f}"
            
            balance_hist = [] if is_waste else r.get('balance_history', [])
            if not balance_hist or rank > 100:
                photo = self.empty_sparkline_photo
            else:
                img = generate_sparkline_image(balance_hist, width=100, height=28)
                photo = ImageTk.PhotoImage(img)
            
            sb_htf_str = str(r.get('sb_require_htf_bias', False))
            
            row_id = self.opt_tree.insert(
                "", tk.END,
                image=photo,
                values=(rank, roi, dd, calmar, trades, wr, pf, methods_str, sl_str, trail_str, ote_str, htf_str, vp_str, smt_str, sweep_str, disp_str, disc_str, concurr, rrr, dl, sb_htf_str)
            )
            if balance_hist:
                self.opt_images[row_id] = photo

    def sort_treeview(self, tree, col, reverse):
        l = [(tree.set(k, col), k) for k in tree.get_children('')]
        try:
            def parse_val(v):
                clean_v = str(v).replace('%', '').replace('$', '').replace('+', '').strip()
                if clean_v in ["Waste of time", "N/A", "-", ""]:
                    return -999999.0 if reverse else 999999.0
                try:
                    return float(clean_v)
                except ValueError:
                    return clean_v
            
            l.sort(key=lambda t: parse_val(t[0]), reverse=reverse)
        except Exception:
            l.sort(key=lambda t: t[0], reverse=reverse)
            
        for index, (val, k) in enumerate(l):
            tree.move(k, '', index)
            
        tree.heading(col, command=lambda: self.sort_treeview(tree, col, not reverse))

    def _apply_selected_opt_to_vars(self):
        selected_item = self.opt_tree.selection()
        if not selected_item:
            raise ValueError("Please select a row from the results table first.")
            
        values = self.opt_tree.item(selected_item)['values']
        methods_str = values[7]
        sl_mode_raw = str(values[8])
        import config
        valid_modes = config.FVG_SL_MODES
        sl_mode = sl_mode_raw if sl_mode_raw in valid_modes else config.FVG_SL_NORMAL
        
        trail_type = values[9]
        ote = str(values[10]) == "True"
        htf = str(values[11]) == "True"
        vp = str(values[12]) == "True"
        smt = str(values[13]) == "True"
        sweep = str(values[14]) == "True"
        disp = str(values[15]) == "True"
        disc = str(values[16]) == "True"
        concurr = int(values[17])
        rrr = float(values[18])
        opt_daily_loss_val = float(str(values[19]).replace('%', ''))
        sb_htf = str(values[20]) == "True" if len(values) > 20 else False
        
        # Apply basics
        import tkinter as tk
        self.backtest_symbol_entry.delete(0, tk.END)
        self.backtest_symbol_entry.insert(0, self.opt_symbol_entry.get())
        self.backtest_date_from.delete(0, tk.END)
        self.backtest_date_from.insert(0, self.opt_date_from.get())
        self.backtest_date_to.delete(0, tk.END)
        self.backtest_date_to.insert(0, self.opt_date_to.get())
        self.backtest_balance_entry.delete(0, tk.END)
        self.backtest_balance_entry.insert(0, self.opt_balance_entry.get())
        self.backtest_risk_entry.delete(0, tk.END)
        self.backtest_risk_entry.insert(0, self.opt_risk_entry.get())
        self.backtest_fixed_lot_entry.delete(0, tk.END)
        self.backtest_fixed_lot_entry.insert(0, self.opt_fixed_lot_entry.get())
        self.backtest_use_fixed_lot.set(self.opt_use_fixed_lot.get())
        if hasattr(self, 'risk_mode_var'):
            self.risk_mode_var.set("Fixed" if self.opt_use_fixed_lot.get() else "Risk")
            
        self.bt_spread_cost.delete(0, tk.END)
        self.bt_spread_cost.insert(0, self.opt_spread_cost.get())
        self.bt_slippage.delete(0, tk.END)
        self.bt_slippage.insert(0, self.opt_slippage.get())
        self.bt_commission.delete(0, tk.END)
        self.bt_commission.insert(0, self.opt_commission.get())

        # Apply ICT Methods
        reverse_shorthands = {
            "FVG": "FVG Return",
            "Silver": "Silver Bullet",
            "MMXM": "MMXM",
            "Model 2022": "ICT Model 2022",
            "Model 2025": "ICT Model 2025",
            "ICT2022": "ICT Model 2022",
            "ICT2025": "ICT Model 2025"
        }
        self.bt_methods_listbox.selection_clear(0, tk.END)
        if methods_str and methods_str != "None":
            for sh in methods_str.split("+"):
                full_name = reverse_shorthands.get(sh, sh)
                for idx in range(self.bt_methods_listbox.size()):
                    if self.bt_methods_listbox.get(idx) == full_name:
                        self.bt_methods_listbox.select_set(idx)

        # Synchronously invoke select handler to configure widgets and repopulate trailing listbox items
        self.on_bt_methods_select(None)
        
        self.bt_fvg_sl_mode.set(sl_mode)
        
        # Build trail methods from the OPT tab trailing checkboxes (same source the grid uses)
        opt_ts_methods = []
        if getattr(self, 'opt_trail_fvg', None) and self.opt_trail_fvg.get(): opt_ts_methods.append("FVG Return")
        if getattr(self, 'opt_trail_sb', None) and self.opt_trail_sb.get(): opt_ts_methods.append("Silver Bullet")
        if getattr(self, 'opt_trail_mmxm', None) and self.opt_trail_mmxm.get(): opt_ts_methods.append("MMXM")
        if getattr(self, 'opt_trail_2022', None) and self.opt_trail_2022.get(): opt_ts_methods.append("ICT Model 2022")
        if getattr(self, 'opt_trail_2025', None) and self.opt_trail_2025.get(): opt_ts_methods.append("ICT Model 2025")

        if trail_type == "None" or not trail_type or trail_type == "None (Disabled)":
            self.trail_type_bt.set("None (Disabled)")
            self.bt_trail_methods_listbox.selection_clear(0, tk.END)
        else:
            self.trail_type_bt.set(trail_type)
            self.bt_trail_methods_listbox.selection_clear(0, tk.END)
            for m in opt_ts_methods:
                for idx in range(self.bt_trail_methods_listbox.size()):
                    if self.bt_trail_methods_listbox.get(idx) == m:
                        self.bt_trail_methods_listbox.select_set(idx)

        # Sync the trailing-stop percentage from the OPT tab.
        if hasattr(self, 'trail_pct_bt') and hasattr(self, 'opt_trail_pct'):
            self.trail_pct_bt.delete(0, tk.END)
            self.trail_pct_bt.insert(0, self.opt_trail_pct.get())
            
        # Carry over Pro Strategy Upgrades & ML Filter from Grid Opt to Backtest
        if hasattr(self, 'opt_pro_dol_tp') and hasattr(self, 'bt_pro_dol_tp'):
            self.bt_pro_dol_tp.set(self.opt_pro_dol_tp.get())
            self.bt_pro_killzone.set(self.opt_pro_killzone.get())
            self.bt_pro_htf_poi.set(self.opt_pro_htf_poi.get())
            self.bt_pro_mandatory.set(self.opt_pro_mandatory.get())
            self.bt_pro_regime.set(self.opt_pro_regime.get())
            self.bt_pro_ml_sizing.set(self.opt_pro_ml_sizing.get())
            self.bt_pro_ml_rank.set(self.opt_pro_ml_rank.get())
            self.bt_pro_multi_tf_conf.set(self.opt_pro_multi_tf_conf.get())
            self.bt_pro_multi_tf_gate.set(self.opt_pro_multi_tf_gate.get())
            self.bt_pro_regime_adaptive.set(self.opt_pro_regime_adaptive.get())

            self.bt_ml_filter.set(self.opt_ml_filter.get())
            self.bt_ml_min_confidence.delete(0, tk.END)
            self.bt_ml_min_confidence.insert(0, self.opt_ml_min_confidence.get())
            
            # Flush changes into global config
            self._apply_pro_flags('bt')

        self.ote_filter_backtest.set(ote)
        self.htf_filter_backtest.set(htf)
        self.bt_fvg_recent_sweep_only.set(sweep)
        self.bt_fvg_displacement_only.set(disp)
        self.bt_fvg_discount_premium_only.set(disc)
        self.bt_vp_enabled.set(vp)
        self.bt_smt_enabled.set(smt)
        
        self.bt_max_concurr.delete(0, tk.END)
        self.bt_max_concurr.insert(0, str(concurr))
        
        self.backtest_min_rrr_entry.delete(0, tk.END)
        self.backtest_min_rrr_entry.insert(0, str(rrr))
        self.backtest_max_rrr_entry.delete(0, tk.END)
        self.backtest_max_rrr_entry.insert(0, str(rrr))
        
        self.bt_min_fvg.delete(0, tk.END)
        self.bt_min_fvg.insert(0, self.opt_min_fvg.get())
        
        self.bt_min_conf.delete(0, tk.END)
        self.bt_min_conf.insert(0, self.opt_min_conf.get())
        
        self.bt_daily_loss.delete(0, tk.END)
        self.bt_daily_loss.insert(0, str(opt_daily_loss_val))
        
        self.trade_all_tfs_backtest.set(self.opt_trade_all_tfs.get())
        self.use_ultra_low_tf_backtest.set(self.opt_use_ultra_low_tf.get())
        self.bypass_htf_backtest.set(self.opt_bypass_htf_conf.get())
        self.bt_sb_require_htf_bias.set(sb_htf)
        
        self.bt_require_bos_fvg.set(self.opt_require_bos_fvg.get())
        self.bt_anti_gap.set(self.opt_anti_gap.get())
        self.bt_anti_gap_mult.delete(0, tk.END)
        self.bt_anti_gap_mult.insert(0, self.opt_anti_gap_mult.get())
        
        self.bt_slippage_recovery_bt.set(self.opt_slippage_recovery.get())
        self.use_dynamic_rrr_backtest.set(self.opt_use_dynamic_rrr.get())
        
        self.bt_news_enabled.set(self.opt_news_enabled.get() if hasattr(self, 'opt_news_enabled') else False)
        if hasattr(self, 'opt_news_buffer'):
            self.bt_news_buffer.delete(0, tk.END)
            self.bt_news_buffer.insert(0, self.opt_news_buffer.get())
            
        if hasattr(self, 'opt_session_filter') and hasattr(self, 'bt_session_filter'):
            self.bt_session_filter.set(self.opt_session_filter.get())
            if hasattr(self, 'opt_session_start') and hasattr(self, 'bt_session_start'):
                self.bt_session_start.delete(0, tk.END)
                self.bt_session_start.insert(0, self.opt_session_start.get())
            if hasattr(self, 'opt_session_end') and hasattr(self, 'bt_session_end'):
                self.bt_session_end.delete(0, tk.END)
                self.bt_session_end.insert(0, self.opt_session_end.get())
                
        if getattr(self, 'opt_smt_pair', None) and hasattr(self, 'bt_smt_pair'):
            self.bt_smt_pair.set(self.opt_smt_pair.get())

        # Disable single run use_symbol_profiles when copying a setting
        if hasattr(self, 'bt_use_overrides_var'):
            self.bt_use_overrides_var.set(False)

        import tkinter.messagebox as messagebox
        messagebox.showinfo("Success", "Settings applied to Single Backtest tab!\nSwitching to Backtest tab.")
        self.notebook.select(0)

    def apply_opt_to_backtest(self):
        try:
            self._apply_selected_opt_to_vars()
            messagebox.showinfo("Success", "Settings applied successfully to the Single Run Backtest tab!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_opt_to_live(self):
        try:
            self._apply_selected_opt_to_vars()
            self.copy_settings_to_live()
            messagebox.showinfo("Success", "Settings applied successfully to both the Backtest and Live Trading tabs!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_opt_report(self):
        selected_item = self.opt_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Select a row from the results table first.")
            return
            
        values = self.opt_tree.item(selected_item)['values']
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Save Optimization Configuration Report"
        )
        if not file_path:
            return
            
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("=== SMC/ICT TRADING BOT OPTIMIZATION CONFIGURATION REPORT ===\n\n")
                f.write(f"Symbol: {self.opt_symbol_entry.get()}\n")
                f.write(f"Period: {self.opt_date_from.get()} to {self.opt_date_to.get()}\n")
                f.write(f"Risk Setting: {self.opt_risk_entry.get()}% per trade\n\n")
                f.write(f"Rank: #{values[0]}\n")
                f.write(f"ROI %: {values[1]}\n")
                f.write(f"Max DD %: {values[2]}\n")
                f.write(f"Calmar Ratio: {values[3]}\n")
                f.write(f"Total Trades: {values[4]}\n")
                f.write(f"Win Rate %: {values[5]}\n")
                f.write(f"Profit Factor: {values[6]}\n\n")
                f.write("--- Selected Settings ---\n")
                f.write(f"Methods Enabled: {values[7]}\n")
                f.write(f"Stop Loss Mode: {values[8]}\n")
                f.write(f"Trailing Stop Type: {values[9]}\n")
                f.write(f"OTE Filter: {values[10]}\n")
                f.write(f"HTF Filter: {values[11]}\n")
                f.write(f"VP Filter: {values[12]}\n")
                f.write(f"SMT Filter: {values[13]}\n")
                f.write(f"Recent Sweep Only: {values[14]}\n")
                f.write(f"Displacement Filter: {values[15]}\n")
                f.write(f"Discount/Premium Filter: {values[16]}\n")
                f.write(f"Max Concurrent Trades: {values[17]}\n")
                f.write(f"Minimum RRR: {values[18]}\n")
                f.write(f"Daily Loss Limit %: {values[19]}\n")
                if len(values) > 20:
                    f.write(f"MTF Alignment: {values[20]}\n")
                # Find matching configuration in grid_opt_state['opt_results']
                matched_r = None
                if hasattr(self, 'grid_opt_state') and self.grid_opt_state and 'opt_results' in self.grid_opt_state:
                    methods_str = values[7]
                    sl_mode = values[8]
                    trail_type = str(values[9])
                    ote = str(values[10]) == "True"
                    htf = str(values[11]) == "True"
                    vp = str(values[12]) == "True"
                    smt = str(values[13]) == "True"
                    sweep = str(values[14]) == "True"
                    disp = str(values[15]) == "True"
                    disc = str(values[16]) == "True"
                    concurr = int(values[17])
                    rrr = float(values[18])
                    dl_val = float(str(values[19]).replace('%', ''))
                    sb_htf = str(values[20]) == "True" if len(values) > 20 else False
                    
                    for r in self.grid_opt_state['opt_results']:
                        r_methods_str = "+".join([m.split()[0] for m in r['methods']])
                        if (r_methods_str == methods_str and
                            r['fvg_sl_mode'] == sl_mode and
                            str(r['trail_type']) == trail_type and
                            r['use_ote_filter'] == ote and
                            r['use_htf_filter'] == htf and
                            r['fvg_recent_sweep_only'] == sweep and
                            r['fvg_displacement_only'] == disp and
                            r['fvg_discount_premium_only'] == disc and
                            r['max_concurrent'] == concurr and
                            abs(r['rrr'] - rrr) < 0.001 and
                            abs(r['daily_loss'] - dl_val) < 0.01):
                            matched_r = r
                            break
                            
                if matched_r and not matched_r.get('is_waste', False):
                    balance_hist = matched_r.get('balance_history', [])
                    if balance_hist:
                        braille_spark = get_braille_sparkline(balance_hist, length=15)
                        f.write(f"Balance Curve: {braille_spark}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {e}")

    def export_opt_csv(self):
        if not hasattr(self, 'grid_opt_state') or not self.grid_opt_state or not self.grid_opt_state.get('opt_results'):
            messagebox.showerror("Error", "No optimization results to export.")
            return
            
        from tkinter import simpledialog
        regime_label = simpledialog.askstring("Regime Label", "Enter market regime for these results (e.g., trend, range, chop, high-vol, weak-trend):")
        if regime_label is None: # User cancelled
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Optimization Results to CSV"
        )
        if not file_path:
            return
            
        try:
            import pandas as pd
            results = self.grid_opt_state['opt_results']
            rows = []
            for r in results:
                row = dict(r)
                if 'methods' in row and isinstance(row['methods'], list):
                    row['methods'] = "|".join(row['methods'])
                row['regime_label'] = regime_label.strip()
                rows.append(row)
                
            df = pd.DataFrame(rows)
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Results exported to {file_path}. You can now load this via regime_manager.py!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV: {e}")

    def start_oos_test(self):
        import traceback
        try:
            self._start_oos_test_impl()
        except Exception as e:
            messagebox.showerror("Crash", f"Error starting OOS Test: {e}\n{traceback.format_exc()}")
            
    def _start_oos_test_impl(self):
        import datetime
        import time
        import threading
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info("start_oos_test clicked.")
        self._apply_pro_flags('bt')  # sync Backtest-tab v17 Pro toggles into config
        
        if not hasattr(self, 'grid_opt_state') or not self.grid_opt_state.get('opt_results'):
            logger.warning("No grid_opt_state or opt_results found.")
            messagebox.showwarning("No Results", "You must run an optimization first before running an OOS test.")
            return
            
        try:
            min_calmar = float(self.oos_min_calmar.get())
            logger.info(f"Min Calmar Threshold: {min_calmar}")
        except ValueError:
            logger.error("Invalid min calmar input.")
            messagebox.showwarning("Invalid Input", "Please enter a valid number for Min Calmar.")
            return
            
        oos_date_str = self.oos_date_from.get().strip()
        logger.info(f"OOS Date string: {oos_date_str}")
        try:
            if " " in oos_date_str:
                oos_date_from = datetime.datetime.strptime(oos_date_str, "%Y-%m-%d %H:%M:%S")
            else:
                oos_date_from = datetime.datetime.strptime(oos_date_str, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid OOS date input.")
            messagebox.showwarning("Invalid Date", "Please enter a valid date in YYYY-MM-DD format.")
            return
            
        # Check against opt date_to
        opt_date_to = self.grid_opt_state.get('fixed_params', {}).get('date_to')
        if opt_date_to:
            if isinstance(opt_date_to, str):
                try:
                    if " " in opt_date_to:
                        opt_date_to = datetime.datetime.strptime(opt_date_to, "%Y-%m-%d %H:%M:%S")
                    else:
                        opt_date_to = datetime.datetime.strptime(opt_date_to, "%Y-%m-%d")
                except ValueError:
                    opt_date_to = None
                    
        if opt_date_to and oos_date_from < opt_date_to:
            logger.warning(f"OOS start date {oos_date_from} overlaps with Opt end date {opt_date_to}.")
            response = messagebox.askyesno("Date Overlap Warning", "The Out-of-Sample start date overlaps with the Optimization data period. This will cause data leakage.\n\nDo you want to continue anyway?")
            if not response:
                logger.info("User cancelled OOS test due to date overlap.")
                return
                
        # Filter results
        valid_results = []
        for r in self.grid_opt_state['opt_results']:
            if r.get('calmar', -999) > min_calmar and not r.get('is_waste', False):
                valid_results.append(r)
                
        logger.info(f"Found {len(valid_results)} valid results with Calmar > {min_calmar} out of {len(self.grid_opt_state['opt_results'])} total.")
        if not valid_results:
            messagebox.showinfo("No Results", f"No optimization results found with Calmar > {min_calmar}.")
            return
            
        # Clear treeview
        for item in self.oos_tree.get_children():
            self.oos_tree.delete(item)
            
        self.oos_run_btn.config(state=tk.DISABLED)
        self.oos_progress_label.config(text=f"Starting OOS test on {len(valid_results)} combos...")
        
        threading.Thread(target=self.run_oos_test_thread, args=(valid_results, oos_date_from), daemon=True).start()

    def run_oos_test_thread(self, valid_results, date_from):
        import sys
        import time
        import datetime
        import MetaTrader5 as mt5
        import logging
        import config
        import utils
        import simulation
        import backtester
        
        date_to = datetime.datetime.now()
        fp = self.grid_opt_state['fixed_params']
        symbol = fp['symbol']
        
        total = len(valid_results)
        completed = 0
        oos_outcomes = []
        
        for i, res in enumerate(valid_results):
            self.ui_update_queue.put(lambda c=completed+1, t=total: self.oos_progress_label.config(text=f"Running {c}/{t} ..."))
            
            methods = res['methods']
            sl_mode = res['fvg_sl_mode']
            trail_type = res['trail_type']
            use_ote = res['use_ote_filter']
            use_htf = res['use_htf_filter']
            disp_only = res['fvg_displacement_only']
            disc_prem = res['fvg_discount_premium_only']
            sweep_only = res['fvg_recent_sweep_only']
            concurr = res['max_concurrent']
            rrr = res['rrr']
            dl_val = res['daily_loss']
            vp_val = res.get('use_vp', False)
            smt_val = res.get('use_smt', False)
            
            config.DAILY_LOSS_LIMIT_PCT = dl_val
            config.USE_VOLUME_PROFILE = vp_val
            config.USE_SMT_DIVERGENCE = smt_val
            if hasattr(config, 'MAX_CONCURRENT_TRADES'):
                config.MAX_CONCURRENT_TRADES = concurr
            
            trail_pct = fp.get('trail_pct', 0.5)
            
            try:
                trades, metrics = backtester.combined_backtest(
                    symbols=[symbol],
                    date_from=date_from,
                    date_to=date_to,
                    initial_balance=fp['initial_balance'],
                    risk_percent=fp['risk_percent'],
                    fixed_lot=fp['fixed_lot'],
                    risk_mode=fp['risk_mode'],
                    trailing_methods=fp['trailing_methods'],
                    ict_params=fp['ict_params'],
                    ict_method=methods,
                    min_rrr=rrr,
                    use_dynamic_rrr=fp['use_dynamic_rrr'],
                    trade_on_all_tfs=fp['trade_all_tfs'],
                    use_ultra_low_tf=fp['use_ultra_low_tf'],
                    fvg_sl_mode=sl_mode,
                    spread_cost=fp['spread_cost'],
                    slippage_points=fp['slippage_pts'],
                    commission_per_lot=fp['commission'],
                    session_filter=fp['session_filter'],
                    session_start=fp['session_start'],
                    session_end=fp['session_end'],
                    progress_callback=None,
                    use_htf_filter=use_htf,
                    use_ote_filter=use_ote,
                    bypass_htf_conf=fp['bypass_htf_conf'],
                    trail_type=trail_type,
                    trail_params={'trail_pct': trail_pct},
                    require_bos_fvg=fp.get('require_bos_fvg', False),
                    enable_slippage_recovery=fp.get('slippage_recovery', False),
                    clear_cache=False,
                    anti_gap_enabled=fp.get('anti_gap_enabled', False),
                    anti_gap_mult=fp.get('anti_gap_mult', 2.0),
                    fvg_sl_spread_buffer=fp.get('fvg_sl_spread_buffer', 2.0),
                    limit_touch_fill=fp.get('limit_touch_fill', False),
                    fvg_displacement_only=disp_only,
                    fvg_discount_premium_only=disc_prem,
                    fvg_recent_sweep_only=sweep_only,
                    use_symbol_profiles=False, 
                    ml_filter=self.bt_ml_filter.get(), 
                    ml_min_confidence=float(self.bt_ml_min_confidence.get()) / 100.0 if getattr(self, 'bt_ml_min_confidence', None) else 0.60,
                    use_smt_divergence=smt_val,
                    smt_correlated_pair=fp.get('smt_correlated_pair', 'DXY'),
                    use_volume_profile=vp_val,
                )
                
                trades_count = len(trades)
                is_waste = False
                if trades_count == 1:
                    t = trades[0]
                    import pandas as pd
                    entry_t = pd.to_datetime(t.get("entry_time"))
                    exit_t = pd.to_datetime(t.get("exit_time"))
                    if entry_t and exit_t:
                        try:
                            if (exit_t - entry_t).total_seconds() / 86400.0 > 10.0:
                                is_waste = True
                        except Exception:
                            pass
                            
                if is_waste:
                    roi, dd, wr, pf, calmar = -999.0, 0.0, 0.0, 0.0, -999.0
                else:
                    roi = metrics.get('roi_pct', 0.0)
                    dd = metrics.get('max_drawdown_pct', 0.0)
                    wr = metrics.get('overall_win_rate', metrics.get('win_rate', 0.0))
                    pf = metrics.get('profit_factor', 0.0)
                    calmar = roi / max(0.1, dd)
                    
                sorted_opt = sorted(self.grid_opt_state['opt_results'], key=lambda x: x.get('calmar', -999.0), reverse=True)
                orig_rank = sorted_opt.index(res) + 1 if res in sorted_opt else "?"
                
                oos_outcomes.append({
                    'orig_rank': orig_rank,
                    'oos_calmar': calmar,
                    'oos_roi': roi,
                    'oos_max_dd': dd,
                    'oos_win_rate': wr,
                    'oos_trades': trades_count,
                    'methods': methods,
                    'sl_mode': sl_mode,
                    'trail_type': str(trail_type),
                    'concurr': concurr,
                    'rrr': rrr,
                    'daily_loss': dl_val
                })
                
            except Exception as e:
                import traceback
                logging.getLogger(__name__).error(f"OOS run failed: {e}\n{traceback.format_exc()}")
                
            completed += 1
            
        oos_outcomes.sort(key=lambda x: x['oos_calmar'], reverse=True)
        
        def update_ui():
            for outcome in oos_outcomes:
                methods_str = "+".join([m.split()[0] for m in outcome['methods']])
                vals = (
                    outcome['orig_rank'],
                    f"{outcome['oos_calmar']:.2f}",
                    f"{outcome['oos_roi']:.2f}",
                    f"{outcome['oos_max_dd']:.2f}",
                    f"{outcome['oos_win_rate']:.1f}",
                    outcome['oos_trades'],
                    methods_str,
                    outcome['sl_mode'],
                    outcome['trail_type'],
                    outcome['concurr'],
                    f"{outcome['rrr']:.1f}",
                    f"{outcome['daily_loss']:.1f}"
                )
                self.oos_tree.insert("", "end", values=vals)
                
            self.oos_progress_label.config(text=f"Done! Evaluated {len(oos_outcomes)} configs.")
            self.oos_run_btn.config(state=tk.NORMAL)
            
        self.ui_update_queue.put(update_ui)


    def run_grid_optimization(self):
        import random
        random.seed(42)
        config.bt_stop_event.clear()
        
        use_mt5_data = self.opt_use_mt5_data.get() if hasattr(self, 'opt_use_mt5_data') else False
        config.OFFLINE_BACKTESTING = not use_mt5_data
        
        symbols_str = self.opt_symbol_entry.get()
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        if not symbols:
            messagebox.showerror("Error", "Enter at least one symbol.")
            return
            
        if not mt5.initialize():
            if use_mt5_data:
                messagebox.showerror("MT5 Connection Error", "Failed to connect to MetaTrader 5 while 'Use MT5 data' is checked. Please ensure MT5 terminal is open.")
                return
            else:
                messagebox.showerror("Error", "Failed to connect to MT5. Ensure terminal is open.")
                return
            
        try:
            date_from = datetime.datetime.strptime(self.opt_date_from.get().strip(), "%Y-%m-%d")
            date_to = datetime.datetime.strptime(self.opt_date_to.get().strip(), "%Y-%m-%d")
            # Include the entire day of date_to
            date_to = date_to.replace(hour=23, minute=59, second=59)
        except ValueError:
            messagebox.showerror("Error", "Date format must be YYYY-MM-DD")
            return
            
        try:
            initial_balance = float(self.opt_balance_entry.get())
            risk_percent = float(self.opt_risk_entry.get())
            fixed_lot = float(self.opt_fixed_lot_entry.get())
            spread_cost = float(self.opt_spread_cost.get())
            slippage_pts = int(self.opt_slippage.get())
            commission = float(self.opt_commission.get())
            session_start = int(self.opt_session_start.get())
            session_end = int(self.opt_session_end.get())
            anti_gap_enabled = self.opt_anti_gap.get()
            anti_gap_mult = float(self.opt_anti_gap_mult.get())
            min_fvg_size = float(self.opt_min_fvg.get())
            
            # Read new settings on main thread
            trade_all_tfs = self.opt_trade_all_tfs.get()
            use_ultra_low_tf = self.opt_use_ultra_low_tf.get()
            bypass_htf_conf = self.opt_bypass_htf_conf.get()
            require_bos_fvg = self.opt_require_bos_fvg.get()
            slippage_recovery = self.opt_slippage_recovery.get()
            use_dynamic_rrr = self.opt_use_dynamic_rrr.get()
            
            min_conf = int(self.opt_min_conf.get())
            daily_loss = float(self.opt_daily_loss.get())
            trail_pct = float(self.opt_trail_pct.get())
            
            session_filter = self.opt_session_filter.get()
        except ValueError:
            messagebox.showerror("Error", "Enter valid numbers for all basic settings.")
            return
            
        risk_mode = "Fixed" if self.opt_use_fixed_lot.get() else "Risk"
        
        individual_methods = []
        if self.opt_sweep_fvg.get(): individual_methods.append("FVG Return")
        if self.opt_sweep_sb.get(): individual_methods.append("Silver Bullet")
        if self.opt_sweep_mmxm.get(): individual_methods.append("MMXM")
        if self.opt_sweep_2022.get(): individual_methods.append("ICT Model 2022")
        if self.opt_sweep_2025.get(): individual_methods.append("ICT Model 2025")
        
        if not individual_methods:
            messagebox.showerror("Error", "Select at least one method to sweep.")
            return
            
        # Extract trailing stop methods selection checkboxes on main thread
        trailing_methods = []
        if self.opt_trail_fvg.get(): trailing_methods.append("FVG Return")
        if self.opt_trail_sb.get(): trailing_methods.append("Silver Bullet")
        if self.opt_trail_mmxm.get(): trailing_methods.append("MMXM")
        if self.opt_trail_2022.get(): trailing_methods.append("ICT Model 2022")
        if self.opt_trail_2025.get(): trailing_methods.append("ICT Model 2025")
        if not trailing_methods:
            trailing_methods = [] # No trailing methods selected
            
        sl_modes = []
        if self.opt_sweep_sl_normal.get(): sl_modes.append("Normal")
        if self.opt_sweep_sl_last_sweep.get(): sl_modes.append("Last Sweep")
        if self.opt_sweep_sl_candle_ext.get(): sl_modes.append("Candle Extreme")
        if not sl_modes: sl_modes = ["Normal"]
        
        trail_types = []
        if self.opt_sweep_trail_none.get(): trail_types.append("None (Disabled)")
        if self.opt_sweep_trail_partial.get(): trail_types.append("True Partial (50% at 1R)")
        if self.opt_sweep_trail_pct.get(): trail_types.append("Percentage Trail")
        if self.opt_sweep_trail_atr.get(): trail_types.append("Profit Lock (50%)")
        if not trail_types: trail_types = ["None (Disabled)"]
        
        ote_vals = [True, False] if self.opt_sweep_ote.get() else [self.opt_default_ote.get()]
        htf_vals = [True, False] if self.opt_sweep_htf.get() else [self.opt_default_htf.get()]
        disp_vals = [True, False] if self.opt_sweep_displacement.get() else [self.opt_default_displacement.get()]
        disc_vals = [True, False] if self.opt_sweep_disc_prem.get() else [self.opt_default_disc_prem.get()]
        sweep_vals = [True, False] if self.opt_sweep_recent_sweep.get() else [self.opt_default_recent_sweep.get()]
        sb_htf_vals = [True, False] if getattr(self, 'opt_sweep_sb_htf_bias', None) and self.opt_sweep_sb_htf_bias.get() else [getattr(self, 'opt_default_sb_htf_bias', None) and self.opt_default_sb_htf_bias.get()]
        vp_vals = [True, False] if self.opt_sweep_vp.get() else [self.opt_default_vp.get()]
        smt_vals = [True, False] if self.opt_sweep_smt.get() else [self.opt_default_smt.get()]
        
        # Override FVG-specific parameters if neither FVG Return nor Silver Bullet is being swept
        if "FVG Return" not in individual_methods and "Silver Bullet" not in individual_methods:
            sl_modes = ["Normal"]
            disp_vals = [False]
            disc_vals = [False]
            sweep_vals = [False]
            sb_htf_vals = [False]

        sweep_combined_profitable = self.opt_sweep_combined_profitable.get()
        
        try:
            c_min = int(self.opt_concurr_min.get())
            c_max = int(self.opt_concurr_max.get())
            concurrencies = list(range(c_min, c_max + 1))
        except ValueError:
            concurrencies = [2]
            
        try:
            r_min = float(self.opt_rrr_min.get())
            r_max = float(self.opt_rrr_max.get())
            r_step = float(self.opt_rrr_step.get())
            rrr_values = []
            curr = r_min
            if r_step > 0:
                while curr <= r_max:
                    rrr_values.append(round(curr, 2))
                    curr += r_step
            if not rrr_values: rrr_values = [r_min]
        except ValueError:
            rrr_values = [1.5]
            
        if self.opt_sweep_daily_loss.get():
            try:
                dl_min = float(self.opt_dl_min.get())
                dl_max = float(self.opt_dl_max.get())
                dl_step = float(self.opt_dl_step.get())
                daily_loss_values = []
                curr = dl_min
                if dl_step > 0:
                    while curr <= dl_max:
                        daily_loss_values.append(round(curr, 2))
                        curr += dl_step
                if not daily_loss_values: daily_loss_values = [dl_min]
            except ValueError:
                daily_loss_values = [daily_loss]
        else:
            daily_loss_values = [daily_loss]
            
        try:
            import os
            n_cores = int(self.opt_cpu_cores.get())
        except ValueError:
            import os
            n_cores = max(1, os.cpu_count() // 2)
            
        self.opt_run_btn.config(state=tk.DISABLED)
        self.opt_stop_btn.config(state=tk.NORMAL)
        self.opt_pause_btn.config(state=tk.NORMAL)
        self.opt_resume_btn.config(state=tk.DISABLED)
        self.opt_save_btn.config(state=tk.DISABLED)
        self.opt_load_btn.config(state=tk.DISABLED)
        
        self._apply_pro_flags('bt')  # sync Backtest-tab v17 Pro toggles into config (grid must use same Pro flags as single backtest)
        self.opt_progress_bar['value'] = 0
        self.opt_progress_label.config(text="Starting optimization...")
        self.opt_elapsed_label.config(text="")
        
        for item in self.opt_tree.get_children():
            self.opt_tree.delete(item)
        self.opt_images.clear()
            
        grid_combos = [[m] for m in individual_methods]
        import itertools
        raw_grid = list(itertools.product(
            grid_combos, sl_modes, trail_types, ote_vals, htf_vals, disp_vals, disc_vals, sweep_vals, 
            vp_vals, smt_vals, concurrencies, rrr_values, daily_loss_values, sb_htf_vals
        ))
        
        grid = []
        for combo in raw_grid:
            (methods, sl_mode, trail_type, use_ote, use_htf, disp_only, disc_prem,
             sweep_only, vp_val, smt_val, concurr, rrr, dl_val, sb_htf_bias) = combo
            grid.append({
                'ict_method': methods,
                'fvg_sl_mode': sl_mode,
                'trail_type': trail_type,
                'use_ote_filter': use_ote,
                'use_htf_filter': use_htf,
                'fvg_displacement_only': disp_only,
                'fvg_discount_premium_only': disc_prem,
                'fvg_recent_sweep_only': sweep_only,
                'use_volume_profile': vp_val,
                'use_smt_divergence': smt_val,
                'max_concurrent_trades': concurr,
                'min_rrr': rrr,
                'daily_loss_limit': dl_val,
                'sb_require_htf_bias': sb_htf_bias
            })
        
        ict_params = get_ict_model_parameters("Default", symbols[0])
        fixed_params = {
            'symbol': symbols[0],
            'date_from': date_from,
            'date_to': date_to,
            'initial_balance': initial_balance,
            'risk_percent': risk_percent,
            'fixed_lot': fixed_lot,
            'risk_mode': risk_mode,
            'trailing_methods': trailing_methods,
            'ict_params': ict_params,
            'use_dynamic_rrr': use_dynamic_rrr,
            'trade_all_tfs': trade_all_tfs,
            'use_ultra_low_tf': use_ultra_low_tf,
            'spread_cost': spread_cost,
            'slippage_pts': slippage_pts,
            'commission': commission,
            'session_filter': session_filter,
            'session_start': session_start,
            'session_end': session_end,
            'bypass_htf_conf': bypass_htf_conf,
            'trail_pct': trail_pct,
            'require_bos_fvg': require_bos_fvg,
            'slippage_recovery': slippage_recovery,
            'anti_gap_enabled': anti_gap_enabled,
            'anti_gap_mult': anti_gap_mult,
            'min_fvg_size': min_fvg_size,
            'min_conf': min_conf,
            'smt_correlated_pair': self.opt_smt_pair.get() if hasattr(self, 'opt_smt_pair') else "AUTO",
            'news_filter_enabled': self.opt_news_enabled.get() if hasattr(self, 'opt_news_enabled') else False,
            'news_filter_buffer': int(self.opt_news_buffer.get()) if hasattr(self, 'opt_news_buffer') and self.opt_news_buffer.get().isdigit() else 30,
            'pro_flags': {
                'dol_tp': bool(getattr(self, "opt_pro_dol_tp").get()) if hasattr(self, "opt_pro_dol_tp") else False,
                'killzone': bool(getattr(self, "opt_pro_killzone").get()) if hasattr(self, "opt_pro_killzone") else False,
                'htf_poi': bool(getattr(self, "opt_pro_htf_poi").get()) if hasattr(self, "opt_pro_htf_poi") else False,
                'mandatory': bool(getattr(self, "opt_pro_mandatory").get()) if hasattr(self, "opt_pro_mandatory") else False,
                'regime': bool(getattr(self, "opt_pro_regime").get()) if hasattr(self, "opt_pro_regime") else False,
                'ml_sizing': bool(getattr(self, "opt_pro_ml_sizing").get()) if hasattr(self, "opt_pro_ml_sizing") else False,
                'ml_rank': bool(getattr(self, "opt_pro_ml_rank").get()) if hasattr(self, "opt_pro_ml_rank") else False,
                'multi_tf_conf': bool(getattr(self, "opt_pro_multi_tf_conf").get()) if hasattr(self, "opt_pro_multi_tf_conf") else False,
                'multi_tf_gate': bool(getattr(self, "opt_pro_multi_tf_gate").get()) if hasattr(self, "opt_pro_multi_tf_gate") else False,
                'regime_adaptive': bool(getattr(self, "opt_pro_regime_adaptive").get()) if hasattr(self, "opt_pro_regime_adaptive") else False,
            },
            'ml_filter': bool(getattr(self, "opt_ml_filter").get()) if hasattr(self, "opt_ml_filter") else False,
            'ml_min_confidence': float(getattr(self, "opt_ml_min_confidence").get()) / 100.0 if hasattr(self, "opt_ml_min_confidence") and getattr(self, "opt_ml_min_confidence").get() else 0.60,
        }
        
        self.grid_opt_state = {
            'is_running': True,
            'is_paused': False,
            'current_index': 0,
            'grid': grid,
            'opt_results': [],
            'phase': 1,
            'sweep_combined_profitable': sweep_combined_profitable,
            'phase2_grid': [],
            'phase2_index': 0,
            'fixed_params': fixed_params,
            'n_cores': n_cores,
            'elapsed_time': 0.0,
            'profitable_ind': [],
            'sweep_ranges': {
                'sl_modes': sl_modes,
                'trail_types': trail_types,
                'ote_vals': ote_vals,
                'htf_vals': htf_vals,
                'disp_vals': disp_vals,
                'disc_vals': disc_vals,
                'sweep_vals': sweep_vals,
                'vp_vals': vp_vals,
                'smt_vals': smt_vals,
                'concurrencies': concurrencies,
                'rrr_values': rrr_values,
                'daily_loss_values': daily_loss_values,
                'sb_htf_vals': sb_htf_vals
            }
        }
        
        threading.Thread(
            target=self.run_grid_opt_thread,
            daemon=True
        ).start()

    def run_grid_opt_thread(self):
        import sys
        import time
        import itertools
        import traceback
        import utils
        import simulation
        import indicators
        import backtester
        import multiprocessing as mp
        import MetaTrader5 as mt5
        from opt_worker import init_worker, run_combo
        
        state = self.grid_opt_state
        fp = state['fixed_params']
        fp['offline_backtesting'] = config.OFFLINE_BACKTESTING
        symbol = fp['symbol']
        
        if isinstance(fp['date_from'], str):
            if " " in fp['date_from']:
                fp['date_from'] = datetime.datetime.strptime(fp['date_from'], "%Y-%m-%d %H:%M:%S")
            else:
                fp['date_from'] = datetime.datetime.strptime(fp['date_from'], "%Y-%m-%d")
        if isinstance(fp['date_to'], str):
            if " " in fp['date_to']:
                fp['date_to'] = datetime.datetime.strptime(fp['date_to'], "%Y-%m-%d %H:%M:%S")
            else:
                fp['date_to'] = datetime.datetime.strptime(fp['date_to'], "%Y-%m-%d")
                
        date_from = fp['date_from']
        date_to = fp['date_to']
        
        self.ui_update_queue.put(lambda: self.opt_progress_label.config(text="Pre-loading historical data..."))
        m15_context = utils.get_data(symbol, mt5.TIMEFRAME_M15, 200, live=False, date_to=date_from)
        m15_df = utils.get_data_by_date(symbol, mt5.TIMEFRAME_M15, date_from, date_to)
        m5_context = utils.get_data(symbol, mt5.TIMEFRAME_M5, 200, live=False, date_to=date_from)
        m5_df = utils.get_data_by_date(symbol, mt5.TIMEFRAME_M5, date_from, date_to)
        h1_context = utils.get_data(symbol, mt5.TIMEFRAME_H1, 200, live=False, date_to=date_from)
        h1_df = utils.get_data_by_date(symbol, mt5.TIMEFRAME_H1, date_from, date_to)
        h4_df = utils.get_data_by_date(symbol, mt5.TIMEFRAME_H4, date_from - datetime.timedelta(days=200), date_to)
        d1_df = utils.get_data_by_date(symbol, mt5.TIMEFRAME_D1, date_from - datetime.timedelta(days=200), date_to)
        w1_df = utils.get_data_by_date(symbol, mt5.TIMEFRAME_W1, date_from - datetime.timedelta(days=365), date_to)
        
        if True in state['sweep_ranges'].get('smt_vals', [False]):
            smt_pair_str = fp.get('smt_correlated_pair', 'AUTO')
            if smt_pair_str.upper() == "AUTO":
                smt_pairs = ["DXY"]
                if "EURUSD" in symbol or "GBPUSD" in symbol or "AUDUSD" in symbol or "NZDUSD" in symbol:
                    smt_pairs = ["DXY", "USDCAD", "USDCHF"] if "GBP" not in symbol else ["DXY", "EURUSD"]
                elif "USD" in symbol:
                    smt_pairs = ["EURUSD", "GBPUSD"]
            else:
                smt_pairs = [p.strip() for p in smt_pair_str.split(',') if p.strip()]
                
            smt_date_from = date_from - datetime.timedelta(days=20)
            import re
            suffix_match = re.search(r'([a-z.]+)$', symbol)
            suffix = suffix_match.group(1) if suffix_match else ""
            for p in smt_pairs:
                pair_to_fetch = p + suffix if suffix else p
                c_df = utils.get_data_by_date(pair_to_fetch, mt5.TIMEFRAME_M15, smt_date_from, date_to)
                if c_df is None or c_df.empty:
                    pair_to_fetch = p
                utils.get_data_by_date(pair_to_fetch, mt5.TIMEFRAME_M15, smt_date_from, date_to)
                utils.get_data_by_date(pair_to_fetch, mt5.TIMEFRAME_M5, smt_date_from, date_to)
                utils.get_data_by_date(pair_to_fetch, mt5.TIMEFRAME_H1, smt_date_from, date_to)

        # Pre-load M1 data for accurate trade fill simulation (matches standard backtest)
        self.ui_update_queue.put(lambda: self.opt_progress_label.config(text="Pre-loading M1 data for simulation precision..."))
        m1_df = utils.get_data_by_date(symbol, mt5.TIMEFRAME_M1, date_from - datetime.timedelta(days=1), date_to + datetime.timedelta(days=30))
        # Pre-load H4/D1/W1 context data for HTF trend lookups
        h4_context = utils.get_data(symbol, mt5.TIMEFRAME_H4, 200, live=False, date_to=date_from)
        d1_context = utils.get_data(symbol, mt5.TIMEFRAME_D1, 200, live=False, date_to=date_from)
        w1_context = utils.get_data(symbol, mt5.TIMEFRAME_W1, 200, live=False, date_to=date_from)
        
        if m15_df is None or m15_df.empty or h1_df is None or h1_df.empty:
            self.ui_update_queue.put(lambda: messagebox.showerror("Optimization Error", "Failed to download historical rates from MT5."))
            self.ui_update_queue.put(lambda: self.opt_progress_label.config(text="Failed!"))
            self.ui_update_queue.put(lambda: self.opt_run_btn.config(state=tk.NORMAL))
            self.ui_update_queue.put(lambda: self.opt_stop_btn.config(state=tk.DISABLED))
            self.ui_update_queue.put(lambda: self.opt_pause_btn.config(state=tk.DISABLED))
            self.ui_update_queue.put(lambda: self.opt_resume_btn.config(state=tk.DISABLED))
            self.ui_update_queue.put(lambda: self.opt_save_btn.config(state=tk.DISABLED))
            self.ui_update_queue.put(lambda: self.opt_load_btn.config(state=tk.NORMAL))
            state['is_running'] = False
            return

        cached_data = {
            (symbol, mt5.TIMEFRAME_M15, 'context'): m15_context,
            (symbol, mt5.TIMEFRAME_M15): m15_df,
            (symbol, mt5.TIMEFRAME_M5, 'context'): m5_context,
            (symbol, mt5.TIMEFRAME_M5): m5_df,
            (symbol, mt5.TIMEFRAME_H1, 'context'): h1_context,
            (symbol, mt5.TIMEFRAME_H1): h1_df,
            (symbol, mt5.TIMEFRAME_H4): h4_df,
            (symbol, mt5.TIMEFRAME_H4, 'context'): h4_context,
            (symbol, mt5.TIMEFRAME_D1): d1_df,
            (symbol, mt5.TIMEFRAME_D1, 'context'): d1_context,
            (symbol, mt5.TIMEFRAME_W1): w1_df,
            (symbol, mt5.TIMEFRAME_W1, 'context'): w1_context,
            (symbol, mt5.TIMEFRAME_M1): m1_df.sort_index() if m1_df is not None and not m1_df.empty else m1_df,
        }
        
        n_cores = state['n_cores']
        start_time = time.time() - state.get('elapsed_time', 0.0)
        
        orig_max_concurr = config.MAX_CONCURRENT_TRADES
        orig_min_fvg_size = config.MIN_FVG_SIZE_SPREADS
        orig_daily_loss = config.DAILY_LOSS_LIMIT_PCT
        orig_min_conf = config.MIN_CONFLUENCE_SCORE
        
        # Inject mock symbol info to avoid MT5 concurrent initialization crashes in workers
        try:
            import MetaTrader5 as mt5
            si = mt5.symbol_info(symbol)
            if si:
                fp['mock_si'] = {
                    'spread': getattr(si, 'spread', 0),
                    'point': getattr(si, 'point', 0.00001),
                    'trade_stops_level': getattr(si, 'trade_stops_level', 0),
                    'trade_calc_mode': getattr(si, 'trade_calc_mode', 0),
                    'trade_contract_size': getattr(si, 'trade_contract_size', 100000),
                    'trade_tick_value': getattr(si, 'trade_tick_value', 0),
                    'trade_tick_size': getattr(si, 'trade_tick_size', getattr(si, 'point', 0.00001)),
                    'volume_step': getattr(si, 'volume_step', 0.01),
                    'volume_min': getattr(si, 'volume_min', 0.01),
                    'volume_max': getattr(si, 'volume_max', 1000.0)
                }
        except Exception:
            pass

        # Save cached_data to disk to prevent multiprocessing Pickling/IPC limits (STATUS_BREAKPOINT)
        import pickle
        import os
        cache_path = os.path.join(os.getcwd(), 'opt_cache_data.pkl')
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            init_cached_data = cache_path
        except Exception as e:
            print("Failed to pickle cache, using direct dict:", e)
            init_cached_data = cached_data
            
        try:
            mp_ctx = mp.get_context('spawn')
            with mp_ctx.Pool(n_cores, initializer=init_worker, initargs=(init_cached_data, fp), maxtasksperchild=None) as pool:
                # 1. Phase 1
                if state['phase'] == 1:
                    grid = state['grid']
                    total_runs = len(grid)
                    batch_size = max(n_cores * 2, 50)
                    
                    while state['current_index'] < total_runs and not config.bt_stop_event.is_set():
                        was_paused = False
                        while state.get('is_paused', False) and not config.bt_stop_event.is_set():
                            was_paused = True
                            self.ui_update_queue.put(lambda: self.opt_progress_label.config(text="Paused"))
                            self.ui_update_queue.put(lambda: self.opt_save_btn.config(state=tk.NORMAL))
                            time.sleep(0.5)
                            
                        if was_paused:
                            start_time = time.time() - state.get('elapsed_time', 0.0)
                            
                        if config.bt_stop_event.is_set():
                            break
                            
                        idx_start = state['current_index']
                        batch = grid[idx_start : idx_start + batch_size]
                        
                        self.ui_update_queue.put(lambda: self.opt_pause_btn.config(state=tk.NORMAL))
                        self.ui_update_queue.put(lambda: self.opt_save_btn.config(state=tk.DISABLED))
                        
                        last_ui_update = 0.0
                        for batch_idx, res in enumerate(pool.imap_unordered(run_combo, batch), 1):
                            if config.bt_stop_event.is_set():
                                pool.terminate()
                                break
                                
                            if res is not None:
                                state['opt_results'].append(res)
                                # Only enqueue treeview row live if profitable to avoid Tkinter Treeview lockup
                                if res.get('roi', 0.0) > 0 and not res.get('is_waste', False):
                                    self.ui_update_queue.put(lambda r=res: self.add_opt_result_row(r))
                                
                            state['current_index'] += 1
                            curr_idx = state['current_index']
                            
                            now = time.time()
                            if now - last_ui_update >= 0.15 or curr_idx == total_runs:
                                last_ui_update = now
                                pct = curr_idx / total_runs * 100
                                state['elapsed_time'] = now - start_time
                                elapsed_str = f"{int(state['elapsed_time']//60)}m {int(state['elapsed_time']%60)}s"
                                roi_val = res['roi'] if res else 0.0
                                dd_val = res['dd'] if res else 0.1
                                
                                if res and res.get('is_waste'):
                                    lbl_text = f"Phase 1: combo {curr_idx}/{total_runs} (Waste of time aborted)"
                                else:
                                    lbl_text = f"Phase 1: combo {curr_idx}/{total_runs} (Calmar={roi_val/max(0.1, dd_val):.2f})"
                                
                                self.ui_update_queue.put(lambda p=pct, l=lbl_text, e=elapsed_str: self.update_opt_ui(p, l, e))
                            
                    if not config.bt_stop_event.is_set():
                        state['phase'] = 2
                        state['phase2_index'] = 0
                
                # 2. Phase 2
                if state['phase'] == 2 and state.get('sweep_combined_profitable', False) and not config.bt_stop_event.is_set():
                    if not state.get('phase2_grid'):
                        profitable_ind = set()
                        for r in state['opt_results']:
                            if len(r['methods']) == 1 and r['roi'] > 0 and not r.get('is_waste', False):
                                profitable_ind.add(r['methods'][0])
                        profitable_ind = list(profitable_ind)
                        state['profitable_ind'] = profitable_ind
                        
                        if len(profitable_ind) >= 2:
                            sr = state['sweep_ranges']
                            raw_phase2 = list(itertools.product(
                                [profitable_ind], sr['sl_modes'], sr['trail_types'], sr['ote_vals'], sr['htf_vals'],
                                sr['disp_vals'], sr['disc_vals'], sr['sweep_vals'], sr['vp_vals'], sr['smt_vals'], sr['concurrencies'], sr['rrr_values'], sr['daily_loss_values'], sr['sb_htf_vals']
                            ))
                            comb_grid = []
                            for combo in raw_phase2:
                                (methods, sl_mode, trail_type, use_ote, use_htf, disp_only, disc_prem,
                                 sweep_only, vp_val, smt_val, concurr, rrr, dl_val, sb_htf_bias) = combo
                                comb_grid.append({
                                    'ict_method': methods,
                                    'fvg_sl_mode': sl_mode,
                                    'trail_type': trail_type,
                                    'use_ote_filter': use_ote,
                                    'use_htf_filter': use_htf,
                                    'fvg_displacement_only': disp_only,
                                    'fvg_discount_premium_only': disc_prem,
                                    'fvg_recent_sweep_only': sweep_only,
                                    'use_volume_profile': vp_val,
                                    'use_smt_divergence': smt_val,
                                    'max_concurrent_trades': concurr,
                                    'min_rrr': rrr,
                                    'daily_loss_limit': dl_val,
                                    'sb_require_htf_bias': sb_htf_bias
                                })
                            state['phase2_grid'] = comb_grid
                            
                    phase2_grid = state['phase2_grid']
                    total_comb_runs = len(phase2_grid)
                    batch_size = max(n_cores * 2, 50)
                    
                    while state['phase2_index'] < total_comb_runs and not config.bt_stop_event.is_set():
                        was_paused = False
                        while state.get('is_paused', False) and not config.bt_stop_event.is_set():
                            was_paused = True
                            self.ui_update_queue.put(lambda: self.opt_progress_label.config(text="Paused"))
                            self.ui_update_queue.put(lambda: self.opt_save_btn.config(state=tk.NORMAL))
                            time.sleep(0.5)
                            
                        if was_paused:
                            start_time = time.time() - state.get('elapsed_time', 0.0)
                            
                        if config.bt_stop_event.is_set():
                            break
                            
                        idx_start = state['phase2_index']
                        batch = phase2_grid[idx_start : idx_start + batch_size]
                        
                        self.ui_update_queue.put(lambda: self.opt_pause_btn.config(state=tk.NORMAL))
                        self.ui_update_queue.put(lambda: self.opt_save_btn.config(state=tk.DISABLED))
                        
                        last_p2_ui_update = 0.0
                        for batch_idx, res in enumerate(pool.imap_unordered(run_combo, batch), 1):
                            if config.bt_stop_event.is_set():
                                pool.terminate()
                                break
                                
                            if res is not None:
                                state['opt_results'].append(res)
                                if res.get('roi', 0.0) > 0 and not res.get('is_waste', False):
                                    self.ui_update_queue.put(lambda r=res: self.add_opt_result_row(r))
                                
                            state['phase2_index'] += 1
                            curr_idx = state['phase2_index']
                            
                            now = time.time()
                            if now - last_p2_ui_update >= 0.15 or curr_idx == total_comb_runs:
                                last_p2_ui_update = now
                                pct = curr_idx / total_comb_runs * 100
                                state['elapsed_time'] = now - start_time
                                elapsed_str = f"{int(state['elapsed_time']//60)}m {int(state['elapsed_time']%60)}s"
                                
                                methods_str = '+'.join([m.split()[0] for m in state['profitable_ind']])
                                lbl_text = f"Phase 2: Combined {curr_idx}/{total_comb_runs} ({methods_str})"
                                self.ui_update_queue.put(lambda p=pct, l=lbl_text, e=elapsed_str: self.update_opt_ui(p, l, e))
                            
            state['opt_results'].sort(key=lambda x: x.get('calmar', -999.0), reverse=True)
            self.ui_update_queue.put(lambda: self.display_sorted_opt_results(state['opt_results']))
            
            # Clean up pickle file
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
            
            state['elapsed_time'] = time.time() - start_time
            elapsed_str = f"{int(state['elapsed_time']//60)}m {int(state['elapsed_time']%60)}s"
            
            if config.bt_stop_event.is_set():
                self.ui_update_queue.put(lambda: self.opt_progress_label.config(text="Optimization stopped!"))
            else:
                self.ui_update_queue.put(lambda: self.opt_progress_label.config(text="Optimization completed!"))
            self.ui_update_queue.put(lambda e=elapsed_str: self.opt_elapsed_label.config(text=f"Total: {e}"))
            
        except Exception as main_ex:
            logger.error("Optimization thread main loop crashed: %s\n%s", main_ex, traceback.format_exc())
            self.ui_update_queue.put(lambda msg=str(main_ex): messagebox.showerror("Optimization Error", f"Crashed: {msg}"))
            self.ui_update_queue.put(lambda: self.opt_progress_label.config(text="Error!"))
        finally:
            config.MAX_CONCURRENT_TRADES = orig_max_concurr
            config.MIN_FVG_SIZE_SPREADS = orig_min_fvg_size
            config.DAILY_LOSS_LIMIT_PCT = orig_daily_loss
            config.MIN_CONFLUENCE_SCORE = orig_min_conf
            
            self.ui_update_queue.put(lambda: self.opt_run_btn.config(state=tk.NORMAL))
            self.ui_update_queue.put(lambda: self.opt_stop_btn.config(state=tk.DISABLED))
            self.ui_update_queue.put(lambda: self.opt_pause_btn.config(state=tk.DISABLED))
            self.ui_update_queue.put(lambda: self.opt_resume_btn.config(state=tk.DISABLED))
            self.ui_update_queue.put(lambda: self.opt_save_btn.config(state=tk.NORMAL))
            self.ui_update_queue.put(lambda: self.opt_load_btn.config(state=tk.NORMAL))
            state['is_running'] = False
            # Don't shut down MT5 here — doing so forces subsequent single backtests
            # to re-download historical data from MT5 (which may return partial results
            # on the first query after reconnect), causing non-deterministic results.

    def copy_settings_to_live(self):
        """Copies all backtest settings to the live trading controls tab."""
        # 0. Flush any unsaved active method overrides to memory, and temporarily restore UI to base settings
        active_bt_symbol = None
        active_bt_method = None
        if self.bt_use_overrides_var.get() and self.bt_currently_selected_override_symbol:
            active_bt_symbol = self.bt_currently_selected_override_symbol
            if self.bt_use_method_overrides_var.get() and self.bt_currently_selected_override_method:
                active_bt_method = self.bt_currently_selected_override_method
                self.save_bt_method_settings(active_bt_symbol, active_bt_method)
            else:
                self.bt_symbol_overrides[active_bt_symbol] = self.get_current_ui_backtest_settings()
            
            # Temporarily revert UI to base settings to perform a clean copy
            if self.bt_base_cached_settings:
                self.apply_backtest_settings_to_ui(self.bt_base_cached_settings)

        # 1. Symbols
        self.live_symbols_entry.delete(0, tk.END)
        self.live_symbols_entry.insert(0, self.backtest_symbol_entry.get())
        
        # 2. Risk & Lots
        self.risk_entry.delete(0, tk.END)
        self.risk_entry.insert(0, self.backtest_risk_entry.get())
        self.fixed_lot_entry.delete(0, tk.END)
        self.fixed_lot_entry.insert(0, self.backtest_fixed_lot_entry.get())
        self.risk_mode_var.set("Fixed" if self.backtest_use_fixed_lot.get() else "Risk")
        
        # 3. RRR & Dynamic RRR
        self.min_rrr_entry.delete(0, tk.END)
        self.min_rrr_entry.insert(0, self.backtest_min_rrr_entry.get())
        self.use_dynamic_rrr_live.set(self.use_dynamic_rrr_backtest.get())
        
        # 4. Multi-TF & Ultra Low TF
        self.trade_all_tfs_live.set(self.trade_all_tfs_backtest.get())
        self.use_ultra_low_tf_live.set(self.use_ultra_low_tf_backtest.get())
        
        # 5. Session Filter
        self.session_filter_live.set(self.bt_session_filter.get())
        self.session_start_live.delete(0, tk.END)
        self.session_start_live.insert(0, self.bt_session_start.get())
        self.session_end_live.delete(0, tk.END)
        self.session_end_live.insert(0, self.bt_session_end.get())
        
        # 6. HTF and OTE Filters
        self.htf_filter_live.set(self.htf_filter_backtest.get())
        self.bypass_htf_live.set(self.bypass_htf_backtest.get())
        self.ote_filter_live.set(self.ote_filter_backtest.get())
        
        # 7. Risk Management (Daily Loss, Max Concurrent, Min FVG, Min Conf)
        # Note: Do not copy daily_loss from backtest to live, because bt_daily_loss is a per-strategy limit
        # while live_daily_loss is an account-wide global drawdown threshold. Overwriting it can cause the global circuit breaker to halt scanning.
        # self.live_daily_loss.delete(0, tk.END)
        # self.live_daily_loss.insert(0, self.bt_daily_loss.get())
        self.live_max_concurr.delete(0, tk.END)
        self.live_max_concurr.insert(0, self.bt_max_concurr.get())
        self.live_min_fvg.delete(0, tk.END)
        self.live_min_fvg.insert(0, self.bt_min_fvg.get())
        self.live_min_conf.delete(0, tk.END)
        self.live_min_conf.insert(0, self.bt_min_conf.get())
        
        # 8. Trailing stop settings
        self.trail_type_live.set(self.trail_type_bt.get())
        self.trail_pct_live.delete(0, tk.END)
        self.trail_pct_live.insert(0, self.trail_pct_bt.get())
        
        # 9. Anti-Gap Settings
        self.anti_gap_enabled_live.set(self.bt_anti_gap.get())
        self.anti_gap_atr_mult_live.delete(0, tk.END)
        self.anti_gap_atr_mult_live.insert(0, self.bt_anti_gap_mult.get())
        
        # 10. Slippage Recovery
        self.slippage_recovery_live.set(self.bt_slippage_recovery_bt.get())
        config.slippage_recovery_tracker['enabled'] = self.bt_slippage_recovery_bt.get()
        
        # 11. Methods & Trail methods selection listboxes
        self.live_methods_listbox.selection_clear(0, tk.END)
        for idx in self.bt_methods_listbox.curselection():
            self.live_methods_listbox.selection_set(idx)
            
        self.live_trail_methods_listbox.selection_clear(0, tk.END)
        for idx in self.bt_trail_methods_listbox.curselection():
            self.live_trail_methods_listbox.selection_set(idx)
            
        # 12. FVG Advanced Filters
        self.live_fvg_sl_mode.set(self.bt_fvg_sl_mode.get())
        self.live_require_bos_fvg.set(self.bt_require_bos_fvg.get())
        self.live_fvg_displacement_only.set(self.bt_fvg_displacement_only.get())
        self.live_fvg_discount_premium_only.set(self.bt_fvg_discount_premium_only.get())
        self.live_fvg_recent_sweep_only.set(self.bt_fvg_recent_sweep_only.get())
        self.live_sb_require_htf_bias.set(self.bt_sb_require_htf_bias.get())
        
        # Manually invoke live methods select listbox callback to configure widgets state
        selected = [self.live_methods_listbox.get(i) for i in self.live_methods_listbox.curselection()]
        fvg_state = tk.NORMAL if "FVG Return" in selected else tk.DISABLED
        for child in self.live_fvg_sl_frame.winfo_children():
            child.configure(state=fvg_state)
        self.live_require_bos_cb.config(state=fvg_state)
        for child in self.live_fvg_filters_frame.winfo_children():
            child.configure(state=fvg_state)
            
        # 13. Method and Symbol Overrides
        import copy
        self.live_symbol_overrides = copy.deepcopy(self.bt_symbol_overrides)
        self.live_use_overrides_var.set(self.bt_use_overrides_var.get())
        self.live_use_method_overrides_var.set(self.bt_use_method_overrides_var.get())
        
        # 14. Restore Backtest UI to its active override state
        if active_bt_symbol:
            if active_bt_method:
                self.apply_backtest_settings_to_ui(self.get_merged_bt_settings(active_bt_symbol, active_bt_method))
            else:
                self.apply_backtest_settings_to_ui(self.bt_symbol_overrides[active_bt_symbol])
                
        # Trigger Live UI updates after restore
        self.on_toggle_live_overrides()
        
        # Mirror active symbol dropdown selection
        if active_bt_symbol and active_bt_symbol in self.live_override_symbol_dropdown["values"]:
            self.live_override_symbol_dropdown.set(active_bt_symbol)
            self.live_currently_selected_override_symbol = active_bt_symbol
        else:
            self.live_currently_selected_override_symbol = None
            self.live_override_symbol_dropdown.set("")
            
        self.on_toggle_live_method_overrides()
        
        # Mirror active method dropdown selection
        if active_bt_symbol and active_bt_method and self.live_use_method_overrides_var.get():
            if active_bt_method in self.live_override_method_dropdown["values"]:
                self.live_override_method_dropdown.set(active_bt_method)
                self.live_currently_selected_override_method = active_bt_method
        elif not self.live_use_method_overrides_var.get():
            self.live_currently_selected_override_method = None
            self.live_override_method_dropdown.set("")
            
        # Apply the final active override to the live UI visually
        if self.live_currently_selected_override_symbol:
            if self.live_use_method_overrides_var.get() and self.live_currently_selected_override_method:
                self.apply_live_settings_to_ui(self.get_merged_live_settings(self.live_currently_selected_override_symbol, self.live_currently_selected_override_method))
            else:
                self.apply_live_settings_to_ui(self.live_symbol_overrides[self.live_currently_selected_override_symbol])
            
        messagebox.showinfo("Settings Copied", "All backtesting settings have been successfully copied to the Live Trading tab!")

    def save_backtest_report_to_file(self, all_results, symbols, initial_balance, date_from, date_to):
        """Automatically saves the entire backtesting session details, comparison summary,
        and all trade logs to a timestamped text file in the workspace directory.
        """
        import os
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_report_{timestamp}.txt"
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BackTestReports")
        os.makedirs(report_dir, exist_ok=True)
        filepath = os.path.join(report_dir, filename)
        
        # Save raw trades for ML analysis
        raw_filename = f"raw_backtest_{timestamp}.pkl"
        raw_filepath = os.path.join(report_dir, raw_filename)
        try:
            import pickle
            with open(raw_filepath, 'wb') as rf:
                pickle.dump({'results': all_results, 'symbols': symbols}, rf)
        except Exception as e:
            logger.error("Failed to save raw backtest data: %s", e)
        
        methods_str = ", ".join([self.bt_methods_listbox.get(i) for i in self.bt_methods_listbox.curselection()])
        trail_methods_str = ", ".join([self.bt_trail_methods_listbox.get(i) for i in self.bt_trail_methods_listbox.curselection()])
        
        report = []
        report.append("=" * 80)
        report.append(f"  ICT STRATEGY BACKTEST REPORT - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        report.append("── BACKTEST SETTINGS & CONFIGURATION ──────────────────────────")
        report.append(f"  Symbols:            {', '.join(symbols)}")
        report.append(f"  Date Period:        {date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}")
        report.append(f"  Initial Balance:    ${initial_balance:,.2f}")
        report.append(f"  Risk Mode:          {'Fixed Lot' if self.backtest_use_fixed_lot.get() else 'Risk %'}")
        report.append(f"  Risk Size:          {self.backtest_fixed_lot_entry.get() + ' Lots' if self.backtest_use_fixed_lot.get() else self.backtest_risk_entry.get() + '%'}")
        report.append(f"  Min/Max/Step RRR:   {self.backtest_min_rrr_entry.get()} / {self.backtest_max_rrr_entry.get()} / {self.backtest_incr_rrr_entry.get()}")
        report.append(f"  Dynamic RRR:        {'Enabled' if self.use_dynamic_rrr_backtest.get() else 'Disabled'}")
        report.append(f"  Multi-Timeframe:    {'Enabled' if self.trade_all_tfs_backtest.get() else 'Disabled'}")
        report.append(f"  Ultra Low TF:       {'Enabled' if self.use_ultra_low_tf_backtest.get() else 'Disabled'}")
        report.append(f"  Spread Cost:        {self.bt_spread_cost.get()}")
        report.append(f"  Slippage (pts):     {self.bt_slippage.get()}")
        report.append(f"  Commission/Lot:     ${self.bt_commission.get()}")
        report.append(f"  Session Filter:     {'Enabled' if self.bt_session_filter.get() else 'Disabled'} ({self.bt_session_start.get()}:00 - {self.bt_session_end.get()}:00)")
        report.append(f"  HTF Filter:         {'Enabled' if self.htf_filter_backtest.get() else 'Disabled'} (Bypass if Conf >= 2: {'Yes' if self.bypass_htf_backtest.get() else 'No'})")
        report.append(f"  OTE Filter:         {'Enabled' if self.ote_filter_backtest.get() else 'Disabled'}")
        report.append(f"  Daily Loss Limit:   {self.bt_daily_loss.get()}%")
        report.append(f"  Max Concurrent:     {self.bt_max_concurr.get()}")
        report.append(f"  Min FVG Size:       {self.bt_min_fvg.get()} spreads")
        report.append(f"  Min Confluence:     {self.bt_min_conf.get()}")
        report.append(f"  Anti-Gap SL:        {'Enabled' if self.bt_anti_gap.get() else 'Disabled'} (x{self.bt_anti_gap_mult.get()})")
        report.append(f"  Slippage Recovery:  {'Enabled' if self.bt_slippage_recovery_bt.get() else 'Disabled'}")
        report.append(f"  ICT Methods:        {methods_str}")
        report.append(f"  FVG SL Mode:        {self.bt_fvg_sl_mode.get()}")
        report.append(f"  Require BoS:        {'Yes' if self.bt_require_bos_fvg.get() else 'No'}")
        report.append(f"  Displacement Fltr:  {'Enabled' if self.bt_fvg_displacement_only.get() else 'Disabled'}")
        report.append(f"  Discount/Premium:   {'Enabled' if self.bt_fvg_discount_premium_only.get() else 'Disabled'}")
        report.append(f"  Recent Sweep Only:  {'Enabled' if self.bt_fvg_recent_sweep_only.get() else 'Disabled'}")
        trail_label = self.trail_type_bt.get() if trail_methods_str else "None (Disabled)"
        report.append(f"  Trailing Stop:      {trail_label} Stop (methods: {trail_methods_str}, pct: {self.trail_pct_bt.get()}%)")
        report.append("")
        
        # Comparison summary if multiple runs
        is_opt = isinstance(all_results[0][0], str)
        if len(all_results) > 1:
            report.append("── COMPARISON SUMMARY ─────────────────────────────────────────")
            hdr_lbl = "Configuration" if is_opt else "RRR"
            max_label_len = max(len(str(rrr)) if is_opt else len(f"{rrr:.2f}") for rrr, _, _ in all_results)
            col_width = max(len(hdr_lbl), max_label_len)
            comp_header = f"{hdr_lbl:<{col_width}} | {'Trades':>6} | {'WR%':>6} | {'Net PnL':>12} | {'Max DD':>10} | {'Max DD%':>8} | {'Sharpe':>7} | {'PF':>6}"
            report.append(comp_header)
            report.append("-" * len(comp_header))
            
            for rrr, trades, metrics in all_results:
                pf = metrics.get('profit_factor', 0)
                pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
                wr = metrics.get('win_rate', 0)
                pnl = metrics.get('net_profit', 0)
                dd = metrics.get('max_drawdown', 0)
                dd_pct = metrics.get('max_drawdown_pct', 0)
                sh = metrics.get('sharpe_ratio', 0)
                tc = metrics.get('total_trades', 0)
                
                label_str = f"{rrr:<{col_width}}" if is_opt else f"{rrr:>{col_width}.2f}"
                line = f"{label_str} | {tc:>6} | {wr:>6.1f} | ${pnl:>11,.2f} | ${dd:>9,.2f} | {dd_pct:>7.2f}% | {sh:>7.2f} | {pf_str:>6}"
                report.append(line)
            report.append("")
            
        # Detailed metrics and trade log for each run
        for idx, (rrr, trades, metrics) in enumerate(all_results):
            run_title = f"RUN #{idx+1} DETAILS ({rrr})" if is_opt else f"RUN DETAILS FOR RRR {rrr:.2f}"
            report.append("=" * 80)
            report.append(f"  {run_title}")
            report.append("=" * 80)
            
            pf = metrics.get('profit_factor', 0)
            pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
            avg_w = metrics.get('avg_win', 0)
            avg_l = abs(metrics.get('avg_loss', 0))
            real_rrr = (avg_w / avg_l) if avg_l > 0 else 0
            real_rrr_str = f"1:{real_rrr:.2f}" if avg_l > 0 else "N/A"
            
            summary_part = f"""
── PERFORMANCE METRICS ──────────────────────
  Initial Balance:    ${initial_balance:,.2f}
  Final Balance:      ${metrics['final_balance']:,.2f}
  Net Profit:         ${metrics.get('net_profit', 0):,.2f}
  ROI:                {metrics.get('roi_pct', 0):.2f}%
  Total Trades:       {metrics['total_trades']}
  Win Rate (SL/TP):   {metrics['win_rate']:.2f}%   ({metrics.get('sltp_wins',0)}/{metrics.get('sltp_total',0)} SL/TP)
  Overall WR (all):   {metrics.get('overall_win_rate', 0):.2f}%   ({metrics['win_count']}/{metrics['total_trades']} wins)
  Profit Factor:      {pf_str}
  Expectancy/Trade:   ${metrics.get('expectancy', 0):,.2f}
  Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):.2f}
  Maximal Drawdown:   ${metrics['max_drawdown']:,.2f} ({metrics.get('max_drawdown_pct', 0):.2f}%)
  Largest Win/Loss:   ${metrics.get('highest_win', 0):,.2f} / ${metrics.get('highest_loss', 0):,.2f}
  Avg Win/Loss:       ${metrics.get('avg_win', 0):,.2f} / ${metrics.get('avg_loss', 0):,.2f}
  Realized RRR:       {real_rrr_str}
  Total Commission:   ${metrics.get('total_commission', 0):,.2f}
  Avg Duration:       {metrics.get('avg_duration_hrs', 0):.1f} hours
"""
            report.append(summary_part)
            
            # Per symbol stats
            sym_stats = metrics.get('symbol_stats', {})
            if sym_stats:
                report.append("  Per Symbol:")
                for sym, st in sorted(sym_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
                    wr = (st['wins'] / st['trades'] * 100) if st['trades'] > 0 else 0
                    report.append(f"    {sym:<10} | Trades: {st['trades']:>4} | Wins: {st['wins']:>3} | WR: {wr:>5.1f}% | PnL: ${st['pnl']:>12,.2f}")
                report.append("")
                
            # Per TF stats
            tf_stats = metrics.get('tf_stats', {})
            if tf_stats:
                report.append("  Per Timeframe:")
                for tf, st in sorted(tf_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
                    wr = (st['wins'] / st['trades'] * 100) if st['trades'] > 0 else 0
                    report.append(f"    {tf:<6} | Trades: {st['trades']:>4} | Wins: {st['wins']:>3} | WR: {wr:>5.1f}% | PnL: ${st['pnl']:>12,.2f}")
                report.append("")
                
            # Per method stats
            mt_stats = metrics.get('method_stats', {})
            if mt_stats:
                report.append("  Per Method:")
                for m, st in sorted(mt_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
                    wr = (st['wins'] / st['trades'] * 100) if st['trades'] > 0 else 0
                    report.append(f"    {m:<15} | Trades: {st['trades']:>4} | Wins: {st['wins']:>3} | WR: {wr:>5.1f}% | PnL: ${st['pnl']:>12,.2f}")
                report.append("")

                # Per-method drawdown (only when method overrides are active)
                has_method_overrides = getattr(self, 'bt_use_method_overrides_var', None) and self.bt_use_method_overrides_var.get()
                if has_method_overrides and any(st.get('max_dd', 0) > 0 for st in mt_stats.values()):
                    worst_method = max(mt_stats.items(), key=lambda x: x[1].get('max_dd', 0))
                    report.append("  Method Drawdown Analysis (Override Mode):")
                    for m, st in sorted(mt_stats.items(), key=lambda x: x[1].get('max_dd', 0), reverse=True):
                        dd = st.get('max_dd', 0)
                        dd_pct = st.get('max_dd_pct', 0)
                        marker = " ◄ WORST" if m == worst_method[0] else ""
                        report.append(f"    {m:<15} | Max DD: ${dd:>11,.2f} ({dd_pct:.2f}%) | PnL: ${st['pnl']:>12,.2f}{marker}")
                    report.append("")
                
            # Complete Trade Log Table
            if trades:
                report.append("── COMPLETE SIMULATED TRADE LOG ────────��─────────────────────────────────────────────────────────")
                header = f"{'#':>3} | {'Symbol':<10} | {'TF':<4} | {'Method':<13} | {'Dir':<4} | {'Entry Time':<19} | {'Entry':>10} | {'Exit Time':<19} | {'Exit':>10} | {'Outcome':<7} | {'Lots':>5} | {'PnL':>11} | {'Balance':>12}"
                report.append(header)
                report.append("-" * 150)
                
                trades_sorted = sorted(trades, key=lambda x: x.get('exit_time', x.get('entry_time')))
                for t_idx, t in enumerate(trades_sorted, 1):
                    entry_ts = str(t['entry_time'])[:19] if t.get('entry_time') else 'N/A'
                    exit_ts = str(t['exit_time'])[:19] if t.get('exit_time') else 'N/A'
                    line = (f"{t_idx:>3} | {t['symbol']:<10} | {t.get('timeframe','N/A'):<4} | {t.get('ict_method','N/A'):<13} | "
                            f"{t.get('trade_direction','N/A'):<4} | {entry_ts:<19} | {t['entry_price']:>10.5f} | "
                            f"{exit_ts:<19} | {t['exit_price']:>10.5f} | {t.get('outcome','N/A'):<7} | "
                            f"{t.get('lot_size',0):>5.2f} | ${t.get('pnl',0):>10,.2f} | ${t.get('balance',0):>11,.2f}")
                    report.append(line)
            report.append("\n\n")
            
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
            
        logger.info("Detailed backtest report auto-saved to: %s", filepath)

    def show_multi_backtest_results(self, all_results, symbols, initial_balance, date_from, date_to):
        if not all_results:
            messagebox.showerror("Backtest", "No tests completed.")
            return
            
        # Automatically save backtest report to file
        try:
            self.save_backtest_report_to_file(all_results, symbols, initial_balance, date_from, date_to)
        except Exception as e:
            logger.error("Failed to auto-save backtest report: %s\n%s", e, traceback.format_exc())

        popup = tk.Toplevel(self)
        popup.title(f"Backtest Results (Realistic) - {len(all_results)} Runs")
        popup.geometry("840x560")
        
        main_notebook = ttk.Notebook(popup)
        main_notebook.pack(fill=tk.BOTH, expand=True)
        
        is_opt = isinstance(all_results[0][0], str)

        if len(all_results) > 1:
            comp_frame = ttk.Frame(main_notebook)
            main_notebook.add(comp_frame, text="Comparison Summary")
            
            comp_text = scrolledtext.ScrolledText(comp_frame, font=('Consolas', 10))
            comp_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            hdr_lbl = "Configuration" if is_opt else "RRR"
            max_label_len = max(len(str(rrr)) if is_opt else len(f"{rrr:.2f}") for rrr, _, _ in all_results)
            col_width = max(len(hdr_lbl), max_label_len)
            comp_header = f"{hdr_lbl:<{col_width}} | {'Trades':>6} | {'WR%':>6} | {'Net PnL':>12} | {'Max DD':>10} | {'Max DD%':>8} | {'Sharpe':>7} | {'PF':>6}\n"
            comp_header += "-" * (len(comp_header) - 1) + "\n"
            comp_content = [comp_header]
            
            for rrr, trades, metrics in all_results:
                pf = metrics.get('profit_factor', 0)
                pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
                wr = metrics.get('win_rate', 0)
                pnl = metrics.get('net_profit', 0)
                dd = metrics.get('max_drawdown', 0)
                dd_pct = metrics.get('max_drawdown_pct', 0)
                sh = metrics.get('sharpe_ratio', 0)
                tc = metrics.get('total_trades', 0)
                
                label_str = f"{rrr:<{col_width}}" if is_opt else f"{rrr:>{col_width}.2f}"
                line = f"{label_str} | {tc:>6} | {wr:>6.1f} | ${pnl:>11,.2f} | ${dd:>9,.2f} | {dd_pct:>7.2f}% | {sh:>7.2f} | {pf_str:>6}\n"
                comp_content.append(line)
            
            comp_text.insert(tk.END, "".join(comp_content))
            comp_text.config(state=tk.DISABLED)

        # Then add a tab for each run containing its own notebook
        for idx, (rrr, trades, metrics) in enumerate(all_results):
            if len(all_results) > 1:
                tab_name = f"Run {idx+1}" if is_opt else f"RRR {rrr:.2f}"
            else:
                tab_name = "Results"
            run_frame = ttk.Frame(main_notebook)
            main_notebook.add(run_frame, text=tab_name)
            
            sub_notebook = ttk.Notebook(run_frame)
            sub_notebook.pack(fill=tk.BOTH, expand=True)
            self.populate_single_result(sub_notebook, trades, metrics, symbols, initial_balance, date_from, date_to)

    def populate_single_result(self, result_notebook, trades, metrics, symbols, initial_balance, date_from, date_to):
        if trades is None or len(trades) == 0:
            err_frame = ttk.Frame(result_notebook)
            result_notebook.add(err_frame, text="Summary")
            ttk.Label(err_frame, text="No trades executed on this run.").pack(padx=20, pady=20)
            return

        # Summary tab
        summary_frame = ttk.Frame(result_notebook)
        result_notebook.add(summary_frame, text="Summary")

        pf = metrics.get('profit_factor', 0)
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        
        avg_w = metrics.get('avg_win', 0)
        avg_l = abs(metrics.get('avg_loss', 0))
        real_rrr = (avg_w / avg_l) if avg_l > 0 else 0
        real_rrr_str = f"1:{real_rrr:.2f}" if avg_l > 0 else "N/A"

        methods_str = ", ".join([self.bt_methods_listbox.get(i) for i in self.bt_methods_listbox.curselection()])
        trail_methods_str = ", ".join([self.bt_trail_methods_listbox.get(i) for i in self.bt_trail_methods_listbox.curselection()])
        
        summary = f"""{'='*60}
  BACKTEST SETTINGS & CONFIGURATION
{'='*60}
  Symbols:            {', '.join(symbols)}
  Date Period:        {date_from.strftime('%Y-%m-%d')} → {date_to.strftime('%Y-%m-%d')}
  Initial Balance:    ${initial_balance:,.2f}
  Risk Mode:          {"Fixed Lot" if self.backtest_use_fixed_lot.get() else "Risk %"}
  Risk Size:          {self.backtest_fixed_lot_entry.get() + " Lots" if self.backtest_use_fixed_lot.get() else self.backtest_risk_entry.get() + "%"}
  Min/Max/Step RRR:   {self.backtest_min_rrr_entry.get()} / {self.backtest_max_rrr_entry.get()} / {self.backtest_incr_rrr_entry.get()}
  Dynamic RRR:        {"Enabled" if self.use_dynamic_rrr_backtest.get() else "Disabled"}
  Multi-Timeframe:    {"Enabled" if self.trade_all_tfs_backtest.get() else "Disabled"}
  Ultra Low TF:       {"Enabled" if self.use_ultra_low_tf_backtest.get() else "Disabled"}
  Spread Cost:        {self.bt_spread_cost.get()}
  Slippage (pts):     {self.bt_slippage.get()}
  Commission/Lot:     ${self.bt_commission.get()}
  Session Filter:     {"Enabled" if self.bt_session_filter.get() else "Disabled"} ({self.bt_session_start.get()}:00 - {self.bt_session_end.get()}:00)
  HTF Filter:         {"Enabled" if self.htf_filter_backtest.get() else "Disabled"} (Bypass if Conf >= 2: {"Yes" if self.bypass_htf_backtest.get() else "No"})
  OTE Filter:         {"Enabled" if self.ote_filter_backtest.get() else "Disabled"}
  Daily Loss Limit:   {self.bt_daily_loss.get()}%
  Max Concurrent:     {self.bt_max_concurr.get()}
  Min FVG Size:       {self.bt_min_fvg.get()} spreads
  Min Confluence:     {self.bt_min_conf.get()}
  Anti-Gap SL:        {"Enabled" if self.bt_anti_gap.get() else "Disabled"} (x{self.bt_anti_gap_mult.get()})
  Slippage Recovery:  {"Enabled" if self.bt_slippage_recovery_bt.get() else "Disabled"}
  ICT Methods:        {methods_str}
  FVG SL Mode:        {self.bt_fvg_sl_mode.get()}
  Require BoS:        {"Yes" if self.bt_require_bos_fvg.get() else "No"}
  Displacement Fltr:  {"Enabled" if self.bt_fvg_displacement_only.get() else "Disabled"}
  Discount/Premium:   {"Enabled" if self.bt_fvg_discount_premium_only.get() else "Disabled"}
  Recent Sweep Only:  {"Enabled" if self.bt_fvg_recent_sweep_only.get() else "Disabled"}
  Trailing Stop:      {"None (Disabled)" if not trail_methods_str else self.trail_type_bt.get()} Stop (methods: {trail_methods_str}, pct: {self.trail_pct_bt.get()}%)

{'='*60}
  BACKTEST PERFORMANCE REPORT — {', '.join(symbols)}
  Period: {date_from.strftime('%Y-%m-%d')} ��� {date_to.strftime('%Y-%m-%d')}
{'='*60}

── ACCOUNT ──────────────────────────────────
  Initial Balance:    ${initial_balance:>12,.2f}
  Final Balance:      ${metrics['final_balance']:>12,.2f}
  Net Profit:         ${metrics.get('net_profit', 0):>12,.2f}
  ROI:                {metrics.get('roi_pct', 0):>11.2f}%
  Peak Balance:       ${metrics.get('peak_balance', 0):>12,.2f}

── PERFORMANCE ──────────────────────────────
  Total Trades:       {metrics['total_trades']:>8}
  Win Rate (SL/TP):   {metrics['win_rate']:>7.2f}%   ({metrics.get('sltp_wins',0)}/{metrics.get('sltp_total',0)} SL/TP trades)
  Overall WR (all):   {metrics.get('overall_win_rate', 0):>7.2f}%   ({metrics['win_count']}/{metrics['total_trades']} all trades)
  Trail Exits:        {metrics.get('trail_total',0):>8}   (wins: {metrics.get('trail_wins',0)})
  Profit Factor:      {pf_str:>8}
  Expectancy/Trade:   ${metrics.get('expectancy', 0):>12,.2f}
  Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):>8.2f}

── RISK ─────────────────────────────────────
  Maximal Drawdown:   ${metrics['max_drawdown']:>12,.2f}
  Relative Drawdown%: {metrics.get('max_drawdown_pct', 0):>7.2f}%
  Max Consec Wins:    {metrics.get('max_consec_wins', 0):>8}
  Max Consec Losses:  {metrics.get('max_consec_losses', 0):>8}

── WIN / LOSS DETAILS ───────────────────────
  Avg Win:            ${metrics.get('avg_win', 0):>12,.2f}
  Avg Loss:           ${metrics.get('avg_loss', 0):>12,.2f}
  Realized RRR:       {real_rrr_str:>13}
  Largest Win:        ${metrics.get('highest_win', 0):>12,.2f}
  Largest Loss:       ${metrics.get('highest_loss', 0):>12,.2f}
  Gross Profit:       ${metrics.get('gross_profit', 0):>12,.2f}
  Gross Loss:         ${metrics.get('gross_loss', 0):>12,.2f}
  Total Commission:   ${metrics.get('total_commission', 0):>12,.2f}

── TIMING ──────────────────────────��────────
  Avg Trade Duration: {metrics.get('avg_duration_hrs', 0):>6.1f} hours
"""
        # Per-symbol breakdown
        sym_stats = metrics.get('symbol_stats', {})
        if sym_stats:
            summary += "\n── PER SYMBOL ───────────────────────────────\n"
            summary += f"  {'Symbol':<12} {'Trades':>7} {'Wins':>5} {'WR%':>7} {'PnL':>14}\n"
            summary += f"  {'-'*48}\n"
            for sym, st in sorted(sym_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
                wr = (st['wins'] / st['trades'] * 100) if st['trades'] > 0 else 0
                summary += f"  {sym:<12} {st['trades']:>7} {st['wins']:>5} {wr:>6.1f}% ${st['pnl']:>13,.2f}\n"

        # Per-TF breakdown
        tf_stats = metrics.get('tf_stats', {})
        if tf_stats:
            summary += "\n── PER TIMEFRAME ────────────────────────────\n"
            summary += f"  {'TF':<6} {'Trades':>7} {'Wins':>5} {'WR%':>7} {'PnL':>14}\n"
            summary += f"  {'-'*42}\n"
            for tf, st in sorted(tf_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
                wr = (st['wins'] / st['trades'] * 100) if st['trades'] > 0 else 0
                summary += f"  {tf:<6} {st['trades']:>7} {st['wins']:>5} {wr:>6.1f}% ${st['pnl']:>13,.2f}\n"

        # Per-method breakdown
        mt_stats = metrics.get('method_stats', {})
        if mt_stats:
            summary += "\n── PER METHOD ───────────────────────────────\n"
            summary += f"  {'Method':<15} {'Trades':>7} {'Wins':>5} {'WR%':>7} {'PnL':>14}\n"
            summary += f"  {'-'*50}\n"
            for m, st in sorted(mt_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
                wr = (st['wins'] / st['trades'] * 100) if st['trades'] > 0 else 0
                summary += f"  {m:<15} {st['trades']:>7} {st['wins']:>5} {wr:>6.1f}% ${st['pnl']:>13,.2f}\n"

        # Per-method drawdown analysis (only shown when method overrides are active)
        has_method_overrides = getattr(self, 'bt_use_method_overrides_var', None) and self.bt_use_method_overrides_var.get()
        if has_method_overrides and mt_stats and any(st.get('max_dd', 0) > 0 for st in mt_stats.values()):
            worst_method = max(mt_stats.items(), key=lambda x: x[1].get('max_dd', 0))
            summary += "\n── METHOD DRAWDOWN ANALYSIS (Override Mode) ─\n"
            summary += f"  {'Method':<15} {'Max DD $':>12} {'Max DD %':>9} {'Trades':>7} {'PnL':>14}\n"
            summary += f"  {'-'*60}\n"
            for m, st in sorted(mt_stats.items(), key=lambda x: x[1].get('max_dd', 0), reverse=True):
                dd = st.get('max_dd', 0)
                dd_pct = st.get('max_dd_pct', 0)
                marker = " ◄ WORST" if m == worst_method[0] else ""
                summary += f"  {m:<15} ${dd:>11,.2f} {dd_pct:>8.2f}% {st['trades']:>7} ${st['pnl']:>13,.2f}{marker}\n"

        # Outcome breakdown
        out_stats = metrics.get('outcome_stats', {})
        if out_stats:
            summary += "\n── EXIT OUTCOMES ────────────────────────────\n"
            summary += f"  {'Outcome':<10} {'Count':>7} {'PnL':>14}\n"
            summary += f"  {'-'*34}\n"
            for o, st in sorted(out_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                summary += f"  {o:<10} {st['count']:>7} ${st['pnl']:>13,.2f}\n"

        # Confluence score breakdown
        conf_stats = metrics.get('confluence_stats', {})
        if conf_stats:
            summary += "\n── CONFLUENCE SCORE ANALYSIS ─────────────────\n"
            summary += f"  {'Score':>5} {'Trades':>7} {'Wins':>5} {'WR%':>7} {'PnL':>14}\n"
            summary += f"  {'-'*42}\n"
            for score in sorted(conf_stats.keys()):
                st = conf_stats[score]
                wr = (st['wins'] / st['trades'] * 100) if st['trades'] > 0 else 0
                summary += f"  {score:>5} {st['trades']:>7} {st['wins']:>5} {wr:>6.1f}% ${st['pnl']:>13,.2f}\n"

        summary += f"""\n{'='*60}
  ICT FEATURES APPLIED
{'='*60}
  ✓ Fair Value Gap (FVG) detection + FVG-only trades
  ✓ Market Structure (BOS/MSS) analysis
  ✓ Confluence scoring (7 ICT factors)
  ✓ Displacement quality filtering
  ✓ Structure-based TP (liquidity pool targeting)
  ✓ OB + FVG confluence detection
  ✓ Spread-adaptive parameters (no hardcoded values)
  ✓ M1 drilldown for precise entry/exit timing
  ✓ Spread cost + slippage + commission
  ✓ Pessimistic intra-bar TP/SL resolution
  ✓ Trailing stop (Buy & Sell) simulated
  ✓ Session filter applied
  ✓ Lot normalization & max-lot splitting
  ✓ Min stop distance enforced
"""
        # ML Filter stats (if ML-filtered backtest was run)
        if metrics.get('ml_total_signals'):
            summary += f"""\n{'='*60}
  🧠 ML FILTER STATISTICS
{'='*60}
  Training Period:     {metrics.get('ml_train_period', 'N/A')}
  Test Period:         {metrics.get('ml_test_period', 'N/A')}
  Training Trades:     {metrics.get('ml_train_trades', 0)}
  RF Model Accuracy:   {metrics.get('ml_rf_accuracy', 0):.1f}%
  Confidence Threshold:{metrics.get('ml_confidence_threshold', 60):.0f}%

  Total Test Signals:  {metrics.get('ml_total_signals', 0)}
  ML Passed:           {metrics.get('ml_passed', 0)}
  ML Rejected:         {metrics.get('ml_rejected', 0)}
  Filter Rate:         {metrics.get('ml_filter_rate', 0):.1f}% of signals rejected
"""

        # Day of Week & Hour breakdown
        dow_stats = {i: {'trades': 0, 'wins': 0, 'pnl': 0.0} for i in range(7)}
        hour_stats = {i: {'trades': 0, 'wins': 0, 'pnl': 0.0} for i in range(24)}
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for t in trades:
            if 'entry_time' in t and pd.notna(t['entry_time']):
                dt = pd.to_datetime(t['entry_time'])
                dow = dt.dayofweek
                hour = dt.hour
                
                dow_stats[dow]['trades'] += 1
                hour_stats[hour]['trades'] += 1
                
                if t.get('pnl', 0) > 0:
                    dow_stats[dow]['wins'] += 1
                    hour_stats[hour]['wins'] += 1
                    
                dow_stats[dow]['pnl'] += t.get('pnl', 0)
                hour_stats[hour]['pnl'] += t.get('pnl', 0)

        # Build DOW summary
        if any(st['trades'] > 0 for st in dow_stats.values()):
            summary += "\n── PER DAY OF WEEK ──────────────────────────\n"
            summary += f"  {'Day':<12} {'Trades':>7} {'Wins':>5} {'WR%':>7} {'PnL':>14}\n"
            summary += f"  {'-'*48}\n"
            for i in range(7):
                st = dow_stats[i]
                if st['trades'] > 0:
                    wr = (st['wins'] / st['trades'] * 100)
                    summary += f"  {dow_names[i]:<12} {st['trades']:>7} {st['wins']:>5} {wr:>6.1f}% ${st['pnl']:>13,.2f}\n"

        # Build Hour summary
        if any(st['trades'] > 0 for st in hour_stats.values()):
            summary += "\n── PER HOUR (ENTRY) ─────────────────────────\n"
            summary += f"  {'Hour':<12} {'Trades':>7} {'Wins':>5} {'WR%':>7} {'PnL':>14}\n"
            summary += f"  {'-'*48}\n"
            for i in range(24):
                st = hour_stats[i]
                if st['trades'] > 0:
                    wr = (st['wins'] / st['trades'] * 100)
                    hour_str = f"{i:02d}:00"
                    summary += f"  {hour_str:<12} {st['trades']:>7} {st['wins']:>5} {wr:>6.1f}% ${st['pnl']:>13,.2f}\n"

        summary_text = scrolledtext.ScrolledText(summary_frame, font=('Consolas', 10))
        summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        summary_text.insert(tk.END, summary)
        summary_text.config(state=tk.DISABLED)

        # ML Report tab
        if metrics.get('ml_report'):
            ml_frame = ttk.Frame(result_notebook)
            result_notebook.add(ml_frame, text="ML Training Report")
            ml_text = scrolledtext.ScrolledText(ml_frame, wrap=tk.WORD, font=('Consolas', 10),
                                                 bg="#1e1e1e", fg="#d4d4d4")
            ml_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            ml_text.insert(tk.END, metrics['ml_report'])
            ml_text.config(state=tk.DISABLED)

        # Trades tab closes
        trades_by_exit = sorted(trades, key=lambda x: x.get('exit_time', x.get('entry_time')))
        trades_frame = ttk.Frame(result_notebook)
        result_notebook.add(trades_frame, text="Trades (by Close)")
        trades_text = scrolledtext.ScrolledText(trades_frame, wrap=tk.NONE, font=('Consolas', 9))
        trades_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        header = f"{'#':>3} | {'Symbol':<10} | {'TF':<4} | {'Method':<13} | {'Dir':<4} | {'OB Type':<8} | {'HTF':<8} | {'Conf':>4} | {'Confluences':<25} | {'Entry Time':<19} | {'Entry':>12} | {'Exit Time':<19} | {'Exit':>12} | {'TP':>12} | {'SL':>12} | {'RRR':>5} | {'Out':<5} | {'Duration':<16} | {'Lots':>7} | {'PnL':>12} | {'Comm':>8} | {'Balance':>12}\n"
        header += "-" * 265 + "\n"
        trades_text.insert(tk.END, header)
        for idx, t in enumerate(trades_by_exit, 1):
            entry_ts = str(t['entry_time'])[:19] if t.get('entry_time') else 'N/A'
            exit_ts = str(t['exit_time'])[:19] if t.get('exit_time') else 'N/A'
            line = (f"{idx:>3} | {t['symbol']:<10} | {t.get('timeframe','N/A'):<4} | {t.get('ict_method','N/A'):<13} | "
                    f"{t.get('trade_direction','N/A'):<4} | {t.get('order_block_type','N/A'):<8} | {t.get('htf_trend','N/A'):<8} | "
                    f"{t.get('confluence_score',0):>4} | {str(t.get('confluence_details',''))[:25]:<25} | "
                    f"{entry_ts:<19} | {t['entry_price']:>12.5f} | "
                    f"{exit_ts:<19} | {t['exit_price']:>12.5f} | "
                    f"{t.get('tp_price',0):>12.5f} | {t.get('sl_price',0):>12.5f} | "
                    f"{t.get('rrr',0):>5.2f} | {t.get('outcome','N/A'):<5} | {str(t.get('duration','N/A'))[:16]:<16} | "
                    f"{t.get('lot_size',0):>7.2f} | "
                    f"${t.get('pnl',0):>11,.2f} | ${t.get('commission',0):>7,.2f} | ${t.get('balance',0):>11,.2f}\n")
            trades_text.insert(tk.END, line)
        trades_text.config(state=tk.DISABLED)

        # Entry-order tab
        entry_frame = ttk.Frame(result_notebook)
        result_notebook.add(entry_frame, text="Trades (by Entry)")
        entry_text = scrolledtext.ScrolledText(entry_frame, wrap=tk.NONE, font=('Consolas', 9))
        entry_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        header2 = f"{'#':>3} | {'Symbol':<10} | {'TF':<4} | {'Method':<13} | {'Dir':<4} | {'Entry Time':<19} | {'Entry':>12} | {'Exit Time':<19} | {'Exit':>12} | {'Out':<5} | {'Duration':<16} | {'Lots':>7} | {'PnL':>12}\n"
        header2 += "-" * 160 + "\n"
        entry_text.insert(tk.END, header2)
        trades_by_entry = sorted(trades, key=lambda x: x.get('entry_time'))
        for idx, t in enumerate(trades_by_entry, 1):
            entry_ts = str(t['entry_time'])[:19] if t.get('entry_time') else 'N/A'
            exit_ts = str(t['exit_time'])[:19] if t.get('exit_time') else 'N/A'
            line = (f"{idx:>3} | {t['symbol']:<10} | {t.get('timeframe','N/A'):<4} | {t.get('ict_method','N/A'):<13} | "
                    f"{t.get('trade_direction','N/A'):<4} | "
                    f"{entry_ts:<19} | {t['entry_price']:>12.5f} | "
                    f"{exit_ts:<19} | {t['exit_price']:>12.5f} | "
                    f"{t.get('outcome','N/A'):<5} | {str(t.get('duration','N/A'))[:16]:<16} | "
                    f"{t.get('lot_size',0):>7.2f} | "
                    f"${t['pnl']:>11,.2f}\n")
            entry_text.insert(tk.END, line)
        entry_text.config(state=tk.DISABLED)

        # Chart tab
        if 'balance_history' in metrics and len(metrics['balance_history']) > 1:
            chart_frame = ttk.Frame(result_notebook)
            result_notebook.add(chart_frame, text="Balance Chart")
            # Sort arrays chronologically to prevent backwards lines
            sorted_pairs = sorted(zip(metrics['timestamps'], metrics['balance_history']), key=lambda x: x[0])
            ts_sorted = [p[0] for p in sorted_pairs]
            bal_sorted = [p[1] for p in sorted_pairs]

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            
            # Use plot with fill
            ax.plot(ts_sorted, bal_sorted, color='#00a8ff', linewidth=2, label="Balance", alpha=0.9)
            
            min_bal = min(bal_sorted) if bal_sorted else 0
            max_bal = max(bal_sorted) if bal_sorted else 0
            padding = (max_bal - min_bal) * 0.05
            ax.fill_between(ts_sorted, bal_sorted, min_bal - padding, color='#00a8ff', alpha=0.15)
            ax.set_ylim(bottom=min_bal - padding)

            ax.set_title("Account Balance Over Time", color='white', pad=15, fontsize=14, fontweight='bold')
            ax.set_xlabel("Date", color='#aaaaaa', fontsize=10)
            ax.set_ylabel("Balance ($)", color='#aaaaaa', fontsize=10)
            
            # Grid
            ax.grid(color='#333333', linestyle='--', linewidth=1, alpha=0.7)
            
            # Ticks
            ax.tick_params(axis='x', colors='#aaaaaa', labelsize=9)
            ax.tick_params(axis='y', colors='#aaaaaa', labelsize=9)
            
            # Remove borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#333333')
            ax.spines['bottom'].set_color('#333333')

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            fig.autofmt_xdate(rotation=45)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def get_selected_trail_type(self, mode='live'):
        """Return the selected trailing stop type string for the given mode."""
        if mode == 'backtest':
            return self.trail_type_bt.get()
        return self.trail_type_live.get()

    def get_trail_params(self, mode='live'):
        """Return a dict of trailing stop parameters from the UI."""
        try:
            if mode == 'backtest':
                return {
                    'trail_pct': float(self.trail_pct_bt.get()),
                }
            else:
                return {
                    'trail_pct': float(self.trail_pct_live.get()),
                }
        except (ValueError, AttributeError):
            return {}

    def _toggle_anti_gap(self):
        """Toggle Anti-Gap SL Protection on/off."""
        config.ANTI_GAP_SL_ENABLED = self.anti_gap_enabled_live.get()
        try:
            config.ANTI_GAP_ATR_MULTIPLIER = float(self.anti_gap_atr_mult_live.get())
        except ValueError:
            config.ANTI_GAP_ATR_MULTIPLIER = 2.0
        state = "ENABLED" if config.ANTI_GAP_SL_ENABLED else "DISABLED"
        logger.info("[ANTI-GAP] Anti-Gap SL Protection %s (ATR × %.1f)", state, config.ANTI_GAP_ATR_MULTIPLIER)

    def _toggle_slippage_recovery(self):
        """Toggle the global slippage recovery tracker."""
        config.slippage_recovery_tracker['enabled'] = self.slippage_recovery_live.get()
        state = "ENABLED" if config.slippage_recovery_tracker['enabled'] else "DISABLED"
        logger.info("[RECOVERY] Slippage recovery %s by user.", state)

    def launch_mt5_terminal(self):
        """Launch MetaTrader 5 Terminal executable (supports Wine on Linux & Native Windows)."""
        import os
        import sys
        import subprocess

        wine_candidates = [
            (
                "/home/mohammed/.wine_mt5",
                "/home/mohammed/.wine_mt5/drive_c/Program Files/MetaTrader 5 EXNESS/terminal64.exe",
            ),
            (
                os.path.expanduser("~/.wine_mt5"),
                os.path.expanduser("~/.wine_mt5/drive_c/Program Files/MetaTrader 5 EXNESS/terminal64.exe"),
            ),
            (
                os.path.expanduser("~/.wine"),
                os.path.expanduser("~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"),
            ),
            (
                os.path.expanduser("~/.wine"),
                os.path.expanduser("~/.wine/drive_c/Program Files (x86)/MetaTrader 5/terminal64.exe"),
            ),
        ]

        if sys.platform.startswith("win"):
            win_paths = [
                r"C:\Program Files\MetaTrader 5\terminal64.exe",
                r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
                r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe",
            ]
            for exe in win_paths:
                if os.path.exists(exe):
                    try:
                        subprocess.Popen([exe])
                        logger.info(f"[MT5 LAUNCHER] Launched MT5 Terminal: {exe}")
                        messagebox.showinfo("MT5 Launcher", f"Successfully launched MetaTrader 5 Terminal:\n{exe}")
                        return
                    except Exception as e:
                        logger.error(f"[MT5 LAUNCHER] Failed to launch MT5: {e}")
                        messagebox.showerror("Launcher Error", f"Failed to launch MT5: {e}")
                        return
            messagebox.showerror("MT5 Not Found", "Could not locate terminal64.exe on Windows.")
        else:
            # Linux (Wine)
            for prefix, exe in wine_candidates:
                if os.path.exists(exe):
                    try:
                        env = os.environ.copy()
                        if prefix and os.path.exists(prefix):
                            env["WINEPREFIX"] = prefix
                        wine_bin = "wine-stable" if subprocess.call(["which", "wine-stable"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0 else "wine"
                        subprocess.Popen([wine_bin, exe], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        logger.info(f"[MT5 LAUNCHER] Launched MT5 under Wine ({wine_bin}, WINEPREFIX={prefix}): {exe}")
                        messagebox.showinfo("MT5 Launcher", f"Successfully launched MetaTrader 5 under Wine!\nPrefix: {prefix}\nExecutable: {exe}")
                        return
                    except Exception as e:
                        logger.error(f"[MT5 LAUNCHER] Failed to launch MT5 under Wine: {e}")
                        messagebox.showerror("Launcher Error", f"Failed to launch MT5 under Wine: {e}")
                        return

            # Dynamic search across user home dir if candidate paths don't match
            try:
                res = subprocess.run(
                    ["find", os.path.expanduser("~"), "-name", "terminal64.exe"],
                    capture_output=True, text=True, timeout=5
                )
                found_files = [p.strip() for p in res.stdout.splitlines() if p.strip()]
                if found_files:
                    exe = found_files[0]
                    prefix = "/home/mohammed/.wine_mt5" if ".wine_mt5" in exe else os.path.expanduser("~/.wine")
                    env = os.environ.copy()
                    if os.path.exists(prefix):
                        env["WINEPREFIX"] = prefix
                    wine_bin = "wine-stable" if subprocess.call(["which", "wine-stable"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0 else "wine"
                    subprocess.Popen([wine_bin, exe], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    logger.info(f"[MT5 LAUNCHER] Found and launched MT5: {exe}")
                    messagebox.showinfo("MT5 Launcher", f"Launched MT5:\n{exe}")
                    return
            except Exception as e:
                logger.error(f"[MT5 LAUNCHER] Search error: {e}")

            messagebox.showerror("MT5 Not Found", "Could not locate terminal64.exe in Wine directories.")

    def start_live_trading(self):
        self._apply_pro_flags()  # sync v17 Pro Strategy toggles into config
        # Auto-save overrides if profiles checkbox is checked - Live
        if self.live_use_overrides_var.get():
            if self.live_currently_selected_override_symbol:
                if self.live_use_method_overrides_var.get() and self.live_currently_selected_override_method:
                    self.save_live_method_settings(self.live_currently_selected_override_symbol, self.live_currently_selected_override_method)
                else:
                    self.live_symbol_overrides[self.live_currently_selected_override_symbol] = self.get_current_ui_live_settings()
            
            import json, os
            profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbol_profiles.json")
            existing_profiles = {}
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, "r", encoding="utf-8") as f:
                        existing_profiles = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading existing profiles on save: {e}")
            
            # Merge self.live_symbol_overrides into existing_profiles
            for sym, settings in self.live_symbol_overrides.items():
                save_settings = {k: v for k, v in settings.items() if k not in ('ict_method', 'trail_methods')}
                if "methods" not in save_settings and sym in existing_profiles and "methods" in existing_profiles[sym]:
                    save_settings["methods"] = existing_profiles[sym]["methods"]
                existing_profiles[sym] = save_settings
                
            try:
                with open(profile_path, "w", encoding="utf-8") as f:
                    json.dump(existing_profiles, f, indent=2)
                logger.info("Auto-saved active live symbol profiles overrides to symbol_profiles.json")
            except Exception as e:
                logger.error(f"Failed to auto-save live symbol profiles: {e}")

        # Resolve base (global default) settings to launch with
        # Always use the current UI settings as the base so user changes take immediate effect.
        # Symbol profiles (if enabled) will override these on a per-symbol basis.
        base_settings = self.get_current_ui_live_settings()

        try:
            risk_percent = float(base_settings.get('risk_percent', 1.0))
            if self.live_use_overrides_var.get():
                config.DAILY_LOSS_LIMIT_PCT = 0.0
            else:
                config.DAILY_LOSS_LIMIT_PCT = float(base_settings.get('daily_loss_limit', 0.0))
            config.MAX_CONCURRENT_TRADES = int(base_settings.get('max_concurrent_trades', 3))
            config.MIN_FVG_SIZE_SPREADS = float(base_settings.get('min_fvg_size', 0.5))
            config.MIN_CONFLUENCE_SCORE = int(base_settings.get('min_confluence_score', 1))
            try:
                config.PENDING_LIMIT_EXPIRY_HOURS = float(base_settings.get('pending_limit_expiry_hours', 2.0))
            except ValueError:
                config.PENDING_LIMIT_EXPIRY_HOURS = 2.0
            config.ANTI_GAP_SL_ENABLED = bool(base_settings.get('anti_gap_enabled', False))
            try:
                config.ANTI_GAP_ATR_MULTIPLIER = float(base_settings.get('anti_gap_mult', 2.0))
            except ValueError:
                config.ANTI_GAP_ATR_MULTIPLIER = 2.0
            config.slippage_recovery_tracker['enabled'] = bool(base_settings.get('slippage_recovery', False))
            
        except (ValueError, TypeError, KeyError) as e:
            messagebox.showerror("Error", f"Enter valid default settings: {e}")
            return

        fixed_lot = None
        risk_mode = "Fixed" if base_settings.get('use_fixed_lot', False) else "Risk"
        if risk_mode == "Fixed":
            try:
                fixed_lot = float(base_settings.get('fixed_lot', 0.1))
            except (ValueError, TypeError):
                messagebox.showerror("Error", "Enter valid fixed lot.")
                return

        symbols = [s.strip() for s in self.live_symbols_entry.get().split(",") if s.strip()]
        if not symbols:
            messagebox.showerror("Error", "Enter at least one symbol.")
            return

        sel_in = self.live_methods_listbox.curselection()
        selected_live_methods = [self.live_methods_listbox.get(i) for i in sel_in]
        if not selected_live_methods:
            selected_live_methods = config.CORE_METHODS[:]

        sel_ts = self.live_trail_methods_listbox.curselection()
        selected_trail_methods = [self.live_trail_methods_listbox.get(i) for i in sel_ts]

        # Use base_settings instead of UI widget reads for the global/fallback parameters passed to live thread:
        use_dynamic_rrr = bool(base_settings.get('use_dynamic_rrr', True))
        try:
            min_rrr = float(base_settings.get('min_rrr', 1.5))
        except ValueError:
            min_rrr = 1.5
        session_filter = bool(base_settings.get('session_filter', True))
        try:
            session_start = int(base_settings.get('session_start', 7))
            session_end = int(base_settings.get('session_end', 21))
        except ValueError:
            session_start = 7
            session_end = 21
        live_fvg_sl_mode = base_settings.get('fvg_sl_mode', 'Normal')
        trade_all_tfs = bool(base_settings.get('trade_on_all_tfs', True))
        use_ultra_low_tf = bool(base_settings.get('use_ultra_low_tf', False))
        use_htf_filter = bool(base_settings.get('use_htf_filter', True))
        use_ote_filter = bool(base_settings.get('use_ote_filter', True))
        bypass_htf_conf = bool(base_settings.get('bypass_htf_conf', False))
        trail_type = base_settings.get('trail_type', 'Swing')
        
        # Resolve trail params from base settings:
        trail_params = {'trail_pct': float(base_settings.get('trail_pct', 0.5))}
        
        require_bos_fvg = bool(base_settings.get('require_bos_fvg', False))
        ml_enabled = bool(base_settings.get('ml_enabled', False))
        try:
            ml_min_confidence = float(base_settings.get('ml_min_confidence', 60.0)) / 100.0
        except ValueError:
            ml_min_confidence = 0.60
        fvg_displacement_only = bool(base_settings.get('fvg_displacement_only', True))
        fvg_discount_premium_only = bool(base_settings.get('fvg_discount_premium_only', True))
        fvg_recent_sweep_only = bool(base_settings.get('fvg_recent_sweep_only', False))
        sb_require_htf_bias = bool(base_settings.get('sb_require_htf_bias', False))
        live_use_symbol_profiles = self.live_use_overrides_var.get()

        config.stop_event.clear()
        config.live_trading_running = True
        self.status_label.config(text="Status: Live Trading Running")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        config.live_trading_thread = threading.Thread(
            target=self.run_trading_with_error_handling,
            args=(risk_percent, risk_mode, fixed_lot, symbols, selected_live_methods, selected_trail_methods,
                  use_dynamic_rrr, min_rrr, session_filter, session_start, session_end, live_fvg_sl_mode, trade_all_tfs,
                  use_ultra_low_tf, use_htf_filter, use_ote_filter, bypass_htf_conf, trail_type, trail_params, require_bos_fvg, ml_enabled,
                  ml_min_confidence, fvg_displacement_only, fvg_discount_premium_only, fvg_recent_sweep_only, sb_require_htf_bias, live_use_symbol_profiles),
            daemon=True
        )
        config.live_trading_thread.start()

    def run_trading_with_error_handling(self, risk_percent, risk_mode, fixed_lot, symbols, selected_live_methods, selected_trail_methods,
                                        use_dynamic_rrr, min_rrr, session_filter, session_start, session_end, live_fvg_sl_mode,
                                        trade_on_all_tfs, use_ultra_low_tf, use_htf_filter, use_ote_filter, bypass_htf_conf, trail_type, trail_params,
                                        require_bos_fvg, ml_enabled, ml_min_confidence, fvg_displacement_only, fvg_discount_premium_only, fvg_recent_sweep_only,
                                        sb_require_htf_bias, use_symbol_profiles):
        try:
            ict_params = get_ict_model_parameters("Default", symbols[0] if symbols else None)
            ict_method = selected_live_methods
            run_live_trading(risk_percent, risk_mode, fixed_lot, symbols,
                             trailing_methods=selected_trail_methods,
                             use_dynamic_rrr=use_dynamic_rrr,
                             fvg_sl_mode=live_fvg_sl_mode,
                             ict_method=ict_method, ict_params=ict_params, min_rrr=min_rrr,
                             trade_on_all_tfs=trade_on_all_tfs,
                             use_ultra_low_tf=use_ultra_low_tf,
                             session_filter=session_filter,
                             session_start=session_start, session_end=session_end,
                             use_htf_filter=use_htf_filter,
                             use_ote_filter=use_ote_filter,
                             bypass_htf_conf=bypass_htf_conf,
                             trail_type=trail_type,
                             trail_params=trail_params,
                             require_bos_fvg=require_bos_fvg,
                             ml_enabled=ml_enabled,
                             ml_min_confidence=ml_min_confidence,
                             fvg_displacement_only=fvg_displacement_only,
                             fvg_discount_premium_only=fvg_discount_premium_only,
                             fvg_recent_sweep_only=fvg_recent_sweep_only,
                             sb_require_htf_bias=sb_require_htf_bias,
                             use_symbol_profiles=use_symbol_profiles)
        except Exception as e:
            logger.error("Trading error: %s\n%s", e, traceback.format_exc())
            self.after(0, self.handle_trading_error)

    def handle_trading_error(self):
        self.stop_live_trading()
        messagebox.showerror("Error", "Trading stopped due to error. Check log.")

    def stop_live_trading(self):
        config.live_trading_running = False
        config.stop_event.set()
        self.status_label.config(text="Status: Stopped")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def on_closing(self):
        self.is_running = False
        for after_id in self.after_ids:
            try:
                self.after_cancel(after_id)
            except Exception:
                pass
        self.after_ids.clear()
        self.stop_live_trading()
        save_trade_memory()
        self.quit()
        self.destroy()

    # _create_chart_tab and _plot_chart are in ChartMixin (ui_chart_mixin.py)
    # _create_ai_tab, _refresh_ai_runs, run_ai_analyst, _ai_analyst_thread are in AIAnalystMixin (ui_ai_mixin.py)

    def check_symbol_count_for_overrides(self):
        symbols_str = self.backtest_symbol_entry.get()
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        if len(symbols) >= 1:
            self.bt_use_overrides_cb.config(state=tk.NORMAL)
            if self.bt_use_overrides_var.get():
                current_values = self.bt_override_symbol_dropdown["values"]
                if list(current_values) != symbols:
                    self.bt_override_symbol_dropdown["values"] = symbols
                    
                    # Initialize overrides for any new symbols that aren't in self.bt_symbol_overrides yet
                    import json, os
                    disk_profiles = {}
                    profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbol_profiles.json")
                    if os.path.exists(profile_path):
                        try:
                            with open(profile_path, 'r', encoding="utf-8") as f:
                                disk_profiles = json.load(f)
                        except Exception:
                            pass
                    for sym in symbols:
                        if sym not in self.bt_symbol_overrides:
                            self.bt_symbol_overrides[sym] = self.resolve_symbol_overrides(sym, disk_profiles)
                            
                    # If the active symbol is no longer in the list, switch to the first symbol
                    active = self.bt_override_symbol_dropdown.get()
                    if active not in symbols:
                        if symbols:
                            self.bt_override_symbol_dropdown.set(symbols[0])
                            self.on_select_override_symbol(None)
                    else:
                        self.bt_currently_selected_override_symbol = active
        else:
            if self.bt_use_overrides_var.get():
                self.bt_use_overrides_var.set(False)
                self.on_toggle_bt_overrides()
            self.bt_use_overrides_cb.config(state=tk.DISABLED)

    def on_toggle_bt_overrides(self):
        symbols_str = self.backtest_symbol_entry.get()
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        
        if self.bt_use_overrides_var.get():
            # Caching base settings
            self.bt_base_cached_settings = self.get_current_ui_backtest_settings()
            
            # Load disk profiles
            import json, os
            disk_profiles = {}
            profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbol_profiles.json")
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, 'r', encoding="utf-8") as f:
                        disk_profiles = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading symbol_profiles.json: {e}")
            
            # Resolve/Merge for all symbols
            for sym in symbols:
                self.bt_symbol_overrides[sym] = self.resolve_symbol_overrides(sym, disk_profiles)
            
            # Populate and enable dropdown
            self.bt_override_symbol_dropdown["values"] = symbols
            self.bt_override_symbol_dropdown.configure(state="readonly")
            self.bt_use_method_overrides_cb.config(state=tk.NORMAL)
            if symbols:
                self.bt_override_symbol_dropdown.set(symbols[0])
                self.bt_currently_selected_override_symbol = symbols[0]
                self.apply_backtest_settings_to_ui(self.bt_symbol_overrides[symbols[0]])
        else:
            # Save current overrides back to the active symbol if any
            if self.bt_currently_selected_override_symbol:
                if self.bt_use_method_overrides_var.get() and self.bt_currently_selected_override_method:
                    self.save_bt_method_settings(self.bt_currently_selected_override_symbol, self.bt_currently_selected_override_method)
                else:
                    self.bt_symbol_overrides[self.bt_currently_selected_override_symbol] = self.get_current_ui_backtest_settings()
            
            # Disable dropdown
            self.bt_override_symbol_dropdown.set("")
            self.bt_override_symbol_dropdown.configure(state="disabled")
            self.bt_use_method_overrides_var.set(False)
            self.bt_use_method_overrides_cb.config(state=tk.DISABLED)
            self.bt_override_method_dropdown.set("")
            self.bt_override_method_dropdown.configure(state="disabled")
            
            # Restore cached base settings
            if self.bt_base_cached_settings:
                self.apply_backtest_settings_to_ui(self.bt_base_cached_settings)
            
            self.bt_currently_selected_override_symbol = None
            self.bt_currently_selected_override_method = None

    def on_select_override_symbol(self, event):
        new_symbol = self.bt_override_symbol_dropdown.get()
        prev_symbol = self.bt_currently_selected_override_symbol
        if not new_symbol:
            return
            
        # Save current edits back to the previous selected symbol
        if prev_symbol:
            if self.bt_use_method_overrides_var.get() and self.bt_currently_selected_override_method:
                self.save_bt_method_settings(prev_symbol, self.bt_currently_selected_override_method)
            else:
                current_ui = self.get_current_ui_backtest_settings()
                existing_methods = self.bt_symbol_overrides.get(prev_symbol, {}).get("methods", {})
                self.bt_symbol_overrides[prev_symbol] = current_ui
                if existing_methods:
                    self.bt_symbol_overrides[prev_symbol]["methods"] = existing_methods
            
        # Load new symbol's overrides
        if new_symbol in self.bt_symbol_overrides:
            if self.bt_use_method_overrides_var.get():
                active_method = self.bt_override_method_dropdown.get()
                if active_method:
                    self.apply_backtest_settings_to_ui(self.get_merged_bt_settings(new_symbol, active_method))
                    self.bt_currently_selected_override_method = active_method
                else:
                    self.apply_backtest_settings_to_ui(self.bt_symbol_overrides[new_symbol])
            else:
                self.apply_backtest_settings_to_ui(self.bt_symbol_overrides[new_symbol])
            
        self.bt_currently_selected_override_symbol = new_symbol

    def resolve_symbol_overrides(self, symbol, disk_profiles):
        # 1. Exact match in memory overrides
        if symbol in self.bt_symbol_overrides:
            return self.bt_symbol_overrides[symbol]
            
        # 2. Exact match in disk profiles
        if symbol in disk_profiles:
            profile = disk_profiles[symbol].copy()
            # Strip session-only keys that should not overwrite current UI listbox selections
            profile.pop('ict_method', None)
            profile.pop('trail_methods', None)
            return profile
            
        # 3. Partial/Fallback matches in disk profiles
        symbol_upper = symbol.upper()
        for k, v in disk_profiles.items():
            k_upper = k.upper()
            if k_upper in symbol_upper or symbol_upper in k_upper:
                profile = v.copy()
                profile.pop('ict_method', None)
                profile.pop('trail_methods', None)
                return profile
            if k_upper == "GOLD" and "XAU" in symbol_upper:
                profile = v.copy()
                profile.pop('ict_method', None)
                profile.pop('trail_methods', None)
                return profile
            if k_upper == "BTC" and "BTC" in symbol_upper:
                profile = v.copy()
                profile.pop('ict_method', None)
                profile.pop('trail_methods', None)
                return profile
                
        # 4. Fallback to cached base settings
        return self.bt_base_cached_settings.copy()

    def on_toggle_bt_method_overrides(self):
        symbol = self.bt_currently_selected_override_symbol
        if not symbol:
            return
            
        if self.bt_use_method_overrides_var.get():
            # Save base settings before loading method overrides
            current_ui = self.get_current_ui_backtest_settings()
            existing_methods = self.bt_symbol_overrides.get(symbol, {}).get("methods", {})
            self.bt_symbol_overrides[symbol] = current_ui
            if existing_methods:
                self.bt_symbol_overrides[symbol]["methods"] = existing_methods
            
            # Populate override dropdown with ONLY the methods currently selected in the Methods listbox
            selected_methods = [self.bt_methods_listbox.get(i) for i in self.bt_methods_listbox.curselection()]
            if not selected_methods:
                selected_methods = config.CORE_METHODS[:]
            self.bt_override_method_dropdown["values"] = selected_methods
            self.bt_override_method_dropdown.configure(state="readonly")
            self.bt_override_method_dropdown.set(selected_methods[0])
            self.bt_currently_selected_override_method = selected_methods[0]
            
            self.apply_backtest_settings_to_ui(self.get_merged_bt_settings(symbol, selected_methods[0]))
            # Update FVG-specific widget states based on the selected override method
            self._update_bt_fvg_widgets_for_override(selected_methods[0])
        else:
            if self.bt_currently_selected_override_method:
                self.save_bt_method_settings(symbol, self.bt_currently_selected_override_method)
                
            self.bt_override_method_dropdown.set("")
            self.bt_override_method_dropdown.configure(state="disabled")
            
            self.bt_session_filter_cb.config(state=tk.NORMAL)
            self.bt_session_start.config(state=tk.NORMAL)
            self.bt_session_end.config(state=tk.NORMAL)
            
            self.apply_backtest_settings_to_ui(self.bt_symbol_overrides[symbol])
            self.bt_currently_selected_override_method = None

    def on_select_override_method(self, event):
        symbol = self.bt_currently_selected_override_symbol
        new_method = self.bt_override_method_dropdown.get()
        prev_method = self.bt_currently_selected_override_method
        
        if symbol and prev_method:
            self.save_bt_method_settings(symbol, prev_method)
            
        if symbol and new_method:
            self.apply_backtest_settings_to_ui(self.get_merged_bt_settings(symbol, new_method))
            # Update FVG-specific widget states based on the newly selected method
            self._update_bt_fvg_widgets_for_override(new_method)
            
        self.bt_currently_selected_override_method = new_method

    def _update_bt_fvg_widgets_for_override(self, method):
        """Enable/disable FVG-specific widgets based on the override method."""
        fvg_state = tk.NORMAL if method == "FVG Return" else tk.DISABLED
        for child in self.bt_fvg_sl_frame.winfo_children():
            child.configure(state=fvg_state)
        self.bt_require_bos_cb.config(state=fvg_state)
        for child in self.bt_fvg_filters_frame.winfo_children():
            child.configure(state=fvg_state)
            
        session_state = tk.DISABLED if method == "Silver Bullet" else tk.NORMAL
        self.bt_session_filter_cb.config(state=session_state)
        self.bt_session_start.config(state=session_state)
        self.bt_session_end.config(state=session_state)

    def save_bt_method_settings(self, symbol, method):
        if not symbol or not method:
            return
        current_ui = self.get_current_ui_backtest_settings()
        base_settings = self.bt_symbol_overrides.setdefault(symbol, {})
        
        # Check differences
        method_overrides = {}
        for k, v in current_ui.items():
            if k in ["methods", "ict_method", "trail_methods"]:
                continue
            if k in base_settings:
                if base_settings[k] != v:
                    method_overrides[k] = v
            else:
                method_overrides[k] = v
                
        if method_overrides:
            base_settings.setdefault("methods", {})[method] = method_overrides
        else:
            if "methods" in base_settings and method in base_settings["methods"]:
                del base_settings["methods"][method]

    def get_merged_bt_settings(self, symbol, method):
        if not symbol:
            return {}
        base_settings = self.bt_symbol_overrides.get(symbol, {}).copy()
        if method:
            method_settings = base_settings.get("methods", {}).get(method, {})
            for k, v in method_settings.items():
                if k != "methods":
                    base_settings[k] = v
        return base_settings

    def get_current_ui_backtest_settings(self):
        settings = {}
        # Entry widgets
        try: settings['risk_percent'] = float(self.backtest_risk_entry.get())
        except ValueError: settings['risk_percent'] = 1.0
        
        try: settings['fixed_lot'] = float(self.backtest_fixed_lot_entry.get())
        except ValueError: settings['fixed_lot'] = 0.1
        
        try: settings['min_rrr'] = float(self.backtest_min_rrr_entry.get())
        except ValueError: settings['min_rrr'] = 1.5
        
        try: settings['max_rrr'] = float(self.backtest_max_rrr_entry.get())
        except ValueError: settings['max_rrr'] = 1.5
        
        try: settings['step_rrr'] = float(self.backtest_incr_rrr_entry.get())
        except ValueError: settings['step_rrr'] = 0.5
        
        try: settings['spread_cost'] = float(self.bt_spread_cost.get())
        except ValueError: settings['spread_cost'] = 0.0
        
        try: settings['slippage_points'] = int(self.bt_slippage.get())
        except ValueError: settings['slippage_points'] = 5
        
        try: settings['commission_per_lot'] = float(self.bt_commission.get())
        except ValueError: settings['commission_per_lot'] = 7.0
        
        try: settings['session_start'] = int(self.bt_session_start.get())
        except ValueError: settings['session_start'] = 7
        
        try: settings['session_end'] = int(self.bt_session_end.get())
        except ValueError: settings['session_end'] = 21
        
        try: settings['daily_loss_limit'] = float(self.bt_daily_loss.get().replace('%', ''))
        except ValueError: settings['daily_loss_limit'] = 0.0
        
        try: settings['max_concurrent_trades'] = int(self.bt_max_concurr.get())
        except ValueError: settings['max_concurrent_trades'] = 3
        
        try: settings['min_fvg_size'] = float(self.bt_min_fvg.get())
        except ValueError: settings['min_fvg_size'] = 0.5
        
        try: settings['min_confluence_score'] = int(self.bt_min_conf.get())
        except ValueError: settings['min_confluence_score'] = 1
        
        try: settings['anti_gap_mult'] = float(self.bt_anti_gap_mult.get())
        except ValueError: settings['anti_gap_mult'] = 2.0
        
        try: settings['fvg_sl_spread_buffer'] = float(self.bt_sl_spread_buffer.get())
        except ValueError: settings['fvg_sl_spread_buffer'] = 2.0
        
        try: settings['trail_pct'] = float(self.trail_pct_bt.get())
        except ValueError: settings['trail_pct'] = 0.5

        # BooleanVar / StringVar / IntVar widgets
        settings['use_fixed_lot'] = self.backtest_use_fixed_lot.get()
        settings['use_dynamic_rrr'] = self.use_dynamic_rrr_backtest.get()
        settings['fvg_sl_mode'] = self.bt_fvg_sl_mode.get()
        settings['trade_on_all_tfs'] = self.trade_all_tfs_backtest.get()
        settings['use_ultra_low_tf'] = self.use_ultra_low_tf_backtest.get()
        settings['session_filter'] = self.bt_session_filter.get()
        settings['use_htf_filter'] = self.htf_filter_backtest.get()
        settings['use_ote_filter'] = self.ote_filter_backtest.get()
        settings['bypass_htf_conf'] = self.bypass_htf_backtest.get()
        settings['trail_type'] = self.trail_type_bt.get()
        settings['require_bos_fvg'] = self.bt_require_bos_fvg.get()
        settings['fvg_displacement_only'] = self.bt_fvg_displacement_only.get()
        settings['fvg_discount_premium_only'] = self.bt_fvg_discount_premium_only.get()
        settings['fvg_recent_sweep_only'] = self.bt_fvg_recent_sweep_only.get()
        settings['sb_require_htf_bias'] = self.bt_sb_require_htf_bias.get()
        settings['anti_gap_enabled'] = self.bt_anti_gap.get()
        settings['limit_touch_fill'] = self.bt_limit_touch_fill.get()
        settings['slippage_recovery'] = self.bt_slippage_recovery_bt.get()
        settings['smart_optimize'] = self.bt_smart_optimize.get()
        settings['ml_filter'] = self.bt_ml_filter.get()
        settings['pro_dol_tp'] = self.bt_pro_dol_tp.get() if hasattr(self, 'bt_pro_dol_tp') else False
        settings['pro_killzone'] = self.bt_pro_killzone.get() if hasattr(self, 'bt_pro_killzone') else False
        settings['pro_htf_poi'] = self.bt_pro_htf_poi.get() if hasattr(self, 'bt_pro_htf_poi') else False
        settings['pro_mandatory'] = self.bt_pro_mandatory.get() if hasattr(self, 'bt_pro_mandatory') else False
        settings['pro_regime'] = self.bt_pro_regime.get() if hasattr(self, 'bt_pro_regime') else False
        settings['pro_ml_sizing'] = self.bt_pro_ml_sizing.get() if hasattr(self, 'bt_pro_ml_sizing') else False
        settings['pro_ml_rank'] = self.bt_pro_ml_rank.get() if hasattr(self, 'bt_pro_ml_rank') else False
        settings['pro_multi_tf_conf'] = self.bt_pro_multi_tf_conf.get() if hasattr(self, 'bt_pro_multi_tf_conf') else False
        settings['pro_multi_tf_gate'] = self.bt_pro_multi_tf_gate.get() if hasattr(self, 'bt_pro_multi_tf_gate') else False
        settings['pro_regime_adaptive'] = self.bt_pro_regime_adaptive.get() if hasattr(self, 'bt_pro_regime_adaptive') else False
        settings['use_smt_divergence'] = self.bt_smt_enabled.get() if hasattr(self, 'bt_smt_enabled') else False
        settings['smt_correlated_pair'] = self.bt_smt_pair.get() if hasattr(self, 'bt_smt_pair') else "DXY"
        settings['use_mt5_data'] = self.bt_use_mt5_data.get() if hasattr(self, 'bt_use_mt5_data') else False
        
        # Override global config for Live
        config.NEWS_FILTER_ENABLED = self.live_news_enabled.get() if hasattr(self, 'live_news_enabled') else False
        try:
            config.NEWS_FILTER_BUFFER_MINS = int(self.live_news_buffer.get()) if hasattr(self, 'live_news_buffer') else 30
        except ValueError:
            config.NEWS_FILTER_BUFFER_MINS = 30
        settings['use_volume_profile'] = self.bt_vp_enabled.get() if hasattr(self, 'bt_vp_enabled') else False
        
        try: settings['ml_min_confidence'] = float(self.bt_ml_min_confidence.get())
        except ValueError: settings['ml_min_confidence'] = 60.0

        # Listboxes
        settings['ict_method'] = [self.bt_methods_listbox.get(i) for i in self.bt_methods_listbox.curselection()]
        if hasattr(self, 'bt_trail_methods_listbox'):
            settings['trail_methods'] = [self.bt_trail_methods_listbox.get(i) for i in self.bt_trail_methods_listbox.curselection()]
        else:
            settings['trail_methods'] = []
            
        return settings

    def apply_backtest_settings_to_ui(self, settings):
        if not settings:
            return
            
        def set_entry(entry_widget, val):
            if entry_widget and val is not None:
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, str(val))

        def set_var(var_widget, val):
            if var_widget is not None and val is not None:
                var_widget.set(val)

        if 'risk_percent' in settings: set_entry(self.backtest_risk_entry, settings['risk_percent'])
        if 'fixed_lot' in settings: set_entry(self.backtest_fixed_lot_entry, settings['fixed_lot'])
        if 'min_rrr' in settings: set_entry(self.backtest_min_rrr_entry, settings['min_rrr'])
        if 'max_rrr' in settings: set_entry(self.backtest_max_rrr_entry, settings['max_rrr'])
        if 'step_rrr' in settings: set_entry(self.backtest_incr_rrr_entry, settings['step_rrr'])
        if 'spread_cost' in settings: set_entry(self.bt_spread_cost, settings['spread_cost'])
        if 'slippage_points' in settings: set_entry(self.bt_slippage, settings['slippage_points'])
        if 'commission_per_lot' in settings: set_entry(self.bt_commission, settings['commission_per_lot'])
        if 'session_start' in settings: set_entry(self.bt_session_start, settings['session_start'])
        if 'session_end' in settings: set_entry(self.bt_session_end, settings['session_end'])
        if 'daily_loss_limit' in settings: set_entry(self.bt_daily_loss, settings['daily_loss_limit'])
        if 'max_concurrent_trades' in settings: set_entry(self.bt_max_concurr, settings['max_concurrent_trades'])
        if 'min_fvg_size' in settings: set_entry(self.bt_min_fvg, settings['min_fvg_size'])
        if 'min_confluence_score' in settings: set_entry(self.bt_min_conf, settings['min_confluence_score'])
        if 'anti_gap_mult' in settings: set_entry(self.bt_anti_gap_mult, settings['anti_gap_mult'])
        if 'trail_pct' in settings: set_entry(self.trail_pct_bt, settings['trail_pct'])
        if 'ml_min_confidence' in settings: set_entry(self.bt_ml_min_confidence, settings['ml_min_confidence'])

        if 'use_fixed_lot' in settings: set_var(self.backtest_use_fixed_lot, settings['use_fixed_lot'])
        if 'use_dynamic_rrr' in settings: set_var(self.use_dynamic_rrr_backtest, settings['use_dynamic_rrr'])
        if 'fvg_sl_mode' in settings: set_var(self.bt_fvg_sl_mode, settings['fvg_sl_mode'])
        if 'trade_on_all_tfs' in settings: set_var(self.trade_all_tfs_backtest, settings['trade_on_all_tfs'])
        if 'use_ultra_low_tf' in settings: set_var(self.use_ultra_low_tf_backtest, settings['use_ultra_low_tf'])
        if 'session_filter' in settings: set_var(self.bt_session_filter, settings['session_filter'])
        if 'use_htf_filter' in settings: set_var(self.htf_filter_backtest, settings['use_htf_filter'])
        if 'use_ote_filter' in settings: set_var(self.ote_filter_backtest, settings['use_ote_filter'])
        if 'bypass_htf_conf' in settings: set_var(self.bypass_htf_backtest, settings['bypass_htf_conf'])
        if 'trail_type' in settings: set_var(self.trail_type_bt, settings['trail_type'])
        if 'require_bos_fvg' in settings: set_var(self.bt_require_bos_fvg, settings['require_bos_fvg'])
        if 'fvg_displacement_only' in settings: set_var(self.bt_fvg_displacement_only, settings['fvg_displacement_only'])
        if 'fvg_discount_premium_only' in settings: set_var(self.bt_fvg_discount_premium_only, settings['fvg_discount_premium_only'])
        if 'fvg_recent_sweep_only' in settings: set_var(self.bt_fvg_recent_sweep_only, settings['fvg_recent_sweep_only'])
        if 'sb_require_htf_bias' in settings: set_var(self.bt_sb_require_htf_bias, settings['sb_require_htf_bias'])
        if 'anti_gap_enabled' in settings: set_var(self.bt_anti_gap, settings['anti_gap_enabled'])
        if 'slippage_recovery' in settings: set_var(self.bt_slippage_recovery_bt, settings['slippage_recovery'])
        if 'smart_optimize' in settings: set_var(self.bt_smart_optimize, settings['smart_optimize'])
        if 'ml_filter' in settings: set_var(self.bt_ml_filter, settings['ml_filter'])
        if 'pro_dol_tp' in settings and hasattr(self, 'bt_pro_dol_tp'): set_var(self.bt_pro_dol_tp, settings['pro_dol_tp'])
        if 'pro_killzone' in settings and hasattr(self, 'bt_pro_killzone'): set_var(self.bt_pro_killzone, settings['pro_killzone'])
        if 'pro_htf_poi' in settings and hasattr(self, 'bt_pro_htf_poi'): set_var(self.bt_pro_htf_poi, settings['pro_htf_poi'])
        if 'pro_mandatory' in settings and hasattr(self, 'bt_pro_mandatory'): set_var(self.bt_pro_mandatory, settings['pro_mandatory'])
        if 'pro_regime' in settings and hasattr(self, 'bt_pro_regime'): set_var(self.bt_pro_regime, settings['pro_regime'])
        if 'pro_ml_sizing' in settings and hasattr(self, 'bt_pro_ml_sizing'): set_var(self.bt_pro_ml_sizing, settings['pro_ml_sizing'])
        if 'pro_ml_rank' in settings and hasattr(self, 'bt_pro_ml_rank'): set_var(self.bt_pro_ml_rank, settings['pro_ml_rank'])
        if 'pro_multi_tf_conf' in settings and hasattr(self, 'bt_pro_multi_tf_conf'): set_var(self.bt_pro_multi_tf_conf, settings['pro_multi_tf_conf'])
        if 'pro_multi_tf_gate' in settings and hasattr(self, 'bt_pro_multi_tf_gate'): set_var(self.bt_pro_multi_tf_gate, settings['pro_multi_tf_gate'])
        if 'pro_regime_adaptive' in settings and hasattr(self, 'bt_pro_regime_adaptive'): set_var(self.bt_pro_regime_adaptive, settings['pro_regime_adaptive'])
        if 'use_mt5_data' in settings:
            if hasattr(self, 'bt_use_mt5_data'): set_var(self.bt_use_mt5_data, settings['use_mt5_data'])
            if hasattr(self, 'opt_use_mt5_data'): set_var(self.opt_use_mt5_data, settings['use_mt5_data'])
            config.OFFLINE_BACKTESTING = not settings['use_mt5_data']

        # Listboxes — skip when method override is active (methods/trails are not per-method settings)
        if not (hasattr(self, 'bt_use_method_overrides_var') and self.bt_use_method_overrides_var.get()):
            if 'ict_method' in settings:
                methods = settings['ict_method']
                if isinstance(methods, list):
                    self.bt_methods_listbox.selection_clear(0, tk.END)
                    all_methods = self.bt_methods_listbox.get(0, tk.END)
                    for idx, m in enumerate(all_methods):
                        if m in methods:
                            self.bt_methods_listbox.selection_set(idx)
                    # Synchronously invoke select handler to configure widgets and repopulate trailing listbox items
                    self.on_bt_methods_select(None)

            if 'trail_methods' in settings:
                trail_methods = settings['trail_methods']
                if isinstance(trail_methods, list) and hasattr(self, 'bt_trail_methods_listbox'):
                    self.bt_trail_methods_listbox.selection_clear(0, tk.END)
                    all_trails = self.bt_trail_methods_listbox.get(0, tk.END)
                    for idx, m in enumerate(all_trails):
                        if m in trail_methods:
                            self.bt_trail_methods_listbox.selection_set(idx)

    def check_live_symbol_count_for_overrides(self):
        symbols_str = self.live_symbols_entry.get()
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        if len(symbols) >= 1:
            self.live_use_overrides_cb.config(state=tk.NORMAL)
            if self.live_use_overrides_var.get():
                current_values = self.live_override_symbol_dropdown["values"]
                if list(current_values) != symbols:
                    self.live_override_symbol_dropdown["values"] = symbols
                    
                    # Initialize overrides for any new symbols that aren't in self.live_symbol_overrides yet
                    import json, os
                    disk_profiles = {}
                    profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbol_profiles.json")
                    if os.path.exists(profile_path):
                        try:
                            with open(profile_path, 'r', encoding="utf-8") as f:
                                disk_profiles = json.load(f)
                        except Exception:
                            pass
                    for sym in symbols:
                        if sym not in self.live_symbol_overrides:
                            self.live_symbol_overrides[sym] = self.resolve_live_symbol_overrides(sym, disk_profiles)
                            
                    # If the active symbol is no longer in the list, switch to the first symbol
                    active = self.live_override_symbol_dropdown.get()
                    if active not in symbols:
                        if symbols:
                            self.live_override_symbol_dropdown.set(symbols[0])
                            self.on_select_live_override_symbol(None)
                    else:
                        self.live_currently_selected_override_symbol = active
        else:
            if self.live_use_overrides_var.get():
                self.live_use_overrides_var.set(False)
                self.on_toggle_live_overrides()
            self.live_use_overrides_cb.config(state=tk.DISABLED)

    def on_toggle_live_overrides(self):
        symbols_str = self.live_symbols_entry.get()
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        
        if self.live_use_overrides_var.get():
            # Caching base settings
            self.live_base_cached_settings = self.get_current_ui_live_settings()
            
            # Load disk profiles
            import json, os
            disk_profiles = {}
            profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbol_profiles.json")
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, 'r', encoding="utf-8") as f:
                        disk_profiles = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading symbol_profiles.json: {e}")
            
            # Resolve/Merge for all symbols
            for sym in symbols:
                self.live_symbol_overrides[sym] = self.resolve_live_symbol_overrides(sym, disk_profiles)
            
            # Populate and enable dropdown
            self.live_override_symbol_dropdown["values"] = symbols
            self.live_override_symbol_dropdown.configure(state="readonly")
            self.live_use_method_overrides_cb.config(state=tk.NORMAL)
            if symbols:
                self.live_override_symbol_dropdown.set(symbols[0])
                self.live_currently_selected_override_symbol = symbols[0]
                self.apply_live_settings_to_ui(self.live_symbol_overrides[symbols[0]])
        else:
            # Save current overrides back to the active symbol if any
            if self.live_currently_selected_override_symbol:
                if self.live_use_method_overrides_var.get() and self.live_currently_selected_override_method:
                    self.save_live_method_settings(self.live_currently_selected_override_symbol, self.live_currently_selected_override_method)
                else:
                    self.live_symbol_overrides[self.live_currently_selected_override_symbol] = self.get_current_ui_live_settings()
            
            # Disable dropdown
            self.live_override_symbol_dropdown.set("")
            self.live_override_symbol_dropdown.configure(state="disabled")
            self.live_use_method_overrides_var.set(False)
            self.live_use_method_overrides_cb.config(state=tk.DISABLED)
            self.live_override_method_dropdown.set("")
            self.live_override_method_dropdown.configure(state="disabled")
            
            # Restore cached base settings
            if self.live_base_cached_settings:
                self.apply_live_settings_to_ui(self.live_base_cached_settings)
            
            self.live_currently_selected_override_symbol = None
            self.live_currently_selected_override_method = None

    def on_select_live_override_symbol(self, event):
        new_symbol = self.live_override_symbol_dropdown.get()
        prev_symbol = self.live_currently_selected_override_symbol
        if not new_symbol:
            return
            
        # Save current edits back to the previous selected symbol
        if prev_symbol:
            if self.live_use_method_overrides_var.get() and self.live_currently_selected_override_method:
                self.save_live_method_settings(prev_symbol, self.live_currently_selected_override_method)
            else:
                current_ui = self.get_current_ui_live_settings()
                existing_methods = self.live_symbol_overrides.get(prev_symbol, {}).get("methods", {})
                self.live_symbol_overrides[prev_symbol] = current_ui
                if existing_methods:
                    self.live_symbol_overrides[prev_symbol]["methods"] = existing_methods
            
        # Load new symbol's overrides
        if new_symbol in self.live_symbol_overrides:
            if self.live_use_method_overrides_var.get():
                active_method = self.live_override_method_dropdown.get()
                if active_method:
                    self.apply_live_settings_to_ui(self.get_merged_live_settings(new_symbol, active_method))
                    self.live_currently_selected_override_method = active_method
                else:
                    self.apply_live_settings_to_ui(self.live_symbol_overrides[new_symbol])
            else:
                self.apply_live_settings_to_ui(self.live_symbol_overrides[new_symbol])
            
        self.live_currently_selected_override_symbol = new_symbol

    def resolve_live_symbol_overrides(self, symbol, disk_profiles):
        # 1. Exact match in memory overrides
        if symbol in self.live_symbol_overrides:
            return self.live_symbol_overrides[symbol]
            
        # 2. Exact match in disk profiles
        if symbol in disk_profiles:
            profile = disk_profiles[symbol].copy()
            # Strip session-only keys that should not overwrite current UI listbox selections
            profile.pop('ict_method', None)
            profile.pop('trail_methods', None)
            return profile
            
        # 3. Partial/Fallback matches in disk profiles
        symbol_upper = symbol.upper()
        for k, v in disk_profiles.items():
            k_upper = k.upper()
            if k_upper in symbol_upper or symbol_upper in k_upper:
                profile = v.copy()
                profile.pop('ict_method', None)
                profile.pop('trail_methods', None)
                return profile
            if k_upper == "GOLD" and "XAU" in symbol_upper:
                profile = v.copy()
                profile.pop('ict_method', None)
                profile.pop('trail_methods', None)
                return profile
            if k_upper == "BTC" and "BTC" in symbol_upper:
                profile = v.copy()
                profile.pop('ict_method', None)
                profile.pop('trail_methods', None)
                return profile
                
        # 4. Fallback to cached base settings
        return self.live_base_cached_settings.copy()

    def on_toggle_live_method_overrides(self):
        symbol = self.live_currently_selected_override_symbol
        if not symbol:
            return
            
        if self.live_use_method_overrides_var.get():
            # Save base settings before loading method overrides
            current_ui = self.get_current_ui_live_settings()
            existing_methods = self.live_symbol_overrides.get(symbol, {}).get("methods", {})
            self.live_symbol_overrides[symbol] = current_ui
            if existing_methods:
                self.live_symbol_overrides[symbol]["methods"] = existing_methods
            
            # Populate override dropdown with ONLY the methods currently selected in the Methods listbox
            selected_methods = [self.live_methods_listbox.get(i) for i in self.live_methods_listbox.curselection()]
            if not selected_methods:
                selected_methods = config.CORE_METHODS[:]
            self.live_override_method_dropdown["values"] = selected_methods
            self.live_override_method_dropdown.configure(state="readonly")
            self.live_override_method_dropdown.set(selected_methods[0])
            self.live_currently_selected_override_method = selected_methods[0]
            
            self.apply_live_settings_to_ui(self.get_merged_live_settings(symbol, selected_methods[0]))
            # Update FVG-specific widget states based on the selected override method
            self._update_live_fvg_widgets_for_override(selected_methods[0])
        else:
            if self.live_currently_selected_override_method:
                self.save_live_method_settings(symbol, self.live_currently_selected_override_method)
                
            self.live_override_method_dropdown.set("")
            self.live_override_method_dropdown.configure(state="disabled")
            
            self.session_filter_live_cb.config(state=tk.NORMAL)
            self.session_start_live.config(state=tk.NORMAL)
            self.session_end_live.config(state=tk.NORMAL)
            
            self.apply_live_settings_to_ui(self.live_symbol_overrides[symbol])
            self.live_currently_selected_override_method = None

    def on_select_live_override_method(self, event):
        symbol = self.live_currently_selected_override_symbol
        new_method = self.live_override_method_dropdown.get()
        prev_method = self.live_currently_selected_override_method
        
        if symbol and prev_method:
            self.save_live_method_settings(symbol, prev_method)
            
        if symbol and new_method:
            self.apply_live_settings_to_ui(self.get_merged_live_settings(symbol, new_method))
            # Update FVG-specific widget states based on the newly selected method
            self._update_live_fvg_widgets_for_override(new_method)
            
        self.live_currently_selected_override_method = new_method

    def _update_live_fvg_widgets_for_override(self, method):
        """Enable/disable FVG-specific widgets based on the override method."""
        fvg_state = tk.NORMAL if method == "FVG Return" else tk.DISABLED
        for child in self.live_fvg_sl_frame.winfo_children():
            child.configure(state=fvg_state)
        self.live_require_bos_cb.config(state=fvg_state)
        for child in self.live_fvg_filters_frame.winfo_children():
            child.configure(state=fvg_state)
            
        session_state = tk.DISABLED if method == "Silver Bullet" else tk.NORMAL
        self.session_filter_live_cb.config(state=session_state)
        self.session_start_live.config(state=session_state)
        self.session_end_live.config(state=session_state)

    def save_live_method_settings(self, symbol, method):
        if not symbol or not method:
            return
        current_ui = self.get_current_ui_live_settings()
        base_settings = self.live_symbol_overrides.setdefault(symbol, {})
        
        # Check differences
        method_overrides = {}
        for k, v in current_ui.items():
            if k in ["methods", "ict_method", "trail_methods"]:
                continue
            if k in base_settings:
                if base_settings[k] != v:
                    method_overrides[k] = v
            else:
                method_overrides[k] = v
                
        if method_overrides:
            base_settings.setdefault("methods", {})[method] = method_overrides
        else:
            if "methods" in base_settings and method in base_settings["methods"]:
                del base_settings["methods"][method]

    def get_merged_live_settings(self, symbol, method):
        if not symbol:
            return {}
        base_settings = self.live_symbol_overrides.get(symbol, {}).copy()
        if method:
            method_settings = base_settings.get("methods", {}).get(method, {})
            for k, v in method_settings.items():
                if k != "methods":
                    base_settings[k] = v
        return base_settings

    def get_current_ui_live_settings(self):
        settings = {}
        # Entry widgets
        try: settings['risk_percent'] = float(self.risk_entry.get())
        except ValueError: settings['risk_percent'] = 1.0
        
        try: settings['fixed_lot'] = float(self.fixed_lot_entry.get())
        except ValueError: settings['fixed_lot'] = 0.1
        
        try: settings['min_rrr'] = float(self.min_rrr_entry.get())
        except ValueError: settings['min_rrr'] = 1.5
        
        try: settings['session_start'] = int(self.session_start_live.get())
        except ValueError: settings['session_start'] = 7
        
        try: settings['session_end'] = int(self.session_end_live.get())
        except ValueError: settings['session_end'] = 21
        
        try: settings['daily_loss_limit'] = float(self.live_daily_loss.get())
        except ValueError: settings['daily_loss_limit'] = 0.0
        
        try: settings['max_concurrent_trades'] = int(self.live_max_concurr.get())
        except ValueError: settings['max_concurrent_trades'] = 3
        
        try: settings['min_fvg_size'] = float(self.live_min_fvg.get())
        except ValueError: settings['min_fvg_size'] = 0.5
        
        try: settings['min_confluence_score'] = int(self.live_min_conf.get())
        except ValueError: settings['min_confluence_score'] = 1

        try: settings['pending_limit_expiry_hours'] = float(self.live_limit_expiry.get())
        except ValueError: settings['pending_limit_expiry_hours'] = 2.0
        
        try: settings['anti_gap_mult'] = float(self.anti_gap_atr_mult_live.get())
        except ValueError: settings['anti_gap_mult'] = 2.0
        
        try: settings['trail_pct'] = float(self.trail_pct_live.get())
        except ValueError: settings['trail_pct'] = 0.5
        
        try: settings['ml_min_confidence'] = float(self.ml_live_min_conf.get())
        except ValueError: settings['ml_min_confidence'] = 60.0

        # BooleanVar / StringVar / IntVar widgets
        settings['use_fixed_lot'] = (self.risk_mode_var.get() == "Fixed")
        settings['use_dynamic_rrr'] = self.use_dynamic_rrr_live.get()
        settings['fvg_sl_mode'] = self.live_fvg_sl_mode.get()
        settings['trade_on_all_tfs'] = self.trade_all_tfs_live.get()
        settings['use_ultra_low_tf'] = self.use_ultra_low_tf_live.get()
        settings['session_filter'] = self.session_filter_live.get()
        settings['use_htf_filter'] = self.htf_filter_live.get()
        settings['use_ote_filter'] = self.ote_filter_live.get()
        settings['bypass_htf_conf'] = self.bypass_htf_live.get()
        settings['trail_type'] = self.trail_type_live.get()
        settings['require_bos_fvg'] = self.live_require_bos_fvg.get()
        settings['fvg_displacement_only'] = self.live_fvg_displacement_only.get()
        settings['fvg_discount_premium_only'] = self.live_fvg_discount_premium_only.get()
        settings['fvg_recent_sweep_only'] = self.live_fvg_recent_sweep_only.get()
        settings['sb_require_htf_bias'] = self.live_sb_require_htf_bias.get()
        settings['anti_gap_enabled'] = self.anti_gap_enabled_live.get()
        settings['slippage_recovery'] = self.slippage_recovery_live.get()
        settings['ml_enabled'] = self.ml_live_enabled.get()
        settings['pro_dol_tp'] = self.pro_dol_tp.get() if hasattr(self, 'pro_dol_tp') else False
        settings['pro_killzone'] = self.pro_killzone.get() if hasattr(self, 'pro_killzone') else False
        settings['pro_htf_poi'] = self.pro_htf_poi.get() if hasattr(self, 'pro_htf_poi') else False
        settings['pro_mandatory'] = self.pro_mandatory.get() if hasattr(self, 'pro_mandatory') else False
        settings['pro_regime'] = self.pro_regime.get() if hasattr(self, 'pro_regime') else False
        settings['pro_ml_sizing'] = self.pro_ml_sizing.get() if hasattr(self, 'pro_ml_sizing') else False
        settings['pro_ml_rank'] = self.pro_ml_rank.get() if hasattr(self, 'pro_ml_rank') else False
        settings['pro_multi_tf_conf'] = self.pro_multi_tf_conf.get() if hasattr(self, 'pro_multi_tf_conf') else False
        settings['pro_multi_tf_gate'] = self.pro_multi_tf_gate.get() if hasattr(self, 'pro_multi_tf_gate') else False
        settings['pro_regime_adaptive'] = self.pro_regime_adaptive.get() if hasattr(self, 'pro_regime_adaptive') else False

        # Listboxes
        settings['ict_method'] = [self.live_methods_listbox.get(i) for i in self.live_methods_listbox.curselection()]
        if hasattr(self, 'live_trail_methods_listbox'):
            settings['trail_methods'] = [self.live_trail_methods_listbox.get(i) for i in self.live_trail_methods_listbox.curselection()]
        else:
            settings['trail_methods'] = []
            
        return settings

    def apply_live_settings_to_ui(self, settings):
        if not settings:
            return
            
        def set_entry(entry_widget, val):
            if entry_widget and val is not None:
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, str(val))

        def set_var(var_widget, val):
            if var_widget is not None and val is not None:
                var_widget.set(val)

        if 'risk_percent' in settings: set_entry(self.risk_entry, settings['risk_percent'])
        if 'fixed_lot' in settings: set_entry(self.fixed_lot_entry, settings['fixed_lot'])
        if 'min_rrr' in settings: set_entry(self.min_rrr_entry, settings['min_rrr'])
        if 'session_start' in settings: set_entry(self.session_start_live, settings['session_start'])
        if 'session_end' in settings: set_entry(self.session_end_live, settings['session_end'])
        if 'daily_loss_limit' in settings: set_entry(self.live_daily_loss, settings['daily_loss_limit'])
        if 'max_concurrent_trades' in settings: set_entry(self.live_max_concurr, settings['max_concurrent_trades'])
        if 'min_fvg_size' in settings: set_entry(self.live_min_fvg, settings['min_fvg_size'])
        if 'min_confluence_score' in settings: set_entry(self.live_min_conf, settings['min_confluence_score'])
        if 'pending_limit_expiry_hours' in settings: set_entry(self.live_limit_expiry, settings['pending_limit_expiry_hours'])
        if 'anti_gap_mult' in settings: set_entry(self.anti_gap_atr_mult_live, settings['anti_gap_mult'])
        if 'trail_pct' in settings: set_entry(self.trail_pct_live, settings['trail_pct'])
        if 'ml_min_confidence' in settings: set_entry(self.ml_live_min_conf, settings['ml_min_confidence'])

        if 'use_fixed_lot' in settings:
            set_var(self.risk_mode_var, "Fixed" if settings['use_fixed_lot'] else "Risk")
        if 'use_dynamic_rrr' in settings: set_var(self.use_dynamic_rrr_live, settings['use_dynamic_rrr'])
        if 'fvg_sl_mode' in settings: set_var(self.live_fvg_sl_mode, settings['fvg_sl_mode'])
        if 'trade_on_all_tfs' in settings: set_var(self.trade_all_tfs_live, settings['trade_on_all_tfs'])
        if 'use_ultra_low_tf' in settings: set_var(self.use_ultra_low_tf_live, settings['use_ultra_low_tf'])
        if 'session_filter' in settings: set_var(self.session_filter_live, settings['session_filter'])
        if 'use_htf_filter' in settings: set_var(self.htf_filter_live, settings['use_htf_filter'])
        if 'use_ote_filter' in settings: set_var(self.ote_filter_live, settings['use_ote_filter'])
        if 'bypass_htf_conf' in settings: set_var(self.bypass_htf_live, settings['bypass_htf_conf'])
        if 'trail_type' in settings: set_var(self.trail_type_live, settings['trail_type'])
        if 'require_bos_fvg' in settings: set_var(self.live_require_bos_fvg, settings['require_bos_fvg'])
        if 'fvg_displacement_only' in settings: set_var(self.live_fvg_displacement_only, settings['fvg_displacement_only'])
        if 'fvg_discount_premium_only' in settings: set_var(self.live_fvg_discount_premium_only, settings['fvg_discount_premium_only'])
        if 'fvg_recent_sweep_only' in settings: set_var(self.live_fvg_recent_sweep_only, settings['fvg_recent_sweep_only'])
        if 'sb_require_htf_bias' in settings: set_var(self.live_sb_require_htf_bias, settings['sb_require_htf_bias'])
        if 'anti_gap_enabled' in settings: set_var(self.anti_gap_enabled_live, settings['anti_gap_enabled'])
        if 'slippage_recovery' in settings: set_var(self.slippage_recovery_live, settings['slippage_recovery'])
        if 'ml_enabled' in settings: set_var(self.ml_live_enabled, settings['ml_enabled'])
        if 'pro_dol_tp' in settings and hasattr(self, 'pro_dol_tp'): set_var(self.pro_dol_tp, settings['pro_dol_tp'])
        if 'pro_killzone' in settings and hasattr(self, 'pro_killzone'): set_var(self.pro_killzone, settings['pro_killzone'])
        if 'pro_htf_poi' in settings and hasattr(self, 'pro_htf_poi'): set_var(self.pro_htf_poi, settings['pro_htf_poi'])
        if 'pro_mandatory' in settings and hasattr(self, 'pro_mandatory'): set_var(self.pro_mandatory, settings['pro_mandatory'])
        if 'pro_regime' in settings and hasattr(self, 'pro_regime'): set_var(self.pro_regime, settings['pro_regime'])
        if 'pro_ml_sizing' in settings and hasattr(self, 'pro_ml_sizing'): set_var(self.pro_ml_sizing, settings['pro_ml_sizing'])
        if 'pro_ml_rank' in settings and hasattr(self, 'pro_ml_rank'): set_var(self.pro_ml_rank, settings['pro_ml_rank'])
        if 'pro_multi_tf_conf' in settings and hasattr(self, 'pro_multi_tf_conf'): set_var(self.pro_multi_tf_conf, settings['pro_multi_tf_conf'])
        if 'pro_multi_tf_gate' in settings and hasattr(self, 'pro_multi_tf_gate'): set_var(self.pro_multi_tf_gate, settings['pro_multi_tf_gate'])
        if 'pro_regime_adaptive' in settings and hasattr(self, 'pro_regime_adaptive'): set_var(self.pro_regime_adaptive, settings['pro_regime_adaptive'])

        # Listboxes — skip when method override is active (methods/trails are not per-method settings)
        if not (hasattr(self, 'live_use_method_overrides_var') and self.live_use_method_overrides_var.get()):
            if 'ict_method' in settings:
                methods = settings['ict_method']
                if isinstance(methods, list):
                    self.live_methods_listbox.selection_clear(0, tk.END)
                    all_methods = self.live_methods_listbox.get(0, tk.END)
                    for idx, m in enumerate(all_methods):
                        if m in methods:
                            self.live_methods_listbox.selection_set(idx)
                    # Synchronously invoke select handler to configure widgets and repopulate trailing listbox items
                    self.on_live_methods_select(None)

            if 'trail_methods' in settings:
                trail_methods = settings['trail_methods']
                if isinstance(trail_methods, list) and hasattr(self, 'live_trail_methods_listbox'):
                    self.live_trail_methods_listbox.selection_clear(0, tk.END)
                    all_trails = self.live_trail_methods_listbox.get(0, tk.END)
                    for idx, m in enumerate(all_trails):
                        if m in trail_methods:
                            self.live_trail_methods_listbox.selection_set(idx)

    def on_bt_methods_select(self, event=None):
        selected = [self.bt_methods_listbox.get(i) for i in self.bt_methods_listbox.curselection()]
        fvg_state = tk.NORMAL if ("FVG Return" in selected or "Silver Bullet" in selected) else tk.DISABLED
        for child in self.bt_fvg_sl_frame.winfo_children():
            child.configure(state=fvg_state)
        self.bt_require_bos_cb.config(state=fvg_state)
        for child in self.bt_fvg_filters_frame.winfo_children():
            child.configure(state=fvg_state)
            
        if hasattr(self, 'bt_session_filter_cb'):
            session_state = tk.DISABLED if len(selected) == 1 and selected[0] == "Silver Bullet" else tk.NORMAL
            self.bt_session_filter_cb.config(state=session_state)
            self.bt_session_start.config(state=session_state)
            self.bt_session_end.config(state=session_state)
        if hasattr(self, 'bt_trail_methods_listbox'):
            # Save selection
            selected_trails = [self.bt_trail_methods_listbox.get(i) for i in self.bt_trail_methods_listbox.curselection()]
            self.bt_trail_methods_listbox.delete(0, tk.END)
            for item in selected:
                self.bt_trail_methods_listbox.insert(tk.END, item)
            # Restore selection if it still exists
            all_trails = self.bt_trail_methods_listbox.get(0, tk.END)
            for idx, item in enumerate(all_trails):
                if item in selected_trails:
                    self.bt_trail_methods_listbox.select_set(idx)

    def on_live_methods_select(self, event=None):
        selected = [self.live_methods_listbox.get(i) for i in self.live_methods_listbox.curselection()]
        fvg_state = tk.NORMAL if ("FVG Return" in selected or "Silver Bullet" in selected) else tk.DISABLED
        for child in self.live_fvg_sl_frame.winfo_children():
            child.configure(state=fvg_state)
        self.live_require_bos_cb.config(state=fvg_state)
        for child in self.live_fvg_filters_frame.winfo_children():
            child.configure(state=fvg_state)
            
        if hasattr(self, 'session_filter_live_cb'):
            session_state = tk.DISABLED if len(selected) == 1 and selected[0] == "Silver Bullet" else tk.NORMAL
            self.session_filter_live_cb.config(state=session_state)
            self.session_start_live.config(state=session_state)
            self.session_end_live.config(state=session_state)
        if hasattr(self, 'live_trail_methods_listbox'):
            # Save selection
            selected_trails = [self.live_trail_methods_listbox.get(i) for i in self.live_trail_methods_listbox.curselection()]
            self.live_trail_methods_listbox.delete(0, tk.END)
            for item in selected:
                self.live_trail_methods_listbox.insert(tk.END, item)
            # Restore selection if it still exists
            all_trails = self.live_trail_methods_listbox.get(0, tk.END)
            for idx, item in enumerate(all_trails):
                if item in selected_trails:
                    self.live_trail_methods_listbox.select_set(idx)


    def train_regime_csv(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Grid Opt CSVs with Regime Labels",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_paths:
            return
            
        try:
            from regime_manager import rm
            total_loaded = 0
            for fp in file_paths:
                loaded = rm.learn_from_grid_results(fp)
                total_loaded += loaded
                
            if total_loaded > 0:
                rm.save_json("regime_params.json")
                messagebox.showinfo("Success", f"Successfully trained {total_loaded} regime parameter sets and saved to regime_params.json!")
            else:
                messagebox.showwarning("Warning", "No valid regime data found in selected CSVs. Make sure you exported them with a regime label.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train regime: {e}")

if __name__ == "__main__":
    # Check for version bumps
    try:
        import version_dumper
        import logging
        logger = logging.getLogger()
        new_version = version_dumper.check_and_bump_version(args=[], custom_logger=logger)
        if new_version:
            import config
            config.BOT_VERSION = new_version
    except Exception as e:
        print(f"Version dumper error: {e}")

    app = TradingUI()
    app.mainloop()
