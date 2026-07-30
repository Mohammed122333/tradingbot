def fix_ui():
    with open('c:/Users/adeL/Desktop/GoodBot/Ver2/ui.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    correct_func = """    def _apply_selected_opt_to_vars(self):
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

        if trail_type == "None" or not trail_type:
            self.trail_type_bt.set(config.TRAIL_TYPE_PARTIAL)
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
        self.bypass_htf_backtest.set(self.opt_bypass_htf.get())
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
        if hasattr(self, 'bt_use_symbol_profiles'):
            self.bt_use_symbol_profiles.set(False)

        import tkinter.messagebox as messagebox
        messagebox.showinfo("Success", "Settings applied to Single Backtest tab!\\nSwitching to Backtest tab.")
        self.notebook.select(0)
"""
    new_lines = lines[:2490] + [correct_func + '\n'] + lines[2842:]

    with open('c:/Users/adeL/Desktop/GoodBot/Ver2/ui.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Done")

if __name__ == '__main__':
    fix_ui()
