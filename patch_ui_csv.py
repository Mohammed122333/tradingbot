with open('ui.py', 'r') as f:
    content = f.read()

content = content.replace(
"""                f.write(f"HTF Filter: {values[11]}\\n")
            messagebox.showinfo("Success", f"Optimization report saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {e}")""",
"""                f.write(f"HTF Filter: {values[11]}\\n")
            messagebox.showinfo("Success", f"Optimization report saved to {file_path}")
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
            messagebox.showerror("Error", f"Failed to export CSV: {e}")"""
)

with open('ui.py', 'w') as f:
    f.write(content)
print("Added export_opt_csv to ui.py")
