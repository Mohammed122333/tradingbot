"""Metrics computation for backtest results.

Extracted from backtester.py to reduce file size.
"""
import datetime

import numpy as np


def compute_metrics(executed_trades, initial_balance,
                    ml_filter=False, ml_total_signals=0,
                    ml_passed=0, ml_rejected=0):
    """Compute all backtest metrics from a list of executed trades.

    Returns the same ``metrics`` dict that ``combined_backtest`` expects.
    """
    total_trades = len(executed_trades)

    # Win rate based ONLY on SL/TP exits (trails excluded)
    sltp_trades = [t for t in executed_trades if t.get('outcome') in ('TP', 'SL')]
    sltp_total = len(sltp_trades)
    sltp_wins = sum(1 for t in sltp_trades if t.get('outcome') == 'TP')
    sltp_losses = sltp_total - sltp_wins
    sltp_win_rate = (sltp_wins / sltp_total * 100) if sltp_total > 0 else 0

    win_count = sum(1 for t in executed_trades if t['pnl'] > 0)
    loss_count = total_trades - win_count
    overall_win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    trail_trades = [t for t in executed_trades if t.get('outcome') not in ('TP', 'SL')]
    trail_total = len(trail_trades)
    trail_wins = sum(1 for t in trail_trades if t['pnl'] > 0)

    # Balance & Mark-to-Market Equity curve reconstruction
    sorted_exits = sorted(executed_trades, key=lambda x: x["exit_time"])
    balances = [initial_balance]
    timestamps = [sorted_exits[0]["entry_time"] - datetime.timedelta(days=1)] if sorted_exits else [datetime.datetime.now()]
    running_balance = initial_balance
    for t in sorted_exits:
        running_balance += t["pnl"]
        t["balance"] = running_balance
    
    # Event-based timeline reconstruction to capture floating equity mark-to-market
    events = []
    for t in executed_trades:
        # Include entry time, exit time, and floating adverse excursion if present
        entry_t = t.get("entry_time")
        exit_t = t.get("exit_time")
        final_pnl = t.get("pnl", 0.0)
        mae = t.get("mae_pnl", min(0.0, final_pnl))
        
        if entry_t:
            events.append((entry_t, 'ENTRY', mae, 0.0))
        if exit_t:
            events.append((exit_t, 'EXIT', 0.0, final_pnl))

    events.sort(key=lambda x: x[0])

    peak_equity = initial_balance
    running_realized = initial_balance
    max_drawdown = 0.0
    max_drawdown_pct = 0.0

    for ev_time, ev_type, ev_mae, ev_pnl in events:
        if ev_type == 'EXIT':
            running_realized += ev_pnl
            balances.append(running_realized)
            timestamps.append(ev_time)
            current_equity = running_realized
        else:
            # Entry/floating event: estimate mark-to-market equity during open position
            current_equity = running_realized + ev_mae

        if current_equity > peak_equity:
            peak_equity = current_equity

        dd_curr = peak_equity - current_equity
        if dd_curr > max_drawdown:
            max_drawdown = dd_curr

        dd_pct_curr = (dd_curr / peak_equity * 100) if peak_equity > 0 else 0.0
        if dd_pct_curr > max_drawdown_pct:
            max_drawdown_pct = dd_pct_curr

    peak_balance = peak_equity

    wins = [t['pnl'] for t in executed_trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in executed_trades if t['pnl'] <= 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    expectancy = (avg_win * (win_count / total_trades) + avg_loss * (loss_count / total_trades)) if total_trades > 0 else 0
    total_commission = sum(t.get('commission', 0) for t in executed_trades)
    net_profit = running_balance - initial_balance
    roi_pct = (net_profit / initial_balance * 100) if initial_balance > 0 else 0

    # Per-symbol breakdown
    symbol_stats = {}
    for t in executed_trades:
        s = t['symbol']
        if s not in symbol_stats:
            symbol_stats[s] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        symbol_stats[s]['trades'] += 1
        if t['pnl'] > 0:
            symbol_stats[s]['wins'] += 1
        symbol_stats[s]['pnl'] += t['pnl']

    # Per-TF breakdown
    tf_stats = {}
    for t in executed_trades:
        tf = t.get('timeframe', 'N/A')
        if tf not in tf_stats:
            tf_stats[tf] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        tf_stats[tf]['trades'] += 1
        if t['pnl'] > 0:
            tf_stats[tf]['wins'] += 1
        tf_stats[tf]['pnl'] += t['pnl']

    # Per-method breakdown
    method_stats = {}
    for t in executed_trades:
        m = t.get('ict_method', 'N/A')
        if m not in method_stats:
            method_stats[m] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        method_stats[m]['trades'] += 1
        if t['pnl'] > 0:
            method_stats[m]['wins'] += 1
        method_stats[m]['pnl'] += t['pnl']

    # Per-method max drawdown
    method_dd_balance = {}
    method_dd_peak = {}
    method_dd_max = {}
    for t in sorted_exits:
        m = t.get('ict_method', 'N/A')
        if m not in method_dd_balance:
            method_dd_balance[m] = initial_balance
            method_dd_peak[m] = initial_balance
            method_dd_max[m] = 0.0
        method_dd_balance[m] += t['pnl']
        if method_dd_balance[m] > method_dd_peak[m]:
            method_dd_peak[m] = method_dd_balance[m]
        dd = method_dd_peak[m] - method_dd_balance[m]
        if dd > method_dd_max[m]:
            method_dd_max[m] = dd

    for m in method_stats:
        method_stats[m]['max_dd'] = method_dd_max.get(m, 0.0)
        method_stats[m]['max_dd_pct'] = (method_dd_max.get(m, 0.0) / running_balance * 100) if running_balance > 0 else 0.0

    # Outcome breakdown
    outcome_stats = {}
    for t in executed_trades:
        o = t.get('outcome', 'N/A')
        if o not in outcome_stats:
            outcome_stats[o] = {'count': 0, 'pnl': 0.0}
        outcome_stats[o]['count'] += 1
        outcome_stats[o]['pnl'] += t['pnl']

    # Confluence score breakdown
    confluence_stats = {}
    for t in executed_trades:
        cs = t.get('confluence_score', 0)
        if cs not in confluence_stats:
            confluence_stats[cs] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        confluence_stats[cs]['trades'] += 1
        if t['pnl'] > 0:
            confluence_stats[cs]['wins'] += 1
        confluence_stats[cs]['pnl'] += t['pnl']

    # Consecutive wins/losses
    max_consec_wins = max_consec_losses = cur_wins = cur_losses = 0
    for t in executed_trades:
        if t['pnl'] > 0:
            cur_wins += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins = 0
        max_consec_wins = max(max_consec_wins, cur_wins)
        max_consec_losses = max(max_consec_losses, cur_losses)

    # Sharpe ratio (daily returns)
    if len(balances) > 2:
        daily_returns = np.diff(balances) / np.array(balances[:-1])
        sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0
    else:
        sharpe = 0

    # Average durations
    durations = []
    for t in executed_trades:
        try:
            d = t.get('duration', '0:00:00')
            if isinstance(d, str) and ':' in d:
                parts = d.split(':')
                if len(parts) == 3:
                    hrs = int(parts[0].split()[-1]) if ' ' in parts[0] else int(parts[0])
                else:
                    hrs = 0
                durations.append(hrs)
        except Exception:
            pass
    avg_duration_hrs = np.mean(durations) if durations else 0

    metrics = {
        "final_balance": running_balance, "peak_balance": peak_balance,
        "max_drawdown": max_drawdown, "max_drawdown_pct": max_drawdown_pct,
        "win_rate": sltp_win_rate,
        "overall_win_rate": overall_win_rate,
        "sltp_total": sltp_total, "sltp_wins": sltp_wins, "sltp_losses": sltp_losses,
        "trail_total": trail_total, "trail_wins": trail_wins,
        "total_trades": total_trades, "win_count": win_count, "loss_count": loss_count,
        "highest_win": max(wins) if wins else 0, "highest_loss": min(losses) if losses else 0,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "gross_profit": gross_profit, "gross_loss": gross_loss,
        "profit_factor": profit_factor, "expectancy": expectancy,
        "total_commission": total_commission,
        "net_profit": net_profit, "roi_pct": roi_pct,
        "sharpe_ratio": sharpe,
        "max_consec_wins": max_consec_wins, "max_consec_losses": max_consec_losses,
        "avg_duration_hrs": avg_duration_hrs,
        "symbol_stats": symbol_stats, "tf_stats": tf_stats,
        "method_stats": method_stats, "outcome_stats": outcome_stats,
        "confluence_stats": confluence_stats,
        "balance_history": balances, "timestamps": timestamps,
    }

    if ml_filter:
        metrics.update({
            "ml_total_signals": ml_total_signals,
            "ml_passed": ml_passed,
            "ml_rejected": ml_rejected,
            "ml_filter_rate": (ml_rejected / ml_total_signals * 100) if ml_total_signals > 0 else 0,
        })

    return metrics
