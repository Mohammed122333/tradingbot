import cProfile
import pstats
import run_repro

# We don't want the UI to block, wait, run_repro uses Tkinter?
# Let's bypass Tkinter and just call backtester directly!

import json
import backtester

def run():
    print("Loading test settings...")
    with open("newtest.json", "r") as f:
        settings = json.load(f)
        
    print("Running backtest directly...")
    # Just run a short backtest
    backtester.combined_backtest(
        symbol="XAUUSDm",
        tf_name="M1",
        date_from="2023-01-01",
        date_to="2023-02-01",
        settings=settings,
        use_htf_filter=True,
        use_ote_filter=True
    )

if __name__ == "__main__":
    cProfile.run('run()', 'profile.stats')
    p = pstats.Stats('profile.stats')
    p.sort_stats('tottime').print_stats(30)
