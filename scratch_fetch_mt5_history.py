import MetaTrader5 as mt5
import pandas as pd
import datetime
import sys

print("Initializing MT5...")
if not mt5.initialize():
    print("MT5 initialize failed:", mt5.last_error())
    sys.exit(1)

from_date = datetime.datetime(2026, 7, 21, 0, 0, 0)
to_date = datetime.datetime(2026, 7, 25, 23, 59, 59)

deals = mt5.history_deals_get(from_date, to_date)
orders = mt5.history_orders_get(from_date, to_date)

print(f"Total deals found from {from_date} to {to_date}: {len(deals) if deals else 0}")
print(f"Total orders found from {from_date} to {to_date}: {len(orders) if orders else 0}")

if deals and len(deals) > 0:
    df_deals = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    if 'time' in df_deals.columns:
        df_deals['time_dt'] = pd.to_datetime(df_deals['time'], unit='s')
    print("\n=== MT5 DEALS LOG ===")
    print(df_deals[['ticket', 'order', 'time_dt', 'symbol', 'type', 'entry', 'volume', 'price', 'profit', 'comment']].to_string())

if orders and len(orders) > 0:
    df_orders = pd.DataFrame(list(orders), columns=orders[0]._asdict().keys())
    if 'time_setup' in df_orders.columns:
        df_orders['time_setup_dt'] = pd.to_datetime(df_orders['time_setup'], unit='s')
    if 'time_done' in df_orders.columns:
        df_orders['time_done_dt'] = pd.to_datetime(df_orders['time_done'], unit='s')
    print("\n=== MT5 ORDERS LOG ===")
    print(df_orders[['ticket', 'time_setup_dt', 'time_done_dt', 'symbol', 'type', 'state', 'volume_initial', 'price_open', 'comment']].to_string())
