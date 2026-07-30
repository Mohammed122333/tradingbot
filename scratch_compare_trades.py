import MetaTrader5 as mt5
import pandas as pd
import datetime

if not mt5.initialize():
    print("MT5 initialize failed")
    exit(1)

from_date = datetime.datetime(2026, 7, 21, 0, 0, 0)
to_date = datetime.datetime(2026, 7, 25, 23, 59, 59)

deals = mt5.history_deals_get(from_date, to_date)
if deals:
    df_deals = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    df_deals['time_dt'] = pd.to_datetime(df_deals['time'], unit='s')
    # Filter out deposit/withdrawal deals (entry 0 is IN, 1 is OUT, 2 is IN/OUT)
    # Exclude deals with 0 volume or non-trade types
    trade_deals = df_deals[(df_deals['symbol'] == 'XAUUSDm') & (df_deals['entry'].isin([0, 1]))].copy()
    
    print(f"Total MT5 XAUUSDm Deals (21-7 to 25-7): {len(trade_deals)}")
    
    # Calculate net profit from MT5 deals
    total_profit = trade_deals['profit'].sum() + trade_deals['swap'].sum() + trade_deals['commission'].sum()
    in_deals = trade_deals[trade_deals['entry'] == 0]
    out_deals = trade_deals[trade_deals['entry'] == 1]
    
    print(f"MT5 Total Executed Entry Positions (IN deals): {len(in_deals)}")
    print(f"MT5 Total Closed Positions (OUT deals): {len(out_deals)}")
    print(f"MT5 Total Net Profit: ${total_profit:.2f}")
    
    print("\n--- MT5 EXECUTED ENTRY DEALS (IN) ---")
    for idx, d in in_deals.iterrows():
        print(f"Deal {d['ticket']} | Order {d['order']} | {d['time_dt']} | Dir: {'BUY' if d['type']==0 else 'SELL'} | Vol: {d['volume']} | Price: {d['price']:.2f} | Comment: {d['comment']}")
        
    print("\n--- MT5 EXECUTED EXIT DEALS (OUT) ---")
    for idx, d in out_deals.iterrows():
        print(f"Deal {d['ticket']} | Position {d['position_id']} | {d['time_dt']} | Dir: {'BUY' if d['type']==0 else 'SELL'} | Vol: {d['volume']} | Price: {d['price']:.2f} | Profit: ${d['profit']:.2f} | Comment: {d['comment']}")

