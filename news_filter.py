import os
import csv
import json
import urllib.request
import logging
from datetime import datetime, timedelta
import pytz

import config

logger = logging.getLogger(__name__)

# Cache for the live API to avoid spamming
_live_news_cache = []
_live_news_last_fetch = None
_LIVE_CACHE_TTL = 3600 * 12  # 12 hours

# Cache for historical CSV parsing
_historical_news_cache = []
_historical_loaded = False

def get_currencies_for_symbol(symbol):
    """Map a trading symbol to its relevant economies."""
    symbol = symbol.upper()
    currencies = []
    
    # Forex pairs
    if len(symbol) == 6:
        currencies = [symbol[:3], symbol[3:]]
        
    # Indices / Commodities mapping
    mapping = {
        "US30": ["USD"],
        "XAUUSD": ["USD"],
        "SPX500": ["USD"],
        "NAS100": ["USD"],
        "WS30": ["USD"],
        "BTCUSD": ["USD"],
        "ETHUSD": ["USD"],
    }
    
    # If it has a suffix like .r, strip it
    clean_sym = symbol.split('.')[0]
    
    if clean_sym in mapping:
        currencies = mapping[clean_sym]
    elif len(clean_sym) == 6:
        currencies = [clean_sym[:3], clean_sym[3:]]
    else:
        # Fallback for unknown stuff: assume USD matters
        currencies = ["USD"]
        
    return set(currencies)

def fetch_live_news():
    """Fetch current week's news from ForexFactory JSON API via Faireconomy."""
    global _live_news_cache, _live_news_last_fetch
    
    now = datetime.now()
    if _live_news_last_fetch and (now - _live_news_last_fetch).total_seconds() < _LIVE_CACHE_TTL:
        return _live_news_cache
        
    try:
        req = urllib.request.Request(
            'https://nfs.faireconomy.media/ff_calendar_thisweek.json',
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        parsed_events = []
        for row in data:
            if row.get('impact') == 'High':  # Only red folder
                # 'date' looks like: "2026-06-28T08:15:00-04:00"
                try:
                    dt = datetime.fromisoformat(row.get('date'))
                    parsed_events.append({
                        'time': dt,
                        'currency': row.get('country', '').upper(),
                        'title': row.get('title', '')
                    })
                except Exception:
                    pass
                    
        _live_news_cache = parsed_events
        _live_news_last_fetch = now
        logger.info(f"NewsFilter: Fetched {len(parsed_events)} High Impact live events.")
        
    except Exception as e:
        logger.error(f"NewsFilter: Failed to fetch live news - {e}")
        
    return _live_news_cache

def load_historical_news():
    """Load historical news from CSV if it exists."""
    global _historical_news_cache, _historical_loaded
    if _historical_loaded:
        return _historical_news_cache
        
    _historical_loaded = True
    csv_path = "historical_news.csv"
    if not os.path.exists(csv_path):
        logger.warning("NewsFilter: WARNING - historical_news.csv not found! Backtest news filter will be inactive until historical_news.csv is provided.")
        return []
        
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('impact', '').lower() == 'high':
                    # Expecting ISO format or standard UTC format in CSV
                    dt_str = row.get('date')
                    try:
                        # try basic isoformat, might be tz naive
                        dt = datetime.fromisoformat(dt_str)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=pytz.UTC)
                            
                        _historical_news_cache.append({
                            'time': dt,
                            'currency': row.get('country', '').upper(),
                            'title': row.get('title', '')
                        })
                    except Exception:
                        pass
        logger.info(f"NewsFilter: Loaded {len(_historical_news_cache)} High Impact historical events.")
    except Exception as e:
        logger.error(f"NewsFilter: Failed to read historical news CSV - {e}")
        
    return _historical_news_cache

def is_macro_news_embargo(symbol, current_time, is_live=False, buffer_mins=30):
    """
    Check if the `current_time` (naive broker time) is within `buffer_mins` 
    of a High Impact news event for the given symbol's currencies.
    """
    if not getattr(config, 'NEWS_FILTER_ENABLED', False):
        return False, ""
        
    buffer_td = timedelta(minutes=buffer_mins)
    
    # 1. Localize current_time to tz-aware
    broker_tz = pytz.timezone(getattr(config, 'BROKER_TIMEZONE', 'Etc/UTC'))
    try:
        if current_time.tzinfo is None:
            current_time_aware = broker_tz.localize(current_time)
        else:
            current_time_aware = current_time
    except Exception:
        current_time_aware = current_time.replace(tzinfo=pytz.UTC)
        
    # 2. Get relevant currencies
    currencies = get_currencies_for_symbol(symbol)
    
    # 3. Pick event list based on live or backtest
    events = fetch_live_news() if is_live else load_historical_news()
    if not events:
        return False, ""
        
    # 4. Check for embargo
    for event in events:
        if event['currency'] in currencies or event['currency'] == 'USD':
            event_time = event['time']
            diff = abs(current_time_aware - event_time)
            
            if diff <= buffer_td:
                return True, f"{event['currency']} News: {event['title']}"
                
    return False, ""
