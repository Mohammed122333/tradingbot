import os
import pickle
import logging
import pandas as pd
import numpy as np

import config

logger = logging.getLogger()

# ML imports (local only - no API/internet)
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.inspection import permutation_importance
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.inspection import permutation_importance
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False

ML_FEATURE_NAMES = [
    'Direction (Buy=1)', 'Method Index', 'RRR', 'Confluence Score',
    'HTF Aligned', 'OB Aligned', 'Entry Hour', 'Day of Week',
    'SL Distance %', 'TP Distance %', 'Has FVG', 'Has Sweep',
    'Has Displacement', 'Has Structure', 'In OTE Zone',
    'FVG Size Normalized',
    'Displacement Ratio',
    'Session Timing Strictness',
    'Premium Discount Depth',
    'Opposing Liq Distance',
    'Spread to ATR Ratio',
    'FVG Upper Wick Ratio',
    'FVG Lower Wick Ratio',
    'FVG Gap Ratio',
    'Is Silver Bullet Window',
    'Candle 3 Overlap Ratio'
]

# Global ML model holder — loaded once, used by live trading & backtest
_ml_live_model = {
    'loaded': False,
    'rf': None,
    'gb': None,
    'nn': None,
    'xgb': None,
    'lgbm': None,
    'cat': None,
    'best_model': None,       # The auto-selected best model pipeline
    'best_model_name': 'rf',  # Name of the best model
    'scaler': None,
    'win_class_idx': 0,
    'metadata': {},  # symbol, tf, train info
}

ML_MODEL_FILE = "ict_ml_model.pkl"


def ml_extract_features_from_trades(trades, symbol, timeframe_value):
    """Extract ICT-style feature vectors from completed backtest trades.
    Uses the same analysis infrastructure as the backtester to 'read' the chart
    the way ICT would: structure, FVGs, sweeps, OTE zones, displacement, etc.
    """
    features = []
    labels = []
    trade_meta = []  # Keep metadata for reporting

    for t in trades:
        if t.get('outcome') in ('CANCELLED', 'TIMEOUT'):
            continue

        # === Label: Win (TP/TRAIL) vs Loss (SL) ===
        is_win = 1 if t.get('outcome') in ('TP', 'TRAIL') else 0

        # === Core trade features ===
        entry_price = t.get('entry_price', 0)
        exit_price = t.get('exit_price', 0)
        sl_price = t.get('sl_price', 0)
        tp_price = t.get('tp_price', 0)
        rrr = t.get('rrr', 0)
        risk = t.get('risk', 0)
        reward = t.get('reward', 0)
        confluence = t.get('confluence_score', 0)

        # Direction encoding
        is_buy = 1 if t.get('trade_direction') == 'Buy' else 0

        # Method encoding (one-hot style via index)
        method = t.get('ict_method', 'Unknown')
        method_idx = config.ICT_METHODS.index(method) if method in config.ICT_METHODS else -1

        # HTF trend alignment
        htf = t.get('htf_trend', 'neutral')
        htf_aligned = 0
        if (is_buy and htf == 'bullish') or (not is_buy and htf == 'bearish'):
            htf_aligned = 1
        elif htf == 'neutral':
            htf_aligned = 0.5

        # OB type alignment with trade direction
        ob_type = t.get('order_block_type', 'neutral')
        ob_aligned = 1 if (
            (is_buy and ob_type == 'bullish') or
            (not is_buy and ob_type == 'bearish')
        ) else 0

        # Duration in hours
        dur_str = t.get('duration', '0:00:00')
        try:
            dur = pd.Timedelta(dur_str)
            dur_hours = dur.total_seconds() / 3600
        except Exception:
            dur_hours = 0

        # Entry hour (session timing — critical ICT concept)
        entry_time = t.get('entry_time')
        if isinstance(entry_time, pd.Timestamp):
            from detectors import convert_time_to_ny_hour
            entry_hour = convert_time_to_ny_hour(entry_time) + entry_time.minute / 60
        else:
            entry_hour = 12  # default

        # Day of week
        if isinstance(entry_time, pd.Timestamp):
            day_of_week = entry_time.dayofweek  # 0=Mon, 4=Fri
        else:
            day_of_week = 2

        # Risk-to-reward ratio capped
        rrr_capped = min(rrr, 10) if rrr else 0

        # SL distance as percentage of entry
        sl_dist_pct = abs(entry_price - sl_price) / entry_price * 100 if entry_price > 0 else 0

        # TP distance as percentage of entry
        tp_dist_pct = abs(tp_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

        # Confluence details parsing
        conf_details = t.get('confluence_details', '')
        has_fvg_conf = 1 if 'FVG' in conf_details else 0
        has_sweep_conf = 1 if 'Sweep' in conf_details or 'sweep' in conf_details else 0
        has_displacement = 1 if 'Displacement' in conf_details or 'displacement' in conf_details else 0
        has_structure = 1 if 'Structure' in conf_details or 'structure' in conf_details else 0
        has_ote = 1 if 'OTE' in conf_details else 0

        # 16. FVG Size Normalized
        fvg_size_normalized = (t.get('fvg_size', 0.0) / entry_price) * 1000 if entry_price > 0 else 0.0
        
        # 17. Displacement Ratio
        displacement_ratio = t.get('fvg_body_ratio', 1.0)
        
        # 18. Session Timing Strictness (min distance in hours to macro centers)
        dist_london = min(abs(entry_hour - 3.5), 24 - abs(entry_hour - 3.5))
        dist_am = min(abs(entry_hour - 10.5), 24 - abs(entry_hour - 10.5))
        dist_pm = min(abs(entry_hour - 14.5), 24 - abs(entry_hour - 14.5))
        session_timing_strictness = min(dist_london, dist_am, dist_pm)
        
        # 19. Premium Discount Depth
        premium_discount_depth = abs(entry_price - sl_price) / max(1e-8, abs(tp_price - sl_price)) if abs(tp_price - sl_price) > 0 else 0.5
        
        # 20. Opposing Liq Distance
        opposing_liq_distance = tp_dist_pct
        
        # 21. Spread to ATR Ratio (volatility-adjusted proxy)
        spread_to_atr_ratio = sl_dist_pct

        # 22. FVG Upper Wick Ratio
        fvg_upper_wick_ratio = t.get('fvg_upper_wick_ratio', 0.1)
        
        # 23. FVG Lower Wick Ratio
        fvg_lower_wick_ratio = t.get('fvg_lower_wick_ratio', 0.1)
        
        # 24. FVG Gap Ratio
        fvg_gap_ratio = t.get('fvg_gap_ratio', 0.3)
        
        # 25. Is Silver Bullet Window
        is_sb_window = t.get('fvg_is_sb_window', 0.0)
        
        # 26. Candle 3 Overlap/Body Ratio
        c3_body_ratio = t.get('fvg_c3_body_ratio', 0.5)

        feature_vec = [
            is_buy,                # 0: Direction
            method_idx,            # 1: Method index
            rrr_capped,            # 2: RRR
            confluence,            # 3: Confluence score
            htf_aligned,           # 4: HTF alignment
            ob_aligned,            # 5: OB alignment
            entry_hour,            # 6: Entry hour
            day_of_week,           # 7: Day of week
            sl_dist_pct,           # 8: SL distance %
            tp_dist_pct,           # 9: TP distance %
            has_fvg_conf,          # 10: Has FVG confluence
            has_sweep_conf,        # 11: Has sweep confluence
            has_displacement,      # 12: Has displacement
            has_structure,         # 13: Has structure confluence
            has_ote,               # 14: In OTE zone
            fvg_size_normalized,   # 15: FVG Size Normalized
            displacement_ratio,    # 16: Displacement Ratio
            session_timing_strictness, # 17: Session Timing Strictness
            premium_discount_depth, # 18: Premium/Discount Depth
            opposing_liq_distance,  # 19: Opposing Liq Distance
            spread_to_atr_ratio,   # 20: Spread to ATR Ratio
            fvg_upper_wick_ratio,  # 21: Upper wick ratio
            fvg_lower_wick_ratio,  # 22: Lower wick ratio
            fvg_gap_ratio,         # 23: Gap ratio
            is_sb_window,          # 24: SB window flag
            c3_body_ratio          # 25: Candle 3 body ratio
        ]

        features.append(feature_vec)
        labels.append(is_win)
        trade_meta.append({
            'method': method,
            'outcome': t.get('outcome'),
            'rrr': rrr,
            'entry_time': entry_time,
            'confluence': confluence,
            'htf_trend': htf,
            'direction': t.get('trade_direction'),
            'pnl_direction': (exit_price - entry_price) if is_buy else (entry_price - exit_price),
        })

    return np.array(features) if features else np.array([]), np.array(labels), trade_meta


def ml_train_and_analyze(trades, symbol, timeframe_value):
    """Train ML models on backtest trade data and produce ICT-style analysis.
    Returns a comprehensive report dict with method rankings, feature importance,
    model accuracy, and ICT-style trade reading explanations.
    """
    if not ML_AVAILABLE:
        return {'error': 'scikit-learn not installed. Run: pip install scikit-learn'}

    X, y, meta = ml_extract_features_from_trades(trades, symbol, timeframe_value)

    if len(X) < 20:
        return {'error': f'Not enough trades for ML analysis ({len(X)} trades, need 20+). Run a longer backtest.'}

    if len(np.unique(y)) < 2:
        return {'error': 'All trades have the same outcome. ML needs both wins and losses to learn.'}

    # === Model Training ===
    
    # Random Forest (robust, handles non-linear patterns)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=42, class_weight='balanced', n_jobs=-1
    )

    # Gradient Boosting (captures sequential patterns)
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, random_state=42
    )
    
    # Neural Network (Deep Learning alternative to CNN/LSTM for tabular features)
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
        max_iter=500, random_state=42
    )

    # State-of-the-Art XGBoost Classifier for pattern recognition
    xgb_available = False
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss', n_jobs=-1
        )
        xgb_available = True
    except Exception as e:
        logger.warning("[ML] XGBoost not available for training, fallback to standard models: %s", e)
        xgb = None

    # State-of-the-Art LightGBM Classifier
    lgb_available = False
    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_available = True
    except Exception as e:
        logger.warning("[ML] LightGBM not available for training: %s", e)
        lgbm = None

    # State-of-the-Art CatBoost Classifier
    cat_available = False
    try:
        from catboost import CatBoostClassifier
        cat = CatBoostClassifier(
            iterations=150, depth=6, learning_rate=0.08,
            random_seed=42, verbose=0, thread_count=-1
        )
        cat_available = True
    except Exception as e:
        logger.warning("[ML] CatBoost not available for training: %s", e)
        cat = None

    # TimeSeriesSplit ensures that the training set only uses past data
    cv = TimeSeriesSplit(n_splits=min(5, max(2, int(len(y) / 10))))
    
    # Create pipelines to scale during evaluation automatically
    rf_pipe = make_pipeline(StandardScaler(), rf)
    gb_pipe = make_pipeline(StandardScaler(), gb)
    nn_pipe = make_pipeline(StandardScaler(), nn)

    rf_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring='accuracy')
    gb_scores = cross_val_score(gb_pipe, X, y, cv=cv, scoring='accuracy')
    nn_scores = cross_val_score(nn_pipe, X, y, cv=cv, scoring='accuracy')

    xgb_scores = np.array([0.0])
    if xgb_available and xgb is not None:
        try:
            xgb_pipe = make_pipeline(StandardScaler(), xgb)
            xgb_scores = cross_val_score(xgb_pipe, X, y, cv=cv, scoring='accuracy')
        except Exception as e:
            logger.error("[ML] XGBoost CV failed: %s", e)

    lgb_scores = np.array([0.0])
    if lgb_available and lgbm is not None:
        try:
            lgb_pipe = make_pipeline(StandardScaler(), lgbm)
            lgb_scores = cross_val_score(lgb_pipe, X, y, cv=cv, scoring='accuracy')
        except Exception as e:
            logger.error("[ML] LightGBM CV failed: %s", e)

    cat_scores = np.array([0.0])
    if cat_available and cat is not None:
        try:
            cat_pipe = make_pipeline(StandardScaler(), cat)
            cat_scores = cross_val_score(cat_pipe, X, y, cv=cv, scoring='accuracy')
        except Exception as e:
            logger.error("[ML] CatBoost CV failed: %s", e)

    # Train final models on all data
    rf_pipe.fit(X, y)
    gb_pipe.fit(X, y)
    nn_pipe.fit(X, y)

    if xgb_available and xgb is not None:
        try:
            xgb_pipe.fit(X, y)
        except Exception as e:
            logger.error("[ML] XGBoost fit failed: %s", e)

    if lgb_available and lgbm is not None:
        try:
            lgb_pipe.fit(X, y)
        except Exception as e:
            logger.error("[ML] LightGBM fit failed: %s", e)

    if cat_available and cat is not None:
        try:
            cat_pipe.fit(X, y)
        except Exception as e:
            logger.error("[ML] CatBoost fit failed: %s", e)

    # Feature Importance (using permutation importance on the unscaled features directly with the pipeline)
    perm_imp = permutation_importance(rf_pipe, X, y, n_repeats=10, random_state=42)
    rf_importance = rf_pipe.named_steps['randomforestclassifier'].feature_importances_
    perm_importance = perm_imp.importances_mean

    # === Method-by-Method Analysis (ICT Style) ===
    method_analysis = {}
    for i, m in enumerate(meta):
        method = m['method']
        if method not in method_analysis:
            method_analysis[method] = {
                'total': 0, 'wins': 0, 'losses': 0,
                'rrrs': [], 'confluences': [], 'entry_hours': [],
                'htf_aligned_wins': 0, 'htf_aligned_total': 0,
                'ote_wins': 0, 'ote_total': 0,
                'fvg_wins': 0, 'fvg_total': 0,
                'displacement_wins': 0, 'displacement_total': 0,
                'sweep_wins': 0, 'sweep_total': 0,
                'pnl_pts': [],
                'buy_wins': 0, 'buy_total': 0,
                'sell_wins': 0, 'sell_total': 0,
                'best_hours': {}, 'best_days': {},
            }
        ma = method_analysis[method]
        ma['total'] += 1
        is_win = y[i]
        if is_win:
            ma['wins'] += 1
        else:
            ma['losses'] += 1
        ma['rrrs'].append(m['rrr'])
        ma['confluences'].append(m['confluence'])
        ma['pnl_pts'].append(m['pnl_direction'])

        # Direction breakdown
        if m['direction'] == 'Buy':
            ma['buy_total'] += 1
            if is_win: ma['buy_wins'] += 1
        else:
            ma['sell_total'] += 1
            if is_win: ma['sell_wins'] += 1

        # Entry hour analysis
        hour = int(X[i][6])
        ma['entry_hours'].append(hour)
        if hour not in ma['best_hours']:
            ma['best_hours'][hour] = {'wins': 0, 'total': 0}
        ma['best_hours'][hour]['total'] += 1
        if is_win:
            ma['best_hours'][hour]['wins'] += 1

        # Day of week analysis
        dow = int(X[i][7])
        if dow not in ma['best_days']:
            ma['best_days'][dow] = {'wins': 0, 'total': 0}
        ma['best_days'][dow]['total'] += 1
        if is_win:
            ma['best_days'][dow]['wins'] += 1

        # Confluence factor tracking
        if X[i][4] >= 0.75:  # HTF aligned
            ma['htf_aligned_total'] += 1
            if is_win: ma['htf_aligned_wins'] += 1
        if X[i][14]:  # OTE
            ma['ote_total'] += 1
            if is_win: ma['ote_wins'] += 1
        if X[i][10]:  # FVG
            ma['fvg_total'] += 1
            if is_win: ma['fvg_wins'] += 1
        if X[i][12]:  # Displacement
            ma['displacement_total'] += 1
            if is_win: ma['displacement_wins'] += 1
        if X[i][11]:  # Sweep
            ma['sweep_total'] += 1
            if is_win: ma['sweep_wins'] += 1

    # Sort methods by win rate
    method_rankings = []
    for method, stats in method_analysis.items():
        wr = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
        avg_rrr = np.mean(stats['rrrs']) if stats['rrrs'] else 0
        avg_pnl = np.mean(stats['pnl_pts']) if stats['pnl_pts'] else 0
        expectancy = (wr / 100 * avg_rrr) - ((1 - wr / 100) * 1.0)

        # Find best trading hours
        best_hours = sorted(
            [(h, d['wins'] / d['total'] * 100) for h, d in stats['best_hours'].items() if d['total'] >= 3],
            key=lambda x: -x[1]
        )[:3]

        # Find best days
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        best_days = sorted(
            [(day_names[d], data['wins'] / data['total'] * 100, data['total'])
             for d, data in stats['best_days'].items() if data['total'] >= 2],
            key=lambda x: -x[1]
        )[:3]

        method_rankings.append({
            'method': method,
            'total_trades': stats['total'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'win_rate': wr,
            'avg_rrr': avg_rrr,
            'avg_pnl_pts': avg_pnl,
            'expectancy': expectancy,
            'avg_confluence': np.mean(stats['confluences']) if stats['confluences'] else 0,
            'htf_wr': stats['htf_aligned_wins'] / stats['htf_aligned_total'] * 100 if stats['htf_aligned_total'] > 0 else 0,
            'ote_wr': stats['ote_wins'] / stats['ote_total'] * 100 if stats['ote_total'] > 0 else 0,
            'fvg_wr': stats['fvg_wins'] / stats['fvg_total'] * 100 if stats['fvg_total'] > 0 else 0,
            'disp_wr': stats['displacement_wins'] / stats['displacement_total'] * 100 if stats['displacement_total'] > 0 else 0,
            'sweep_wr': stats['sweep_wins'] / stats['sweep_total'] * 100 if stats['sweep_total'] > 0 else 0,
            'buy_wr': stats['buy_wins'] / stats['buy_total'] * 100 if stats['buy_total'] > 0 else 0,
            'sell_wr': stats['sell_wins'] / stats['sell_total'] * 100 if stats['sell_total'] > 0 else 0,
            'buy_total': stats['buy_total'],
            'sell_total': stats['sell_total'],
            'best_hours': best_hours,
            'best_days': best_days,
        })

    method_rankings.sort(key=lambda x: -x['expectancy'])

    # === ML Predictions: What the model learned ===
    # Use the trained model to identify the "ideal" trade profile
    rf_predictions = rf_pipe.predict(X)
    gb_predictions = gb_pipe.predict(X)

    # Probability of win for each trade
    rf_proba = rf_pipe.predict_proba(X)
    win_class_idx = list(rf.classes_).index(1) if 1 in rf.classes_ else 0
    win_probas = rf_proba[:, win_class_idx]

    # Find the feature thresholds that maximize win probability
    high_prob_mask = win_probas >= 0.65
    ideal_profile = {}
    if high_prob_mask.sum() >= 5:
        high_prob_X = X[high_prob_mask]
        ideal_profile = {
            'avg_confluence': np.mean(high_prob_X[:, 3]),
            'avg_rrr': np.mean(high_prob_X[:, 2]),
            'best_entry_hours': np.median(high_prob_X[:, 6]),
            'htf_alignment_avg': np.mean(high_prob_X[:, 4]),
            'ote_pct': np.mean(high_prob_X[:, 14]) * 100,
            'displacement_pct': np.mean(high_prob_X[:, 12]) * 100,
            'count': int(high_prob_mask.sum()),
        }

    # === Best-Model Auto-Selection ===
    model_scores = {
        'rf': rf_scores.mean(),
        'gb': gb_scores.mean(),
        'nn': nn_scores.mean(),
    }
    model_pipes = {
        'rf': rf_pipe,
        'gb': gb_pipe,
        'nn': nn_pipe,
    }
    if xgb_available and xgb is not None:
        model_scores['xgb'] = xgb_scores.mean()
        model_pipes['xgb'] = xgb_pipe
    if lgb_available and lgbm is not None:
        model_scores['lgbm'] = lgb_scores.mean()
        model_pipes['lgbm'] = lgb_pipe
    if cat_available and cat is not None:
        model_scores['cat'] = cat_scores.mean()
        model_pipes['cat'] = cat_pipe

    best_model_name = max(model_scores, key=model_scores.get)
    best_model_pipe = model_pipes[best_model_name]
    best_model_acc = model_scores[best_model_name] * 100

    _MODEL_DISPLAY_NAMES = {
        'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'nn': 'Neural Network (MLP)',
        'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'cat': 'CatBoost',
    }
    logger.info("[ML] 🏆 Best model auto-selected: %s (%.1f%% CV accuracy)",
                _MODEL_DISPLAY_NAMES.get(best_model_name, best_model_name), best_model_acc)

    return {
        'rf_accuracy': rf_scores.mean() * 100,
        'rf_accuracy_std': rf_scores.std() * 100,
        'gb_accuracy': gb_scores.mean() * 100,
        'gb_accuracy_std': gb_scores.std() * 100,
        'nn_accuracy': nn_scores.mean() * 100,
        'nn_accuracy_std': nn_scores.std() * 100,
        'xgb_accuracy': xgb_scores.mean() * 100 if xgb_available else 0.0,
        'xgb_accuracy_std': xgb_scores.std() * 100 if xgb_available else 0.0,
        'lgb_accuracy': lgb_scores.mean() * 100 if lgb_available else 0.0,
        'lgb_accuracy_std': lgb_scores.std() * 100 if lgb_available else 0.0,
        'cat_accuracy': cat_scores.mean() * 100 if cat_available else 0.0,
        'cat_accuracy_std': cat_scores.std() * 100 if cat_available else 0.0,
        'best_model_name': best_model_name,
        'best_model_accuracy': best_model_acc,
        'feature_importance': list(zip(ML_FEATURE_NAMES, rf_importance.tolist(), perm_importance.tolist())),
        'method_rankings': method_rankings,
        'total_trades': len(y),
        'overall_win_rate': np.mean(y) * 100,
        'avg_ml_conf_all': np.mean(win_probas) * 100,
        'avg_ml_conf_wins': np.mean(win_probas[y == 1]) * 100 if np.any(y == 1) else 0.0,
        'avg_ml_conf_losses': np.mean(win_probas[y == 0]) * 100 if np.any(y == 0) else 0.0,
        'ideal_profile': ideal_profile,
        'model_rf': rf_pipe,
        'model_gb': gb_pipe,
        'model_nn': nn_pipe,
        'model_xgb': xgb_pipe if xgb_available else None,
        'model_lgbm': lgb_pipe if lgb_available else None,
        'model_cat': cat_pipe if cat_available else None,
        'model_best': best_model_pipe,
    }


def ml_generate_ict_report(results):
    """Generate a human-readable ICT-style analysis report from ML results."""
    if 'error' in results:
        return f"❌ ML Analysis Error: {results['error']}"

    lines = []
    lines.append("=" * 80)
    lines.append("  🧠 ML ICT CHART READER — ANALYSIS REPORT")
    lines.append("  Reads charts like ICT using Machine Learning (100% Local)")
    lines.append("=" * 80)
    lines.append("")

    # Model accuracy
    _MODEL_DISPLAY_NAMES = {
        'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'nn': 'Neural Network (MLP)',
        'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'cat': 'CatBoost',
    }
    best_name = results.get('best_model_name', 'rf')
    best_display = _MODEL_DISPLAY_NAMES.get(best_name, best_name)

    lines.append("━━━ MODEL PERFORMANCE ━━━")
    lines.append(f"  Random Forest Accuracy:     {results['rf_accuracy']:.1f}% ± {results['rf_accuracy_std']:.1f}%")
    lines.append(f"  Gradient Boosting Accuracy: {results['gb_accuracy']:.1f}% ± {results['gb_accuracy_std']:.1f}%")
    lines.append(f"  Deep Learning (NN) Acc:     {results['nn_accuracy']:.1f}% ± {results['nn_accuracy_std']:.1f}%")
    if results.get('xgb_accuracy', 0.0) > 0:
        lines.append(f"  XGBoost Pattern Acc:        {results['xgb_accuracy']:.1f}% ± {results['xgb_accuracy_std']:.1f}%")
    if results.get('lgb_accuracy', 0.0) > 0:
        lines.append(f"  LightGBM Accuracy:          {results['lgb_accuracy']:.1f}% ± {results['lgb_accuracy_std']:.1f}%")
    if results.get('cat_accuracy', 0.0) > 0:
        lines.append(f"  CatBoost Accuracy:          {results['cat_accuracy']:.1f}% ± {results['cat_accuracy_std']:.1f}%")
    lines.append("")
    lines.append(f"  🏆 BEST MODEL SELECTED:     {best_display} ({results.get('best_model_accuracy', 0.0):.1f}%)")
    lines.append(f"     → This model will be used for live ML scoring")
    lines.append("")
    lines.append(f"  Total Trades Analyzed:      {results['total_trades']}")
    lines.append(f"  Overall Win Rate:           {results['overall_win_rate']:.1f}%")
    lines.append(f"  Avg ML Confidence (All):    {results.get('avg_ml_conf_all', 0.0):.1f}%")
    lines.append(f"  Avg ML Confidence (Wins):   {results.get('avg_ml_conf_wins', 0.0):.1f}%")
    lines.append(f"  Avg ML Confidence (Losses): {results.get('avg_ml_conf_losses', 0.0):.1f}%")
    lines.append("")

    # Walk-Forward results (if available)
    if results.get('walk_forward'):
        wf = results['walk_forward']
        lines.append("━━━ 🔄 WALK-FORWARD VALIDATION (Out-Of-Sample Reality Check) ━━━")
        lines.append(f"  Folds Completed:            {wf['n_folds']}")
        lines.append(f"  Avg OOS Accuracy:           {wf['avg_oos_accuracy']:.1f}%")
        lines.append(f"  Avg OOS Win Precision:      {wf['avg_oos_precision']:.1f}%")
        lines.append(f"  OOS Accuracy Std Dev:       {wf['oos_accuracy_std']:.1f}%")
        lines.append(f"  Best Fold OOS Accuracy:     {wf['best_fold_accuracy']:.1f}%")
        lines.append(f"  Worst Fold OOS Accuracy:    {wf['worst_fold_accuracy']:.1f}%")
        lines.append(f"  Best OOS Model (most wins): {_MODEL_DISPLAY_NAMES.get(wf['best_oos_model'], wf['best_oos_model'])}")
        lines.append("")
        # Per-fold details
        for i, fold in enumerate(wf['fold_details'], 1):
            train_pct = fold['train_size'] / (fold['train_size'] + fold['test_size']) * 100
            lines.append(f"  Fold {i}: Train={fold['train_size']} ({train_pct:.0f}%) | Test={fold['test_size']} | OOS Acc={fold['oos_accuracy']:.1f}% | Best={_MODEL_DISPLAY_NAMES.get(fold['best_model'], fold['best_model'])}")
        lines.append("")
        # Interpretation
        gap = results.get('best_model_accuracy', 0) - wf['avg_oos_accuracy']
        if gap < 3:
            lines.append("  ✅ EXCELLENT: CV and OOS accuracy are very close — low overfitting risk.")
        elif gap < 8:
            lines.append("  ⚠️ MODERATE: Some gap between CV and OOS — mild overfitting detected.")
            lines.append("     Consider using more training data or simpler model parameters.")
        else:
            lines.append("  ❌ WARNING: Large gap between CV and OOS — significant overfitting.")
            lines.append("     The model memorized training data. Use Walk-Forward results as the real accuracy.")
        lines.append("")

    # Feature importance (what ICT concepts matter most)
    lines.append("━━━ ICT CONCEPT IMPORTANCE (What Matters Most) ━━━")
    lines.append(f"  {'ICT Concept':<25} {'Model Imp.':<12} {'Actual Impact':<12}")
    lines.append("  " + "─" * 49)
    sorted_feats = sorted(results['feature_importance'], key=lambda x: -x[1])
    for name, imp, perm_imp in sorted_feats:
        stars = "★" * min(5, max(1, int(imp * 20 + 0.5)))
        lines.append(f"  {name:<25} {imp:.4f} {stars:<6}  {perm_imp:+.4f}")
    lines.append("")

    # Method-by-method ICT analysis
    lines.append("━━━ ICT METHOD RANKINGS (Best → Worst by Expectancy) ━━━")
    lines.append("")

    for rank, mr in enumerate(results['method_rankings'], 1):
        grade = "🏆" if mr['expectancy'] > 0.5 else ("✅" if mr['expectancy'] > 0 else "⚠️" if mr['expectancy'] > -0.3 else "❌")
        lines.append(f"  {'─' * 72}")
        lines.append(f"  {grade} #{rank}: {mr['method']}")
        lines.append(f"  {'─' * 72}")
        lines.append(f"    Trades: {mr['total_trades']}  |  Wins: {mr['wins']}  |  Losses: {mr['losses']}  |  Win Rate: {mr['win_rate']:.1f}%")
        lines.append(f"    Avg RRR: {mr['avg_rrr']:.2f}  |  Expectancy: {mr['expectancy']:+.3f}  |  Avg Confluence: {mr['avg_confluence']:.1f}")
        lines.append(f"    Avg PnL (pts): {mr['avg_pnl_pts']:+.4f}")
        lines.append("")

        # Direction bias
        if mr['buy_total'] > 0 and mr['sell_total'] > 0:
            lines.append(f"    📊 Direction: Buy WR={mr['buy_wr']:.0f}% ({mr['buy_total']} trades) | Sell WR={mr['sell_wr']:.0f}% ({mr['sell_total']} trades)")
            if abs(mr['buy_wr'] - mr['sell_wr']) > 15:
                better = "BUY" if mr['buy_wr'] > mr['sell_wr'] else "SELL"
                lines.append(f"       ➡ ML INSIGHT: Strong {better} bias detected — consider focusing on {better} setups")
        elif mr['buy_total'] > 0:
            lines.append(f"    📊 Direction: Only BUY trades ({mr['buy_total']}), WR={mr['buy_wr']:.0f}%")
        elif mr['sell_total'] > 0:
            lines.append(f"    📊 Direction: Only SELL trades ({mr['sell_total']}), WR={mr['sell_wr']:.0f}%")

        # Confluence impact
        conf_insights = []
        if mr.get('htf_wr', 0) > 0:
            conf_insights.append(f"HTF Aligned={mr['htf_wr']:.0f}%")
        if mr.get('ote_wr', 0) > 0:
            conf_insights.append(f"OTE Zone={mr['ote_wr']:.0f}%")
        if mr.get('fvg_wr', 0) > 0:
            conf_insights.append(f"FVG={mr['fvg_wr']:.0f}%")
        if mr.get('disp_wr', 0) > 0:
            conf_insights.append(f"Displacement={mr['disp_wr']:.0f}%")
        if mr.get('sweep_wr', 0) > 0:
            conf_insights.append(f"Sweep={mr['sweep_wr']:.0f}%")
        if conf_insights:
            lines.append(f"    🔗 Confluence WR: {' | '.join(conf_insights)}")

        # Best hours
        if mr['best_hours']:
            hours_str = ", ".join([f"{h}:00 ({wr:.0f}%)" for h, wr in mr['best_hours']])
            lines.append(f"    🕐 Best Hours: {hours_str}")

        # Best days
        if mr['best_days']:
            days_str = ", ".join([f"{d} ({wr:.0f}%, {n} trades)" for d, wr, n in mr['best_days']])
            lines.append(f"    📅 Best Days: {days_str}")

        # ICT-style reading
        lines.append("")
        lines.append("    📖 ICT Reading:")
        if mr['expectancy'] > 0.5:
            lines.append("       This is a HIGH-PROBABILITY setup. ICT would consider this a")
            lines.append("       'bread and butter' trade — consistent edge with good R:R.")
            if mr['avg_confluence'] >= 3:
                lines.append(f"       Multiple confluence factors align (avg {mr['avg_confluence']:.1f}/7),")
                lines.append("       confirming institutional order flow direction.")
        elif mr['expectancy'] > 0:
            lines.append("       Positive expectancy but moderate — this method works but needs")
            lines.append("       careful trade selection. Focus on higher confluence setups.")
            if mr['win_rate'] < 50 and mr['avg_rrr'] > 2:
                lines.append(f"       Low WR compensated by high RRR ({mr['avg_rrr']:.1f}x) — classic ICT sniper style.")
        elif mr['expectancy'] > -0.3:
            lines.append("       Marginally negative — this method is breakeven. The edge may")
            lines.append("       be present only under specific conditions (check hours/days above).")
        else:
            lines.append("       ❌ NEGATIVE expectancy — this method is LOSING money.")
            lines.append("       ICT would say: 'If the market structure doesn't support it,")
            lines.append("       don't force the trade.' Consider disabling this method.")
        lines.append("")

    # Ideal trade profile
    if results.get('ideal_profile'):
        ip = results['ideal_profile']
        lines.append("━━━ 🎯 IDEAL TRADE PROFILE (ML Predicted High-Win Trades) ━━━")
        lines.append(f"  Based on {ip['count']} trades with >65% predicted win probability:")
        lines.append(f"  • Confluence Score:  ≥ {ip['avg_confluence']:.1f}")
        lines.append(f"  • RRR Target:        {ip['avg_rrr']:.2f}")
        lines.append(f"  • Entry Around:      {int(ip['best_entry_hours'])}:00 New York time")
        lines.append(f"  • HTF Alignment:     {'Required' if ip['htf_alignment_avg'] > 0.6 else 'Helpful but not critical'}")
        lines.append(f"  • In OTE Zone:       {ip['ote_pct']:.0f}% of high-prob trades")
        lines.append(f"  • Has Displacement:  {ip['displacement_pct']:.0f}% of high-prob trades")
        lines.append("")

    # Summary recommendations
    lines.append("━━━ 🏁 ML RECOMMENDATIONS ━━━")
    winning_methods = [m for m in results['method_rankings'] if m['expectancy'] > 0]
    losing_methods = [m for m in results['method_rankings'] if m['expectancy'] <= -0.3]

    if winning_methods:
        lines.append(f"  ✅ WINNING methods ({len(winning_methods)}):")
        for m in winning_methods:
            lines.append(f"     • {m['method']}: {m['win_rate']:.0f}% WR, {m['avg_rrr']:.1f}x RRR, E[R]={m['expectancy']:+.2f}")
    if losing_methods:
        lines.append(f"  ❌ LOSING methods to DISABLE ({len(losing_methods)}):")
        for m in losing_methods:
            lines.append(f"     • {m['method']}: {m['win_rate']:.0f}% WR, {m['avg_rrr']:.1f}x RRR, E[R]={m['expectancy']:+.2f}")

    # Most important ICT concepts
    top_features = sorted(results['feature_importance'], key=lambda x: -x[1])[:5]
    lines.append("\n  🧠 Top 5 ICT Concepts by Predictive Power:")
    for name, imp, _ in top_features:
        lines.append(f"     • {name}: {imp:.4f} importance")

    lines.append("")

    # Walk-Forward recommendation
    if not results.get('walk_forward'):
        lines.append("  💡 TIP: Run Walk-Forward validation to get realistic OOS accuracy.")
        lines.append("     The CV accuracy above may be optimistic. Walk-Forward tests on")
        lines.append("     truly unseen future data windows to measure real-world performance.")
        lines.append("")

    lines.append("=" * 80)
    lines.append("  END OF ML ICT ANALYSIS")
    lines.append("=" * 80)

    return "\n".join(lines)


def ml_save_model(ml_results, filepath=None, extra_meta=None):
    """Save trained ML model to disk for later use in live trading."""
    if filepath is None:
        filepath = ML_MODEL_FILE
    best_name = ml_results.get('best_model_name', 'rf')
    save_data = {
        'rf': ml_results['model_rf'],
        'gb': ml_results['model_gb'],
        'nn': ml_results['model_nn'],
        'xgb': ml_results.get('model_xgb', None),
        'lgbm': ml_results.get('model_lgbm', None),
        'cat': ml_results.get('model_cat', None),
        'best_model': ml_results.get('model_best', ml_results['model_rf']),
        'best_model_name': best_name,
        'best_model_accuracy': ml_results.get('best_model_accuracy', 0),
        'rf_accuracy': ml_results.get('rf_accuracy', 0),
        'gb_accuracy': ml_results.get('gb_accuracy', 0),
        'nn_accuracy': ml_results.get('nn_accuracy', 0),
        'xgb_accuracy': ml_results.get('xgb_accuracy', 0),
        'lgb_accuracy': ml_results.get('lgb_accuracy', 0),
        'cat_accuracy': ml_results.get('cat_accuracy', 0),
        'overall_win_rate': ml_results.get('overall_win_rate', 0),
        'total_trades': ml_results.get('total_trades', 0),
        'method_rankings': ml_results.get('method_rankings', []),
        'ideal_profile': ml_results.get('ideal_profile', {}),
        'walk_forward': ml_results.get('walk_forward', None),
        'feature_names': ML_FEATURE_NAMES,
        'metadata': extra_meta or {},
    }
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    _MODEL_DISPLAY_NAMES = {
        'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'nn': 'Neural Network (MLP)',
        'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'cat': 'CatBoost',
    }
    logger.info("[ML] Model saved to %s (Best: %s %.1f%%, %d trades)",
                filepath, _MODEL_DISPLAY_NAMES.get(best_name, best_name),
                ml_results.get('best_model_accuracy', 0), ml_results.get('total_trades', 0))
    return filepath


def ml_load_model(filepath=None):
    """Load a trained ML model from disk into the global holder for live/backtest use."""
    global _ml_live_model
    if filepath is None:
        filepath = ML_MODEL_FILE
    if not os.path.exists(filepath):
        logger.error("[ML] Model file not found: %s", filepath)
        return False
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        _ml_live_model['rf'] = data['rf']
        _ml_live_model['gb'] = data['gb']
        _ml_live_model['nn'] = data['nn']
        _ml_live_model['xgb'] = data.get('xgb', None)
        _ml_live_model['lgbm'] = data.get('lgbm', None)
        _ml_live_model['cat'] = data.get('cat', None)

        # Load best model — fall back to RF for backward compatibility with old .pkl files
        best_name = data.get('best_model_name', 'rf')
        _ml_live_model['best_model_name'] = best_name
        if 'best_model' in data and data['best_model'] is not None:
            _ml_live_model['best_model'] = data['best_model']
        else:
            # Old model file without best_model — fall back to RF
            _ml_live_model['best_model'] = data['rf']
            _ml_live_model['best_model_name'] = 'rf'

        # Determine win class index from the RF pipe
        rf = data['rf'].named_steps['randomforestclassifier']
        _ml_live_model['win_class_idx'] = list(rf.classes_).index(1) if 1 in rf.classes_ else 0
        _ml_live_model['metadata'] = data.get('metadata', {})
        _ml_live_model['metadata']['rf_accuracy'] = data.get('rf_accuracy', 0)
        _ml_live_model['metadata']['xgb_accuracy'] = data.get('xgb_accuracy', 0)
        _ml_live_model['metadata']['lgb_accuracy'] = data.get('lgb_accuracy', 0)
        _ml_live_model['metadata']['cat_accuracy'] = data.get('cat_accuracy', 0)
        _ml_live_model['metadata']['best_model_name'] = best_name
        _ml_live_model['metadata']['best_model_accuracy'] = data.get('best_model_accuracy', 0)
        _ml_live_model['metadata']['total_trades'] = data.get('total_trades', 0)
        _ml_live_model['metadata']['method_rankings'] = data.get('method_rankings', [])
        _ml_live_model['loaded'] = True
        _MODEL_DISPLAY_NAMES = {
            'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'nn': 'Neural Network (MLP)',
            'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'cat': 'CatBoost',
        }
        logger.info("[ML] Model loaded from %s (Best: %s %.1f%%, trained on %d trades)",
                    filepath, _MODEL_DISPLAY_NAMES.get(best_name, best_name),
                    data.get('best_model_accuracy', 0), data.get('total_trades', 0))
        return True
    except Exception as e:
        logger.error("[ML] Failed to load model: %s", e)
        _ml_live_model['loaded'] = False
        return False

def ml_score_signal(trade_direction, method, confluence_score, htf_trend,
                    ob_type, entry_price, sl_price, tp_price, rrr,
                    entry_time, confluence_details="", fvg_size=0.0, fvg_body_ratio=1.0,
                    fvg_upper_wick_ratio=0.1, fvg_lower_wick_ratio=0.1,
                    fvg_gap_ratio=0.3, fvg_is_sb_window=0.0, fvg_c3_body_ratio=0.5,
                    best_model_name='RandomForest', symbol=None):
    """Score a trade signal through the loaded ML model.
    Returns: (win_probability, should_take) or (0.0, False) if no model loaded.
    """
    if not _ml_live_model.get('loaded'):
        import os
        from utils import CACHE_DIR
        
        loaded_symbol_model = False
        sym_path = os.path.join(CACHE_DIR, f"ml_model_{symbol}.pkl") if symbol else ""
        if sym_path and os.path.exists(sym_path):
            ml_load_model(sym_path)
            loaded_symbol_model = _ml_live_model.get('loaded', False)
            
        if not loaded_symbol_model:
            ml_load_model()
            
        if not _ml_live_model.get('loaded'):
            return 0.0, True  # No model = pass all signals through

    is_buy = 1 if trade_direction == 'Buy' else 0
    method_idx = config.ICT_METHODS.index(method) if method in config.ICT_METHODS else -1

    htf_aligned = 0
    if (is_buy and htf_trend == 'bullish') or (not is_buy and htf_trend == 'bearish'):
        htf_aligned = 1
    elif htf_trend == 'neutral':
        htf_aligned = 0.5

    ob_aligned = 1 if ((is_buy and ob_type == 'bullish') or (not is_buy and ob_type == 'bearish')) else 0

    if isinstance(entry_time, pd.Timestamp):
        from detectors import convert_time_to_ny_hour
        entry_hour = convert_time_to_ny_hour(entry_time) + entry_time.minute / 60
        day_of_week = entry_time.dayofweek
    else:
        entry_hour = 12
        day_of_week = 2

    rrr_capped = min(rrr, 10) if rrr else 0
    sl_dist_pct = abs(entry_price - sl_price) / entry_price * 100 if entry_price > 0 else 0
    tp_dist_pct = abs(tp_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

    conf_str = confluence_details if isinstance(confluence_details, str) else ", ".join(confluence_details) if confluence_details else ""
    has_fvg_conf = 1 if 'FVG' in conf_str else 0
    has_sweep_conf = 1 if 'Sweep' in conf_str or 'sweep' in conf_str else 0
    has_displacement = 1 if 'Displacement' in conf_str or 'displacement' in conf_str else 0
    has_structure = 1 if 'Structure' in conf_str or 'structure' in conf_str else 0
    has_ote = 1 if 'OTE' in conf_str else 0

    # 16. FVG Size Normalized
    fvg_size_normalized = (fvg_size / entry_price) * 1000 if entry_price > 0 else 0.0

    # 17. Displacement Ratio
    displacement_ratio = fvg_body_ratio

    # 18. Session Timing Strictness (min distance in hours to macro centers)
    dist_london = min(abs(entry_hour - 3.5), 24 - abs(entry_hour - 3.5))
    dist_am = min(abs(entry_hour - 10.5), 24 - abs(entry_hour - 10.5))
    dist_pm = min(abs(entry_hour - 14.5), 24 - abs(entry_hour - 14.5))
    session_timing_strictness = min(dist_london, dist_am, dist_pm)

    # 19. Premium Discount Depth
    premium_discount_depth = abs(entry_price - sl_price) / max(1e-8, abs(tp_price - sl_price)) if abs(tp_price - sl_price) > 0 else 0.5

    # 20. Opposing Liq Distance
    opposing_liq_distance = tp_dist_pct

    # 21. Spread to ATR Ratio (volatility-adjusted proxy)
    spread_to_atr_ratio = sl_dist_pct

    try:
        feature_vec = np.array([[
            is_buy,                # 0: Direction
            method_idx,            # 1: Method index
            rrr_capped,            # 2: RRR
            confluence_score,      # 3: Confluence score
            htf_aligned,           # 4: HTF alignment
            ob_aligned,            # 5: OB alignment
            entry_hour,            # 6: Entry hour
            day_of_week,           # 7: Day of week
            sl_dist_pct,           # 8: SL distance %
            tp_dist_pct,           # 9: TP distance %
            has_fvg_conf,          # 10: Has FVG confluence
            has_sweep_conf,        # 11: Has sweep confluence
            has_displacement,      # 12: Has displacement
            has_structure,         # 13: Has structure confluence
            has_ote,               # 14: In OTE zone
            fvg_size_normalized,   # 15: FVG Size Normalized
            displacement_ratio,    # 16: Displacement Ratio
            session_timing_strictness, # 17: Session Timing Strictness
            premium_discount_depth, # 18: Premium/Discount Depth
            opposing_liq_distance,  # 19: Opposing Liq Distance
            spread_to_atr_ratio,   # 20: Spread to ATR Ratio
            fvg_upper_wick_ratio,  # 21: Upper wick ratio
            fvg_lower_wick_ratio,  # 22: Lower wick ratio
            fvg_gap_ratio,         # 23: Gap ratio
            fvg_is_sb_window,      # 24: SB window flag
            fvg_c3_body_ratio      # 25: Candle 3 body ratio
        ]], dtype=float)
        
        feature_vec = np.nan_to_num(feature_vec)
    
        # Use the best auto-selected model for scoring
        best_pipe = _ml_live_model.get('best_model') or _ml_live_model['rf']
        if hasattr(best_pipe, 'n_features_in_') and feature_vec.shape[1] != best_pipe.n_features_in_:
            if feature_vec.shape[1] < best_pipe.n_features_in_:
                pad_width = best_pipe.n_features_in_ - feature_vec.shape[1]
                feature_vec = np.pad(feature_vec, ((0, 0), (0, pad_width)), mode='constant')
            else:
                feature_vec = feature_vec[:, :best_pipe.n_features_in_]
        
        proba = best_pipe.predict_proba(feature_vec)[0]
        win_class_idx = _ml_live_model['win_class_idx']
        win_class_idx = min(win_class_idx, len(proba) - 1)
        best_proba = proba[win_class_idx]
        
        return best_proba, True
    except Exception as e:
        logger.error("ml_score_signal exception: %s", e)
        return 0.0, False

def ml_score_signals_batch(trades):
    if not _ml_live_model.get('loaded') or _ml_live_model.get('rf') is None:
        import os
        from utils import CACHE_DIR
        
        # Try loading the symbol-specific model for the first trade in the batch
        loaded_symbol_model = False
        if trades:
            first_sym = trades[0].get('symbol', '')
            if first_sym:
                sym_path = os.path.join(CACHE_DIR, f"ml_model_{first_sym}.pkl")
                if os.path.exists(sym_path):
                    ml_load_model(sym_path)
                    loaded_symbol_model = _ml_live_model.get('loaded', False)
        
        # Try loading the default model automatically if symbol model not found/loaded
        if not loaded_symbol_model:
            ml_load_model()
            
        if not _ml_live_model.get('loaded') or _ml_live_model.get('rf') is None:
            return [0.0] * len(trades)

    if not trades:
        return []

    import pandas as pd
    import numpy as np
    from detectors import convert_time_to_ny_hour
    import config

    feature_vecs = []
    
    for trade in trades:
        trade_direction = trade["trade_direction"]
        method = trade["ict_method"]
        confluence_score = trade.get("confluence_score", 0)
        htf_trend = trade.get("htf_trend", "neutral")
        ob_type = trade.get("ob_type", "neutral")
        entry_price = trade["entry_price"]
        sl_price = trade["sl_price"]
        tp_price = trade["tp_price"]
        rrr = trade.get("rrr", trade.get("rr_ratio", 1.0))
        entry_time = trade["entry_time"]
        confluence_details = trade.get("confluence_details", "")
        fvg_size = trade.get("fvg_size", 0.0)
        fvg_body_ratio = trade.get("fvg_body_ratio", 1.0)
        fvg_upper_wick_ratio = trade.get("fvg_upper_wick_ratio", 0.1)
        fvg_lower_wick_ratio = trade.get("fvg_lower_wick_ratio", 0.1)
        fvg_gap_ratio = trade.get("fvg_gap_ratio", 0.3)
        fvg_is_sb_window = trade.get("fvg_is_sb_window", 0.0)
        fvg_c3_body_ratio = trade.get("fvg_c3_body_ratio", 0.5)

        is_buy = 1 if trade_direction == 'Buy' else 0
        method_idx = config.ICT_METHODS.index(method) if method in config.ICT_METHODS else -1

        htf_aligned = 0
        if (is_buy and htf_trend == 'bullish') or (not is_buy and htf_trend == 'bearish'):
            htf_aligned = 1
        elif htf_trend == 'neutral':
            htf_aligned = 0.5

        ob_aligned = 1 if ((is_buy and ob_type == 'bullish') or (not is_buy and ob_type == 'bearish')) else 0

        if isinstance(entry_time, pd.Timestamp):
            entry_hour = convert_time_to_ny_hour(entry_time) + entry_time.minute / 60
            day_of_week = entry_time.dayofweek
        else:
            entry_hour = 12
            day_of_week = 2

        rrr_capped = min(rrr, 10) if rrr else 0
        sl_dist_pct = abs(entry_price - sl_price) / entry_price * 100 if entry_price > 0 else 0
        tp_dist_pct = abs(tp_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

        conf_str = confluence_details if isinstance(confluence_details, str) else ", ".join(confluence_details) if confluence_details else ""
        has_fvg_conf = 1 if 'FVG' in conf_str else 0
        has_sweep_conf = 1 if 'Sweep' in conf_str or 'sweep' in conf_str else 0
        has_displacement = 1 if 'Displacement' in conf_str or 'displacement' in conf_str else 0
        has_structure = 1 if 'Structure' in conf_str or 'structure' in conf_str else 0
        has_ote = 1 if 'OTE' in conf_str else 0

        fvg_size_normalized = (fvg_size / entry_price) * 1000 if entry_price > 0 else 0.0
        displacement_ratio = fvg_body_ratio

        dist_london = min(abs(entry_hour - 3.5), 24 - abs(entry_hour - 3.5))
        dist_am = min(abs(entry_hour - 10.5), 24 - abs(entry_hour - 10.5))
        dist_pm = min(abs(entry_hour - 14.5), 24 - abs(entry_hour - 14.5))
        session_timing_strictness = min(dist_london, dist_am, dist_pm)

        premium_discount_depth = abs(entry_price - sl_price) / max(1e-8, abs(tp_price - sl_price)) if abs(tp_price - sl_price) > 0 else 0.5
        opposing_liq_distance = tp_dist_pct
        spread_to_atr_ratio = sl_dist_pct

        feature_vecs.append([
            is_buy, method_idx, rrr_capped, confluence_score, htf_aligned, ob_aligned,
            entry_hour, day_of_week, sl_dist_pct, tp_dist_pct, has_fvg_conf, has_sweep_conf,
            has_displacement, has_structure, has_ote, fvg_size_normalized, displacement_ratio,
            session_timing_strictness, premium_discount_depth, opposing_liq_distance,
            spread_to_atr_ratio, fvg_upper_wick_ratio, fvg_lower_wick_ratio,
            fvg_gap_ratio, fvg_is_sb_window, fvg_c3_body_ratio
        ])

    try:
        X_raw = np.array(feature_vecs, dtype=float)
        X_raw = np.nan_to_num(X_raw)
        
        # Use the best auto-selected model for batch scoring
        best_pipe = _ml_live_model.get('best_model') or _ml_live_model['rf']
        if hasattr(best_pipe, 'n_features_in_') and X_raw.shape[1] != best_pipe.n_features_in_:
            if X_raw.shape[1] < best_pipe.n_features_in_:
                pad_width = best_pipe.n_features_in_ - X_raw.shape[1]
                X_raw = np.pad(X_raw, ((0, 0), (0, pad_width)), mode='constant')
            else:
                X_raw = X_raw[:, :best_pipe.n_features_in_]
        
        probas = best_pipe.predict_proba(X_raw)
        win_class_idx = _ml_live_model['win_class_idx']
        win_class_idx = min(win_class_idx, probas.shape[1] - 1)
        
        best_probas = probas[:, win_class_idx]
    
        return best_probas.tolist()
    except Exception as e:
        logger.error("ml_score_signals_batch exception: %s", e)
        return [0.0] * len(trades)


def get_ml_risk_multiplier(win_prob):
    """ML Model Calibration & Dynamic Lot Sizing:
    Maps ML model win probability to dynamic position risk multiplier:
      - win_prob < 0.60  -> 0.00 (Reject low-probability setups)
      - win_prob < 0.75  -> 0.50 (Standard risk)
      - win_prob < 0.85  -> 1.00 (High conviction)
      - win_prob >= 0.85 -> 1.25 (Ultra conviction)
    """
    if win_prob <= 0.0:
        return 1.0  # Fallback if ML is uncalibrated/disabled
    if win_prob < 0.60:
        return 0.0
    elif win_prob < 0.75:
        return 0.5
    elif win_prob < 0.85:
        return 1.0
    else:
        return 1.25


def walk_forward_ml_analysis(trades, symbol, timeframe_value, n_folds=5, train_ratio=0.7):
    """Walk-Forward ML Validation — the gold standard for trading ML.
    
    Splits trades chronologically into rolling train/test windows:
      Fold 1: Train on [0..70%], Test on [70%..82%]
      Fold 2: Train on [0..76%], Test on [76%..88%]
      Fold 3: Train on [0..82%], Test on [82%..94%]
      ... etc.
    
    Each fold trains all available models on the training window, then evaluates
    accuracy on the unseen test window. This measures REAL out-of-sample performance.
    
    Returns a dict with per-fold results and aggregated OOS metrics.
    """
    if not ML_AVAILABLE:
        return {'error': 'scikit-learn not installed. Run: pip install scikit-learn'}

    X, y, meta = ml_extract_features_from_trades(trades, symbol, timeframe_value)

    if len(X) < 40:
        return {'error': f'Not enough trades for Walk-Forward ({len(X)} trades, need 40+). Run a longer backtest.'}

    if len(np.unique(y)) < 2:
        return {'error': 'All trades have the same outcome. Walk-Forward needs both wins and losses.'}

    n_total = len(X)
    min_train = max(20, int(n_total * 0.4))  # Minimum training size
    min_test = max(10, int(n_total * 0.05))   # Minimum test size

    import config
    ml_light_mode = getattr(config, 'ML_LIGHT_MODE', False)
    if ml_light_mode:
        n_folds = min(n_folds, 3)

    # Calculate fold boundaries
    # Each fold slides the train/test split forward
    test_size = max(min_test, int(n_total * (1 - train_ratio) / n_folds))
    
    fold_details = []
    all_oos_accuracies = []
    all_oos_precisions = []
    model_win_counts = {}  # Track which model wins each fold

    _MODEL_DISPLAY_NAMES = {
        'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'nn': 'Neural Network (MLP)',
        'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'cat': 'CatBoost',
    }

    for fold_idx in range(n_folds):
        # Expanding window: train grows, test slides forward
        test_end = n_total - (n_folds - fold_idx - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start

        if train_end < min_train or test_start >= test_end or test_end > n_total:
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # Train all available models on this fold
        fold_models = {}
        fold_scores = {}

        try:
            rf_fold = make_pipeline(StandardScaler(), RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                random_state=42, class_weight='balanced', n_jobs=-1
            ))
            rf_fold.fit(X_train, y_train)
            fold_models['rf'] = rf_fold
            fold_scores['rf'] = rf_fold.score(X_test, y_test)
        except Exception:
            pass

        if not ml_light_mode:
            try:
                gb_fold = make_pipeline(StandardScaler(), GradientBoostingClassifier(
                    n_estimators=150, max_depth=5, learning_rate=0.1,
                    min_samples_leaf=5, random_state=42
                ))
                gb_fold.fit(X_train, y_train)
                fold_models['gb'] = gb_fold
                fold_scores['gb'] = gb_fold.score(X_test, y_test)
            except Exception:
                pass

        if not ml_light_mode:
            try:
                nn_fold = make_pipeline(StandardScaler(), MLPClassifier(
                    hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                    max_iter=500, random_state=42
                ))
                nn_fold.fit(X_train, y_train)
                fold_models['nn'] = nn_fold
                fold_scores['nn'] = nn_fold.score(X_test, y_test)
            except Exception:
                pass

        try:
            from xgboost import XGBClassifier
            xgb_fold = make_pipeline(StandardScaler(), XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric='logloss', n_jobs=-1
            ))
            xgb_fold.fit(X_train, y_train)
            fold_models['xgb'] = xgb_fold
            fold_scores['xgb'] = xgb_fold.score(X_test, y_test)
        except Exception:
            pass

        if not ml_light_mode:
            try:
                from lightgbm import LGBMClassifier
                lgb_fold = make_pipeline(StandardScaler(), LGBMClassifier(
                    n_estimators=150, max_depth=6, learning_rate=0.08,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbose=-1
                ))
                lgb_fold.fit(X_train, y_train)
                fold_models['lgbm'] = lgb_fold
                fold_scores['lgbm'] = lgb_fold.score(X_test, y_test)
            except Exception:
                pass

        if not ml_light_mode:
            try:
                from catboost import CatBoostClassifier
                cat_fold = make_pipeline(StandardScaler(), CatBoostClassifier(
                    iterations=150, depth=6, learning_rate=0.08,
                    random_seed=42, verbose=0, thread_count=-1
                ))
                cat_fold.fit(X_train, y_train)
                fold_models['cat'] = cat_fold
                fold_scores['cat'] = cat_fold.score(X_test, y_test)
            except Exception:
                pass

        if not fold_scores:
            continue

        # Find best model for this fold
        best_fold_model = max(fold_scores, key=fold_scores.get)
        best_fold_acc = fold_scores[best_fold_model] * 100
        model_win_counts[best_fold_model] = model_win_counts.get(best_fold_model, 0) + 1

        # Calculate precision (how many predicted wins actually won)
        best_pipe = fold_models[best_fold_model]
        y_pred = best_pipe.predict(X_test)
        predicted_wins = y_pred == 1
        if predicted_wins.sum() > 0:
            precision = (y_test[predicted_wins] == 1).sum() / predicted_wins.sum() * 100
        else:
            precision = 0.0

        all_oos_accuracies.append(best_fold_acc)
        all_oos_precisions.append(precision)

        fold_details.append({
            'fold': fold_idx + 1,
            'train_size': len(y_train),
            'test_size': len(y_test),
            'oos_accuracy': best_fold_acc,
            'oos_precision': precision,
            'best_model': best_fold_model,
            'all_model_scores': {k: v * 100 for k, v in fold_scores.items()},
        })

        logger.info("[ML] Walk-Forward Fold %d: Train=%d, Test=%d, OOS Acc=%.1f%%, Best=%s",
                    fold_idx + 1, len(y_train), len(y_test), best_fold_acc,
                    _MODEL_DISPLAY_NAMES.get(best_fold_model, best_fold_model))

    if not fold_details:
        return {'error': 'Walk-Forward failed — not enough data for any fold.'}

    # Determine overall best OOS model (most fold wins)
    best_oos_model = max(model_win_counts, key=model_win_counts.get) if model_win_counts else 'rf'

    oos_accs = np.array(all_oos_accuracies)
    oos_precs = np.array(all_oos_precisions)

    result = {
        'n_folds': len(fold_details),
        'avg_oos_accuracy': oos_accs.mean(),
        'oos_accuracy_std': oos_accs.std(),
        'avg_oos_precision': oos_precs.mean(),
        'best_fold_accuracy': oos_accs.max(),
        'worst_fold_accuracy': oos_accs.min(),
        'best_oos_model': best_oos_model,
        'model_win_counts': model_win_counts,
        'fold_details': fold_details,
    }

    logger.info("[ML] 🔄 Walk-Forward Complete: %d folds, Avg OOS Acc=%.1f%% ± %.1f%%, Best OOS Model=%s",
                len(fold_details), oos_accs.mean(), oos_accs.std(),
                _MODEL_DISPLAY_NAMES.get(best_oos_model, best_oos_model))

    return result
