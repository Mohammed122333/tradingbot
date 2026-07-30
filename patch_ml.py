with open('ml_engine.py', 'r') as f:
    content = f.read()

content = content.replace(
"""    # Calculate fold boundaries
    # Each fold slides the train/test split forward
    test_size = max(min_test, int(n_total * (1 - train_ratio) / n_folds))""",
"""    import config
    ml_light_mode = getattr(config, 'ML_LIGHT_MODE', False)
    if ml_light_mode:
        n_folds = min(n_folds, 3)

    # Calculate fold boundaries
    # Each fold slides the train/test split forward
    test_size = max(min_test, int(n_total * (1 - train_ratio) / n_folds))"""
)

# GB
content = content.replace(
"""        try:
            gb_fold = make_pipeline(StandardScaler(), GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                min_samples_leaf=5, random_state=42
            ))
            gb_fold.fit(X_train, y_train)
            fold_models['gb'] = gb_fold
            fold_scores['gb'] = gb_fold.score(X_test, y_test)
        except Exception:
            pass""",
"""        if not ml_light_mode:
            try:
                gb_fold = make_pipeline(StandardScaler(), GradientBoostingClassifier(
                    n_estimators=150, max_depth=5, learning_rate=0.1,
                    min_samples_leaf=5, random_state=42
                ))
                gb_fold.fit(X_train, y_train)
                fold_models['gb'] = gb_fold
                fold_scores['gb'] = gb_fold.score(X_test, y_test)
            except Exception:
                pass"""
)

# NN
content = content.replace(
"""        try:
            nn_fold = make_pipeline(StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                max_iter=500, random_state=42
            ))
            nn_fold.fit(X_train, y_train)
            fold_models['nn'] = nn_fold
            fold_scores['nn'] = nn_fold.score(X_test, y_test)
        except Exception:
            pass""",
"""        if not ml_light_mode:
            try:
                nn_fold = make_pipeline(StandardScaler(), MLPClassifier(
                    hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                    max_iter=500, random_state=42
                ))
                nn_fold.fit(X_train, y_train)
                fold_models['nn'] = nn_fold
                fold_scores['nn'] = nn_fold.score(X_test, y_test)
            except Exception:
                pass"""
)

# LGBM
content = content.replace(
"""        try:
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
            pass""",
"""        if not ml_light_mode:
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
                pass"""
)

# CAT
content = content.replace(
"""        try:
            from catboost import CatBoostClassifier
            cat_fold = make_pipeline(StandardScaler(), CatBoostClassifier(
                iterations=150, depth=6, learning_rate=0.08,
                random_seed=42, verbose=0, thread_count=-1
            ))
            cat_fold.fit(X_train, y_train)
            fold_models['cat'] = cat_fold
            fold_scores['cat'] = cat_fold.score(X_test, y_test)
        except Exception:
            pass""",
"""        if not ml_light_mode:
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
                pass"""
)

with open('ml_engine.py', 'w') as f:
    f.write(content)

print("Patch applied cleanly.")
