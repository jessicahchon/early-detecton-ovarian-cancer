# Union Tier Feature Set + Model Evaluations
# RF, SVM, XGB, NN, EN, Stacking (RF + SVM + EN >>> LR meta)
# 7 Repeats x 10 Outer x 5 Inner, Threshold 0.5

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV, cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve)
from scipy.stats import randint, loguniform, uniform
from xgboost import XGBClassifier
import warnings
import time

warnings.filterwarnings('ignore')


def get_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'AUC': roc_auc_score(y_true, y_prob),
        'Acc': accuracy_score(y_true, y_pred),
        'Sens': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Spec': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'F1': f1_score(y_true, y_pred)
    }


# Model configs (consistent with all other pipelines)
MODEL_CONFIGS = {
    'RF': {
        'model_class': RandomForestClassifier,
        'param_distributions': {
            'n_estimators': randint(500, 1001),
            'max_depth': [3, 5, 7, 9],
            'max_features': ['sqrt', 0.5]
        },
        'fixed_params': {'random_state': 42, 'n_jobs': -1, 'bootstrap': True},
        'needs_scaling': False
    },
    'XGB': {
        'model_class': XGBClassifier,
        'param_distributions': {
            'n_estimators': randint(300, 601),
            'max_depth': [3, 5, 7, 9],
            'learning_rate': loguniform(0.001, 0.1)
        },
        'fixed_params': {'random_state': 42, 'eval_metric': 'logloss',
                         'subsample': 1.0, 'verbosity': 0},
        'needs_scaling': False
    },
    'SVM': {
        'model_class': SVC,
        'param_distributions': {
            'C': loguniform(0.1, 100),
            'gamma': loguniform(0.0001, 1.0),
            'kernel': ['linear', 'rbf']
        },
        'fixed_params': {'probability': True, 'random_state': 42},
        'needs_scaling': True
    },
    'EN': {
        'model_class': LogisticRegression,
        'param_distributions': {
            'C': loguniform(0.001, 10),
            'l1_ratio': uniform(0.0, 1.0)
        },
        'fixed_params': {'penalty': 'elasticnet', 'solver': 'saga',
                         'max_iter': 10000, 'random_state': 42},
        'needs_scaling': True
    },
    'NN': {
        'model_class': MLPClassifier,
        'param_distributions': {
            'hidden_layer_sizes': [(8,), (16,), (32,), (16, 8), (32, 16)],
            'alpha': [0.01, 0.1, 1.0]
        },
        'fixed_params': {'random_state': 42, 'max_iter': 500,
                         'early_stopping': True, 'validation_fraction': 0.1,
                         'n_iter_no_change': 10},
        'needs_scaling': True
    }
}

# Stacking base learner configs (RF + SVM + EN >>> LR meta)
STACK_BASE_CONFIGS = {
    'RF': MODEL_CONFIGS['RF'],
    'SVM': MODEL_CONFIGS['SVM'],
    'EN': MODEL_CONFIGS['EN']
}

# Configurations
N_REPEATS = 7
N_OUTER = 10
N_INNER = 5
N_ITER_SEARCH = 20
SEED = 42
THRESHOLD = 0.5

# Data 
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X_all = df[feat_names].values

# Load union feature frequencies
comp_df = pd.read_csv('Union_Feature_Frequencies.csv')

def get_union_at_threshold_fast(threshold):
    mask = ((comp_df['MW_Freq'] >= threshold) |
            (comp_df['EN_Freq'] >= threshold) |
            (comp_df['RF_Freq'] >= threshold) |
            (comp_df['XGB_Freq'] >= threshold) |
            (comp_df['CERP_Freq'] >= threshold))
    return sorted(comp_df.loc[mask, 'miRNA'].tolist())

# Define tiers
tier_thresholds = [70]
tier_defs = {}
for i, thresh in enumerate(tier_thresholds):
    feats = get_union_at_threshold_fast(thresh)
    if len(feats) > 0:
        tier_defs[f"Tier{i+1}_gte{thresh}"] = feats

print("Union Tier Feature Set + Model Evaluations")
print(f"\n Models: RF, XGB, SVM, EN, NN, Stacking")
print(f"\n {N_REPEATS}R x {N_OUTER}F outer x {N_INNER}F inner")

for name, feats in tier_defs.items():
    print(f"\n {name}: {len(feats)} features")

# Main loop
total_start = time.time()

for tier_name, tier_feats in tier_defs.items():
    n_feat = len(tier_feats)
    feat_idx = [i for i, f in enumerate(feat_names) if f in tier_feats]
    X = X_all[:, feat_idx]
    print(f"\n {tier_name}: {n_feat} features")
    
    # Individual models 
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n {tier_name} + {model_name}")
        model_results = []
        mean_fpr = np.linspace(0, 1, 100)
        model_tprs = []
        model_start = time.time()

        for repeat in range(N_REPEATS):
            outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True,
                                       random_state=SEED + repeat)
            for fold_idx, (tv_idx, te_idx) in enumerate(outer_cv.split(X, y)):
                X_tv, X_te = X[tv_idx], X[te_idx]
                y_tv, y_te = y[tv_idx], y[te_idx]

                # Scaling
                if config['needs_scaling']:
                    scaler = StandardScaler()
                    X_tv_use = scaler.fit_transform(X_tv)
                    X_te_use = scaler.transform(X_te)
                else:
                    X_tv_use, X_te_use = X_tv, X_te

                # Inner CV tuning
                inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                           random_state=SEED + repeat * 100 + fold_idx)
                search = RandomizedSearchCV(
                    config['model_class'](**config['fixed_params']),
                    param_distributions=config['param_distributions'],
                    n_iter=N_ITER_SEARCH, cv=inner_cv, scoring='roc_auc',
                    n_jobs=-1, random_state=SEED + repeat * 100 + fold_idx)
                search.fit(X_tv_use, y_tv)

                y_prob = search.best_estimator_.predict_proba(X_te_use)[:, 1]
                y_pred = (y_prob >= THRESHOLD).astype(int)
                m = get_metrics(y_te, y_pred, y_prob)
                m.update({'Repeat': repeat + 1, 'Outer_Fold': fold_idx + 1,
                          'N_Features': n_feat,
                          'Best_Params': str(search.best_params_)})
                model_results.append(m)

                fpr, tpr, _ = roc_curve(y_te, y_prob)
                model_tprs.append(np.interp(mean_fpr, fpr, tpr))
                model_tprs[-1][0] = 0.0

            print(f"\n R{repeat+1} done", flush=True)

        # Save
        res_df = pd.DataFrame(model_results)
        safe = tier_name.replace(' ', '_')
        res_df.to_csv(f'Union_{safe}_{model_name}_results.csv', index=False)

        mean_tpr = np.mean(model_tprs, axis=0); mean_tpr[-1] = 1.0
        pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv(
            f'Union_{safe}_{model_name}_ROC_Data.csv', index=False)

        rt = (time.time() - model_start) / 60
        print(f"\n {model_name}: AUC={res_df['AUC'].mean():.4f}±{res_df['AUC'].std():.4f} "
              f"Sens={res_df['Sens'].mean():.3f} Spec={res_df['Spec'].mean():.3f} "
              f"|S-S|={abs(res_df['Sens'].mean()-res_df['Spec'].mean()):.3f} "
              f"({rt:.1f}min)")

    # Stacking (RF + SVM + EN >>> LR) 
    print(f"\n {tier_name} + Stacking")
    stack_results = []
    stack_tprs = []
    stack_start = time.time()

    for repeat in range(N_REPEATS):
        outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True,
                                   random_state=SEED + repeat)
        for fold_idx, (tv_idx, te_idx) in enumerate(outer_cv.split(X, y)):
            X_tv, X_te = X[tv_idx], X[te_idx]
            y_tv, y_te = y[tv_idx], y[te_idx]

            scaler = StandardScaler()
            X_tv_sc = scaler.fit_transform(X_tv)
            X_te_sc = scaler.transform(X_te)

            inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                       random_state=SEED + repeat * 100 + fold_idx)

            oof_train = np.zeros((X_tv.shape[0], len(STACK_BASE_CONFIGS)))
            oof_test = np.zeros((X_te.shape[0], len(STACK_BASE_CONFIGS)))

            for i, (base_name, cfg) in enumerate(STACK_BASE_CONFIGS.items()):
                if cfg['needs_scaling']:
                    X_tv_base, X_te_base = X_tv_sc, X_te_sc
                else:
                    X_tv_base, X_te_base = X_tv, X_te

                search = RandomizedSearchCV(
                    cfg['model_class'](**cfg['fixed_params']),
                    param_distributions=cfg['param_distributions'],
                    n_iter=N_ITER_SEARCH, cv=inner_cv, scoring='roc_auc',
                    n_jobs=-1, random_state=SEED + repeat * 100 + fold_idx)
                search.fit(X_tv_base, y_tv)

                oof_train[:, i] = cross_val_predict(
                    search.best_estimator_, X_tv_base, y_tv,
                    cv=inner_cv, method='predict_proba')[:, 1]
                oof_test[:, i] = search.best_estimator_.predict_proba(X_te_base)[:, 1]

            meta = LogisticRegression(random_state=42, max_iter=10000)
            meta.fit(oof_train, y_tv)
            y_prob = meta.predict_proba(oof_test)[:, 1]
            y_pred = (y_prob >= THRESHOLD).astype(int)

            m = get_metrics(y_te, y_pred, y_prob)
            m.update({'Repeat': repeat + 1, 'Outer_Fold': fold_idx + 1,
                      'N_Features': n_feat})
            stack_results.append(m)

            fpr, tpr, _ = roc_curve(y_te, y_prob)
            stack_tprs.append(np.interp(mean_fpr, fpr, tpr))
            stack_tprs[-1][0] = 0.0

        print(f"\n R{repeat+1} done", flush=True)

    res_df = pd.DataFrame(stack_results)
    safe = tier_name.replace(' ', '_')
    res_df.to_csv(f'Union_{safe}_Stacking_results.csv', index=False)

    mean_tpr = np.mean(stack_tprs, axis=0); mean_tpr[-1] = 1.0
    pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv(
        f'Union_{safe}_Stacking_ROC_Data.csv', index=False)

    rt = (time.time() - stack_start) / 60
    print(f"\n Stacking: AUC={res_df['AUC'].mean():.4f}±{res_df['AUC'].std():.4f} "
          f"Sens={res_df['Sens'].mean():.3f} Spec={res_df['Spec'].mean():.3f} "
          f"|S-S|={abs(res_df['Sens'].mean()-res_df['Spec'].mean()):.3f} "
          f"({rt:.1f}min)")

print(f"\n Total runtime: {(time.time()-total_start)/60:.1f} min")
