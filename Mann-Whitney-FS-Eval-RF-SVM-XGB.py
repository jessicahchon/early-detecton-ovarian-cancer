# Mann-Whitney FS + RF, SVM, XGBoost Evaluation
# 7 Repeats x 10 Outer x 5 Inner, Threshold 0.5

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from scipy.stats import mannwhitneyu
from itertools import product
import warnings
import time

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("Mann-Whitney FS + RF, SVM, XGBoost")
print("  Feature Selection: Mann-Whitney U test (p < 0.05)")
print("  Inner CV: Hyperparameter tuning (AUC criterion)")
print("  Threshold: 0.5 fixed")
print("  Structure: 7 Repeats x 10 Outer x 5 Inner")
print("="*80, flush=True)

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

def mannwhitney_selection(X, y, alpha=0.05):
    n_features = X.shape[1]
    p_values = np.ones(n_features)
    cancer_mask = y == 1
    control_mask = y == 0
    for i in range(n_features):
        cancer_vals = X[cancer_mask, i]
        control_vals = X[control_mask, i]
        if np.std(cancer_vals) == 0 and np.std(control_vals) == 0:
            continue
        try:
            _, p = mannwhitneyu(cancer_vals, control_vals, alternative='two-sided')
            p_values[i] = p
        except:
            continue
    selected_idx = np.where(p_values < alpha)[0]
    return selected_idx, p_values

MODEL_CONFIGS = {
    'RF': {
        'model_class': RandomForestClassifier,
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_leaf': [1, 5, 9]
        },
        'fixed_params': {'random_state': 42, 'n_jobs': -1}
    },
    'SVM': {
        'model_class': SVC,
        'param_grid': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'fixed_params': {'probability': True, 'random_state': 42}
    },
    'XGB': {
        'model_class': XGBClassifier,
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        },
        'fixed_params': {'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0}
    }
}

# Config
N_REPEATS = 7
N_OUTER = 10
N_INNER = 5
ALPHA = 0.05
THRESHOLD = 0.5

# Data
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X = df[feat_names].values

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class: {np.sum(y==1)} Cancer, {np.sum(y==0)} Control")
print(f"Models: {list(MODEL_CONFIGS.keys())}", flush=True)

# Main loop
all_model_results = {name: [] for name in MODEL_CONFIGS}
mean_fpr = np.linspace(0, 1, 100)
all_tprs = {name: [] for name in MODEL_CONFIGS}

# Per-repeat accumulators
rep_collectors = {name: {'y_true': [], 'y_prob': [], 'y_pred': []}
                  for name in MODEL_CONFIGS}

repeat_summaries = {name: [] for name in MODEL_CONFIGS}

total_start = time.time()

for repeat in range(N_REPEATS):
    print(f"\nREPEAT {repeat+1}/{N_REPEATS}")

    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + repeat)

    # Reset repeat accumulators
    for name in MODEL_CONFIGS:
        rep_collectors[name] = {'y_true': [], 'y_prob': [], 'y_pred': []}

    for outer_fold, (trainval_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"  Fold {outer_fold+1}/{N_OUTER}", end="", flush=True)

        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx], y[test_idx]

        # Mann-Whitney FS on training data only
        selected_idx, _ = mannwhitney_selection(X_trainval, y_trainval, alpha=ALPHA)
        n_selected = len(selected_idx)
        if n_selected == 0:
            selected_idx = np.argsort(_)[:50]
            n_selected = 50

        X_trainval_sel = X_trainval[:, selected_idx]
        X_test_sel = X_test[:, selected_idx]

        scaler = StandardScaler()
        X_trainval_sc = scaler.fit_transform(X_trainval_sel)
        X_test_sc = scaler.transform(X_test_sel)

        for model_name, config in MODEL_CONFIGS.items():
            inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                       random_state=42 + repeat * 100 + outer_fold)
            best_params = None
            best_inner_auc = -1

            param_names = list(config['param_grid'].keys())
            param_values = list(config['param_grid'].values())

            for param_combo in product(*param_values):
                params = dict(zip(param_names, param_combo))
                params.update(config['fixed_params'])

                inner_aucs = []
                for train_idx, val_idx in inner_cv.split(X_trainval_sc, y_trainval):
                    model = config['model_class'](**params)
                    model.fit(X_trainval_sc[train_idx], y_trainval[train_idx])
                    val_prob = model.predict_proba(X_trainval_sc[val_idx])[:, 1]
                    inner_aucs.append(roc_auc_score(y_trainval[val_idx], val_prob))

                mean_auc = np.mean(inner_aucs)
                if mean_auc > best_inner_auc:
                    best_inner_auc = mean_auc
                    best_params = params.copy()

            final_model = config['model_class'](**best_params)
            final_model.fit(X_trainval_sc, y_trainval)

            y_test_prob = final_model.predict_proba(X_test_sc)[:, 1]
            y_test_pred = (y_test_prob >= THRESHOLD).astype(int)

            # Per-fold metrics
            metrics = get_metrics(y_test, y_test_pred, y_test_prob)
            metrics['Repeat'] = repeat + 1
            metrics['Outer_Fold'] = outer_fold + 1
            metrics['N_Selected'] = n_selected
            metrics['Best_Params'] = str(best_params)
            all_model_results[model_name].append(metrics)

            # Accumulate for repeat-level CM
            rep_collectors[model_name]['y_true'].extend(y_test)
            rep_collectors[model_name]['y_prob'].extend(y_test_prob)
            rep_collectors[model_name]['y_pred'].extend(y_test_pred)

            # ROC interpolation per fold
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            all_tprs[model_name].append(np.interp(mean_fpr, fpr, tpr))
            all_tprs[model_name][-1][0] = 0.0

        print(f" | RF={all_model_results['RF'][-1]['AUC']:.3f}, "
              f"SVM={all_model_results['SVM'][-1]['AUC']:.3f}, "
              f"XGB={all_model_results['XGB'][-1]['AUC']:.3f}", flush=True)

    # Repeat-level confusion matrices
    for name in MODEL_CONFIGS:
        yt = rep_collectors[name]['y_true']
        yprob = rep_collectors[name]['y_prob']
        ypred = rep_collectors[name]['y_pred']
        tn, fp, fn, tp = confusion_matrix(yt, ypred).ravel()
        rep_auc = roc_auc_score(yt, yprob)
        repeat_summaries[name].append({
            'Repeat': repeat + 1, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'AUC': rep_auc,
            'Accuracy': (tp + tn) / len(yt),
            'Sens': tp / (tp + fn), 'Spec': tn / (tn + fp),
            'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'F1': f1_score(yt, ypred)
        })

# Summary and save
print(f"\n{'#'*80}")
print("RESULTS (Repeat-level, Mean +/- SD)")
print(f"{'#'*80}")

for name in MODEL_CONFIGS:
    results_df = pd.DataFrame(all_model_results[name])
    repeat_df = pd.DataFrame(repeat_summaries[name])

    print(f"\n--- {name} ---")
    print(f"  AUC:         {repeat_df['AUC'].mean():.4f} +/- {repeat_df['AUC'].std():.4f}")
    print(f"  Accuracy:    {repeat_df['Accuracy'].mean():.4f} +/- {repeat_df['Accuracy'].std():.4f}")
    print(f"  Sensitivity: {repeat_df['Sens'].mean():.4f} +/- {repeat_df['Sens'].std():.4f}")
    print(f"  Specificity: {repeat_df['Spec'].mean():.4f} +/- {repeat_df['Spec'].std():.4f}")
    print(f"  PPV:         {repeat_df['PPV'].mean():.4f} +/- {repeat_df['PPV'].std():.4f}")
    print(f"  NPV:         {repeat_df['NPV'].mean():.4f} +/- {repeat_df['NPV'].std():.4f}")
    print(f"  F1:          {repeat_df['F1'].mean():.4f} +/- {repeat_df['F1'].std():.4f}")
    print(f"  |Sens-Spec|: {abs(repeat_df['Sens'].mean() - repeat_df['Spec'].mean()):.4f}")

    # Save fold-level results
    results_df.to_csv(f'mannwhitney_{name}_results.csv', index=False)

    # Save repeat-level summary
    repeat_df.to_csv(f'mannwhitney_{name}_repeat_summary.csv', index=False)

    # Save ROC data
    mean_tpr = np.mean(all_tprs[name], axis=0)
    mean_tpr[-1] = 1.0
    pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv(
        f'mannwhitney_{name}_ROC_Data.csv', index=False)

print(f"\nSaved per model: *_results.csv (fold), *_repeat_summary.csv (repeat CM), *_ROC_Data.csv")
print(f"Runtime: {(time.time() - total_start)/60:.1f} min")
