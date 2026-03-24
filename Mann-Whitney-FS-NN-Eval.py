# Mann-Whitney FS + Neural Network Evaluation
# 7 Repeats x 10 Outer x 5 Inner, Threshold 0.5

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score, roc_curve
from sklearn.neural_network import MLPClassifier
from scipy.stats import mannwhitneyu
from itertools import product
import warnings
import time

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("Mann-Whitney FS + Neural Network")
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

NN_PARAM_GRID = {
    'hidden_layer_sizes': [(8,), (16,), (32,), (16, 8), (32, 16)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.01, 0.1, 1.0],
    'learning_rate_init': [0.001, 0.01],
    'batch_size': [16, 32, 'auto']
}

NN_FIXED_PARAMS = {
    'random_state': 42,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 10,
    'max_iter': 500
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
print(f"\nNN Hyperparameter Grid:")
for param, values in NN_PARAM_GRID.items():
    print(f"  {param}: {values}")
print(flush=True)

# Main loop
all_nn_results = []
mean_fpr = np.linspace(0, 1, 100)
all_tprs = []
repeat_summaries = []

total_start = time.time()

for repeat in range(N_REPEATS):
    print(f"\nREPEAT {repeat+1}/{N_REPEATS}")

    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + repeat)
    rep_y_true, rep_y_prob, rep_y_pred = [], [], []

    for outer_fold, (trainval_idx, test_idx) in enumerate(outer_cv.split(X, y)):
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

        # Inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                   random_state=42 + repeat * 100 + outer_fold)
        best_params = None
        best_inner_auc = -1

        param_names = list(NN_PARAM_GRID.keys())
        param_values = list(NN_PARAM_GRID.values())
        param_combinations = list(product(*param_values))

        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))
            params.update(NN_FIXED_PARAMS)

            inner_aucs = []
            for train_idx, val_idx in inner_cv.split(X_trainval_sc, y_trainval):
                model = MLPClassifier(**params)
                model.fit(X_trainval_sc[train_idx], y_trainval[train_idx])
                val_prob = model.predict_proba(X_trainval_sc[val_idx])[:, 1]
                inner_aucs.append(roc_auc_score(y_trainval[val_idx], val_prob))

            mean_auc = np.mean(inner_aucs)
            if mean_auc > best_inner_auc:
                best_inner_auc = mean_auc
                best_params = params.copy()

        # Final model
        final_model = MLPClassifier(**best_params)
        final_model.fit(X_trainval_sc, y_trainval)

        y_test_prob = final_model.predict_proba(X_test_sc)[:, 1]
        y_test_pred = (y_test_prob >= THRESHOLD).astype(int)

        # Per-fold metrics
        metrics = get_metrics(y_test, y_test_pred, y_test_prob)
        metrics['Repeat'] = repeat + 1
        metrics['Outer_Fold'] = outer_fold + 1
        metrics['N_Selected'] = n_selected
        metrics['Best_Params'] = str(best_params)
        all_nn_results.append(metrics)

        # Accumulate for repeat-level CM
        rep_y_true.extend(y_test)
        rep_y_prob.extend(y_test_prob)
        rep_y_pred.extend(y_test_pred)

        # ROC interpolation per fold
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        all_tprs.append(np.interp(mean_fpr, fpr, tpr))
        all_tprs[-1][0] = 0.0

        print(f"  Fold {outer_fold+1}: N={n_selected}, "
              f"AUC={metrics['AUC']:.3f}, Sens={metrics['Sens']:.3f}, Spec={metrics['Spec']:.3f}",
              flush=True)

    # Repeat-level confusion matrix (10 folds = 174 samples)
    tn, fp, fn, tp = confusion_matrix(rep_y_true, rep_y_pred).ravel()
    rep_auc = roc_auc_score(rep_y_true, rep_y_prob)
    repeat_summaries.append({
        'Repeat': repeat + 1, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'AUC': rep_auc,
        'Accuracy': (tp + tn) / len(rep_y_true),
        'Sens': tp / (tp + fn), 'Spec': tn / (tn + fp),
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'F1': f1_score(rep_y_true, rep_y_pred)
    })
    print(f"  Repeat {repeat+1} AUC: {rep_auc:.4f}")

# Summary
results_df = pd.DataFrame(all_nn_results)
repeat_df = pd.DataFrame(repeat_summaries)

print(f"\nPerformance (Repeat-level, Mean +/- SD):")
print(f"  AUC:         {repeat_df['AUC'].mean():.4f} +/- {repeat_df['AUC'].std():.4f}")
print(f"  Accuracy:    {repeat_df['Accuracy'].mean():.4f} +/- {repeat_df['Accuracy'].std():.4f}")
print(f"  Sensitivity: {repeat_df['Sens'].mean():.4f} +/- {repeat_df['Sens'].std():.4f}")
print(f"  Specificity: {repeat_df['Spec'].mean():.4f} +/- {repeat_df['Spec'].std():.4f}")
print(f"  PPV:         {repeat_df['PPV'].mean():.4f} +/- {repeat_df['PPV'].std():.4f}")
print(f"  NPV:         {repeat_df['NPV'].mean():.4f} +/- {repeat_df['NPV'].std():.4f}")
print(f"  F1:          {repeat_df['F1'].mean():.4f} +/- {repeat_df['F1'].std():.4f}")
print(f"  |Sens-Spec|: {abs(repeat_df['Sens'].mean() - repeat_df['Spec'].mean()):.4f}")

# Save
results_df.to_csv('mannwhitney_NN_results.csv', index=False)
repeat_df.to_csv('mannwhitney_NN_repeat_summary.csv', index=False)

mean_tpr = np.mean(all_tprs, axis=0)
mean_tpr[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv(
    'mannwhitney_NN_ROC_Data.csv', index=False)

print(f"\nSaved: mannwhitney_NN_results.csv (fold-level)")
print(f"Saved: mannwhitney_NN_repeat_summary.csv (repeat-level CM)")
print(f"Saved: mannwhitney_NN_ROC_Data.csv")
print(f"Runtime: {(time.time() - total_start)/60:.1f} min")
