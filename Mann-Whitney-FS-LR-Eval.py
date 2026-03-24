# Mann-Whitney FS + LR (Elastic Net) Evaluation
# 7 Repeats x 10 Outer x 5 Inner, Threshold 0.5

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score, roc_curve
from scipy.stats import mannwhitneyu
import warnings
import time

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("Mann-Whitney FS + LR (Elastic Net)")
print("  Feature Selection: Mann-Whitney U test (p < 0.05)")
print("  Inner CV: Tune C, l1_ratio (AUC criterion)")
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

# Config
N_REPEATS = 7
N_OUTER = 10
N_INNER = 5
ALPHA = 0.05
THRESHOLD = 0.5
CANDIDATE_C = [0.001, 0.01, 0.05, 0.1, 0.5]
CANDIDATE_L1_RATIO = [0.0, 0.05, 0.1, 0.3]

# Data
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X = df[feat_names].values

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class: {np.sum(y==1)} Cancer, {np.sum(y==0)} Control")
print(f"Grid: C={CANDIDATE_C}, l1_ratio={CANDIDATE_L1_RATIO}")
print(flush=True)

# Full data Mann-Whitney (for reference only)
selected_idx_example, p_values_example = mannwhitney_selection(X, y, alpha=ALPHA)
print(f"Full data features (p < 0.05): {len(selected_idx_example)}")

# Save full feature info
feature_info_df = pd.DataFrame({
    'Biomarker': feat_names, 'P_value': p_values_example,
    'Pass_P05': p_values_example < ALPHA
}).sort_values('P_value')
feature_info_df.to_csv('mannwhitney_all_features.csv', index=False)

# Main loop
all_results = []
feature_selection_counts = np.zeros(X.shape[1])
mean_fpr = np.linspace(0, 1, 100)
all_tprs = []

# Per-repeat accumulators for confusion matrix
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
        selected_idx, p_vals = mannwhitney_selection(X_trainval, y_trainval, alpha=ALPHA)
        n_selected = len(selected_idx)
        if n_selected == 0:
            selected_idx = np.argsort(p_vals)[:50]
            n_selected = 50
        feature_selection_counts[selected_idx] += 1

        X_trainval_sel = X_trainval[:, selected_idx]
        X_test_sel = X_test[:, selected_idx]

        # Inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                   random_state=42 + repeat * 100 + outer_fold)
        best_c, best_l1, best_inner_auc = None, None, -1

        for c in CANDIDATE_C:
            for l1 in CANDIDATE_L1_RATIO:
                inner_aucs = []
                for train_idx, val_idx in inner_cv.split(X_trainval_sel, y_trainval):
                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_trainval_sel[train_idx])
                    X_val_sc = scaler.transform(X_trainval_sel[val_idx])

                    model = LogisticRegression(
                        penalty='elasticnet', solver='saga',
                        l1_ratio=l1, C=c, max_iter=5000, random_state=42)
                    model.fit(X_train_sc, y_trainval[train_idx])
                    val_prob = model.predict_proba(X_val_sc)[:, 1]
                    inner_aucs.append(roc_auc_score(y_trainval[val_idx], val_prob))

                mean_auc = np.mean(inner_aucs)
                if mean_auc > best_inner_auc:
                    best_inner_auc = mean_auc
                    best_c, best_l1 = c, l1

        # Final model
        scaler_final = StandardScaler()
        X_trainval_sc = scaler_final.fit_transform(X_trainval_sel)
        X_test_sc = scaler_final.transform(X_test_sel)

        final_model = LogisticRegression(
            penalty='elasticnet', solver='saga',
            l1_ratio=best_l1, C=best_c, max_iter=5000, random_state=42)
        final_model.fit(X_trainval_sc, y_trainval)

        y_test_prob = final_model.predict_proba(X_test_sc)[:, 1]
        y_test_pred = (y_test_prob >= THRESHOLD).astype(int)

        # Per-fold metrics
        metrics = get_metrics(y_test, y_test_pred, y_test_prob)
        metrics['Repeat'] = repeat + 1
        metrics['Outer_Fold'] = outer_fold + 1
        metrics['N_Selected'] = n_selected
        metrics['Best_C'] = best_c
        metrics['Best_L1'] = best_l1
        all_results.append(metrics)

        # Accumulate for repeat-level CM
        rep_y_true.extend(y_test)
        rep_y_prob.extend(y_test_prob)
        rep_y_pred.extend(y_test_pred)

        # ROC interpolation per fold
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        all_tprs.append(np.interp(mean_fpr, fpr, tpr))
        all_tprs[-1][0] = 0.0

        print(f"  Fold {outer_fold+1}: N={n_selected}, C={best_c}, L1={best_l1}, "
              f"AUC={metrics['AUC']:.3f}, Sens={metrics['Sens']:.3f}, Spec={metrics['Spec']:.3f}",
              flush=True)

    # Repeat-level confusion matrix (sum of 10 folds = 174 samples)
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
results_df = pd.DataFrame(all_results)
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

# Top features
total_folds = N_REPEATS * N_OUTER
ranking_idx = np.argsort(feature_selection_counts)[::-1]
top_features_df = pd.DataFrame({
    'Rank': range(1, len(ranking_idx) + 1),
    'Biomarker': feat_names[ranking_idx],
    'Selection_Count': feature_selection_counts[ranking_idx].astype(int),
    'Frequency_Percent': (feature_selection_counts[ranking_idx] / total_folds * 100).round(2),
    'P_value': p_values_example[ranking_idx]
})
top_features_df = top_features_df[top_features_df['Selection_Count'] > 0]

# Save everything
results_df.to_csv('mannwhitney_pipeline_results.csv', index=False)
repeat_df.to_csv('mannwhitney_LR_repeat_summary.csv', index=False)
top_features_df.to_csv('mannwhitney_top_features.csv', index=False)

mean_tpr = np.mean(all_tprs, axis=0)
mean_tpr[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv(
    'mannwhitney_LR_ROC_Data.csv', index=False)

print(f"\nSaved: mannwhitney_pipeline_results.csv (fold-level)")
print(f"Saved: mannwhitney_LR_repeat_summary.csv (repeat-level CM)")
print(f"Saved: mannwhitney_LR_ROC_Data.csv")
print(f"Saved: mannwhitney_top_features.csv")
print(f"Saved: mannwhitney_all_features.csv")
print(f"Runtime: {(time.time() - total_start)/60:.1f} min")
