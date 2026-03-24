import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, roc_curve)
from scipy.stats import randint
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X = df[feat_names].values

# Settings
n_repeats = 7
n_outer = 10
n_inner = 5
n_iter_search = 20
random_seed = 42

# RF search space (log2 removed: selected 0/70 times in initial run)
param_distributions = {
    'n_estimators': randint(500, 1001),
    'max_depth': [3, 5, 7, 9],
    'max_features': ['sqrt', 0.5]
}

repeat_summaries = []
fold_level_metrics = []
best_params_history = []
feature_selection_counts = np.zeros(X.shape[1])
mean_fpr = np.linspace(0, 1, 100)
all_tprs = []

for r in range(n_repeats):
    repeat_y_true, repeat_y_probs, repeat_y_preds = [], [], []
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_seed + r)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        x_train_outer, x_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        # Scale for consistency with downstream classifiers (SVM, LR in Stacking)
        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train_outer)
        x_test_sc = scaler.transform(x_test_outer)

        # Inner loop: Tune RF selector
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_seed + r)
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1, bootstrap=True),
            param_distributions=param_distributions,
            n_iter=n_iter_search, cv=inner_cv, scoring='roc_auc', n_jobs=-1,
            random_state=random_seed + r + fold_idx
        )
        search.fit(x_train_sc, y_train_outer)
        best_rf_selector = search.best_estimator_

        # Feature selection (importance > mean)
        selector = SelectFromModel(best_rf_selector, threshold='mean', prefit=True)
        selected_mask = selector.get_support()
        n_selected = selected_mask.sum()
        feature_selection_counts += selected_mask.astype(int)

        # Subset to selected features
        x_train_sel = x_train_sc[:, selected_mask]
        x_test_sel = x_test_sc[:, selected_mask]

        # Re-train RF on selected features with best params
        final_rf = RandomForestClassifier(**search.best_params_, random_state=42, n_jobs=-1)
        final_rf.fit(x_train_sel, y_train_outer)
        best_params_history.append(search.best_params_)

        probs = final_rf.predict_proba(x_test_sel)[:, 1]
        preds = (probs >= 0.5).astype(int)

        repeat_y_true.extend(y_test_outer)
        repeat_y_probs.extend(probs)
        repeat_y_preds.extend(preds)

        # Fold-level metrics
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_test_outer, preds, labels=[0,1]).ravel()
        fold_auc = roc_auc_score(y_test_outer, probs)
        fold_sens = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
        fold_spec = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0
        fold_ppv = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
        fold_npv = tn_f / (tn_f + fn_f) if (tn_f + fn_f) > 0 else 0
        fold_acc = (tp_f + tn_f) / (tn_f + fp_f + fn_f + tp_f)
        fold_f1 = 2*tp_f / (2*tp_f + fp_f + fn_f) if (2*tp_f + fp_f + fn_f) > 0 else 0

        fold_level_metrics.append({
            'Repeat': r + 1, 'Fold': fold_idx + 1,
            'AUC': fold_auc, 'Acc': fold_acc,
            'Sens': fold_sens, 'Spec': fold_spec,
            'PPV': fold_ppv, 'NPV': fold_npv, 'F1': fold_f1,
            'N_Selected': n_selected,
            'Best_Params': str(search.best_params_)
        })

        # ROC interpolation
        fpr, tpr, _ = roc_curve(y_test_outer, probs)
        all_tprs.append(np.interp(mean_fpr, fpr, tpr))
        all_tprs[-1][0] = 0.0

        print(f"  Repeat {r+1}, Fold {fold_idx+1}/10 | AUC={fold_auc:.4f} | "
              f"Sens={fold_sens:.3f} Spec={fold_spec:.3f} | N_sel={n_selected}")

    # Repeat-level: pooled confusion matrix
    tn, fp, fn, tp = confusion_matrix(repeat_y_true, repeat_y_preds).ravel()
    rep_auc = roc_auc_score(repeat_y_true, repeat_y_probs)
    repeat_summaries.append({
        'Repeat': r + 1, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'AUC': rep_auc,
        'Accuracy': (tp + tn) / (tn + fp + fn + tp),
        'Sens': tp / (tp + fn), 'Spec': tn / (tn + fp),
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'F1': 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    })
    print(f"Repeat {r+1} RF-RF | AUC={rep_auc:.4f} | "
          f"Sens={tp/(tp+fn):.3f} Spec={tn/(tn+fp):.3f} | Total={tn+fp+fn+tp}\n")

# Save results
pd.DataFrame(repeat_summaries).to_csv(
    'RF_Selection_RF_Evaluation_Summary.csv', index=False)
pd.DataFrame(fold_level_metrics).to_csv(
    'RF_Selection_RF_Fold_Metrics.csv', index=False)
pd.DataFrame({'FPR': mean_fpr, 'TPR': np.mean(all_tprs, axis=0)}).to_csv(
    'RF_Selection_RF_ROC_Data.csv', index=False)

# Feature selection frequency
feat_df = pd.DataFrame({
    'Feature': feat_names, 'Count': feature_selection_counts
}).sort_values('Count', ascending=False)
feat_df['Frequency_Percent'] = feat_df['Count'] / 70 * 100
feat_df.to_csv('RF_Selection_Feature_Counts.csv', index=False)

# Parameter frequency analysis
params_df = pd.DataFrame(best_params_history)
with open('RF_Selection_Param_Frequencies.txt', 'w') as f:
    f.write("RF Selector Parameter Frequency Analysis\n")
    f.write(f"Total folds: {len(best_params_history)}\n")
    for col in params_df.columns:
        f.write(f"\nParameter: {col}\n")
        vc = params_df[col].value_counts()
        for val, cnt in vc.items():
            f.write(f"  {val}: {cnt} ({cnt/len(params_df)*100:.1f}%)\n")

# Print summary
print("=" * 60)
rep_df = pd.DataFrame(repeat_summaries)
for col in ['AUC', 'Accuracy', 'Sens', 'Spec', 'PPV', 'NPV', 'F1']:
    print(f"{col}: {rep_df[col].mean():.4f} +/- {rep_df[col].std():.4f}")
print(f"|S-S|: {abs(rep_df.Sens.mean() - rep_df.Spec.mean()):.4f}")
print(f"Avg features selected: {feat_df.Count.sum()/70:.0f}")
