import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV,
                                     cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, roc_curve)
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X = df[feat_names].values

# Settings
N_REPEATS = 7
N_OUTER = 10
N_INNER = 5
N_ITER_SEARCH = 20
RANDOM_SEED = 42
THRESHOLD = 0.5

# EN selector search space (identical across all EN FS pipelines)
EN_PARAM_DIST = {
    'C': loguniform(0.001, 10),
    'l1_ratio': uniform(0.0, 1.0)
}

# Base learner configs: RF + SVM + XGB (no EN — EN is the selector)
BASE_CONFIGS = {
    'RF': {
        'model_class': RandomForestClassifier,
        'param_distributions': {
            'n_estimators': randint(500, 1001),
            'max_depth': [3, 5, 7, 9],
            'max_features': ['sqrt', 0.5]
        },
        'fixed_params': {
            'random_state': 42, 'n_jobs': -1, 'bootstrap': True
        }
    },
    'SVM': {
        'model_class': SVC,
        'param_distributions': {
            'C': loguniform(0.1, 100),
            'gamma': loguniform(0.0001, 1.0),
            'kernel': ['linear', 'rbf']
        },
        'fixed_params': {'probability': True, 'random_state': 42}
    },
    'XGB': {
        'model_class': XGBClassifier,
        'param_distributions': {
            'n_estimators': randint(300, 601),
            'max_depth': [3, 5, 7, 9],
            'learning_rate': loguniform(0.001, 0.1)
        },
        'fixed_params': {
            'random_state': 42, 'eval_metric': 'logloss',
            'subsample': 1.0, 'verbosity': 0
        }
    }
}

print("=" * 60)
print("EN Feature Selection -> Stacking Ensemble Evaluation")
print(f"Base learners: {', '.join(BASE_CONFIGS.keys())}")
print(f"Meta-learner: Logistic Regression")
print("=" * 60)

repeat_summaries = []
fold_level_metrics = []
en_params_history = []
feature_selection_counts = np.zeros(X.shape[1])
mean_fpr = np.linspace(0, 1, 100)
all_tprs = []

for r in range(N_REPEATS):
    repeat_y_true, repeat_y_probs, repeat_y_preds = [], [], []
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True,
                               random_state=RANDOM_SEED + r)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        x_train_outer, x_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train_outer)
        x_test_sc = scaler.transform(x_test_outer)

        # ===== STEP 1: EN Feature Selection (per-fold tuned) =====
        inner_cv_en = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                      random_state=RANDOM_SEED + r)
        en_search = RandomizedSearchCV(
            LogisticRegression(penalty='elasticnet', solver='saga',
                               max_iter=10000, random_state=42),
            param_distributions=EN_PARAM_DIST,
            n_iter=N_ITER_SEARCH, cv=inner_cv_en, scoring='roc_auc',
            n_jobs=-1, random_state=RANDOM_SEED + r + fold_idx
        )
        en_search.fit(x_train_sc, y_train_outer)
        best_en = en_search.best_estimator_

        en_params_history.append({
            'Repeat': r + 1, 'Fold': fold_idx + 1,
            'C': en_search.best_params_['C'],
            'l1_ratio': en_search.best_params_['l1_ratio']
        })

        # Select features with non-zero coefficients
        coef = best_en.coef_.ravel()
        selected_mask = np.abs(coef) > 0
        n_selected = selected_mask.sum()

        # Fallback: if all coefficients are zero, keep top 20 by magnitude
        if n_selected == 0:
            top_idx = np.argsort(np.abs(coef))[-20:]
            selected_mask[top_idx] = True
            n_selected = selected_mask.sum()
            print(f"  WARNING: R{r+1} F{fold_idx+1} all coefs zero, "
                  f"fallback to top {n_selected}")

        feature_selection_counts += selected_mask.astype(int)

        x_train_sel = x_train_sc[:, selected_mask]
        x_test_sel = x_test_sc[:, selected_mask]

        # ===== STEP 2: Tune base learners & generate meta-features =====
        inner_cv_base = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                        random_state=RANDOM_SEED + r)

        oof_train = np.zeros((x_train_sel.shape[0], len(BASE_CONFIGS)))
        oof_test = np.zeros((x_test_sel.shape[0], len(BASE_CONFIGS)))
        base_params_str = {}

        for i, (name, cfg) in enumerate(BASE_CONFIGS.items()):
            search = RandomizedSearchCV(
                cfg['model_class'](**cfg['fixed_params']),
                param_distributions=cfg['param_distributions'],
                n_iter=N_ITER_SEARCH, cv=inner_cv_base, scoring='roc_auc',
                n_jobs=-1, random_state=RANDOM_SEED + r + fold_idx
            )
            search.fit(x_train_sel, y_train_outer)
            best_base = search.best_estimator_
            base_params_str[name] = str(search.best_params_)

            # Out-of-fold predictions for meta-features
            oof_train[:, i] = cross_val_predict(
                best_base, x_train_sel, y_train_outer,
                cv=inner_cv_base, method='predict_proba'
            )[:, 1]
            # Test predictions from best base
            oof_test[:, i] = best_base.predict_proba(x_test_sel)[:, 1]

        # ===== STEP 3: Meta-learner =====
        meta_model = LogisticRegression(random_state=42, max_iter=10000)
        meta_model.fit(oof_train, y_train_outer)

        probs = meta_model.predict_proba(oof_test)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)

        repeat_y_true.extend(y_test_outer)
        repeat_y_probs.extend(probs)
        repeat_y_preds.extend(preds)

        # Fold-level metrics
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(
            y_test_outer, preds, labels=[0, 1]).ravel()
        fold_auc = roc_auc_score(y_test_outer, probs)
        fold_sens = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
        fold_spec = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0
        fold_ppv = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
        fold_npv = tn_f / (tn_f + fn_f) if (tn_f + fn_f) > 0 else 0
        fold_acc = (tp_f + tn_f) / (tn_f + fp_f + fn_f + tp_f)
        fold_f1 = (2 * tp_f / (2 * tp_f + fp_f + fn_f)
                   if (2 * tp_f + fp_f + fn_f) > 0 else 0)

        fold_level_metrics.append({
            'Repeat': r + 1, 'Fold': fold_idx + 1,
            'AUC': fold_auc, 'Acc': fold_acc,
            'Sens': fold_sens, 'Spec': fold_spec,
            'PPV': fold_ppv, 'NPV': fold_npv, 'F1': fold_f1,
            'N_Selected': n_selected,
            'Best_EN_Params': str(en_search.best_params_),
            'Best_RF_Params': base_params_str.get('RF', ''),
            'Best_SVM_Params': base_params_str.get('SVM', ''),
            'Best_XGB_Params': base_params_str.get('XGB', '')
        })

        # ROC interpolation
        fpr, tpr, _ = roc_curve(y_test_outer, probs)
        all_tprs.append(np.interp(mean_fpr, fpr, tpr))
        all_tprs[-1][0] = 0.0

        print(f"  R{r+1} F{fold_idx+1:2d}/10 | AUC={fold_auc:.4f} | "
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
        'F1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    })
    print(f"  Repeat {r+1} EN->Stacking | AUC={rep_auc:.4f} | "
          f"Sens={tp/(tp+fn):.3f} Spec={tn/(tn+fp):.3f} | "
          f"Total={tn+fp+fn+tp}\n")

# ===== Save results =====
pd.DataFrame(repeat_summaries).to_csv(
    'EN_Selection_Stacking_Evaluation_Summary.csv', index=False)
pd.DataFrame(fold_level_metrics).to_csv(
    'EN_Selection_Stacking_Fold_Metrics.csv', index=False)

mean_tpr = np.mean(all_tprs, axis=0)
mean_tpr[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv(
    'EN_Selection_Stacking_ROC_Data.csv', index=False)

# EN selector parameter frequency
en_params_df = pd.DataFrame(en_params_history)
with open('EN_Selection_Stacking_EN_Param_Frequencies.txt', 'w') as f:
    f.write("EN Selector Parameter Frequency Analysis\n")
    f.write(f"Total folds: {len(en_params_history)}\n")
    f.write("\nParameter: C\n")
    c_vals = en_params_df['C']
    f.write(f"  Min: {c_vals.min():.6f}, Median: {c_vals.median():.6f}, "
            f"Max: {c_vals.max():.6f}\n")
    for lo, hi in [(0, 0.01), (0.01, 0.1), (0.1, 1), (1, 10)]:
        cnt = ((c_vals >= lo) & (c_vals < hi)).sum()
        f.write(f"  [{lo}, {hi}): {cnt} ({cnt/len(en_params_df)*100:.1f}%)\n")
    f.write("\nParameter: l1_ratio\n")
    l1_vals = en_params_df['l1_ratio']
    f.write(f"  Min: {l1_vals.min():.4f}, Median: {l1_vals.median():.4f}, "
            f"Max: {l1_vals.max():.4f}\n")
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        cnt = ((l1_vals >= lo) & (l1_vals < hi)).sum()
        f.write(f"  [{lo}, {hi}): {cnt} ({cnt/len(en_params_df)*100:.1f}%)\n")

# Feature selection counts
feat_df = pd.DataFrame({
    'Feature': feat_names,
    'Count': feature_selection_counts.astype(int)
}).sort_values('Count', ascending=False)
feat_df['Frequency_Percent'] = feat_df['Count'] / (N_REPEATS * N_OUTER) * 100
feat_df.to_csv('EN_Selection_Stacking_Feature_Counts.csv', index=False)

# Print summary
print("=" * 60)
print("EN Selection -> Stacking Summary")
print("=" * 60)
rep_df = pd.DataFrame(repeat_summaries)
for col in ['AUC', 'Accuracy', 'Sens', 'Spec', 'PPV', 'NPV', 'F1']:
    print(f"{col}: {rep_df[col].mean():.4f} +/- {rep_df[col].std():.4f}")
print(f"|S-S|: {abs(rep_df.Sens.mean() - rep_df.Spec.mean()):.4f}")

fold_df = pd.DataFrame(fold_level_metrics)
print(f"\nAvg features selected: {fold_df.N_Selected.mean():.0f} "
      f"(range: {fold_df.N_Selected.min()}-{fold_df.N_Selected.max()})")

print("\nFiles saved:")
print("  EN_Selection_Stacking_Evaluation_Summary.csv")
print("  EN_Selection_Stacking_Fold_Metrics.csv")
print("  EN_Selection_Stacking_ROC_Data.csv")
print("  EN_Selection_Stacking_EN_Param_Frequencies.txt")
print("  EN_Selection_Stacking_Feature_Counts.csv")
