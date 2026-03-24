import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
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

# EN selector search space (identical to EN_EN_revised.py)
# C: loguniform spans 4 orders of magnitude (0.001 ~ 10)
# l1_ratio: uniform on [0, 1], linear mixing between L1 and L2
EN_PARAM_DIST = {
    'C': loguniform(0.001, 10),
    'l1_ratio': uniform(0.0, 1.0)
}

# Evaluator search spaces
EVAL_CONFIGS = {
    'RF': {
        'model_class': RandomForestClassifier,
        'param_distributions': {
            'n_estimators': randint(500, 1001),
            'max_depth': [3, 5, 7, 9],
            # min_samples_leaf excluded: redundant with max_depth (per professor)
            'max_features': ['sqrt', 0.5]
        },
        'fixed_params': {'random_state': 42, 'n_jobs': -1, 'bootstrap': True}
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

for model_name, config in EVAL_CONFIGS.items():
    print("=" * 60)
    print(f"EN Feature Selection -> {model_name} Evaluation")
    print("=" * 60)

    repeat_summaries = []
    fold_level_metrics = []
    en_params_history = []
    eval_params_history = []
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

            # Scale (required for EN selector; SVM also benefits)
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
            en_params_history.append(en_search.best_params_)

            # Select features with non-zero coefficients
            selected_mask = (np.abs(best_en.coef_[0]) > 1e-6)
            n_selected = selected_mask.sum()
            feature_selection_counts += selected_mask.astype(int)

            x_train_sel = x_train_sc[:, selected_mask]
            x_test_sel = x_test_sc[:, selected_mask]

            # ===== STEP 2: Evaluator tuning on selected features =====
            inner_cv_eval = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                            random_state=RANDOM_SEED + r)
            eval_search = RandomizedSearchCV(
                config['model_class'](**config['fixed_params']),
                param_distributions=config['param_distributions'],
                n_iter=N_ITER_SEARCH, cv=inner_cv_eval, scoring='roc_auc',
                n_jobs=-1, random_state=RANDOM_SEED + r + fold_idx
            )
            eval_search.fit(x_train_sel, y_train_outer)
            best_model = eval_search.best_estimator_
            eval_params_history.append(eval_search.best_params_)

            # Predict on test fold
            probs = best_model.predict_proba(x_test_sel)[:, 1]
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
                'Best_Eval_Params': str(eval_search.best_params_)
            })

            # ROC interpolation
            fpr, tpr, _ = roc_curve(y_test_outer, probs)
            all_tprs.append(np.interp(mean_fpr, fpr, tpr))
            all_tprs[-1][0] = 0.0

            print(f"  R{r+1} F{fold_idx+1:2d}/10 | AUC={fold_auc:.4f} | "
                  f"Sens={fold_sens:.3f} Spec={fold_spec:.3f} | N_sel={n_selected}")

        # Repeat-level: pooled confusion matrix (TN+FP+FN+TP = 174)
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
        print(f"  Repeat {r+1} EN->{model_name} | AUC={rep_auc:.4f} | "
              f"Sens={tp/(tp+fn):.3f} Spec={tn/(tn+fp):.3f} | "
              f"Total={tn+fp+fn+tp}\n")

    # ===== Save results =====
    pd.DataFrame(repeat_summaries).to_csv(
        f'EN_Selection_{model_name}_Evaluation_Summary.csv', index=False)
    pd.DataFrame(fold_level_metrics).to_csv(
        f'EN_Selection_{model_name}_Fold_Metrics.csv', index=False)

    mean_tpr = np.mean(all_tprs, axis=0)
    mean_tpr[-1] = 1.0
    pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv(
        f'EN_Selection_{model_name}_ROC_Data.csv', index=False)

    # Evaluator parameter frequency
    eval_params_df = pd.DataFrame(eval_params_history)
    with open(f'EN_Selection_{model_name}_Param_Frequencies.txt', 'w') as f:
        f.write(f"{model_name} Evaluator Parameter Frequency Analysis\n")
        f.write(f"Total folds: {len(eval_params_history)}\n")
        for col in eval_params_df.columns:
            f.write(f"\nParameter: {col}\n")
            if (eval_params_df[col].dtype in [np.float64, np.int64]
                    and eval_params_df[col].nunique() > 10):
                vals = eval_params_df[col]
                f.write(f"  Min: {vals.min():.6f}, Median: {vals.median():.6f}, "
                        f"Max: {vals.max():.6f}\n")
                f.write(pd.cut(vals, bins=5).value_counts().sort_index().to_string())
                f.write("\n")
            else:
                vc = eval_params_df[col].astype(str).value_counts()
                for val, cnt in vc.items():
                    f.write(f"  {val}: {cnt} ({cnt/len(eval_params_df)*100:.1f}%)\n")

    # Print summary
    print("-" * 60)
    rep_df = pd.DataFrame(repeat_summaries)
    for col in ['AUC', 'Accuracy', 'Sens', 'Spec', 'PPV', 'NPV', 'F1']:
        print(f"{col}: {rep_df[col].mean():.4f} +/- {rep_df[col].std():.4f}")
    print(f"|S-S|: {abs(rep_df.Sens.mean() - rep_df.Spec.mean()):.4f}")

    fold_df = pd.DataFrame(fold_level_metrics)
    print(f"Avg features selected: {fold_df.N_Selected.mean():.0f} "
          f"(range: {fold_df.N_Selected.min()}-{fold_df.N_Selected.max()})")
    print()

    # Reset for next model
    eval_params_history = []

print("\nFiles saved:")
for m in EVAL_CONFIGS:
    print(f"  EN_Selection_{m}_Evaluation_Summary.csv")
    print(f"  EN_Selection_{m}_Fold_Metrics.csv")
    print(f"  EN_Selection_{m}_ROC_Data.csv")
    print(f"  EN_Selection_{m}_Param_Frequencies.txt")
