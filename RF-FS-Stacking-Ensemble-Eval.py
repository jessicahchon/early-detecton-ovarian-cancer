import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
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

# RF selector search space (identical to RF_RF_revised.py)
RF_SEL_PARAM_DIST = {
    'n_estimators': randint(500, 1001),
    'max_depth': [3, 5, 7, 9],
    'max_features': ['sqrt', 0.5]
}

# Base learner configs: LR(EN) + SVM + XGB (no RF — RF is the selector)
BASE_CONFIGS = {
    'LR': {
        'model_class': LogisticRegression,
        'param_distributions': {
            'C': [0.001, 0.01, 0.05, 0.1, 0.5],
            'l1_ratio': [0.0, 0.1, 0.2, 0.3]
        },
        'fixed_params': {
            'penalty': 'elasticnet', 'solver': 'saga',
            'max_iter': 10000, 'random_state': 42
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
print("RF Feature Selection -> Stacking Ensemble Evaluation")
print(f"Base learners: {', '.join(BASE_CONFIGS.keys())}")
print(f"Meta-learner: Logistic Regression")
print("=" * 60)

repeat_summaries = []
fold_level_metrics = []
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

        # Scale for consistency across all models
        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train_outer)
        x_test_sc = scaler.transform(x_test_outer)

        # ===== STEP 1: RF Feature Selection (per-fold tuned) =====
        inner_cv_sel = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                       random_state=RANDOM_SEED + r)
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1, bootstrap=True),
            param_distributions=RF_SEL_PARAM_DIST,
            n_iter=N_ITER_SEARCH, cv=inner_cv_sel, scoring='roc_auc',
            n_jobs=-1, random_state=RANDOM_SEED + r + fold_idx
        )
        rf_search.fit(x_train_sc, y_train_outer)

        selector = SelectFromModel(rf_search.best_estimator_,
                                   threshold='mean', prefit=True)
        selected_mask = selector.get_support()
        n_selected = selected_mask.sum()
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
            'Best_RF_Params': str(rf_search.best_params_),
            'Best_LR_Params': base_params_str.get('LR', ''),
            'Best_SVM_Params': base_params_str.get('SVM', ''),
            'Best_XGB_Params': base_params_str.get('XGB', '')
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
    print(f"  Repeat {r+1} RF->Stacking | AUC={rep_auc:.4f} | "
          f"Sens={tp/(tp+fn):.3f} Spec={tn/(tn+fp):.3f} | "
          f"Total={tn+fp+fn+tp}\n")

# ===== Save results =====
pd.DataFrame(repeat_summaries).to_csv(
    'RF_Selection_Stacking_Evaluation_Summary.csv', index=False)
pd.DataFrame(fold_level_metrics).to_csv(
    'RF_Selection_Stacking_Fold_Metrics.csv', index=False)

mean_tpr = np.mean(all_tprs, axis=0)
mean_tpr[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv(
    'RF_Selection_Stacking_ROC_Data.csv', index=False)

# Print summary
print("=" * 60)
print("RF Selection -> Stacking Summary")
print("=" * 60)
rep_df = pd.DataFrame(repeat_summaries)
for col in ['AUC', 'Accuracy', 'Sens', 'Spec', 'PPV', 'NPV', 'F1']:
    print(f"{col}: {rep_df[col].mean():.4f} +/- {rep_df[col].std():.4f}")
print(f"|S-S|: {abs(rep_df.Sens.mean() - rep_df.Spec.mean()):.4f}")

fold_df = pd.DataFrame(fold_level_metrics)
print(f"\nAvg features selected: {fold_df.N_Selected.mean():.0f} "
      f"(range: {fold_df.N_Selected.min()}-{fold_df.N_Selected.max()})")

print("\nFiles saved:")
print("  RF_Selection_Stacking_Evaluation_Summary.csv")
print("  RF_Selection_Stacking_Fold_Metrics.csv")
print("  RF_Selection_Stacking_ROC_Data.csv")
