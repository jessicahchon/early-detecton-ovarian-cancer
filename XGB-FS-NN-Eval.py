import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
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

# XGB selector search space (identical across all XGB FS pipelines)
XGB_SEL_PARAM_DIST = {
    'n_estimators': randint(300, 601),
    'max_depth': [3, 5, 7, 9],
    'learning_rate': loguniform(0.001, 0.1)
}
XGB_SEL_FIXED = {
    'random_state': 42, 'eval_metric': 'logloss',
    'subsample': 1.0, 'verbosity': 0
}

# NN evaluator search space (identical to ElasticNet_NN)
# Small networks to prevent overfitting with 174 samples.
NN_PARAM_DIST = {
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

model_name = 'NN'

print("=" * 60)
print(f"XGB Feature Selection -> {model_name} Evaluation")
print("=" * 60)

repeat_summaries = []
fold_level_metrics = []
sel_params_history = []
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

        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train_outer)
        x_test_sc = scaler.transform(x_test_outer)

        # XGB Feature Selection (per-fold tuned)
        inner_cv_sel = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                       random_state=RANDOM_SEED + r)
        xgb_search = RandomizedSearchCV(
            XGBClassifier(**XGB_SEL_FIXED),
            param_distributions=XGB_SEL_PARAM_DIST,
            n_iter=N_ITER_SEARCH, cv=inner_cv_sel, scoring='roc_auc',
            n_jobs=-1, random_state=RANDOM_SEED + r + fold_idx
        )
        xgb_search.fit(x_train_sc, y_train_outer)
        sel_params_history.append(xgb_search.best_params_)

        selector = SelectFromModel(xgb_search.best_estimator_,
                                   threshold='mean', prefit=True)
        selected_mask = selector.get_support()
        n_selected = selected_mask.sum()
        feature_selection_counts += selected_mask.astype(int)

        x_train_sel = x_train_sc[:, selected_mask]
        x_test_sel = x_test_sc[:, selected_mask]

        # Evaluator tuning on selected features
        inner_cv_eval = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                        random_state=RANDOM_SEED + r)
        eval_search = RandomizedSearchCV(
            MLPClassifier(**NN_FIXED_PARAMS),
            param_distributions=NN_PARAM_DIST,
            n_iter=N_ITER_SEARCH, cv=inner_cv_eval, scoring='roc_auc',
            n_jobs=-1, random_state=RANDOM_SEED + r + fold_idx
        )
        eval_search.fit(x_train_sel, y_train_outer)
        best_model = eval_search.best_estimator_
        eval_params_history.append(eval_search.best_params_)

        probs = best_model.predict_proba(x_test_sel)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)

        repeat_y_true.extend(y_test_outer)
        repeat_y_probs.extend(probs)
        repeat_y_preds.extend(preds)

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
            'Best_Sel_Params': str(xgb_search.best_params_),
            'Best_Eval_Params': str(eval_search.best_params_)
        })

        fpr, tpr, _ = roc_curve(y_test_outer, probs)
        all_tprs.append(np.interp(mean_fpr, fpr, tpr))
        all_tprs[-1][0] = 0.0

        print(f"  R{r+1} F{fold_idx+1:2d}/10 | AUC={fold_auc:.4f} | "
              f"Sens={fold_sens:.3f} Spec={fold_spec:.3f} | N_sel={n_selected}")

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
    print(f"  Repeat {r+1} XGB->{model_name} | AUC={rep_auc:.4f} | "
          f"Sens={tp/(tp+fn):.3f} Spec={tn/(tn+fp):.3f} | Total={tn+fp+fn+tp}\n")

# Save results
pd.DataFrame(repeat_summaries).to_csv(
    f'XGB_Selection_{model_name}_Evaluation_Summary.csv', index=False)
pd.DataFrame(fold_level_metrics).to_csv(
    f'XGB_Selection_{model_name}_Fold_Metrics.csv', index=False)

mean_tpr = np.mean(all_tprs, axis=0); mean_tpr[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv(
    f'XGB_Selection_{model_name}_ROC_Data.csv', index=False)

pd.DataFrame({'Feature': feat_names, 'Count': feature_selection_counts.astype(int)
}).sort_values('Count', ascending=False).to_csv(
    f'XGB_Selection_{model_name}_Feature_Counts.csv', index=False)

eval_params_df = pd.DataFrame(eval_params_history)
with open(f'XGB_Selection_NN_Param_Frequencies.txt', 'w') as f:
    f.write(f"NN Evaluator Parameter Frequency Analysis\n")
    f.write(f"Total folds: {len(eval_params_history)}\n")
    for col in eval_params_df.columns:
        f.write(f"\nParameter: {col}\n")
        if eval_params_df[col].dtype in [np.float64, np.int64] and eval_params_df[col].nunique() > 10:
            vals = eval_params_df[col]
            f.write(f"  Min: {vals.min():.6f}, Median: {vals.median():.6f}, Max: {vals.max():.6f}\n")
            f.write(pd.cut(vals, bins=5).value_counts().sort_index().to_string() + "\n")
        else:
            vc = eval_params_df[col].astype(str).value_counts()
            for val, cnt in vc.items():
                f.write(f"  {val}: {cnt} ({cnt/len(eval_params_df)*100:.1f}%)\n")

print("=" * 60)
rep_df = pd.DataFrame(repeat_summaries)
for col in ['AUC', 'Accuracy', 'Sens', 'Spec', 'PPV', 'NPV', 'F1']:
    print(f"{col}: {rep_df[col].mean():.4f} +/- {rep_df[col].std():.4f}")
print(f"|S-S|: {abs(rep_df.Sens.mean() - rep_df.Spec.mean()):.4f}")
fold_df = pd.DataFrame(fold_level_metrics)
print(f"Avg features selected: {fold_df.N_Selected.mean():.0f} "
      f"(range: {fold_df.N_Selected.min()}-{fold_df.N_Selected.max()})")
