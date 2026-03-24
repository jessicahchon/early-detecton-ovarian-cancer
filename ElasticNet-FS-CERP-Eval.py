import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, roc_curve)
import warnings

warnings.filterwarnings('ignore')


# CERP Classifier (Weighted Voting) 
# Partitions features into r random subsets, builds a decision tree on each,
# repeats n_ensembles times, combines via training-AUC-weighted voting.
class CERPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, r=3, n_ensembles=11, random_state=42):
        self.r = r
        self.n_ensembles = n_ensembles
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.trees_ = []
        self.partitions_ = []
        self.tree_weights_ = []
        rng = np.random.RandomState(self.random_state)

        for ens in range(self.n_ensembles):
            indices = rng.permutation(n_features)
            partitions = np.array_split(indices, self.r)
            ensemble_trees = []
            ensemble_weights = []

            for part in partitions:
                if len(part) == 0:
                    continue
                tree = DecisionTreeClassifier(random_state=42)
                tree.fit(X[:, part], y)
                # Weight = training AUC
                train_proba = tree.predict_proba(X[:, part])
                if train_proba.shape[1] == 2:
                    weight = roc_auc_score(y, train_proba[:, 1])
                else:
                    weight = 0.5
                ensemble_trees.append(tree)
                ensemble_weights.append(weight)

            self.trees_.append(ensemble_trees)
            self.partitions_.append(partitions)
            self.tree_weights_.append(ensemble_weights)
        return self

    def predict_proba(self, X):
        weighted_sum = np.zeros((X.shape[0], 2))
        total_weight = 0

        for ens_idx in range(len(self.trees_)):
            partitions = self.partitions_[ens_idx]
            for tree_idx, tree in enumerate(self.trees_[ens_idx]):
                part = partitions[tree_idx]
                if len(part) == 0:
                    continue
                weight = self.tree_weights_[ens_idx][tree_idx]
                proba = tree.predict_proba(X[:, part])
                if proba.shape[1] == 2:
                    weighted_sum += weight * proba
                    total_weight += weight

        if total_weight > 0:
            weighted_sum /= total_weight
        return weighted_sum

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_params(self, deep=True):
        return {'r': self.r, 'n_ensembles': self.n_ensembles,
                'random_state': self.random_state}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


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

# EN selector search space 
EN_PARAM_DIST = {
    'C': loguniform(0.001, 10),
    'l1_ratio': uniform(0.0, 1.0)
}

# CERP hyperparameter grid
# r: number of feature partitions per ensemble
# n_ensembles: number of random partition repetitions
CERP_PARAM_GRID = {
    'r': [3, 5, 7],
    'n_ensembles': [11, 15, 17, 19, 21]
}

# Storage
repeat_summaries = []
fold_level_metrics = []
en_params_history = []
cerp_params_history = []
feature_selection_counts = np.zeros(X.shape[1])
mean_fpr = np.linspace(0, 1, 100)
all_tprs = []

print("=" * 60)
print("EN Feature Selection -> CERP (Weighted Voting) Evaluation")
print("=" * 60)

for r in range(N_REPEATS):
    repeat_y_true, repeat_y_probs, repeat_y_preds = [], [], []
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True,
                               random_state=RANDOM_SEED + r)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        x_train_outer, x_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        # Scale features (EN selector requires scaling;
        # CERP uses decision trees which are scale-invariant)
        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train_outer)
        x_test_sc = scaler.transform(x_test_outer)

        # EN Feature Selection (inside CV) 
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

        # CERP Evaluation on selected features 
        inner_cv_cerp = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                        random_state=RANDOM_SEED + r)
        cerp_search = GridSearchCV(
            CERPClassifier(random_state=42),
            param_grid=CERP_PARAM_GRID,
            cv=inner_cv_cerp, scoring='roc_auc', n_jobs=-1
        )
        cerp_search.fit(x_train_sel, y_train_outer)
        best_cerp = cerp_search.best_estimator_
        cerp_params_history.append(cerp_search.best_params_)

        # Predict on test fold
        probs = best_cerp.predict_proba(x_test_sel)[:, 1]
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
            'Best_CERP_Params': str(cerp_search.best_params_)
        })

        # ROC interpolation
        fpr, tpr, _ = roc_curve(y_test_outer, probs)
        all_tprs.append(np.interp(mean_fpr, fpr, tpr))
        all_tprs[-1][0] = 0.0

        print(f"  R{r+1} F{fold_idx+1:2d}/10 | AUC={fold_auc:.4f} | "
              f"Sens={fold_sens:.3f} Spec={fold_spec:.3f} | "
              f"N_sel={n_selected} | r={cerp_search.best_params_['r']} "
              f"ens={cerp_search.best_params_['n_ensembles']}")

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
    print(f"  Repeat {r+1} EN->CERP | AUC={rep_auc:.4f} | "
          f"Sens={tp/(tp+fn):.3f} Spec={tn/(tn+fp):.3f} | "
          f"Total={tn+fp+fn+tp}\n")

# ===== Save results =====

pd.DataFrame(repeat_summaries).to_csv(
    'EN_Selection_CERP_Evaluation_Summary.csv', index=False)
pd.DataFrame(fold_level_metrics).to_csv(
    'EN_Selection_CERP_Fold_Metrics.csv', index=False)

mean_tpr = np.mean(all_tprs, axis=0)
mean_tpr[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv(
    'EN_Selection_CERP_ROC_Data.csv', index=False)

# EN selector parameter frequency
en_params_df = pd.DataFrame(en_params_history)
with open('EN_Selection_CERP_EN_Param_Frequencies.txt', 'w') as f:
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

# CERP parameter frequency
cerp_params_df = pd.DataFrame(cerp_params_history)
with open('EN_Selection_CERP_CERP_Param_Frequencies.txt', 'w') as f:
    f.write("CERP Evaluator Parameter Frequency Analysis\n")
    f.write(f"Total folds: {len(cerp_params_history)}\n")
    for col in cerp_params_df.columns:
        f.write(f"\nParameter: {col}\n")
        vc = cerp_params_df[col].astype(str).value_counts()
        for val, cnt in vc.items():
            f.write(f"  {val}: {cnt} ({cnt/len(cerp_params_df)*100:.1f}%)\n")

# Print summary
print("=" * 60)
print("EN Selection -> CERP (Weighted Voting) Summary")
print("=" * 60)
rep_df = pd.DataFrame(repeat_summaries)
for col in ['AUC', 'Accuracy', 'Sens', 'Spec', 'PPV', 'NPV', 'F1']:
    print(f"{col}: {rep_df[col].mean():.4f} +/- {rep_df[col].std():.4f}")
print(f"|S-S|: {abs(rep_df.Sens.mean() - rep_df.Spec.mean()):.4f}")

fold_df = pd.DataFrame(fold_level_metrics)
print(f"\nAvg features selected per fold: {fold_df.N_Selected.mean():.0f} "
      f"(range: {fold_df.N_Selected.min()}-{fold_df.N_Selected.max()})")

print("\nFiles saved:")
print("  EN_Selection_CERP_Evaluation_Summary.csv")
print("  EN_Selection_CERP_Fold_Metrics.csv")
print("  EN_Selection_CERP_ROC_Data.csv")
print("  EN_Selection_CERP_EN_Param_Frequencies.txt")
print("  EN_Selection_CERP_CERP_Param_Frequencies.txt")
