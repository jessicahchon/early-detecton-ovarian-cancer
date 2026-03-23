# Mann-Whitney FS + CERP Evaluation
# Majority Voting & Weighted Voting
# 7 Repeats x 10 Outer x 5 Inner, Threshold 0.5

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score, roc_curve
from scipy.stats import mannwhitneyu
from itertools import product
from collections import Counter
from joblib import Parallel, delayed
import ast
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

def nearest_odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


class CERPClassifier:
    """CERP - Moon et al. (2007)"""
    def __init__(self, n_partitions=None, n_ensembles=15,
                 min_samples_leaf=5, random_state=None):
        self.n_ensembles = nearest_odd(n_ensembles)
        self.n_partitions = n_partitions
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.ensembles_ = []
        self.feature_partitions_ = []
        self.tree_weights_ = []  # training AUC per tree for weighted voting
        self.tree_selection_log_ = []

    def _create_partitions(self, n_features, r, rng):
        indices = rng.permutation(n_features)
        subset_size = max(1, n_features // r)
        partitions = []
        for i in range(0, n_features, subset_size):
            partition = indices[i:i + subset_size]
            if len(partition) >= 1:
                partitions.append(partition)
        if len(partitions) % 2 == 0 and len(partitions) > 1:
            partitions = partitions[:-1]
        return partitions

    def _build_tree(self, X, y, rng):
        full_tree = DecisionTreeClassifier(
            criterion='gini', min_samples_leaf=self.min_samples_leaf,
            random_state=rng.integers(0, 2**31))
        full_tree.fit(X, y)

        path = full_tree.cost_complexity_pruning_path(X, y)
        ccp_alphas = path.ccp_alphas[path.ccp_alphas >= 0]
        if len(ccp_alphas) <= 1:
            return full_tree
        if len(ccp_alphas) > 10:
            ccp_alphas = ccp_alphas[np.linspace(0, len(ccp_alphas) - 1, 10, dtype=int)]

        min_class_count = np.min(np.bincount(y))
        if min_class_count < 2:
            return full_tree

        # 1-SE Rule pruning
        n_folds = min(10, min_class_count)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng.integers(0, 2**31))
        cv_errors, cv_stds = [], []

        for alpha in ccp_alphas:
            fold_errors = []
            for tr_idx, val_idx in cv.split(X, y):
                tree = DecisionTreeClassifier(
                    criterion='gini', min_samples_leaf=self.min_samples_leaf,
                    ccp_alpha=alpha, random_state=rng.integers(0, 2**31))
                tree.fit(X[tr_idx], y[tr_idx])
                fold_errors.append(1 - tree.score(X[val_idx], y[val_idx]))
            cv_errors.append(np.mean(fold_errors))
            cv_stds.append(np.std(fold_errors) / np.sqrt(len(fold_errors)))

        min_idx = np.argmin(cv_errors)
        threshold = cv_errors[min_idx] + cv_stds[min_idx]
        valid_indices = np.where(cv_errors <= threshold)[0]
        best_alpha = ccp_alphas[valid_indices[-1]]

        optimal_tree = DecisionTreeClassifier(
            criterion='gini', min_samples_leaf=self.min_samples_leaf,
            ccp_alpha=best_alpha, random_state=rng.integers(0, 2**31))
        optimal_tree.fit(X, y)
        return optimal_tree

    def fit(self, X, y, r=None, feature_names=None):
        rng = np.random.default_rng(self.random_state)
        self.optimal_r_ = r if r is not None else (self.n_partitions if self.n_partitions else 5)
        if self.optimal_r_ % 2 == 0:
            self.optimal_r_ += 1

        self.ensembles_, self.feature_partitions_ = [], []
        self.tree_weights_, self.tree_selection_log_ = [], []

        for e_idx in range(self.n_ensembles):
            partitions = self._create_partitions(X.shape[1], self.optimal_r_, rng)
            trees, features, weights = [], [], []
            for t_idx, part_idx in enumerate(partitions):
                tree = self._build_tree(X[:, part_idx], y, rng)
                trees.append(tree)
                features.append(part_idx)

                # Compute training AUC as weight for weighted voting
                try:
                    train_prob = tree.predict_proba(X[:, part_idx])[:, 1]
                    tree_auc = roc_auc_score(y, train_prob)
                except:
                    tree_auc = 0.5
                weights.append(tree_auc)

                used_in_tree = set(tree.tree_.feature)
                if -2 in used_in_tree:
                    used_in_tree.remove(-2)
                if feature_names is not None:
                    for f_idx in part_idx[list(used_in_tree)]:
                        self.tree_selection_log_.append({
                            'Ensemble': e_idx + 1, 'Tree': t_idx + 1,
                            'Feature': feature_names[f_idx]
                        })

            if len(trees) % 2 == 0 and len(trees) > 1:
                trees, features, weights = trees[:-1], features[:-1], weights[:-1]
            self.ensembles_.append(trees)
            self.feature_partitions_.append(features)
            self.tree_weights_.append(weights)
        return self

    def predict_proba_majority(self, X):
        """
        Majority Voting (Hard Voting):
        Each tree casts a binary vote (0 or 1). Within each ensemble,
        the class with more than half the votes wins. The final prediction
        is the proportion of ensembles that voted for class 1.
        """
        n_samples = X.shape[0]
        ensemble_preds = []
        for trees, feature_sets in zip(self.ensembles_, self.feature_partitions_):
            votes = np.zeros(n_samples)
            for tree, feats in zip(trees, feature_sets):
                votes += tree.predict(X[:, feats])
            ensemble_preds.append((votes > len(trees) / 2).astype(int))
        prob_pos = np.array(ensemble_preds).mean(axis=0)
        return np.column_stack([1 - prob_pos, prob_pos])

    def predict_proba_weighted(self, X):
        """
        Weighted Voting (Performance-Based):
        Each tree's predicted probability is weighted by its training AUC.
        Trees that performed better on training data have more influence.
        Final probability = sum(weight_i * prob_i) / sum(weight_i),
        where weight_i is the training AUC of tree i.
        """
        n_samples = X.shape[0]
        weighted_sum = np.zeros(n_samples)
        total_weight = 0.0

        for trees, feature_sets, weights in zip(
                self.ensembles_, self.feature_partitions_, self.tree_weights_):
            for tree, feats, w in zip(trees, feature_sets, weights):
                proba = tree.predict_proba(X[:, feats])
                if proba.shape[1] == 2:
                    prob_pos = proba[:, 1]
                else:
                    prob_pos = proba.ravel()
                weighted_sum += w * prob_pos
                total_weight += w

        if total_weight == 0:
            prob_pos = np.full(n_samples, 0.5)
        else:
            prob_pos = weighted_sum / total_weight

        return np.column_stack([1 - prob_pos, prob_pos])


# Config
N_REPEATS = 7
N_OUTER = 10
N_INNER = 5
ALPHA = 0.05
THRESHOLD = 0.5
R_CANDIDATES = [3, 5, 7]
N_ENS_CANDIDATES = [11, 15, 17, 19, 21]
N_JOBS = 4


def evaluate_one_param(r_val, n_ens, X_tv, y_tv, inner_cv, seed):
    """Evaluate one (r, n_ensembles) combo via inner CV — for parallel"""
    aucs = []
    for tr, va in inner_cv.split(X_tv, y_tv):
        clf = CERPClassifier(n_ensembles=n_ens, random_state=seed)
        clf.fit(X_tv[tr], y_tv[tr], r=r_val)
        try:
            aucs.append(roc_auc_score(y_tv[va], clf.predict_proba_weighted(X_tv[va])[:, 1]))
        except:
            aucs.append(0.5)
    return r_val, n_ens, np.mean(aucs)

# Data
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X = df[feat_names].values

print(f"\n{'='*80}")
print("Mann-Whitney FS + CERP Evaluation")
print(f"  Feature Selection: Mann-Whitney U test (p < 0.05)")
print(f"  Voting: Majority & Weighted")
print(f"  Structure: {N_REPEATS} Repeats x {N_OUTER} Outer x {N_INNER} Inner")
print(f"{'='*80}")
print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class: {np.sum(y==1)} Cancer, {np.sum(y==0)} Control")
print(f"CERP Grid: r={R_CANDIDATES}, n_ens={N_ENS_CANDIDATES}")
print(flush=True)

# Main loop
all_majority_results = []
all_weighted_results = []
mean_fpr = np.linspace(0, 1, 100)
all_tprs_maj = []
all_tprs_wt = []

total_start = time.time()

for repeat in range(N_REPEATS):
    print(f"\nREPEAT {repeat+1}/{N_REPEATS}")
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + repeat)

    for outer_fold, (trainval_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        fold_start = time.time()

        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx], y[test_idx]

        # Mann-Whitney FS on training data only
        selected_idx, pvals = mannwhitney_selection(X_trainval, y_trainval, alpha=ALPHA)
        n_selected = len(selected_idx)
        if n_selected == 0:
            selected_idx = np.argsort(pvals)[:50]
            n_selected = 50

        X_trainval_sel = X_trainval[:, selected_idx]
        X_test_sel = X_test[:, selected_idx]
        sel_feat_names = feat_names[selected_idx]

        # Inner CV for CERP hyperparameter tuning (parallelized)
        inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                   random_state=42 + repeat * 100 + outer_fold)
        seed = 42 + repeat * 100 + outer_fold

        param_combos = [(r, n) for r, n in product(R_CANDIDATES, N_ENS_CANDIDATES)
                        if n_selected // r >= 2]
        if len(param_combos) == 0:
            param_combos = [(3, 11)]

        results_par = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_one_param)(r, n, X_trainval_sel, y_trainval, inner_cv, seed)
            for r, n in param_combos
        )

        best_params = None
        best_inner_auc = -1
        for r_val, n_ens_val, mean_auc in results_par:
            if mean_auc > best_inner_auc:
                best_inner_auc = mean_auc
                best_params = {'r': r_val, 'n_ensembles': n_ens_val}

        # Train final model
        final_clf = CERPClassifier(
            n_partitions=best_params['r'],
            n_ensembles=best_params['n_ensembles'],
            random_state=42 + repeat * 100 + outer_fold)
        final_clf.fit(X_trainval_sel, y_trainval, r=best_params['r'],
                      feature_names=sel_feat_names)

        # Majority Voting
        y_prob_maj = final_clf.predict_proba_majority(X_test_sel)[:, 1]
        y_pred_maj = (y_prob_maj >= THRESHOLD).astype(int)
        metrics_maj = get_metrics(y_test, y_pred_maj, y_prob_maj)
        metrics_maj['Repeat'] = repeat + 1
        metrics_maj['Outer_Fold'] = outer_fold + 1
        metrics_maj['N_Selected'] = n_selected
        metrics_maj['Best_Params'] = str(best_params)
        all_majority_results.append(metrics_maj)

        fpr_maj, tpr_maj, _ = roc_curve(y_test, y_prob_maj)
        all_tprs_maj.append(np.interp(mean_fpr, fpr_maj, tpr_maj))
        all_tprs_maj[-1][0] = 0.0

        # Weighted Voting
        y_prob_wt = final_clf.predict_proba_weighted(X_test_sel)[:, 1]
        y_pred_wt = (y_prob_wt >= THRESHOLD).astype(int)
        metrics_wt = get_metrics(y_test, y_pred_wt, y_prob_wt)
        metrics_wt['Repeat'] = repeat + 1
        metrics_wt['Outer_Fold'] = outer_fold + 1
        metrics_wt['N_Selected'] = n_selected
        metrics_wt['Best_Params'] = str(best_params)
        all_weighted_results.append(metrics_wt)

        fpr_wt, tpr_wt, _ = roc_curve(y_test, y_prob_wt)
        all_tprs_wt.append(np.interp(mean_fpr, fpr_wt, tpr_wt))
        all_tprs_wt[-1][0] = 0.0

        fold_time = time.time() - fold_start
        print(f"  Fold {outer_fold+1}: n={n_selected}, r={best_params['r']}, "
              f"b={best_params['n_ensembles']}, "
              f"MV={metrics_maj['AUC']:.3f}, WV={metrics_wt['AUC']:.3f} "
              f"({fold_time:.0f}s)", flush=True)

total_time = (time.time() - total_start) / 60

# Summary
maj_df = pd.DataFrame(all_majority_results)
wt_df = pd.DataFrame(all_weighted_results)

for label, res_df in [("Majority Voting", maj_df), ("Weighted Voting", wt_df)]:
    print(f"\n--- {label} (Mean +/- SD) ---")
    print(f"  AUC:         {res_df['AUC'].mean():.4f} +/- {res_df['AUC'].std():.4f}")
    print(f"  Accuracy:    {res_df['Acc'].mean():.4f} +/- {res_df['Acc'].std():.4f}")
    print(f"  Sensitivity: {res_df['Sens'].mean():.4f} +/- {res_df['Sens'].std():.4f}")
    print(f"  Specificity: {res_df['Spec'].mean():.4f} +/- {res_df['Spec'].std():.4f}")
    print(f"  PPV:         {res_df['PPV'].mean():.4f} +/- {res_df['PPV'].std():.4f}")
    print(f"  NPV:         {res_df['NPV'].mean():.4f} +/- {res_df['NPV'].std():.4f}")
    print(f"  F1:          {res_df['F1'].mean():.4f} +/- {res_df['F1'].std():.4f}")
    print(f"  |Sens-Spec|: {abs(res_df['Sens'].mean() - res_df['Spec'].mean()):.4f}")

r_counts = Counter()
ens_counts = Counter()
for p in maj_df['Best_Params']:
    d = ast.literal_eval(p)
    r_counts[d['r']] += 1
    ens_counts[d['n_ensembles']] += 1

print(f"\nOptimal r: {dict(r_counts)}")
print(f"Optimal n_ensembles: {dict(ens_counts)}")
print(f"Features selected: {maj_df['N_Selected'].mean():.1f} +/- {maj_df['N_Selected'].std():.1f}")

# Save results
maj_df.to_csv('mannwhitney_CERP_majority_results.csv', index=False)
wt_df.to_csv('mannwhitney_CERP_weighted_results.csv', index=False)

mean_tpr_maj = np.mean(all_tprs_maj, axis=0)
mean_tpr_maj[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr_maj}).to_csv(
    'mannwhitney_CERP_majority_ROC_Data.csv', index=False)

mean_tpr_wt = np.mean(all_tprs_wt, axis=0)
mean_tpr_wt[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr_wt}).to_csv(
    'mannwhitney_CERP_weighted_ROC_Data.csv', index=False)

print(f"\nSaved: mannwhitney_CERP_majority_results.csv")
print(f"Saved: mannwhitney_CERP_weighted_results.csv")
print(f"Saved: mannwhitney_CERP_majority_ROC_Data.csv")
print(f"Saved: mannwhitney_CERP_weighted_ROC_Data.csv")
print(f"Runtime: {total_time:.1f} min")
