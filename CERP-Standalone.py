# CERP Standalone Evaluation (No Feature Selection)
# 1-SE pruning (Breiman et al. 1984) + Majority & Weighted Voting
# 7 Repeats x 10 Outer x 5 Inner, Threshold 0.5
# Parallelized inner CV with joblib

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, roc_curve)
from itertools import product
from collections import Counter
from joblib import Parallel, delayed
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


def nearest_odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


class CERPClassifier:
    # CERP - Moon et al. (2007) with 1-SE cost-complexity pruning (Breiman et al. 1984)
    def __init__(self, n_partitions=None, n_ensembles=15,
                 min_samples_leaf=5, random_state=None):
        self.n_ensembles = nearest_odd(n_ensembles)
        self.n_partitions = n_partitions
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.ensembles_ = []
        self.feature_partitions_ = []
        self.tree_weights_ = []
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
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=rng.integers(0, 2**31))
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
        self.optimal_r_ = r if r else (self.n_partitions or 5)
        if self.optimal_r_ % 2 == 0:
            self.optimal_r_ += 1

        self.ensembles_, self.feature_partitions_, self.tree_weights_ = [], [], []
        self.tree_selection_log_ = []

        for e_idx in range(self.n_ensembles):
            partitions = self._create_partitions(X.shape[1], self.optimal_r_, rng)
            trees, features, weights = [], [], []
            for t_idx, part_idx in enumerate(partitions):
                tree = self._build_tree(X[:, part_idx], y, rng)
                trees.append(tree)
                features.append(part_idx)
                try:
                    w = roc_auc_score(y, tree.predict_proba(X[:, part_idx])[:, 1])
                except:
                    w = 0.5
                weights.append(w)

                # Log features actually used in tree splits
                if feature_names is not None:
                    used = set(tree.tree_.feature)
                    if -2 in used:
                        used.remove(-2)
                    for f_idx in part_idx[list(used)]:
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

    def predict_proba(self, X):
        # Weighted voting by training AUC
        n = X.shape[0]
        wsum, wtot = np.zeros(n), 0.0
        for trees, fsets, wts in zip(self.ensembles_, self.feature_partitions_, self.tree_weights_):
            for tree, feats, w in zip(trees, fsets, wts):
                p = tree.predict_proba(X[:, feats])
                wsum += w * (p[:, 1] if p.shape[1] == 2 else p.ravel())
                wtot += w
        prob = wsum / wtot if wtot > 0 else np.full(n, 0.5)
        return np.column_stack([1 - prob, prob])

    def predict_proba_majority(self, X):
        # Majority voting: binary vote within ensemble, average across ensembles
        n = X.shape[0]
        ensemble_preds = []
        for trees, fsets in zip(self.ensembles_, self.feature_partitions_):
            votes = np.zeros(n)
            for tree, feats in zip(trees, fsets):
                votes += tree.predict(X[:, feats])
            ensemble_preds.append((votes > len(trees) / 2).astype(int))
        prob_pos = np.array(ensemble_preds).mean(axis=0)
        return np.column_stack([1 - prob_pos, prob_pos])


def evaluate_one_param(r_val, n_ens, X_tv, y_tv, inner_cv, seed):
    # Evaluate one (r, n_ensembles) combo via inner CV — for parallel execution
    aucs = []
    for tr, va in inner_cv.split(X_tv, y_tv):
        clf = CERPClassifier(n_ensembles=n_ens, random_state=seed)
        clf.fit(X_tv[tr], y_tv[tr], r=r_val)
        try:
            aucs.append(roc_auc_score(y_tv[va], clf.predict_proba(X_tv[va])[:, 1]))
        except:
            aucs.append(0.5)
    return r_val, n_ens, np.mean(aucs)


# Configurations
N_REPEATS = 7
N_OUTER = 10
N_INNER = 5
RANDOM_SEED = 42
THRESHOLD = 0.5
N_JOBS = 4  # M2 Air: 4 performance cores

R_CANDIDATES = [3, 5, 7]
N_ENS_CANDIDATES = [11, 15, 17, 19, 21]

# Data 
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X = df[feat_names].values

print(f"\n{'='*80}")
print("CERP Standalone (No FS, 1-SE pruning, Weighted Voting)")
print(f"  {N_REPEATS}R x {N_OUTER}F outer x {N_INNER}F inner")
print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features (ALL)")
print(f"  CERP Grid: r={R_CANDIDATES}, n_ens={N_ENS_CANDIDATES}")
print(f"  Parallel: {N_JOBS} jobs")
print(f"{'='*80}\n", flush=True)

all_results_wt = []
all_results_maj = []
cerp_params_history = []
mean_fpr = np.linspace(0, 1, 100)
all_tprs_wt = []
all_tprs_maj = []
all_tree_logs = []

total_start = time.time()

for repeat in range(N_REPEATS):
    print(f"REPEAT {repeat+1}/{N_REPEATS}")
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True,
                               random_state=RANDOM_SEED + repeat)

    for fold_idx, (tv_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        fold_start = time.time()
        X_tv, X_te = X[tv_idx], X[te_idx]
        y_tv, y_te = y[tv_idx], y[te_idx]

        # CERP hyperparameter tuning (parallelized inner CV)
        inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                   random_state=RANDOM_SEED + repeat * 100 + fold_idx)
        seed = RANDOM_SEED + repeat * 100 + fold_idx

        param_combos = [(r, n) for r, n in product(R_CANDIDATES, N_ENS_CANDIDATES)]

        results_parallel = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_one_param)(r, n, X_tv, y_tv, inner_cv, seed)
            for r, n in param_combos
        )

        best_params, best_auc = None, -1
        for r_val, n_ens, mean_auc in results_parallel:
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = {'r': r_val, 'n_ensembles': n_ens}

        if best_params is None:
            best_params = {'r': 3, 'n_ensembles': 11}
        cerp_params_history.append(best_params)

        # Final model on full training set
        final = CERPClassifier(n_ensembles=best_params['n_ensembles'],
                               random_state=seed)
        final.fit(X_tv, y_tv, r=best_params['r'], feature_names=feat_names)

        # Collect feature selection log
        for log in final.tree_selection_log_:
            log.update({'Repeat': repeat + 1, 'Fold': fold_idx + 1})
            all_tree_logs.append(log)

        # Weighted Voting
        y_prob_wt = final.predict_proba(X_te)[:, 1]
        y_pred_wt = (y_prob_wt >= THRESHOLD).astype(int)
        m_wt = get_metrics(y_te, y_pred_wt, y_prob_wt)
        m_wt.update({'Repeat': repeat + 1, 'Outer_Fold': fold_idx + 1,
                     'Best_Params': str(best_params)})
        all_results_wt.append(m_wt)

        fpr_wt, tpr_wt, _ = roc_curve(y_te, y_prob_wt)
        all_tprs_wt.append(np.interp(mean_fpr, fpr_wt, tpr_wt))
        all_tprs_wt[-1][0] = 0.0

        # Majority Voting
        y_prob_maj = final.predict_proba_majority(X_te)[:, 1]
        y_pred_maj = (y_prob_maj >= THRESHOLD).astype(int)
        m_maj = get_metrics(y_te, y_pred_maj, y_prob_maj)
        m_maj.update({'Repeat': repeat + 1, 'Outer_Fold': fold_idx + 1,
                      'Best_Params': str(best_params)})
        all_results_maj.append(m_maj)

        fpr_maj, tpr_maj, _ = roc_curve(y_te, y_prob_maj)
        all_tprs_maj.append(np.interp(mean_fpr, fpr_maj, tpr_maj))
        all_tprs_maj[-1][0] = 0.0

        print(f"  F{fold_idx+1}: r={best_params['r']}, "
              f"ens={best_params['n_ensembles']}, "
              f"MV={m_maj['AUC']:.3f}, WV={m_wt['AUC']:.3f} "
              f"({time.time()-fold_start:.0f}s)", flush=True)

total_time = (time.time() - total_start) / 60

# Save
wt_df = pd.DataFrame(all_results_wt)
wt_df.to_csv('CERP_standalone_weighted_results.csv', index=False)
maj_df = pd.DataFrame(all_results_maj)
maj_df.to_csv('CERP_standalone_majority_results.csv', index=False)

mean_tpr_wt = np.mean(all_tprs_wt, axis=0); mean_tpr_wt[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr_wt}).to_csv(
    'CERP_standalone_weighted_ROC_Data.csv', index=False)
mean_tpr_maj = np.mean(all_tprs_maj, axis=0); mean_tpr_maj[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr_maj}).to_csv(
    'CERP_standalone_majority_ROC_Data.csv', index=False)

# CERP param frequency
r_counts, ens_counts = Counter(), Counter()
for p in cerp_params_history:
    r_counts[p['r']] += 1; ens_counts[p['n_ensembles']] += 1
with open('CERP_standalone_Param_Frequencies.txt', 'w') as f:
    f.write(f"r: {dict(r_counts)}\nn_ensembles: {dict(ens_counts)}\n")

# Feature selection log (features used in tree splits)
log_df = pd.DataFrame(all_tree_logs)
log_df.to_csv('CERP_standalone_tree_selection_log.csv', index=False)

# Feature frequency summary
feat_freq = log_df['Feature'].value_counts().reset_index()
feat_freq.columns = ['Feature', 'Usage_Count']
feat_freq['Usage_Percent'] = feat_freq['Usage_Count'] / (N_REPEATS * N_OUTER) * 100
feat_freq.to_csv('CERP_standalone_feature_frequencies.csv', index=False)

# Summary
for label, res_df in [("Majority Voting", maj_df), ("Weighted Voting", wt_df)]:
    print(f"\n{'='*60}")
    print(f"CERP Standalone {label} (Mean +/- SD, 70 folds)")
    print(f"{'='*60}")
    for col in ['AUC', 'Acc', 'Sens', 'Spec', 'PPV', 'NPV', 'F1']:
        print(f"  {col:5}: {res_df[col].mean():.4f} +/- {res_df[col].std():.4f}")
    print(f"  |S-S|: {abs(res_df['Sens'].mean() - res_df['Spec'].mean()):.4f}")

print(f"\n  r: {dict(r_counts)}, n_ens: {dict(ens_counts)}")
print(f"  Unique features used: {log_df['Feature'].nunique()} / {len(feat_names)}")
print(f"  Runtime: {total_time:.1f} min")

print(f"\nSaved:")
print(f"  CERP_standalone_weighted_results.csv")
print(f"  CERP_standalone_majority_results.csv")
print(f"  CERP_standalone_weighted_ROC_Data.csv")
print(f"  CERP_standalone_majority_ROC_Data.csv")
print(f"  CERP_standalone_Param_Frequencies.txt")
print(f"  CERP_standalone_tree_selection_log.csv")
print(f"  CERP_standalone_feature_frequencies.csv")
