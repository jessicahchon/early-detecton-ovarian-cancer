# XGB FS + CERP Evaluation 
# 1-SE pruning + Weighted Voting only
# 7 Repeats x 10 Outer x 5 Inner, Threshold 0.5

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, roc_curve)
from scipy.stats import randint, loguniform
from itertools import product
from joblib import Parallel, delayed
from collections import Counter
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
    """CERP - Moon et al. (2007) with 1-SE cost-complexity pruning, weighted voting"""
    def __init__(self, n_partitions=None, n_ensembles=15,
                 min_samples_leaf=5, random_state=None):
        self.n_ensembles = nearest_odd(n_ensembles)
        self.n_partitions = n_partitions
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.ensembles_ = []
        self.feature_partitions_ = []
        self.tree_weights_ = []

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

    def fit(self, X, y, r=None):
        rng = np.random.default_rng(self.random_state)
        self.optimal_r_ = r if r else (self.n_partitions or 5)
        if self.optimal_r_ % 2 == 0:
            self.optimal_r_ += 1

        self.ensembles_, self.feature_partitions_, self.tree_weights_ = [], [], []
        for _ in range(self.n_ensembles):
            partitions = self._create_partitions(X.shape[1], self.optimal_r_, rng)
            trees, features, weights = [], [], []
            for part_idx in partitions:
                tree = self._build_tree(X[:, part_idx], y, rng)
                trees.append(tree)
                features.append(part_idx)
                try:
                    w = roc_auc_score(y, tree.predict_proba(X[:, part_idx])[:, 1])
                except:
                    w = 0.5
                weights.append(w)
            if len(trees) % 2 == 0 and len(trees) > 1:
                trees, features, weights = trees[:-1], features[:-1], weights[:-1]
            self.ensembles_.append(trees)
            self.feature_partitions_.append(features)
            self.tree_weights_.append(weights)
        return self

    def predict_proba(self, X):
        """Weighted voting by training AUC"""
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
        """Majority voting: binary vote within ensemble, average across ensembles"""
        n = X.shape[0]
        ensemble_preds = []
        for trees, fsets in zip(self.ensembles_, self.feature_partitions_):
            votes = np.zeros(n)
            for tree, feats in zip(trees, fsets):
                votes += tree.predict(X[:, feats])
            ensemble_preds.append((votes > len(trees) / 2).astype(int))
        prob_pos = np.array(ensemble_preds).mean(axis=0)
        return np.column_stack([1 - prob_pos, prob_pos])


# Configurations
N_REPEATS = 7
N_OUTER = 10
N_INNER = 5
N_ITER_SEARCH = 20
RANDOM_SEED = 42
THRESHOLD = 0.5

XGB_SEL_PARAM_DIST = {
    'n_estimators': randint(300, 601),
    'max_depth': [3, 5, 7, 9],
    'learning_rate': loguniform(0.001, 0.1)
}
XGB_SEL_FIXED = {
    'random_state': 42, 'eval_metric': 'logloss',
    'subsample': 1.0, 'verbosity': 0
}
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
            aucs.append(roc_auc_score(y_tv[va], clf.predict_proba(X_tv[va])[:, 1]))
        except:
            aucs.append(0.5)
    return r_val, n_ens, np.mean(aucs)

# Data 
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X = df[feat_names].values

print(f"\n{'='*80}")
print("XGB FS + CERP (1-SE pruning, Weighted Voting)")
print(f"  {N_REPEATS}R x {N_OUTER}F outer x {N_INNER}F inner")
print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  CERP Grid: r={R_CANDIDATES}, n_ens={N_ENS_CANDIDATES}")
print(f"{'='*80}\n", flush=True)

all_results_wt = []
all_results_maj = []
xgb_params_history = []
cerp_params_history = []
feature_selection_counts = np.zeros(X.shape[1])
mean_fpr = np.linspace(0, 1, 100)
all_tprs_wt = []
all_tprs_maj = []

total_start = time.time()

for repeat in range(N_REPEATS):
    print(f"REPEAT {repeat+1}/{N_REPEATS}")
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True,
                               random_state=RANDOM_SEED + repeat)

    for fold_idx, (tv_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        fold_start = time.time()
        X_tv, X_te = X[tv_idx], X[te_idx]
        y_tv, y_te = y[tv_idx], y[te_idx]

        # XGB Feature Selection (XGB is tree-based but scaling kept for consistency)
        scaler = StandardScaler()
        X_tv_sc = scaler.fit_transform(X_tv)
        X_te_sc = scaler.transform(X_te)

        xgb_search = RandomizedSearchCV(
            XGBClassifier(**XGB_SEL_FIXED),
            param_distributions=XGB_SEL_PARAM_DIST, n_iter=N_ITER_SEARCH,
            cv=StratifiedKFold(n_splits=N_INNER, shuffle=True,
                               random_state=RANDOM_SEED + repeat),
            scoring='roc_auc', n_jobs=-1,
            random_state=RANDOM_SEED + repeat + fold_idx
        )
        xgb_search.fit(X_tv_sc, y_tv)
        xgb_params_history.append(xgb_search.best_params_)

        selector = SelectFromModel(xgb_search.best_estimator_,
                                   threshold='mean', prefit=True)
        sel_mask = selector.get_support()
        n_sel = sel_mask.sum()
        feature_selection_counts += sel_mask.astype(int)

        # CERP uses unscaled data
        X_tv_sel = X_tv[:, sel_mask]
        X_te_sel = X_te[:, sel_mask]

        # CERP tuning (parallelized inner CV)
        inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                   random_state=RANDOM_SEED + repeat * 100 + fold_idx)
        seed = RANDOM_SEED + repeat * 100 + fold_idx

        param_combos = [(r, n) for r, n in product(R_CANDIDATES, N_ENS_CANDIDATES)
                        if n_sel // r >= 2]
        if len(param_combos) == 0:
            param_combos = [(3, 11)]

        results_par = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_one_param)(r, n, X_tv_sel, y_tv, inner_cv, seed)
            for r, n in param_combos
        )

        best_params, best_auc = None, -1
        for r_val, n_ens, mean_auc in results_par:
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = {'r': r_val, 'n_ensembles': n_ens}

        if best_params is None:
            best_params = {'r': 3, 'n_ensembles': 11}
        cerp_params_history.append(best_params)

        # Final model
        final = CERPClassifier(n_ensembles=best_params['n_ensembles'],
                               random_state=RANDOM_SEED + repeat * 100 + fold_idx)
        final.fit(X_tv_sel, y_tv, r=best_params['r'])

        # Weighted Voting
        y_prob_wt = final.predict_proba(X_te_sel)[:, 1]
        y_pred_wt = (y_prob_wt >= THRESHOLD).astype(int)
        m_wt = get_metrics(y_te, y_pred_wt, y_prob_wt)
        m_wt.update({'Repeat': repeat+1, 'Outer_Fold': fold_idx+1,
                     'N_Selected': n_sel, 'Best_Params': str(best_params)})
        all_results_wt.append(m_wt)

        fpr_wt, tpr_wt, _ = roc_curve(y_te, y_prob_wt)
        all_tprs_wt.append(np.interp(mean_fpr, fpr_wt, tpr_wt))
        all_tprs_wt[-1][0] = 0.0

        # Majority Voting
        y_prob_maj = final.predict_proba_majority(X_te_sel)[:, 1]
        y_pred_maj = (y_prob_maj >= THRESHOLD).astype(int)
        m_maj = get_metrics(y_te, y_pred_maj, y_prob_maj)
        m_maj.update({'Repeat': repeat+1, 'Outer_Fold': fold_idx+1,
                      'N_Selected': n_sel, 'Best_Params': str(best_params)})
        all_results_maj.append(m_maj)

        fpr_maj, tpr_maj, _ = roc_curve(y_te, y_prob_maj)
        all_tprs_maj.append(np.interp(mean_fpr, fpr_maj, tpr_maj))
        all_tprs_maj[-1][0] = 0.0

        print(f"  F{fold_idx+1}: n={n_sel}, r={best_params['r']}, "
              f"ens={best_params['n_ensembles']}, "
              f"MV={m_maj['AUC']:.3f}, WV={m_wt['AUC']:.3f} "
              f"({time.time()-fold_start:.0f}s)", flush=True)

total_time = (time.time() - total_start) / 60

# Save results
wt_df = pd.DataFrame(all_results_wt)
wt_df.to_csv('XGB_Selection_CERP_weighted_results.csv', index=False)
maj_df = pd.DataFrame(all_results_maj)
maj_df.to_csv('XGB_Selection_CERP_majority_results.csv', index=False)

mean_tpr_wt = np.mean(all_tprs_wt, axis=0); mean_tpr_wt[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr_wt}).to_csv(
    'XGB_Selection_CERP_weighted_ROC_Data.csv', index=False)
mean_tpr_maj = np.mean(all_tprs_maj, axis=0); mean_tpr_maj[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr_maj}).to_csv(
    'XGB_Selection_CERP_majority_ROC_Data.csv', index=False)

xgb_df = pd.DataFrame(xgb_params_history)
with open('XGB_Selection_CERP_XGB_Param_Frequencies.txt', 'w') as f:
    f.write("XGB Selector Parameter Frequency\n")
    f.write(f"Total folds: {len(xgb_params_history)}\n\n")
    for col in xgb_df.columns:
        f.write(f"Parameter: {col}\n")
        if xgb_df[col].dtype in [np.float64] and xgb_df[col].nunique() > 10:
            f.write(f"  Min: {xgb_df[col].min():.6f}, Median: {xgb_df[col].median():.6f}, "
                    f"Max: {xgb_df[col].max():.6f}\n")
        else:
            vc = xgb_df[col].astype(str).value_counts()
            for val, cnt in vc.items():
                f.write(f"  {val}: {cnt} ({cnt/len(xgb_df)*100:.1f}%)\n")
        f.write("\n")

r_counts, ens_counts = Counter(), Counter()
for p in cerp_params_history:
    r_counts[p['r']] += 1; ens_counts[p['n_ensembles']] += 1
with open('XGB_Selection_CERP_CERP_Param_Frequencies.txt', 'w') as f:
    f.write(f"r: {dict(r_counts)}\nn_ensembles: {dict(ens_counts)}\n")

# Summary 
for label, res_df in [("Majority", maj_df), ("Weighted", wt_df)]:
    print(f"\n{'='*60}")
    print(f"XGB FS + CERP {label} (Mean +/- SD, 70 folds)")
    print(f"{'='*60}")
    for col in ['AUC','Acc','Sens','Spec','PPV','NPV','F1']:
        print(f"  {col:5}: {res_df[col].mean():.4f} +/- {res_df[col].std():.4f}")
    print(f"  |S-S|: {abs(res_df['Sens'].mean() - res_df['Spec'].mean()):.4f}")
print(f"  Features: {res_df['N_Selected'].mean():.0f} +/- {res_df['N_Selected'].std():.0f}")
print(f"  r: {dict(r_counts)}, n_ens: {dict(ens_counts)}")
print(f"  Runtime: {total_time:.1f} min")
