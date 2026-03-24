# Union Feature Set Creation + CERP Evaluation
# 1-SE pruning + Weighted Voting + Parallelized
# 7 Repeats x 10 Outer x 5 Inner, Threshold 0.5
#
# Build union set from 5 FS methods (MW, EN, RF, XGB, CERP)
# Define tiers by consensus count
# Evaluate each tier with corrected CERPClassifier

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


# Load Feature Selection Frequencies
print("Feature Selection Results")

# MW: Selection_Count = number of folds (out of 70) feature was selected
mw_df = pd.read_csv('mannwhitney_top_features.csv')
mw_freq = dict(zip(mw_df['Biomarker'], mw_df['Selection_Count']))
print(f"MW:   {len(mw_df)} features total, {(mw_df['Selection_Count']>=70).sum()} at 70/70")

# EN: Count = number of folds feature was selected
en_df = pd.read_csv('EN_Selection_Feature_Counts.csv')
en_freq = dict(zip(en_df['Feature'], en_df['Count']))
print(f"EN:   {(en_df['Count']>0).sum()} features selected, {(en_df['Count']>=70).sum()} at 70/70")

# RF: Count = number of folds feature was selected
rf_df = pd.read_csv('RF_Selection_Feature_Counts.csv')
rf_freq = dict(zip(rf_df['Feature'], rf_df['Count']))
print(f"RF:   {(rf_df['Count']>0).sum()} features selected, {(rf_df['Count']>=70).sum()} at 70/70")

# XGB: Count = number of folds feature was selected
xgb_df = pd.read_csv('XGB_Selection_XGB_Feature_Counts.csv')
xgb_freq = dict(zip(xgb_df['Feature'], xgb_df['Count']))
print(f"XGB:  {(xgb_df['Count']>0).sum()} features selected, {(xgb_df['Count']>=70).sum()} at 70/70")

# CERP: count unique folds each feature appeared in (out of 70)
cerp_log = pd.read_csv('CERP_standalone_tree_selection_log.csv')
cerp_fold = cerp_log.drop_duplicates(subset=['Feature', 'Repeat', 'Fold'])
cerp_fold_counts = cerp_fold.groupby('Feature').size().reset_index(name='Fold_Count')
cerp_freq = dict(zip(cerp_fold_counts['Feature'], cerp_fold_counts['Fold_Count']))
print(f"CERP: {len(cerp_freq)} features used, {sum(1 for v in cerp_freq.values() if v>=70)} at 70/70")

# All feature names
all_features = set()
all_features.update(mw_df['Biomarker'])
all_features.update(en_df['Feature'])
all_features.update(rf_df['Feature'])
all_features.update(xgb_df['Feature'])
all_features.update(cerp_fold_counts['Feature'])

# Build comparison matrix
comp_df = pd.DataFrame({'miRNA': sorted(all_features)})
comp_df['MW_Freq'] = comp_df['miRNA'].map(mw_freq).fillna(0).astype(int)
comp_df['EN_Freq'] = comp_df['miRNA'].map(en_freq).fillna(0).astype(int)
comp_df['RF_Freq'] = comp_df['miRNA'].map(rf_freq).fillna(0).astype(int)
comp_df['XGB_Freq'] = comp_df['miRNA'].map(xgb_freq).fillna(0).astype(int)
comp_df['CERP_Freq'] = comp_df['miRNA'].map(cerp_freq).fillna(0).astype(int)
comp_df.to_csv('Union_Feature_Frequencies.csv', index=False)
print(f"\nSaved: Union_Feature_Frequencies.csv ({len(comp_df)} features)")


# Define Tiers by Fold-Frequency Threshold
# Advisor's direction: Tier 1 = union of features at 70/70 per method
# Tier 2 = Tier 1 + features at 69/70, etc.

print("Defining Tiers (Fold-Frequency Threshold)")
print("\n Tier 1 = union of 70/70 features across all methods")
print("\n Tier 2 = Tier 1 + 69/70 features, etc.")


def get_union_at_threshold(threshold):
    # Get union of features where any method selected them >= threshold folds
    feats = set()
    for mirna, row in comp_df.iterrows():
        name = comp_df.loc[mirna, 'miRNA']
        if (comp_df.loc[mirna, 'MW_Freq'] >= threshold or
            comp_df.loc[mirna, 'EN_Freq'] >= threshold or
            comp_df.loc[mirna, 'RF_Freq'] >= threshold or
            comp_df.loc[mirna, 'XGB_Freq'] >= threshold or
            comp_df.loc[mirna, 'CERP_Freq'] >= threshold):
            feats.add(name)
    return feats

# Vectorized version for speed
def get_union_at_threshold_fast(threshold):
    mask = ((comp_df['MW_Freq'] >= threshold) |
            (comp_df['EN_Freq'] >= threshold) |
            (comp_df['RF_Freq'] >= threshold) |
            (comp_df['XGB_Freq'] >= threshold) |
            (comp_df['CERP_Freq'] >= threshold))
    return set(comp_df.loc[mask, 'miRNA'])

# Build tiers cumulatively: Only Tier 5 and 6 (Tier 1-4 already completed)
tier_thresholds = [66, 65]
tier_defs = {}
tier_labels = {66: 5, 65: 6}  # Map threshold to tier number
for min_thresh in tier_thresholds:
    feats = sorted(get_union_at_threshold_fast(min_thresh))
    if len(feats) > 0:
        i = tier_labels[min_thresh]
        label = f"Tier{i}_gte{min_thresh}"
        tier_defs[label] = feats
        print(f"  Tier {i} (>= {min_thresh}/70): {len(feats)} features")

# Save tier membership
tier_membership = pd.DataFrame({'miRNA': sorted(all_features)})
for label, feats in tier_defs.items():
    tier_membership[label] = tier_membership['miRNA'].isin(feats).astype(int)
tier_membership.to_csv('Union_Tier_Membership.csv', index=False)
print(f"\nSaved: Union_Tier_Membership.csv")

# =========================================================================
# Part 3: CERP Evaluation with Corrected Implementation
# =========================================================================

print("CERP Evaluation on Union Tiers")
print("\n 1-SE pruning + Weighted Voting + Parallelized")

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
    # CERP - Moon et al. (2007) with 1-SE cost-complexity pruning, weighted voting
    def __init__(self, n_ensembles=15, min_samples_leaf=5, random_state=None):
        self.n_ensembles = nearest_odd(n_ensembles)
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
            cv_stds.append(np.std(fold_errors))

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
        self.r_ = r if r else 5
        if self.r_ % 2 == 0:
            self.r_ += 1

        self.ensembles_, self.feature_partitions_, self.tree_weights_ = [], [], []

        for _ in range(self.n_ensembles):
            partitions = self._create_partitions(X.shape[1], self.r_, rng)
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
        n = X.shape[0]
        wsum, wtot = np.zeros(n), 0.0
        for trees, fsets, wts in zip(self.ensembles_, self.feature_partitions_, self.tree_weights_):
            for tree, feats, w in zip(trees, fsets, wts):
                p = tree.predict_proba(X[:, feats])
                wsum += w * (p[:, 1] if p.shape[1] == 2 else p.ravel())
                wtot += w
        prob = wsum / wtot if wtot > 0 else np.full(n, 0.5)
        return np.column_stack([1 - prob, prob])


def evaluate_one_param(r_val, n_ens, X_tv, y_tv, inner_cv, seed):
    # Evaluate one (r, n_ensembles) combo — for joblib parallel
    aucs = []
    for tr, va in inner_cv.split(X_tv, y_tv):
        clf = CERPClassifier(n_ensembles=n_ens, random_state=seed)
        clf.fit(X_tv[tr], y_tv[tr], r=r_val)
        try:
            aucs.append(roc_auc_score(y_tv[va], clf.predict_proba(X_tv[va])[:, 1]))
        except:
            aucs.append(0.5)
    return r_val, n_ens, np.mean(aucs)


# Config
N_REPEATS = 7
N_OUTER = 10
N_INNER = 5
SEED = 42
THRESHOLD = 0.5
N_JOBS = 4

R_CANDIDATES = [3, 5, 7]
N_ENS_CANDIDATES = [11, 15, 17, 19, 21]

# Data
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X_all = df[feat_names].values

# Evaluate each tier
all_tier_results = []

for tier_name, tier_feats in tier_defs.items():
    n_feat = len(tier_feats)
    print(f"\n{'='*60}")
    print(f"  {tier_name}: {n_feat} features")
    print(f"{'='*60}")

    # Get feature indices
    feat_idx = [i for i, f in enumerate(feat_names) if f in tier_feats]
    X = X_all[:, feat_idx]
    print(f"  X shape: {X.shape}", flush=True)

    tier_results = []
    mean_fpr = np.linspace(0, 1, 100)
    tier_tprs = []

    tier_start = time.time()

    for repeat in range(N_REPEATS):
        print(f"  REPEAT {repeat+1}/{N_REPEATS}")
        outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True,
                                   random_state=SEED + repeat)

        for fold_idx, (tv_idx, te_idx) in enumerate(outer_cv.split(X, y)):
            fold_start = time.time()
            X_tv, X_te = X[tv_idx], X[te_idx]
            y_tv, y_te = y[tv_idx], y[te_idx]

            # Inner CV tuning (parallelized)
            inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                       random_state=SEED + repeat * 100 + fold_idx)
            seed = SEED + repeat * 100 + fold_idx

            # Filter param combos where r is feasible
            param_combos = [(r, n) for r, n in product(R_CANDIDATES, N_ENS_CANDIDATES)
                            if n_feat // r >= 2]

            if len(param_combos) == 0:
                param_combos = [(3, 11)]

            results_par = Parallel(n_jobs=N_JOBS)(
                delayed(evaluate_one_param)(r, n, X_tv, y_tv, inner_cv, seed)
                for r, n in param_combos
            )

            best_params, best_auc = None, -1
            for r_val, n_ens, mean_auc in results_par:
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_params = {'r': r_val, 'n_ensembles': n_ens}

            if best_params is None:
                best_params = {'r': 3, 'n_ensembles': 11}

            # Final model
            final = CERPClassifier(n_ensembles=best_params['n_ensembles'],
                                   random_state=seed)
            final.fit(X_tv, y_tv, r=best_params['r'])

            y_prob = final.predict_proba(X_te)[:, 1]
            y_pred = (y_prob >= THRESHOLD).astype(int)
            m = get_metrics(y_te, y_pred, y_prob)
            m.update({'Repeat': repeat + 1, 'Outer_Fold': fold_idx + 1,
                      'N_Features': n_feat, 'Best_Params': str(best_params)})
            tier_results.append(m)

            fpr, tpr, _ = roc_curve(y_te, y_prob)
            tier_tprs.append(np.interp(mean_fpr, fpr, tpr))
            tier_tprs[-1][0] = 0.0

            print(f"    F{fold_idx+1}: r={best_params['r']}, "
                  f"ens={best_params['n_ensembles']}, "
                  f"AUC={m['AUC']:.3f} ({time.time()-fold_start:.0f}s)", flush=True)

    tier_time = (time.time() - tier_start) / 60
    res_df = pd.DataFrame(tier_results)

    # Save per-tier results
    safe_name = tier_name.replace(' ', '_').replace('(', '').replace(')', '').replace('>=', 'gte')
    res_df.to_csv(f'Union_CERP_{safe_name}_results.csv', index=False)

    mean_tpr = np.mean(tier_tprs, axis=0); mean_tpr[-1] = 1.0
    pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv(
        f'Union_CERP_{safe_name}_ROC_Data.csv', index=False)

    # Summary
    print(f"\n {tier_name} (Mean +/- SD, 70 folds)")
    for col in ['AUC', 'Acc', 'Sens', 'Spec', 'PPV', 'NPV', 'F1']:
        print(f"    {col:5}: {res_df[col].mean():.4f} +/- {res_df[col].std():.4f}")
    print(f"    |S-S|: {abs(res_df['Sens'].mean() - res_df['Spec'].mean()):.4f}")
    print(f"    Runtime: {tier_time:.1f} min")

    all_tier_results.append({
        'Tier': tier_name,
        'N_Features': n_feat,
        'AUC_mean': res_df['AUC'].mean(),
        'AUC_std': res_df['AUC'].std(),
        'Acc_mean': res_df['Acc'].mean(),
        'Sens_mean': res_df['Sens'].mean(),
        'Spec_mean': res_df['Spec'].mean(),
        'PPV_mean': res_df['PPV'].mean(),
        'NPV_mean': res_df['NPV'].mean(),
        'F1_mean': res_df['F1'].mean(),
        'SensSpec_diff': abs(res_df['Sens'].mean() - res_df['Spec'].mean()),
        'Runtime_min': tier_time
    })

# Final summary
print("Final Summary: Union CERP by Tier")
summary_df = pd.DataFrame(all_tier_results)
summary_df.to_csv('Union_CERP_Tier_Summary.csv', index=False)
print(summary_df.to_string(index=False))
print(f"\nSaved: Union_CERP_Tier_Summary.csv")
