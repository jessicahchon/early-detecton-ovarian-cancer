# Quick CERP r-scan: Simple 10-fold CV x 1 repeat
# Purpose: Find approximate optimal r range for different feature set sizes
# NOT for final evaluation — just to set grid boundaries

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import warnings
import time

warnings.filterwarnings('ignore')


def nearest_odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


class CERPQuick:
    """Lightweight CERP for grid scanning (1-SE pruning, weighted voting)"""
    def __init__(self, n_ensembles=15, min_samples_leaf=5, random_state=None):
        self.n_ensembles = nearest_odd(n_ensembles)
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.ensembles_ = []
        self.feature_partitions_ = []
        self.tree_weights_ = []

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
        min_class = np.min(np.bincount(y))
        if min_class < 2:
            return full_tree
        n_folds = min(10, min_class)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng.integers(0, 2**31))
        cv_errors, cv_stds = [], []
        for alpha in ccp_alphas:
            errs = []
            for tr, va in cv.split(X, y):
                t = DecisionTreeClassifier(criterion='gini', min_samples_leaf=self.min_samples_leaf,
                                           ccp_alpha=alpha, random_state=rng.integers(0, 2**31))
                t.fit(X[tr], y[tr])
                errs.append(1 - t.score(X[va], y[va]))
            cv_errors.append(np.mean(errs))
            cv_stds.append(np.std(errs))
        min_idx = np.argmin(cv_errors)
        threshold = cv_errors[min_idx] + cv_stds[min_idx]
        valid = np.where(cv_errors <= threshold)[0]
        best_alpha = ccp_alphas[valid[-1]]
        opt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=self.min_samples_leaf,
                                     ccp_alpha=best_alpha, random_state=rng.integers(0, 2**31))
        opt.fit(X, y)
        return opt

    def fit(self, X, y, r=5):
        rng = np.random.default_rng(self.random_state)
        r = r if r % 2 == 1 else r + 1
        self.ensembles_, self.feature_partitions_, self.tree_weights_ = [], [], []
        for _ in range(self.n_ensembles):
            indices = rng.permutation(X.shape[1])
            sz = max(1, X.shape[1] // r)
            parts = [indices[i:i+sz] for i in range(0, X.shape[1], sz) if len(indices[i:i+sz]) >= 1]
            if len(parts) % 2 == 0 and len(parts) > 1:
                parts = parts[:-1]
            trees, feats, wts = [], [], []
            for p in parts:
                t = self._build_tree(X[:, p], y, rng)
                trees.append(t); feats.append(p)
                try: wts.append(roc_auc_score(y, t.predict_proba(X[:, p])[:, 1]))
                except: wts.append(0.5)
            self.ensembles_.append(trees)
            self.feature_partitions_.append(feats)
            self.tree_weights_.append(wts)
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        wsum, wtot = np.zeros(n), 0.0
        for trees, fsets, wts in zip(self.ensembles_, self.feature_partitions_, self.tree_weights_):
            for t, f, w in zip(trees, fsets, wts):
                p = t.predict_proba(X[:, f])
                wsum += w * (p[:, 1] if p.shape[1] == 2 else p.ravel())
                wtot += w
        prob = wsum / wtot if wtot > 0 else np.full(n, 0.5)
        return np.column_stack([1 - prob, prob])


def scan_one(r, n_ens, X, y, seed):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in cv.split(X, y):
        clf = CERPQuick(n_ensembles=n_ens, random_state=seed)
        clf.fit(X[tr], y[tr], r=r)
        try: aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        except: aucs.append(0.5)
    return r, n_ens, np.mean(aucs), np.std(aucs)


# Data 
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X_all = df[feat_names].values

# Feature sets to scan 
# Load union features
mw_df = pd.read_csv('mannwhitney_top_features.csv')
en_df = pd.read_csv('EN_Selection_Feature_Counts.csv')
rf_df = pd.read_csv('RF_Selection_Feature_Counts.csv')
xgb_df = pd.read_csv('XGB_Selection_XGB_Feature_Counts.csv')
cerp_log = pd.read_csv('CERP_standalone_tree_selection_log.csv')
cerp_fold = cerp_log.drop_duplicates(subset=['Feature', 'Repeat', 'Fold'])
cerp_freq = cerp_fold.groupby('Feature').size().reset_index(name='Fold_Count')

all_feats = set(feat_names)
comp = pd.DataFrame({'miRNA': sorted(all_feats)})
comp['MW'] = comp['miRNA'].map(dict(zip(mw_df['Biomarker'], mw_df['Selection_Count']))).fillna(0)
comp['EN'] = comp['miRNA'].map(dict(zip(en_df['Feature'], en_df['Count']))).fillna(0)
comp['RF'] = comp['miRNA'].map(dict(zip(rf_df['Feature'], rf_df['Count']))).fillna(0)
comp['XGB'] = comp['miRNA'].map(dict(zip(xgb_df['Feature'], xgb_df['Count']))).fillna(0)
comp['CERP'] = comp['miRNA'].map(dict(zip(cerp_freq['Feature'], cerp_freq['Fold_Count']))).fillna(0)

# Scan configs
feature_configs = {
    'Tier1 (>=70/70)': comp[(comp['MW']>=70)|(comp['EN']>=70)|(comp['RF']>=70)|(comp['XGB']>=70)|(comp['CERP']>=70)]['miRNA'].tolist(),
    'Standalone (all)': list(feat_names),
}

r_scan = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
n_ens_fixed = 15  # Fix ensembles, scan r only

print("=" * 70)
print("CERP Quick r-Scan (10-fold CV x 1 repeat, n_ens=15 fixed)")
print("=" * 70)

for config_name, feats in feature_configs.items():
    feat_idx = [i for i, f in enumerate(feat_names) if f in feats]
    X = X_all[:, feat_idx]
    n_feat = X.shape[1]
    print(f"\n {config_name}: {n_feat} features ")
    print(f"  {'r':>4} | feat/part | {'AUC':>12} | time")
    print(f"  {'-'*45}")

    start = time.time()
    results = Parallel(n_jobs=4)(
        delayed(scan_one)(r, n_ens_fixed, X, y, 42)
        for r in r_scan if n_feat // r >= 2
    )

    for r, ne, auc_m, auc_s in sorted(results, key=lambda x: x[0]):
        fpp = n_feat // r
        print(f"  r={r:>3} | {fpp:>8} | {auc_m:.3f}±{auc_s:.3f} |")

    best = max(results, key=lambda x: x[2])
    print(f"  BEST: r={best[0]}, AUC={best[2]:.3f}")
    print(f"  Scan time: {(time.time()-start)/60:.1f} min")
