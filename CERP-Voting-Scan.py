# Quick Majority vs Weighted Voting Scan
# 1-SE pruning (SE), 10-fold CV x 3 repeats
# Full dataset (2578 features) + Union Tier1 (107 features)

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import warnings
import time

warnings.filterwarnings('ignore')


def nearest_odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


class CERPQuick:
    # CERP with 1-SE pruning, both voting methods
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
        if len(ccp_alphas) <= 1: return full_tree
        if len(ccp_alphas) > 10:
            ccp_alphas = ccp_alphas[np.linspace(0, len(ccp_alphas) - 1, 10, dtype=int)]
        min_class = np.min(np.bincount(y))
        if min_class < 2: return full_tree
        n_folds = min(10, min_class)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng.integers(0, 2**31))
        cv_errors, cv_ses = [], []
        for alpha in ccp_alphas:
            errs = []
            for tr, va in cv.split(X, y):
                t = DecisionTreeClassifier(criterion='gini', min_samples_leaf=self.min_samples_leaf,
                                           ccp_alpha=alpha, random_state=rng.integers(0, 2**31))
                t.fit(X[tr], y[tr])
                errs.append(1 - t.score(X[va], y[va]))
            cv_errors.append(np.mean(errs))
            cv_ses.append(np.std(errs) / np.sqrt(len(errs)))  # SE, not SD
        min_idx = np.argmin(cv_errors)
        threshold = cv_errors[min_idx] + cv_ses[min_idx]
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
            if len(parts) % 2 == 0 and len(parts) > 1: parts = parts[:-1]
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

    def predict_proba_weighted(self, X):
        n = X.shape[0]
        wsum, wtot = np.zeros(n), 0.0
        for trees, fsets, wts in zip(self.ensembles_, self.feature_partitions_, self.tree_weights_):
            for t, f, w in zip(trees, fsets, wts):
                p = t.predict_proba(X[:, f])
                wsum += w * (p[:, 1] if p.shape[1] == 2 else p.ravel())
                wtot += w
        prob = wsum / wtot if wtot > 0 else np.full(n, 0.5)
        return prob

    def predict_proba_majority(self, X):
        n = X.shape[0]
        ensemble_preds = []
        for trees, fsets in zip(self.ensembles_, self.feature_partitions_):
            votes = np.zeros(n)
            for t, f in zip(trees, fsets):
                votes += t.predict(X[:, f])
            ensemble_preds.append((votes > len(trees) / 2).astype(int))
        return np.array(ensemble_preds).mean(axis=0)


# Data 
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X_all = df[feat_names].values

# Union Tier1
comp_df = pd.read_csv('Union_Feature_Frequencies.csv')
mask = ((comp_df['MW_Freq'] >= 70) | (comp_df['EN_Freq'] >= 70) |
        (comp_df['RF_Freq'] >= 70) | (comp_df['XGB_Freq'] >= 70) |
        (comp_df['CERP_Freq'] >= 70))
tier1_idx = [i for i, f in enumerate(feat_names) if f in set(comp_df.loc[mask, 'miRNA'])]
X_tier1 = X_all[:, tier1_idx]

configs = {
    'Standalone (2578)': (X_all, 7, 15),
    'Tier1 (107)': (X_tier1, 7, 15),
}

N_REPEATS = 3
SEED = 42

print("=" * 60)
print("Majority vs Weighted Voting Quick Scan")
print("  1-SE pruning (SE), 3R x 10F")
print("=" * 60)

for config_name, (X, r, n_ens) in configs.items():
    print(f"\n--- {config_name}, r={r}, n_ens={n_ens} ---")
    start = time.time()

    results = {'MV': {'aucs': [], 'sens': [], 'specs': []},
               'WV': {'aucs': [], 'sens': [], 'specs': []}}

    for rep in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED + rep)
        for tr, te in cv.split(X, y):
            clf = CERPQuick(n_ensembles=n_ens, random_state=SEED + rep)
            clf.fit(X[tr], y[tr], r=r)

            for label, prob in [('MV', clf.predict_proba_majority(X[te])),
                                ('WV', clf.predict_proba_weighted(X[te]))]:
                pred = (prob >= 0.5).astype(int)
                results[label]['aucs'].append(roc_auc_score(y[te], prob))
                tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
                results[label]['sens'].append(tp/(tp+fn) if (tp+fn) > 0 else 0)
                results[label]['specs'].append(tn/(tn+fp) if (tn+fp) > 0 else 0)

        print(f"  R{rep+1}: MV={np.mean(results['MV']['aucs'][-10:]):.3f}, "
              f"WV={np.mean(results['WV']['aucs'][-10:]):.3f}")

    rt = (time.time() - start) / 60
    print(f"\n  {'':>4} {'AUC':>12} {'Sens':>8} {'Spec':>8} {'|S-S|':>7}")
    for label in ['MV', 'WV']:
        r_ = results[label]
        auc = np.mean(r_['aucs'])
        sens = np.mean(r_['sens'])
        spec = np.mean(r_['specs'])
        ss = abs(sens - spec)
        print(f"  {label:>4} {auc:.3f}±{np.std(r_['aucs']):.3f} {sens:>8.3f} {spec:>8.3f} {ss:>7.3f}")
    print(f"  Time: {rt:.1f} min")
