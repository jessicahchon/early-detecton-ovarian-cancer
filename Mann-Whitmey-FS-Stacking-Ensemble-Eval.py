import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve
from scipy.stats import mannwhitneyu, randint, loguniform
import warnings

warnings.filterwarnings('ignore')

# Load dataset and prepare feature/label arrays
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X = df[feat_names].values

# Experimental design and search iterations
n_repeats = 7
n_outer = 10
n_inner = 5
n_iter_search = 20
random_seed = 42

# Standardized base configurations using probability distributions for RandomizedSearch
base_configs = {
    'RF': {
        'model_class': RandomForestClassifier,
        'param_dist': {
            'n_estimators': randint(500, 1000),
            'max_depth': [5, 7, 9],
            'max_features': [0.5]
        },
        'fixed_params': {'random_state': 42, 'n_jobs': -1}
    },
    'SVM': {
        'model_class': SVC,
        'param_dist': {
            'C': loguniform(0.1, 100),
            'kernel': ['linear', 'rbf']
        },
        'fixed_params': {'probability': True, 'random_state': 42}
    },
    'XGB': {
        'model_class': XGBClassifier,
        'param_dist': {
            'n_estimators': randint(300, 600),
            'max_depth': [8, 9],
            'learning_rate': loguniform(0.001, 0.1)
        },
        'fixed_params': {
            'random_state': 42, 'eval_metric': 'logloss', 
            'subsample': 1.0, 'verbosity': 0
        }
    }
}

# Feature selection via Mann-Whitney U test for univariate analysis
def mannwhitney_selection(X, y, alpha=0.05):
    n_features = X.shape[1]
    p_values = np.ones(n_features)
    cancer_mask, control_mask = (y == 1), (y == 0)
    for i in range(n_features):
        try:
            _, p = mannwhitneyu(X[cancer_mask, i], X[control_mask, i], alternative='two-sided')
            p_values[i] = p
        except: continue
    selected_idx = np.where(p_values < alpha)[0]
    return selected_idx

stacking_results = []
mean_fpr = np.linspace(0, 1, 100)
all_tprs = []

for r in range(n_repeats):
    repeat_y_true, repeat_y_probs, repeat_y_preds = [], [], []
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_seed + r)
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        x_train_out, x_test_out = X[train_idx], X[test_idx]
        y_train_out, y_test_out = y[train_idx], y[test_idx]
        
        # Univariate feature selection within each outer fold to prevent leakage
        selected_indices = mannwhitney_selection(x_train_out, y_train_out, alpha=0.05)
        if len(selected_indices) == 0: selected_indices = np.arange(X.shape[1])
        
        # Feature scaling for consistency across diverse algorithms
        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train_out[:, selected_indices])
        x_test_sc = scaler.transform(x_test_out[:, selected_indices])
        
        # Initialize meta-feature matrices for the stacking layer
        oof_train = np.zeros((x_train_sc.shape[0], len(base_configs)))
        oof_test = np.zeros((x_test_sc.shape[0], len(base_configs)))
        
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_seed + r)
        
        # Optimize base learners and generate out-of-fold meta-features
        for i, (name, cfg) in enumerate(base_configs.items()):
            search = RandomizedSearchCV(
                cfg['model_class'](**cfg['fixed_params']), 
                param_distributions=cfg['param_dist'], 
                n_iter=n_iter_search,
                cv=inner_cv, scoring='roc_auc', n_jobs=-1,
                random_state=random_seed + r + fold_idx
            )
            search.fit(x_train_sc, y_train_out)
            best_base = search.best_estimator_
            
            oof_train[:, i] = cross_val_predict(
                best_base, x_train_sc, y_train_out, cv=inner_cv, method='predict_proba'
            )[:, 1]
            oof_test[:, i] = best_base.predict_proba(x_test_sc)[:, 1]
            
        # Meta-learner fuses predictions from base models
        meta_model = LogisticRegression(random_state=random_seed)
        meta_model.fit(oof_train, y_train_out)
        
        # Generate final ensemble predictions for the held-out outer fold
        stack_probs = meta_model.predict_proba(oof_test)[:, 1]
        
        repeat_y_true.extend(y_test_out)
        repeat_y_probs.extend(stack_probs)
        repeat_y_preds.extend((stack_probs >= 0.5).astype(int))
        
        fpr, tpr, _ = roc_curve(y_test_out, stack_probs)
        all_tprs.append(np.interp(mean_fpr, fpr, tpr))
        all_tprs[-1][0] = 0.0

    # Aggregate performance metrics for each repeat validation
    tn, fp, fn, tp = confusion_matrix(repeat_y_true, repeat_y_preds).ravel()
    rep_auc = roc_auc_score(repeat_y_true, repeat_y_probs)
    stacking_results.append({
        'Repeat': r+1, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'AUC': rep_auc,
        'Accuracy': accuracy_score(repeat_y_true, repeat_y_preds),
        'Sens': tp/(tp+fn), 'Spec': tn/(tn+fp), 'F1': f1_score(repeat_y_true, repeat_y_preds)
    })
    print(f"Repeat {r+1} Stacking AUC: {rep_auc:.4f}")

# Consolidate overall metrics and ROC curve data
pd.DataFrame(stacking_results).to_csv('Univariate_Stacking_Summary.csv', index=False)
mean_tpr = np.mean(all_tprs, axis=0); mean_tpr[-1] = 1.0
pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr}).to_csv('Univariate_Stacking_ROC_Data.csv', index=False)
