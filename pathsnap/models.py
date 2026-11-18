"""Model zoo used in the comparison (configs are also written to configs/models.json)."""
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42

MODEL_SPECS = {
    "logreg": ("Linear", dict(C=1.0, max_iter=500)),
    "gnb": ("Probabilistic", dict(var_smoothing=1e-9)),
    "knn": ("Instance-based", dict(n_neighbors=25, weights="distance")),
    "random_forest": ("Ensemble (bagging)", dict(n_estimators=120, max_depth=18, min_samples_leaf=5,
                                                 max_samples=0.5, n_jobs=2, random_state=SEED)),
    "extra_trees": ("Ensemble (bagging)", dict(n_estimators=120, max_depth=18, min_samples_leaf=5,
                                               n_jobs=2, random_state=SEED)),
    "hist_gb": ("Boosting", dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED)),
    "xgboost": ("Boosting", dict(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", n_jobs=2, random_state=SEED,
                                 eval_metric="logloss")),
    "lightgbm": ("Boosting", dict(n_estimators=300, learning_rate=0.1, num_leaves=31, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, n_jobs=2, random_state=SEED,
                                  verbose=-1)),
    "mlp": ("Neural network", dict(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3,
                                   batch_size=1024, max_iter=60, early_stopping=True,
                                   validation_fraction=0.1, n_iter_no_change=6, random_state=SEED)),
}


def build(name, params=None):
    p = dict(MODEL_SPECS[name][1])
    p.update(params or {})
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(**p))
    if name == "gnb":
        return GaussianNB(**p)
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**p))
    if name == "random_forest":
        return RandomForestClassifier(**p)
    if name == "extra_trees":
        return ExtraTreesClassifier(**p)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(**p)
    if name == "xgboost":
        return XGBClassifier(**p)
    if name == "lightgbm":
        return LGBMClassifier(**p)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(**p))
    raise KeyError(name)
