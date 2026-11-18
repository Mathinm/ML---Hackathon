"""Step 4: Bayesian hyperparameter optimisation (Optuna TPE) for the selected model.
Objective: mean group-aware CV (GroupKFold over training trips) candidate-level Macro F1.
Search uses only the 80 training trips. Tuned vs untuned are then both evaluated on student_test.
Outputs: results/optuna_trials.csv, results/tuned_vs_untuned.csv, configs/best_params.json, models/<name>_tuned.joblib
"""
import argparse, json, os, sys, time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pathsnap.prep import FEATURES
from pathsnap.models import build, SEED
from pathsnap.metrics import classification_metrics
from pathsnap.io import load_trip
from train_models import load_split, eval_on


def space(trial, name):
    if name == "xgboost":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    min_child_weight=trial.suggest_float("min_child_weight", 1, 20, log=True),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                    gamma=trial.suggest_float("gamma", 1e-4, 5, log=True))
    if name == "lightgbm":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
    raise KeyError(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgboost")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--search_stride", type=int, default=6)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--res", default=os.path.join(ROOT, "results"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()

    tr_all = load_split(a.res, "train", 1)
    tr_all = tr_all[tr_all["reliable"] == 1]
    s = tr_all[tr_all["obs"] % a.search_stride == 0].reset_index(drop=True)
    X, y, g = s[FEATURES].to_numpy(), s["y"].to_numpy(), s["trip"].to_numpy()
    splits = list(GroupKFold(n_splits=a.folds).split(X, y, g))

    def objective(trial):
        prm = space(trial, a.model)
        f1s = []
        for i_tr, i_va in splits:
            m = build(a.model, prm).fit(X[i_tr], y[i_tr])
            f1s.append(classification_metrics(y[i_va], m.predict_proba(X[i_va])[:, 1])["macro_f1"])
        trial.set_user_attr("fold_f1", f1s)
        return float(np.mean(f1s))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name=f"{a.model}_tpe")
    t0 = time.time()
    # trial 0 = the untuned default config, so the search record contains the baseline
    BASELINE = {"xgboost": dict(n_estimators=300, learning_rate=0.1, max_depth=6, min_child_weight=1.0,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, gamma=1e-4),
                "lightgbm": dict(n_estimators=300, learning_rate=0.1, num_leaves=31, min_child_samples=20,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1e-3)}
    study.enqueue_trial(BASELINE[a.model])
    study.optimize(objective, n_trials=a.trials, show_progress_bar=False)
    search_s = time.time() - t0
    study.trials_dataframe().to_csv(f"{a.res}/optuna_trials.csv", index=False)
    best = study.best_params
    json.dump(dict(model=a.model, best_params=best, best_cv_macro_f1=study.best_value, trials=a.trials,
                   search_seconds=search_s, objective="mean GroupKFold candidate-level macro F1",
                   folds=a.folds, search_stride=a.search_stride),
              open(os.path.join(ROOT, "configs", "best_params.json"), "w"), indent=2)
    print("best", study.best_value, best, "search_s", round(search_s))

    # ---- untuned vs tuned with IDENTICAL protocol (5-fold GroupKFold on stride-3 train, then student_test)
    tr = tr_all[tr_all["obs"] % 3 == 0].reset_index(drop=True)
    X, y, g = tr[FEATURES].to_numpy(), tr["y"].to_numpy(), tr["trip"].to_numpy()
    te = load_split(a.res, "student_test", 1)
    arcs_te = {t: load_trip(a.data, "student_test", t, with_route=False).arcs for t in te["trip"].unique()}
    rows = []
    for tag, prm in [("untuned", {}), ("tuned", best)]:
        cv = []
        for i_tr, i_va in GroupKFold(n_splits=5).split(X, y, g):
            m = build(a.model, prm).fit(X[i_tr], y[i_tr])
            cv.append(classification_metrics(y[i_va], m.predict_proba(X[i_va])[:, 1])["macro_f1"])
        t1 = time.time()
        m = build(a.model, prm).fit(X, y)
        ttrain = time.time() - t1
        te_m, p = eval_on(m, te, arcs_te)
        if tag == "tuned":
            joblib.dump(m, os.path.join(ROOT, "models", f"{a.model}_tuned.joblib"))
            np.save(f"{a.res}/proba_student_test_{a.model}_tuned.npy", p)
        rows.append(dict(variant=tag, cv_macro_f1=np.mean(cv), cv_macro_f1_std=np.std(cv), train_time_s=ttrain,
                         **{f"test_{k}": v for k, v in te_m.items()}))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(f"{a.res}/tuned_vs_untuned.csv", index=False)
