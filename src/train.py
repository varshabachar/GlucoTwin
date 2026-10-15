"""
Step 5 - Fuse the streams, train the spike predictor, and prove that fusion helps.

Validation = leave-one-subject-out (LOSO): train on 15 people, test on the 16th, repeat 16 times.
The model is ALWAYS tested on a person it has never seen - the honest way to measure a medical model.
(Random row splits leak the same person into train and test and inflate scores. Don't.)
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from src.ehr import EHR_FEATURES
from src.evaluate import event_metrics, per_subject_auroc, row_metrics
from src.features import CGM_FEATURES, MEAL_FEATURES, WEARABLE_FEATURES, add_labels, build_features
from src.twin import TWIN_FEATURES, twin_features_for_patient

ABLATION = [
    ("CGM only", CGM_FEATURES),
    ("+ Wearables", CGM_FEATURES + WEARABLE_FEATURES),
    ("+ Meal log", CGM_FEATURES + WEARABLE_FEATURES + MEAL_FEATURES),
    ("+ EHR (full fusion)", CGM_FEATURES + WEARABLE_FEATURES + MEAL_FEATURES + EHR_FEATURES),
    ("+ Digital twin", CGM_FEATURES + WEARABLE_FEATURES + MEAL_FEATURES + EHR_FEATURES + TWIN_FEATURES),
]
FINAL_SET = ABLATION[-1][0]
FINAL_FEATURES = ABLATION[-1][1]


# ------------------------------------------------------------------ dataset
def build_dataset(tables: dict, meals: dict, ehr: pd.DataFrame, verbose: bool = True):
    frames, twin_params, curves, final_params = [], {}, [], {}
    for pid in sorted(tables):
        df = build_features(tables[pid], meals[pid])
        df = add_labels(df)
        tf, day_params, curve, final = twin_features_for_patient(df, meals[pid])
        df = df.join(tf)
        for c in EHR_FEATURES:
            df[c] = float(ehr.loc[pid, c]) if pid in ehr.index else np.nan
        frames.append(df)
        twin_params[pid] = day_params
        final_params[pid] = final
        if len(curve):
            curve["pid"] = pid
            curves.append(curve)
        if verbose:
            n_lab = int(df["y"].notna().sum())
            pos = int(np.nansum(df["y"]))
            print(f"  [{pid}] rows={len(df):5d}  labelled={n_lab:5d}  spike-ahead rows={pos:4d}  "
                  f"twin k={final['k']:.2f} tp={final['tp']:.0f}min a={final['a']:.2f}")
    data = pd.concat(frames)
    data.index.name = "time"
    curve_df = pd.concat(curves) if curves else pd.DataFrame()
    return data, twin_params, final_params, curve_df


# ------------------------------------------------------------------ baseline rule
def rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """What a simple CGM alarm does: extrapolate the last 30-min trend 2 h ahead."""
    proj = df["glucose"] + df["slope30"].clip(lower=0) * config.HORIZON_MIN
    return 1 / (1 + np.exp(-(proj - config.SPIKE_THRESHOLD) / 10))


# ------------------------------------------------------------------ LOSO training
def _fit(X, y):
    m = lgb.LGBMClassifier(**config.LGBM_PARAMS)
    m.fit(X, y)
    return m


def _inner_threshold(data: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, pids: np.ndarray,
                     train_pids: np.ndarray, can_predict: np.ndarray, train_mask: np.ndarray) -> float:
    """Pick the alert threshold using ONLY the training patients (inner cross-validation).

    Rule: the threshold that warns >= TARGET_SENSITIVITY of spikes with the fewest false alarms per day;
    if no threshold reaches the target, the one with the highest sensitivity.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    groups = np.array_split(rng.permutation(train_pids), min(config.INNER_FOLDS, len(train_pids)))
    inner = np.full(len(data), np.nan)
    for g in groups:
        tr = train_mask & np.isin(pids, train_pids) & ~np.isin(pids, g)
        va = can_predict & np.isin(pids, g)
        if va.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        inner[va] = _fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    sub = np.isin(pids, train_pids)
    tmp = data.loc[sub, ["pid", "glucose_obs"]].copy()
    tmp["_p"] = inner[sub]
    best, best_key = config.ALERT_THRESHOLD, None
    for thr in config.INNER_THR_GRID:
        em = event_metrics(tmp, "_p", thr)
        sens, fa = em["sensitivity"], em["false_alarms_per_day"]
        if not np.isfinite(sens):
            continue
        hit = sens >= config.TARGET_SENSITIVITY
        key = (hit, -fa if hit else sens)
        if best_key is None or key > best_key:
            best, best_key = float(thr), key
    return best


def loso_predict(data: pd.DataFrame, features: list[str], save_models_to=None) -> tuple[np.ndarray, dict]:
    """Out-of-fold risk for every row: each person's predictions come from a model that never saw them.

    Returns (risk per row, alert threshold per patient). Each patient's threshold was chosen on the
    OTHER patients only, so nothing about the test patient leaks into either the model or the alarm.
    """
    pred = np.full(len(data), np.nan)
    thresholds = {}
    pids = data["pid"].to_numpy()
    can_predict = (data["glucose"].notna() & (data["glucose"] < config.SPIKE_THRESHOLD)).to_numpy()
    train_mask = data["y"].notna().to_numpy()
    X = data[features].astype(float)
    y = data["y"].to_numpy()
    for pid in np.unique(pids):
        tr = train_mask & (pids != pid)
        te = can_predict & (pids == pid)
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        model = _fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]
        others = np.array([p for p in np.unique(pids) if p != pid])
        thresholds[pid] = _inner_threshold(data, X, y, pids, others, can_predict, train_mask)
        if save_models_to is not None:
            model.booster_.save_model(str(save_models_to / f"fold_{pid}.txt"))
    return pred, thresholds


def _alerts(data: pd.DataFrame, risk: np.ndarray, thresholds: dict) -> np.ndarray:
    thr = data["pid"].map(thresholds).to_numpy(dtype=float)
    out = (risk >= thr).astype(float)
    out[~np.isfinite(risk)] = np.nan
    return out


def run_training(data: pd.DataFrame, verbose: bool = True) -> dict:
    out_dir = config.ARTIFACTS_DIR
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    results, thr_table = [], []

    data = data.copy()
    rule = rule_baseline(data).to_numpy(dtype=float, copy=True)
    rule[~(data["glucose"] < config.SPIKE_THRESHOLD).to_numpy()] = np.nan
    data["risk_rule"] = rule
    rule_alert = (rule >= 0.5).astype(float)
    rule_alert[~np.isfinite(rule)] = np.nan
    data["alert_rule"] = rule_alert
    rows = [("Rule: trend extrapolation", "risk_rule", "alert_rule")]
    all_thresholds = {}

    for name, feats in ABLATION:
        col = "risk_" + name.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
        if verbose:
            print(f"  training: {name:22s} ({len(feats)} features, {data['pid'].nunique()}-fold LOSO + inner threshold) ...")
        risk, thr = loso_predict(data, feats, save_models_to=models_dir if name == FINAL_SET else None)
        data[col] = risk
        data["alert_" + col] = _alerts(data, risk, thr)
        all_thresholds[name] = thr
        rows.append((name, col, "alert_" + col))

    for name, col, acol in rows:
        rm = row_metrics(data["y"].to_numpy(), data[col].to_numpy())
        ps = per_subject_auroc(data, col)
        em = event_metrics(data, acol, 0.5)  # alert column is already 0/1 with each patient's own threshold
        results.append(dict(model=name, auroc=rm["auroc"], auprc=rm["auprc"],
                            auroc_subject_mean=float(ps.mean()), auroc_subject_sd=float(ps.std()),
                            sensitivity=em["sensitivity"], median_lead_min=em["median_lead_min"],
                            false_alarms_per_day=em["false_alarms_per_day"], alert_precision=em["alert_precision"],
                            prevalence=rm["prevalence"], column=col))
    final_col = rows[-1][1]
    for thr in config.THRESHOLD_GRID:
        thr_table.append(event_metrics(data, final_col, thr))

    data["risk"] = data[final_col]
    data["alert"] = data["alert_" + final_col]
    final_thr = all_thresholds[FINAL_SET]
    (out_dir / "alert_thresholds.json").write_text(json.dumps(
        {"final_model_per_patient": final_thr, "all_models": all_thresholds}, indent=2))
    ablation = pd.DataFrame(results)
    ablation.to_csv(out_dir / "ablation.csv", index=False)
    pd.DataFrame(thr_table).to_csv(out_dir / "thresholds.csv", index=False)

    # final model on everyone (for global SHAP explanation only - per-patient risk stays out-of-fold)
    lab = data["y"].notna().to_numpy()
    final_model = _fit(data.loc[lab, FINAL_FEATURES].astype(float), data.loc[lab, "y"].to_numpy())
    final_model.booster_.save_model(str(models_dir / "final_all_patients.txt"))
    (out_dir / "features.json").write_text(json.dumps({"final": FINAL_FEATURES,
                                                        "ablation": {n: f for n, f in ABLATION}}, indent=2))
    return dict(data=data, ablation=ablation, thresholds=pd.DataFrame(thr_table), final_model=final_model,
                patient_thresholds=final_thr)
