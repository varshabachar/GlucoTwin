"""
Step 6 - Honest evaluation, the way a clinician would judge an alarm system.

Row-level:   AUROC, AUPRC (how well risk scores rank spike vs no-spike moments)
Event-level: of all real spikes, how many got a warning beforehand (sensitivity)?
             how many minutes of warning (lead time)?
             how many alerts were NOT followed by a spike (false alarms per day)?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config
from src.features import spike_onsets


def row_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return dict(auroc=np.nan, auprc=np.nan, n=int(len(y)), prevalence=float(np.mean(y)) if len(y) else np.nan)
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                n=int(len(y)), prevalence=float(np.mean(y)))


def per_subject_auroc(df: pd.DataFrame, col: str) -> pd.Series:
    out = {}
    for pid, d in df.groupby("pid"):
        m = row_metrics(d["y"].to_numpy(), d[col].to_numpy())
        out[pid] = m["auroc"]
    return pd.Series(out)


def event_metrics_patient(d: pd.DataFrame, col: str, thr: float) -> dict:
    """Event-level metrics for one patient's timeline."""
    onsets = spike_onsets(d)
    p = d[col]
    alert = (p >= thr).fillna(False)
    alert_times = d.index[alert.to_numpy()]
    horizon = pd.Timedelta(minutes=config.HORIZON_MIN)
    warned, leads = 0, []
    for o in onsets:
        w = alert_times[(alert_times >= o - horizon) & (alert_times <= o - pd.Timedelta(minutes=config.BIN_MIN))]
        if len(w):
            warned += 1
            leads.append((o - w.min()).total_seconds() / 60)
    # alert episodes: a new alert starts when the previous alert was > ALERT_GAP_MIN ago
    starts = []
    last = None
    for t in alert_times:
        if last is None or (t - last) > pd.Timedelta(minutes=config.ALERT_GAP_MIN):
            starts.append(t)
        last = t
    false_alarms = sum(1 for a in starts if not ((onsets > a) & (onsets <= a + horizon)).any())
    days = p.notna().sum() * config.BIN_MIN / 1440
    return dict(spikes=len(onsets), warned=warned, leads=leads, alerts=len(starts),
                false_alarms=false_alarms, days=days)


def event_metrics(df: pd.DataFrame, col: str, thr: float) -> dict:
    tot = dict(spikes=0, warned=0, leads=[], alerts=0, false_alarms=0, days=0.0)
    for _, d in df.groupby("pid"):
        m = event_metrics_patient(d, col, thr)
        for k in tot:
            tot[k] = tot[k] + m[k]
    return dict(
        threshold=thr,
        spikes=tot["spikes"],
        spikes_warned=tot["warned"],
        sensitivity=tot["warned"] / tot["spikes"] if tot["spikes"] else np.nan,
        median_lead_min=float(np.median(tot["leads"])) if tot["leads"] else np.nan,
        alerts=tot["alerts"],
        alert_precision=(tot["alerts"] - tot["false_alarms"]) / tot["alerts"] if tot["alerts"] else np.nan,
        false_alarms_per_day=tot["false_alarms"] / tot["days"] if tot["days"] else np.nan,
        days_monitored=tot["days"],
    )
