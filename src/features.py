"""
Step 2 - Turn the 5-minute sensor table into model features + the "spike in the next 2 h" label.

Rule followed everywhere: a feature at time t only uses data from time <= t.
Only the LABEL looks into the future (that is what we are trying to predict).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# ------------------------------------------------------------------ feature groups (used for the ablation)
CGM_FEATURES = [
    "glucose", "d15", "d30", "d60", "slope30", "accel15",
    "g_mean_60", "g_std_60", "g_min_60", "g_max_60", "g_mean_3h",
    "g_above_base", "baseline_24h", "hour_sin", "hour_cos",
]
# Wearable signals are expressed RELATIVE TO EACH PERSON'S OWN RECENT NORMAL (trailing 3 days, past only).
# Raw levels (e.g. resting heart rate) differ far more between people than within a person, so raw values
# mostly tell a model WHO someone is - which does not carry over to a new patient.
WEARABLE_FEATURES = [
    "hr_15_dev", "hr_60_dev", "hr_std_60", "hr_rel_rest",
    "rmssd_60_rel", "act_15", "act_60", "act_3h",
    "temp_15_dev", "eda_15_dev",
    "sleep_prev_min", "sleep_prev_hr_dev", "sleep_prev_rmssd_rel",
]
MEAL_FEATURES = [
    "mins_since_meal", "carbs_last_meal", "sugar_last_meal", "fat_last_meal",
    "protein_last_meal", "fiber_last_meal", "carbs_3h", "calories_3h",
]
# EHR_FEATURES live in src/ehr.py, TWIN_FEATURES in src/twin.py


# ------------------------------------------------------------------ small helpers
def _per_bin(tab: pd.DataFrame, col: str) -> pd.Series:
    n = tab[f"{col}_n"].replace(0, np.nan)
    return tab[f"{col}_sum"] / n


def _window_mean(tab: pd.DataFrame, col: str, minutes: int) -> pd.Series:
    """Mean of a raw signal over the last `minutes` (uses sums/counts so sparse data is weighted correctly)."""
    w = f"{minutes}min"
    s = tab[f"{col}_sum"].fillna(0).rolling(w).sum()
    n = tab[f"{col}_n"].fillna(0).rolling(w).sum()
    return (s / n.replace(0, np.nan))


def _slope(series: pd.Series, points: int = 7) -> pd.Series:
    """Least-squares slope over the last `points` readings, in mg/dL per minute."""
    x = np.arange(points, dtype=float)
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    return series.rolling(points, min_periods=points).apply(lambda v: float(np.dot(w, v)), raw=True) / config.BIN_MIN


# ------------------------------------------------------------------ sleep (estimated from the wristband)
def estimate_sleep(tab: pd.DataFrame) -> pd.DataFrame:
    """Rough sleep detector: still wrist + heart rate below the person's recent typical level, at night.

    Not a clinical sleep-staging algorithm - good enough to say "slept ~6 h last night".
    A night runs 18:00 -> 10:00 next day and is only used for predictions after 10:00.
    """
    hr = _per_bin(tab, "hr")
    act = _per_bin(tab, "act")
    hr_ref = hr.rolling("72h", min_periods=12).quantile(0.40)
    candidate = (act < 15) & (hr < hr_ref)
    # smooth out single noisy bins (30-min window inside the same night - all in the past by 10:00)
    smooth = candidate.astype(float).rolling(6, center=True, min_periods=3).median() > 0.5
    hours = tab.index.hour
    in_night = (hours >= 18) | (hours < 10)
    has_data = tab["act_n"].fillna(0) > 0
    is_sleep = smooth & pd.Series(in_night, index=tab.index) & has_data

    night_id = (tab.index - pd.Timedelta(hours=10)).normalize()  # night of date D: D 10:00 -> D+1 10:00
    nights = pd.DataFrame({
        "night": night_id,
        "is_sleep": is_sleep.to_numpy(),
        "in_night": in_night,
        "has_data": has_data.to_numpy(),
        "hr": hr.to_numpy(),
        "ssd": tab["ssd"].fillna(0).to_numpy(),
        "n_pairs": tab["n_pairs"].fillna(0).to_numpy(),
    })
    nn = nights[nights["in_night"]]
    if len(nn):
        grp = nn.groupby("night")
        summary = pd.DataFrame({
            "sleep_min": grp["is_sleep"].sum() * config.BIN_MIN,
            "coverage": grp["has_data"].mean(),
        })
        sl = nn[nn["is_sleep"]].groupby("night")
        summary["sleep_hr"] = sl["hr"].mean()
        ssd_sum, pairs = sl["ssd"].sum(), sl["n_pairs"].sum()
        summary["sleep_rmssd"] = (1000 * np.sqrt(ssd_sum / pairs.replace(0, np.nan))).where(pairs >= 30)
        summary.loc[summary["coverage"] < 0.5, ["sleep_min", "sleep_hr", "sleep_rmssd"]] = np.nan
    else:
        summary = pd.DataFrame(columns=["sleep_min", "sleep_hr", "sleep_rmssd", "coverage"])

    # each row gets the last COMPLETED night: the night that ended at the most recent 10:00
    prev_night = night_id - pd.Timedelta(days=1)
    out = pd.DataFrame(index=tab.index)
    out["is_sleep"] = is_sleep.astype(int)
    for src, dst in [("sleep_min", "sleep_prev_min"), ("sleep_hr", "sleep_prev_hr"), ("sleep_rmssd", "sleep_prev_rmssd")]:
        out[dst] = summary[src].reindex(prev_night).to_numpy() if len(summary) else np.nan
    return out


# ------------------------------------------------------------------ main feature builder
def build_features(tab: pd.DataFrame, meals: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=tab.index)
    df["pid"] = tab["pid"]
    df["glucose_obs"] = tab["glucose_obs"]
    g = tab["glucose_obs"].ffill(limit=config.MAX_FFILL_BINS)
    df["glucose"] = g

    # ---- CGM dynamics
    df["d15"] = g - g.shift(3)
    df["d30"] = g - g.shift(6)
    df["d60"] = g - g.shift(12)
    df["slope30"] = _slope(g, 7)
    df["accel15"] = df["d15"] - (g.shift(3) - g.shift(6))
    r12 = g.rolling(12, min_periods=8)
    df["g_mean_60"], df["g_std_60"], df["g_min_60"], df["g_max_60"] = r12.mean(), r12.std(), r12.min(), r12.max()
    df["g_mean_3h"] = g.rolling(36, min_periods=24).mean()
    df["baseline_24h"] = g.rolling("24h", min_periods=36).quantile(0.20)
    df["g_above_base"] = g - df["baseline_24h"]
    minutes = df.index.hour * 60 + df.index.minute
    df["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
    df["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)

    # ---- wearables
    df["hr"] = _per_bin(tab, "hr")                     # for plots
    df["act"] = _per_bin(tab, "act")                   # for plots (milli-g)
    df["temp"] = _per_bin(tab, "temp")
    df["hr_15"] = _window_mean(tab, "hr", 15)
    df["hr_60"] = _window_mean(tab, "hr", 60)
    df["hr_std_60"] = df["hr"].rolling(12, min_periods=6).std()
    rest = df["hr"].rolling("72h", min_periods=24).quantile(0.10)
    df["hr_rel_rest"] = df["hr_15"] - rest
    ssd60 = tab["ssd"].fillna(0).rolling("60min").sum()
    np60 = tab["n_pairs"].fillna(0).rolling("60min").sum()
    df["rmssd_60"] = np.where(np60 >= 20, 1000 * np.sqrt(ssd60 / np60.replace(0, np.nan)), np.nan)
    df["act_15"] = _window_mean(tab, "act", 15)
    df["act_60"] = _window_mean(tab, "act", 60)
    df["act_3h"] = _window_mean(tab, "act", 180)
    df["temp_15"] = _window_mean(tab, "temp", 15)
    df["eda_15"] = _window_mean(tab, "eda", 15)
    sleep = estimate_sleep(tab)
    df = df.join(sleep)

    # ---- person-relative versions (baseline = this person's trailing 72 h, data up to now only)
    def trailing_median(x: pd.Series) -> pd.Series:
        return x.rolling("72h", min_periods=36).median()

    df["hr_15_dev"] = df["hr_15"] - trailing_median(df["hr"])
    df["hr_60_dev"] = df["hr_60"] - trailing_median(df["hr"])
    df["rmssd_60_rel"] = df["rmssd_60"] / trailing_median(pd.Series(df["rmssd_60"], index=df.index))
    df["temp_15_dev"] = df["temp_15"] - trailing_median(df["temp"])
    df["eda_15_dev"] = df["eda_15"] - trailing_median(df["eda_15"])
    df["sleep_prev_hr_dev"] = df["sleep_prev_hr"] - df["sleep_prev_hr"].expanding(min_periods=1).median()
    df["sleep_prev_rmssd_rel"] = df["sleep_prev_rmssd"] / df["sleep_prev_rmssd"].expanding(min_periods=1).median()

    # ---- meal log (only meals already eaten at time t)
    mt = pd.to_datetime(meals["time"]).to_numpy() if len(meals) else np.array([], dtype="datetime64[ns]")
    order = np.argsort(mt)
    mt = mt[order]
    m = meals.iloc[order].reset_index(drop=True) if len(meals) else meals
    t = df.index.to_numpy()
    idx = np.searchsorted(mt, t, side="right") - 1
    has = idx >= 0
    safe = np.clip(idx, 0, None)
    since = np.full(len(t), np.nan)
    if len(mt):
        since[has] = (t[has] - mt[safe[has]]).astype("timedelta64[s]").astype(float) / 60
    df["mins_since_meal"] = np.clip(np.nan_to_num(since, nan=720), 0, 720)
    for src, dst in [("net_carbs", "carbs_last_meal"), ("sugar", "sugar_last_meal"), ("fat", "fat_last_meal"),
                     ("protein", "protein_last_meal"), ("fiber", "fiber_last_meal")]:
        vals = np.zeros(len(t))
        if len(mt):
            vals[has] = m[src].to_numpy(dtype=float)[safe[has]]
        df[dst] = vals
    for src, dst in [("net_carbs", "carbs_3h"), ("calories", "calories_3h")]:
        if len(mt):
            cum = np.concatenate([[0.0], np.cumsum(m[src].to_numpy(dtype=float))])
            hi = np.searchsorted(mt, t, side="right")
            lo = np.searchsorted(mt, t - np.timedelta64(180, "m"), side="right")
            df[dst] = cum[hi] - cum[lo]
        else:
            df[dst] = 0.0

    df["rel_day"] = (df.index.normalize() - df.index.normalize().min()).days
    df["act_from_hr"] = int(tab["act_from_hr"].iloc[0]) if "act_from_hr" in tab else 0
    df["meal_log_ok"] = int(tab["meal_log_ok"].iloc[0]) if "meal_log_ok" in tab else 1
    if not df["meal_log_ok"].iloc[0]:
        df[MEAL_FEATURES] = np.nan   # meal log unavailable -> "unknown", not "did not eat"
    return df


# ------------------------------------------------------------------ the label
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """y = 1 if glucose is below the threshold now and reaches it within the next 2 hours.

    Rows already at/above the threshold are excluded (we want to predict onset, not continuation).
    """
    steps = config.HORIZON_MIN // config.BIN_MIN
    need = int(np.ceil(steps * config.MIN_FUTURE_COVERAGE))
    obs = df["glucose_obs"]
    fut = obs[::-1].rolling(steps, min_periods=need).max()[::-1].shift(-1)
    y = (fut >= config.SPIKE_THRESHOLD).astype(float)
    y[fut.isna() | df["glucose"].isna() | (df["glucose"] >= config.SPIKE_THRESHOLD)] = np.nan
    df["future_max_2h"] = fut
    df["y"] = y
    return df


def spike_onsets(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Times when glucose crosses the threshold after >=30 min below it (one 'event' per spike)."""
    g = df["glucose_obs"]
    above = g >= config.SPIKE_THRESHOLD
    below_before = (g.shift(1) < config.SPIKE_THRESHOLD) & (g.rolling(7, min_periods=4).max().shift(1) < config.SPIKE_THRESHOLD)
    return df.index[above & below_before]
