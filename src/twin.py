"""
Step 4 - The digital twin: a small physiological model of ONE patient's glucose.

    glucose(t) = baseline(t)                         personal 24-h baseline (from their CGM)
               + k * sum_meals carbs_i * h(t - t_i)  each meal raises glucose with a hump-shaped curve
               - a * activity_effect(t)              recent movement pulls glucose down
               + deviation * exp(-elapsed / tr)      anything unexplained fades back to baseline

h(.) peaks at `tp` minutes after eating. The four numbers (k, tp, a, tr) ARE the twin:
    k  = how strongly this person reacts to carbs          ("carb sensitivity")
    tp = how fast their glucose peaks after a meal
    a  = how much activity helps them
    tr = how fast they recover to baseline

The twin starts from population values (config.TWIN_PRIOR) and is re-calibrated every day on that
patient's own data - so it "learns" the person. It is then used to
    (1) forecast the next 2 hours  -> extra features for the ML model
    (2) answer what-if questions   -> "what if she eats 60 g carbs and walks 15 min?"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter

import config

PARAMS = ["k", "tp", "a", "tr"]
# Features the twin hands to the ML model. Only its time-varying FORECASTS are used: the personal
# parameters themselves are constant within a person and would act as an ID card for that person.
TWIN_FEATURES = [
    "twin_peak_2h", "twin_rise_2h", "twin_g60", "twin_meal_effect_now",
]
TWIN_PARAM_COLUMNS = ["twin_carb_sens", "twin_time_to_peak"]   # kept for display only
STEP = config.BIN_MIN
H_STEPS = config.HORIZON_MIN // STEP


# ------------------------------------------------------------------ building blocks
def meal_kernel(tau: np.ndarray, tp: float) -> np.ndarray:
    """Hump-shaped response to one gram of carbs: 0 before eating, 1.0 at tp minutes, then decays."""
    x = np.clip(tau, 0, None) / tp
    return np.where(tau > 0, x * np.exp(1 - x), 0.0)


def activity_effect(act: np.ndarray) -> np.ndarray:
    """Exponentially smoothed activity (milli-g): a walk keeps helping for a while after it ends."""
    alpha = np.exp(-STEP / config.TWIN_ACTIVITY_TAU)
    return lfilter([1 - alpha], [1, -alpha], np.nan_to_num(act, nan=0.0))


class PatientSeries:
    """All arrays the twin needs for one patient, on the 5-minute grid."""

    def __init__(self, df: pd.DataFrame, meals: pd.DataFrame):
        self.index = df.index
        t0 = df.index[0]
        self.t = ((df.index - t0).total_seconds() / 60).to_numpy(dtype=float)
        self.G = df["glucose"].to_numpy(dtype=float)
        self.G_obs = df["glucose_obs"].to_numpy(dtype=float)
        gb = df["baseline_24h"].copy()
        gb = gb.fillna(df["glucose"].expanding(min_periods=1).quantile(0.2)).fillna(95.0)
        self.Gb = gb.to_numpy(dtype=float)
        self.A = activity_effect(df["act"].to_numpy(dtype=float))
        self.day = df["rel_day"].to_numpy()
        if len(meals):
            mt = ((pd.to_datetime(meals["time"]) - t0).dt.total_seconds() / 60).to_numpy(dtype=float)
            self.meal_t = mt
            self.meal_c = meals["net_carbs"].to_numpy(dtype=float)
        else:
            self.meal_t = np.array([])
            self.meal_c = np.array([])

    def S(self, tp: float, times: np.ndarray | None = None) -> np.ndarray:
        """Total carb 'load' felt at each time from ALL logged meals (hindsight - used for fitting)."""
        times = self.t if times is None else times
        if not len(self.meal_t):
            return np.zeros_like(times)
        return meal_kernel(times[:, None] - self.meal_t[None, :], tp) @ self.meal_c


# ------------------------------------------------------------------ fitting
def _full_theta(theta_free, free: list[str]) -> np.ndarray:
    it = iter(theta_free)
    return np.array([next(it) if p in free else config.TWIN_PRIOR[p] for p in PARAMS], dtype=float)


def _residuals(theta_free, ps: PatientSeries, anchors: np.ndarray, free: list[str]):
    k, tp, a, tr = _full_theta(theta_free, free)
    S = ps.S(tp)
    s = np.array([h // STEP for h in config.TWIN_FIT_HORIZONS])
    j = anchors[:, None]
    js = j + s[None, :]
    D = ps.G[j] - ps.Gb[j] - k * S[j] + a * ps.A[j]
    pred = ps.Gb[j] + k * S[js] - a * ps.A[js] + D * np.exp(-(s * STEP) / tr)
    r = (pred - ps.G_obs[js]) / config.TWIN_OBS_SD
    r = r[np.isfinite(r)]
    full = (k, tp, a, tr)
    prior = np.array([(full[i] - config.TWIN_PRIOR[p]) / config.TWIN_PRIOR_SD[p]
                      for i, p in enumerate(PARAMS) if p in free])
    return np.concatenate([r, prior])


def fit_twin(ps: PatientSeries, upto_index: int | None = None, fit_activity: bool = True) -> dict:
    """Calibrate the twin on data strictly before `upto_index` (None = all data).

    fit_activity=False keeps the activity effect at the population value - used when activity is only
    estimated from heart rate, which also rises after meals and would bias the fit.
    """
    free = [p for p in PARAMS if fit_activity or p != "a"]
    n = len(ps.t)
    last = n if upto_index is None else upto_index
    max_s = max(config.TWIN_FIT_HORIZONS) // STEP
    anchors = np.arange(0, max(last - max_s, 0), 3)  # every 15 min
    anchors = anchors[np.isfinite(ps.G[anchors])] if len(anchors) else anchors
    if len(anchors) < 50:
        return dict(config.TWIN_PRIOR, n_fit=0, activity_fitted=False)
    x0 = np.array([config.TWIN_PRIOR[p] for p in free])
    lo = [config.TWIN_BOUNDS[p][0] for p in free]
    hi = [config.TWIN_BOUNDS[p][1] for p in free]
    scale = {"k": 0.5, "tp": 20, "a": 0.1, "tr": 40}
    res = least_squares(_residuals, x0, bounds=(lo, hi), args=(ps, anchors, free),
                        x_scale=[scale[p] for p in free], max_nfev=200)
    full = _full_theta(res.x, free)
    out = {p: float(v) for p, v in zip(PARAMS, full)}
    out["n_fit"] = int(len(anchors))
    out["activity_fitted"] = bool(fit_activity)
    return out


# ------------------------------------------------------------------ forecasting (what the twin "expects")
def forecast(ps: PatientSeries, rows: np.ndarray, params: dict, meals_known: str = "past") -> np.ndarray:
    """Forecast glucose for the next 2 h from each row. Returns array (len(rows), H_STEPS).

    meals_known="past": only meals eaten up to now are known (real-time setting).
    meals_known="all" : meals in the next 2 h are also known (meal 'announced' when eaten).
    Future activity is assumed to fade (no new exercise) in both cases.
    """
    k, tp, a, tr = (params[p] for p in PARAMS)
    s = np.arange(1, H_STEPS + 1)
    t_now = ps.t[rows]
    fut_t = t_now[:, None] + s[None, :] * STEP
    if len(ps.meal_t):
        known = ps.meal_t[None, :] <= (t_now[:, None] if meals_known == "past" else t_now[:, None] + config.HORIZON_MIN)
        C = ps.meal_c[None, :] * known                                   # (rows, meals)
        Kf = meal_kernel(fut_t[:, :, None] - ps.meal_t[None, None, :], tp)  # (rows, H, meals)
        S_fut = np.einsum("rhm,rm->rh", Kf, C)
        S_now = (meal_kernel(t_now[:, None] - ps.meal_t[None, :], tp) * C).sum(axis=1)
    else:
        S_fut = np.zeros_like(fut_t)
        S_now = np.zeros(len(rows))
    A_now = ps.A[rows]
    A_fut = A_now[:, None] * np.exp(-(s[None, :] * STEP) / config.TWIN_ACTIVITY_TAU)
    Gb = ps.Gb[rows][:, None]
    D = ps.G[rows] - ps.Gb[rows] - k * S_now + a * A_now
    return Gb + k * S_fut - a * A_fut + D[:, None] * np.exp(-(s[None, :] * STEP) / tr)


def twin_features_for_patient(df: pd.DataFrame, meals: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], pd.DataFrame, dict]:
    """Online twin: day d uses parameters fitted only on days < d.

    Returns (features per row, params per day, learning-curve table, final params on all data).
    """
    ps = PatientSeries(df, meals)
    fit_act = not bool(df["act_from_hr"].iloc[0]) if "act_from_hr" in df else True
    feats = pd.DataFrame(index=df.index, columns=TWIN_FEATURES + TWIN_PARAM_COLUMNS, dtype=float)
    day_params, curve = [], []
    days = np.unique(ps.day)
    for d in days:
        rows = np.where(ps.day == d)[0]
        start = rows[0]
        params = (fit_twin(ps, upto_index=start, fit_activity=fit_act) if d > days[0]
                  else dict(config.TWIN_PRIOR, n_fit=0, activity_fitted=False))
        day_params.append(dict(day=int(d), **params))
        ok = rows[np.isfinite(ps.G[rows])]
        if not len(ok):
            continue
        fc = forecast(ps, ok, params, "past")
        feats.iloc[ok, 0] = fc.max(axis=1)
        feats.iloc[ok, 1] = fc.max(axis=1) - ps.G[ok]
        feats.iloc[ok, 2] = fc[:, 11]
        S_now = ps.S(params["tp"], ps.t[ok]) if len(ps.meal_t) else np.zeros(len(ok))
        feats.iloc[ok, 3] = params["k"] * S_now
        feats.iloc[ok, 4] = params["k"]
        feats.iloc[ok, 5] = params["tp"]
        # learning curve: 60-min-ahead error when the meal is logged at eating time
        if d > days[0]:
            tgt = ok + 12
            keep = tgt < len(ps.t)
            ok2, tgt = ok[keep], tgt[keep]
            truth = ps.G_obs[tgt]
            fin = np.isfinite(truth)
            if fin.sum() > 20:
                pers = forecast(ps, ok2, params, "all")[:, 11]
                prior = forecast(ps, ok2, dict(config.TWIN_PRIOR), "all")[:, 11]
                curve.append(dict(
                    day=int(d), days_of_data=int(d - days[0]),
                    mae_twin=float(np.nanmean(np.abs(pers - truth)[fin])),
                    mae_prior=float(np.nanmean(np.abs(prior - truth)[fin])),
                    mae_persistence=float(np.nanmean(np.abs(ps.G[ok2] - truth)[fin])),
                ))
    final = fit_twin(ps, fit_activity=fit_act)
    return feats.astype(float), day_params, pd.DataFrame(curve), final


# ------------------------------------------------------------------ what-if simulator (dashboard)
def simulate_whatif(params: dict, g_now: float, baseline: float, act_effect_now: float,
                    past_meals: list[tuple[float, float]], new_meal_carbs: float = 0.0,
                    walk_minutes: int = 0, walk_start_after_meal: int = 15, walk_intensity: float = 150.0,
                    horizon: int = 180) -> pd.DataFrame:
    """Simulate the next `horizon` minutes for three scenarios.

    past_meals: list of (minutes_ago, net_carbs) for meals in the last 6 h.
    Returns a table with columns: minute, no_change, meal, meal_plus_walk.
    """
    k, tp, a, tr = (params[p] for p in PARAMS)
    m = np.arange(0, horizon + STEP, STEP, dtype=float)
    past = sum(c * meal_kernel(m + ago, tp) for ago, c in past_meals) if past_meals else np.zeros_like(m)
    new = new_meal_carbs * meal_kernel(m, tp)
    alpha = np.exp(-STEP / config.TWIN_ACTIVITY_TAU)

    def act_path(walk: bool):
        act = np.zeros_like(m)
        if walk and walk_minutes > 0:
            act[(m > walk_start_after_meal) & (m <= walk_start_after_meal + walk_minutes)] = walk_intensity
        A = np.empty_like(m)
        prev = act_effect_now
        for i in range(len(m)):
            prev = prev if i == 0 else alpha * prev + (1 - alpha) * act[i]
            A[i] = prev
        return A

    D = g_now - baseline - k * past[0] + a * act_effect_now
    decay = np.exp(-m / tr)

    def run(meal_part, A):
        return baseline + k * (past + meal_part) - a * A + D * decay

    return pd.DataFrame({
        "minute": m,
        "no_change": run(0 * new, act_path(False)),
        "meal": run(new, act_path(False)),
        "meal_plus_walk": run(new, act_path(True)),
    })
