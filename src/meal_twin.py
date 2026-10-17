"""
Step 3b (NEW) - A twin that learns MEAL BY MEAL, and knows WHAT was in the meal.

The twin in src/twin.py is re-fitted once a day on everything that happened before that day.
This module is the per-MEAL version of the same idea, and it adds food composition:

    before each meal   predict the post-meal peak, using parameters fitted ONLY on that person's
                       EARLIER meals (and only on glucose recorded before this meal started)
    after the meal     compare the prediction with the real CGM peak
    then               re-fit, so the next meal is predicted by a slightly better twin

Two extra per-person parameters let the composition of the meal change its response:

    amplitude  x  1 / (1 + fb * fibre / (net_carbs + 1))        fb = fibre blunting
    time-to-peak x (1 + fd * (protein + fat) / (net_carbs + 10))  fd = fat/protein delay

Both priors are ZERO (config.TWIN_COMP_PRIOR), so the twin starts from "composition makes no
difference" and only moves if that person's own meals say otherwise. That makes the comparison in
`compare_variants` an honest test rather than a guaranteed win.

NOTHING here touches src/twin.py, the LightGBM features, or the leave-one-subject-out results.
It is an addition, reported separately.

No leakage, specifically:
  * parameters for meal i come from meals 0..i-1 only;
  * every residual used in that fit has its TARGET time strictly before meal i starts, so a long
    response window from meal i-1 can never reach past the moment meal i is served;
  * what the meal contains is known at prediction time, because the patient logs the meal when they
    eat it. That is the only thing about meal i the forecast is allowed to use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import config
from src.twin import PARAMS, PatientSeries, activity_effect

STEP = config.BIN_MIN
COMP_PARAMS = ["fb", "fd"]
ALL_PARAMS = PARAMS + COMP_PARAMS
NUTRIENTS = ["net_carbs", "fiber", "sugar", "protein", "fat"]

# columns written to artifacts/meal_learning.csv
MEAL_LOG_COLUMNS = [
    "pid", "meal_number", "time", "net_carbs", "fiber", "protein", "fat",
    "glucose_at_meal", "actual_peak", "actual_rise",
    "pred_peak_prior", "pred_peak_carbs", "pred_peak_comp",
    "err_prior", "err_carbs", "err_comp",
    "n_history_meals", "k", "tp", "fb", "fd",
]


# ------------------------------------------------------------------ composition
def meal_composition(meals: pd.DataFrame) -> dict:
    """Per-meal nutrient arrays, with any column the food log does not have treated as zero."""
    n = len(meals)
    out = {}
    for c in NUTRIENTS:
        out[c] = (pd.to_numeric(meals[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                  if c in meals else np.zeros(n))
    return out


def comp_multipliers(comp: dict, fb: float, fd: float) -> tuple[np.ndarray, np.ndarray]:
    """(amplitude multiplier, time-to-peak multiplier) for every meal, from what was in it.

    fb = fd = 0 reproduces the carbs-only twin exactly (both multipliers become 1.0).
    """
    carbs = comp["net_carbs"]
    amp = 1.0 / (1.0 + fb * comp["fiber"] / (carbs + 1.0))
    dly = 1.0 + fd * (comp["protein"] + comp["fat"]) / (carbs + 10.0)
    return amp, dly


def meal_load(t_eval: np.ndarray, meal_t: np.ndarray, meal_c: np.ndarray,
              tp: float, amp: np.ndarray, dly: np.ndarray) -> np.ndarray:
    """Carb 'load' felt at each time in t_eval, with a per-meal height and a per-meal time-to-peak.

    Same hump as src/twin.meal_kernel, but tp varies meal by meal, so a fatty meal peaks later.
    """
    if not len(meal_t):
        return np.zeros(len(t_eval), dtype=float)
    tau = t_eval[:, None] - meal_t[None, :]
    x = np.clip(tau, 0, None) / (tp * dly)[None, :]
    h = np.where(tau > 0, x * np.exp(1.0 - x), 0.0)
    return h @ (meal_c * amp)


# ------------------------------------------------------------------ fitting on a set of meals
def _theta_full(theta_free: np.ndarray, free: list[str]) -> dict:
    """Fill the parameters that are not being fitted from the population prior."""
    prior = {**config.TWIN_PRIOR, **config.TWIN_COMP_PRIOR}
    it = iter(theta_free)
    return {p: (float(next(it)) if p in free else float(prior[p])) for p in ALL_PARAMS}


def _residuals(theta_free, ps: PatientSeries, comp: dict, anchors: np.ndarray,
               steps: np.ndarray, cap: int, free: list[str]):
    """Scaled errors of "from the start of each earlier meal, where did glucose go?" + prior pull."""
    th = _theta_full(theta_free, free)
    k, tp, a, tr = th["k"], th["tp"], th["a"], th["tr"]
    amp, dly = comp_multipliers(comp, th["fb"], th["fd"])

    n = len(ps.t)
    j = anchors[:, None]
    js = j + steps[None, :]
    # a target must exist AND lie strictly before the meal being predicted -> no leakage.
    # jc is js clipped only so the array lookups below stay in bounds; `ok` is what decides
    # which of them actually count, so a clipped index can never contribute a residual.
    jc = np.clip(js, 0, n - 1)
    ok = (js < min(cap, n)) & np.isfinite(ps.G_obs[jc])
    if not ok.any():
        return np.zeros(1)

    need = np.unique(np.concatenate([anchors, jc[ok]]))
    S = np.zeros(n)
    S[need] = meal_load(ps.t[need], ps.meal_t, ps.meal_c, tp, amp, dly)

    D = ps.G[j] - ps.Gb[j] - k * S[j] + a * ps.A[j]
    pred = ps.Gb[jc] + k * S[jc] - a * ps.A[jc] + D * np.exp(-(steps[None, :] * STEP) / tr)
    r = ((pred - ps.G_obs[jc]) / config.TWIN_OBS_SD)[ok]
    r = r[np.isfinite(r)]

    prior = {**config.TWIN_PRIOR, **config.TWIN_COMP_PRIOR}
    prior_sd = {**config.TWIN_PRIOR_SD, **config.TWIN_COMP_PRIOR_SD}
    pull = np.array([(th[p] - prior[p]) / prior_sd[p] for p in free], dtype=float)
    return np.concatenate([r, pull])


def fit_on_meals(ps: PatientSeries, comp: dict, anchors: np.ndarray, cap: int,
                 use_composition: bool, fit_activity: bool, x0: dict | None = None) -> dict:
    """Fit the twin to the response to the meals at row indices `anchors`, using data before `cap`.

    use_composition=False holds fb and fd at zero -> the carbs-only twin, for a fair comparison.
    x0 warm-starts from the previous meal's parameters (which came from even earlier data only).
    """
    prior = {**config.TWIN_PRIOR, **config.TWIN_COMP_PRIOR}
    bounds = {**config.TWIN_BOUNDS, **config.TWIN_COMP_BOUNDS}
    free = [p for p in PARAMS if fit_activity or p != "a"]
    if use_composition:
        free = free + COMP_PARAMS
    if len(anchors) < config.MEAL_LEARN_MIN_HISTORY:
        return dict(prior, n_fit=0, activity_fitted=False, personalised=False)

    steps = np.array([h // STEP for h in config.MEAL_FIT_HORIZONS], dtype=int)
    start = dict(prior) if x0 is None else {p: float(x0.get(p, prior[p])) for p in ALL_PARAMS}
    scale = {"k": 0.5, "tp": 20.0, "a": 0.1, "tr": 40.0, "fb": 1.0, "fd": 0.5}
    lo = [bounds[p][0] for p in free]
    hi = [bounds[p][1] for p in free]
    p0 = [float(np.clip(start[p], bounds[p][0], bounds[p][1])) for p in free]
    try:
        res = least_squares(_residuals, p0, bounds=(lo, hi), x_scale=[scale[p] for p in free],
                            args=(ps, comp, anchors, steps, cap, free), max_nfev=80)
        th = _theta_full(res.x, free)
    except Exception:  # a pathological window -> fall back to the prior rather than crash the run
        th = dict(prior)
    return dict(th, n_fit=int(len(anchors)), activity_fitted=bool(fit_activity), personalised=True)


# ------------------------------------------------------------------ forecasting one meal's peak
def predict_peak(ps: PatientSeries, comp: dict, row: int, params: dict,
                 window_min: int | None = None, order: str | None = None,
                 extra_activity: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Projected glucose for the `window_min` after the meal at row `row`. Returns (peak, curve).

    Everything it uses is known the moment the meal is served: the parameters (fitted on earlier
    meals), glucose and activity up to now, and what is in this meal (the patient logged it).
    `order` applies the literature-based eating-order adjustment to the meal's rise only.
    """
    window_min = config.MEAL_PEAK_WINDOW_MIN if window_min is None else window_min
    k, tp, a, tr = params["k"], params["tp"], params["a"], params["tr"]
    amp, dly = comp_multipliers(comp, params.get("fb", 0.0), params.get("fd", 0.0))
    if order:
        amp = amp * config.ORDER_PEAK_MULTIPLIER.get(order, 1.0)
        dly = dly * config.ORDER_DELAY_MULTIPLIER.get(order, 1.0)

    steps = np.arange(0, window_min // STEP + 1, dtype=int)
    t_now = ps.t[row]
    t_fut = t_now + steps * STEP
    # Only meals already served may be used. Without this mask a meal eaten LATER inside the
    # 3-hour window would leak into the forecast, and the twin would look better than it is.
    # The meal being predicted sits at t_now, so "<=" keeps it: the patient logs it as they eat.
    known = ps.meal_t <= t_now
    S_fut = meal_load(t_fut, ps.meal_t[known], ps.meal_c[known], tp, amp[known], dly[known])
    S_now = meal_load(np.array([t_now]), ps.meal_t[known], ps.meal_c[known], tp, amp[known], dly[known])[0]

    A_fut = ps.A[row] * np.exp(-(steps * STEP) / config.TWIN_ACTIVITY_TAU)
    if extra_activity is not None:
        A_fut = A_fut + extra_activity[: len(steps)]
    Gb = ps.Gb[row]
    D = ps.G[row] - Gb - k * S_now + a * ps.A[row]
    curve = Gb + k * S_fut - a * A_fut + D * np.exp(-(steps * STEP) / tr)
    return float(np.nanmax(curve)), curve


# ------------------------------------------------------------------ the per-meal learning loop
def _scoreable_meals(ps: PatientSeries, df: pd.DataFrame, meals: pd.DataFrame) -> np.ndarray:
    """Indices into `meals` of carb meals whose response window has enough CGM to score."""
    if not len(meals):
        return np.array([], dtype=int)
    n = len(ps.t)
    span = config.MEAL_PEAK_WINDOW_MIN // STEP
    keep = []
    for i, (mt, c) in enumerate(zip(ps.meal_t, ps.meal_c)):
        if c < config.MEAL_MIN_CARBS:
            continue
        row = int(np.searchsorted(ps.t, mt, side="left"))
        if row <= 0 or row + span >= n:
            continue
        if not np.isfinite(ps.G[row]):
            continue
        win = ps.G_obs[row: row + span + 1]
        if np.isfinite(win).mean() < config.MEAL_MIN_COVERAGE:
            continue
        keep.append(i)
    return np.array(keep, dtype=int)


def per_meal_learning(pid: str, df: pd.DataFrame, meals: pd.DataFrame,
                      verbose: bool = True) -> pd.DataFrame:
    """Walk one patient's meals in time order, predicting each before it happens, then learning.

    Returns one row per scoreable meal (see MEAL_LOG_COLUMNS).
    """
    if not len(meals) or not int(df.get("meal_log_ok", pd.Series([1])).iloc[0]):
        return pd.DataFrame(columns=MEAL_LOG_COLUMNS)
    ps = PatientSeries(df, meals)
    comp = meal_composition(meals)
    fit_act = not bool(df["act_from_hr"].iloc[0]) if "act_from_hr" in df else True
    idx = _scoreable_meals(ps, df, meals)
    if len(idx) == 0:
        return pd.DataFrame(columns=MEAL_LOG_COLUMNS)

    prior = {**config.TWIN_PRIOR, **config.TWIN_COMP_PRIOR}
    span = config.MEAL_PEAK_WINDOW_MIN // STEP
    rows_out = []
    warm_carbs, warm_comp = None, None

    for n_seen, mi in enumerate(idx):
        row = int(np.searchsorted(ps.t, ps.meal_t[mi], side="left"))
        history = np.array([int(np.searchsorted(ps.t, ps.meal_t[j], side="left")) for j in idx[:n_seen]],
                           dtype=int)

        # --- fit on earlier meals only, scoring nothing at or after this meal (cap = row)
        p_carbs = fit_on_meals(ps, comp, history, row, False, fit_act, warm_carbs)
        p_comp = fit_on_meals(ps, comp, history, row, True, fit_act, warm_comp)
        if p_carbs.get("personalised"):
            warm_carbs = p_carbs
        if p_comp.get("personalised"):
            warm_comp = p_comp

        # --- predict this meal's peak three ways
        peak_prior, _ = predict_peak(ps, comp, row, prior)
        peak_carbs, _ = predict_peak(ps, comp, row, {**p_carbs, "fb": 0.0, "fd": 0.0})
        peak_comp, _ = predict_peak(ps, comp, row, p_comp)

        # --- what actually happened
        win = ps.G_obs[row: row + span + 1]
        actual = float(np.nanmax(win))
        g_at = float(ps.G[row])

        rows_out.append({
            "pid": pid, "meal_number": n_seen + 1,
            "time": ps.index[row],
            "net_carbs": float(comp["net_carbs"][mi]), "fiber": float(comp["fiber"][mi]),
            "protein": float(comp["protein"][mi]), "fat": float(comp["fat"][mi]),
            "glucose_at_meal": g_at, "actual_peak": actual, "actual_rise": actual - g_at,
            "pred_peak_prior": peak_prior, "pred_peak_carbs": peak_carbs, "pred_peak_comp": peak_comp,
            "err_prior": abs(peak_prior - actual), "err_carbs": abs(peak_carbs - actual),
            "err_comp": abs(peak_comp - actual),
            "n_history_meals": int(len(history)),
            "k": p_comp["k"], "tp": p_comp["tp"], "fb": p_comp["fb"], "fd": p_comp["fd"],
        })

    out = pd.DataFrame(rows_out, columns=MEAL_LOG_COLUMNS)
    if verbose and len(out):
        last = out.iloc[-1]
        print(f"  [{pid}] {len(out):3d} meals scored | peak error: population {out['err_prior'].mean():5.1f}"
              f" -> per-meal {out['err_carbs'].mean():5.1f} -> +composition {out['err_comp'].mean():5.1f} mg/dL"
              f" | final fb={last['fb']:.2f} fd={last['fd']:.2f}")
    return out


def run_all(tables_features: dict, meals: dict, verbose: bool = True) -> pd.DataFrame:
    """Per-meal learning for every patient. `tables_features` is pid -> build_features(...) frame."""
    parts = [per_meal_learning(pid, tables_features[pid], meals.get(pid, pd.DataFrame()), verbose)
             for pid in sorted(tables_features)]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=MEAL_LOG_COLUMNS)


# ------------------------------------------------------------------ honest comparison
def compare_variants(log: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error of the predicted post-meal peak, for each twin variant.

    `late_mae` uses only the second half of each patient's meals: if the twin really learns, it
    should do better once it has seen a few meals than it does over all meals.
    """
    if log.empty:
        return pd.DataFrame()
    late = log.groupby("pid", group_keys=False).apply(
        lambda d: d.iloc[len(d) // 2:], include_groups=False)
    rows = []
    for name, col in [("Population twin (no personalisation)", "err_prior"),
                      ("Per-meal twin, carbs only", "err_carbs"),
                      ("Per-meal twin + food composition", "err_comp")]:
        rows.append(dict(
            variant=name,
            mae_all=float(log[col].mean()),
            mae_late=float(late[col].mean()) if len(late) else np.nan,
            median_all=float(log[col].median()),
            meals=int(log[col].notna().sum()),
        ))
    out = pd.DataFrame(rows)
    base = out.loc[0, "mae_all"]
    out["improvement_vs_population"] = (base - out["mae_all"]) / base
    return out


def learning_curve(log: pd.DataFrame, bin_size: int = 5) -> pd.DataFrame:
    """Average peak error against how many of that person's meals the twin had already seen."""
    if log.empty:
        return pd.DataFrame()
    d = log.copy()
    d["meals_seen"] = (d["n_history_meals"] // bin_size) * bin_size
    g = d.groupby("meals_seen").agg(
        err_prior=("err_prior", "mean"), err_carbs=("err_carbs", "mean"),
        err_comp=("err_comp", "mean"), n=("err_comp", "size")).reset_index()
    return g


# ------------------------------------------------------------------ Indian meal menu (dashboard)
def load_menu(path=None) -> pd.DataFrame:
    """Read data/indian_meals.csv. Approximate values - see the header of that file.

    Returns an empty frame (never raises) if the file is missing, so the dashboard still runs.
    """
    path = config.MENU_FILE if path is None else path
    try:
        m = pd.read_csv(path, comment="#")
    except Exception:
        return pd.DataFrame(columns=["name", "serving", "carbs_g", "fiber_g", "protein_g",
                                     "fat_g", "calories_kcal", "has_veg_protein", "net_carbs"])
    for c in ["carbs_g", "fiber_g", "protein_g", "fat_g", "calories_kcal", "has_veg_protein"]:
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0.0)
    m["net_carbs"] = (m["carbs_g"] - m["fiber_g"]).clip(lower=0)   # same rule as real meals
    return m


def menu_to_comp(item: pd.Series) -> dict:
    """One menu row -> the single-meal composition dict that predict_peak/plan_meal expect."""
    return {"net_carbs": np.array([float(item["net_carbs"])]),
            "fiber": np.array([float(item["fiber_g"])]),
            "sugar": np.array([0.0]),
            "protein": np.array([float(item["protein_g"])]),
            "fat": np.array([float(item["fat_g"])])}


def plan_meal(params: dict, g_now: float, baseline: float, act_effect_now: float,
              comp: dict, orders: list[str] | None = None, walk_minutes: int = 0,
              walk_start_after_meal: int = 15, walk_intensity: float = 150.0,
              horizon: int | None = None, cycle_phase: str | None = None) -> pd.DataFrame:
    """Project a meal eaten NOW under each eating order. Returns minute + one column per order.

    The twin's k, tp (and fb, fd) are LEARNED from this patient's own meals.

    Two things here are NOT learned, and both are doctor-set controls on this simulator only:
      * eating order  -> config.ORDER_PEAK_MULTIPLIER  (see config.ORDER_SOURCE_NOTE)
      * cycle_phase   -> config.CYCLE_K_MULTIPLIER     (see config.CYCLE_SOURCE_NOTE)
    Neither is used anywhere an accuracy number is produced: `predict_peak` and `per_meal_learning`
    never apply them, because this dataset records no eating order and no cycle phase.
    """
    horizon = config.MEAL_PEAK_WINDOW_MIN if horizon is None else horizon
    orders = list(config.ORDER_PEAK_MULTIPLIER) if orders is None else orders
    k, tp, a, tr = params["k"], params["tp"], params["a"], params["tr"]
    if cycle_phase:
        k = k * config.CYCLE_K_MULTIPLIER.get(cycle_phase, 1.0)
        baseline = baseline + config.CYCLE_BASELINE_SHIFT.get(cycle_phase, 0.0)
    m = np.arange(0, horizon + STEP, STEP, dtype=float)

    # a walk started after the meal, smoothed the same way the twin smooths real activity
    act = np.zeros_like(m)
    if walk_minutes > 0:
        act[(m > walk_start_after_meal) & (m <= walk_start_after_meal + walk_minutes)] = walk_intensity
    alpha = np.exp(-STEP / config.TWIN_ACTIVITY_TAU)
    A = np.empty_like(m)
    prev = act_effect_now
    for i in range(len(m)):
        prev = prev if i == 0 else alpha * prev + (1 - alpha) * act[i]
        A[i] = prev

    out = {"minute": m}
    for order in orders:
        amp, dly = comp_multipliers(comp, params.get("fb", 0.0), params.get("fd", 0.0))
        amp = amp * config.ORDER_PEAK_MULTIPLIER.get(order, 1.0)
        dly = dly * config.ORDER_DELAY_MULTIPLIER.get(order, 1.0)
        S = meal_load(m, np.array([0.0]), comp["net_carbs"], tp, amp, dly)
        D = g_now - baseline + a * act_effect_now
        out[order] = baseline + k * S - a * A + D * np.exp(-m / tr)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ final parameters for the dashboard
def fit_final(pid: str, df: pd.DataFrame, meals: pd.DataFrame) -> dict:
    """Twin fitted on ALL of this patient's meals - for the what-if simulator only.

    Fine for a simulator a doctor drives by hand; NOT used for any reported accuracy number,
    where every prediction comes from parameters fitted on earlier meals only.
    """
    prior = {**config.TWIN_PRIOR, **config.TWIN_COMP_PRIOR}
    if not len(meals) or not int(df.get("meal_log_ok", pd.Series([1])).iloc[0]):
        return dict(prior, n_fit=0, personalised=False, meals_used=0)
    ps = PatientSeries(df, meals)
    comp = meal_composition(meals)
    fit_act = not bool(df["act_from_hr"].iloc[0]) if "act_from_hr" in df else True
    idx = _scoreable_meals(ps, df, meals)
    if len(idx) < config.MEAL_LEARN_MIN_HISTORY:
        return dict(prior, n_fit=0, personalised=False, meals_used=int(len(idx)))
    rows = np.array([int(np.searchsorted(ps.t, ps.meal_t[j], side="left")) for j in idx], dtype=int)
    out = fit_on_meals(ps, comp, rows, len(ps.t), True, fit_act, None)
    out["meals_used"] = int(len(idx))
    return out


def composition_available(meals: dict) -> pd.DataFrame:
    """Which nutrient columns each patient's food log actually carries, and whether they are non-zero.

    Step 2 of the plan is only honest if these columns really exist in the raw files, so the
    pipeline prints this table instead of assuming.
    """
    rows = []
    for pid in sorted(meals):
        m = meals[pid]
        r = {"pid": pid, "meals": int(len(m))}
        for c in NUTRIENTS:
            if c not in m or not len(m):
                r[c] = "column absent"
            else:
                v = pd.to_numeric(m[c], errors="coerce").fillna(0.0)
                r[c] = f"{(v > 0).mean():.0%} non-zero" if (v > 0).any() else "all zero"
        rows.append(r)
    return pd.DataFrame(rows).set_index("pid")
