"""
Step 1b - Clock alignment between the three data sources.

Three independent devices (CGM, wristband, phone food log) each keep their own clock. In the public
BIG IDEAs release they are not guaranteed to agree: timestamps are date-shifted for de-identification,
some meal logs are logged late or in a different time zone / daylight-saving setting, and one food log
does not even overlap its CGM recording. Feeding misaligned streams to a model silently destroys the
signal, so we check and correct alignment explicitly and print a report.

1. Wristband vs CGM  - during sleep, heart rate is at its daily minimum and glucose is at its flattest.
                        The hour gap between the two "quiet" windows estimates the wristband clock offset.
2. Meal log vs CGM   - after a carb-containing meal glucose rises 40-80 min later. We test meal-time
                        offsets of -2 h..+2 h and keep one only if it clearly strengthens that rise.
3. Meal-log coverage - if most logged meals fall outside the CGM recording, the meal log is marked
                        unavailable (features become "missing", not "no meals eaten").

This is a one-off calibration per participant, of the kind a deployed system does once at device setup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

STEP = config.BIN_MIN
WRIST_COLS = ["hr_sum", "hr_n", "temp_sum", "temp_n", "eda_sum", "eda_n", "act_sum", "act_n", "ssd", "n_pairs"]


def _hourly_profiles(tab: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """24-value day profiles (z-scored): wristband heart rate and CGM 'busyness' (mean |change|)."""
    hr = tab["hr_sum"] / tab["hr_n"].replace(0, np.nan)
    dg = tab["glucose_obs"].diff().abs()
    hour = tab.index.hour
    out = []
    for series in (hr, dg):
        prof = series.groupby(hour).mean().reindex(range(24))
        if prof.notna().sum() < 18:
            out.append(None)
            continue
        prof = prof.interpolate(limit_direction="both").to_numpy(dtype=float)
        sd = prof.std()
        out.append((prof - prof.mean()) / sd if sd > 0 else None)
    return out[0], out[1]


def _shift_scores(hr_prof: np.ndarray, cgm_prof: np.ndarray) -> np.ndarray:
    """Correlation between the CGM day profile and the HR profile moved back by s hours, s = -12..11."""
    return np.array([np.corrcoef(np.roll(hr_prof, -s), cgm_prof)[0, 1] for s in range(-12, 12)])


def wrist_offset_hours(tab: pd.DataFrame) -> tuple[float, np.ndarray | None]:
    """(best whole-hour wristband clock offset for this person, correlation curve over -12..11 h).

    Both heart rate and glucose are quiet at night and busy by day. The shift that best lines up
    the two 24-h patterns is the wristband's clock offset relative to the CGM.
    """
    hr_prof, cgm_prof = _hourly_profiles(tab)
    if hr_prof is None or cgm_prof is None:
        return np.nan, None
    sc = _shift_scores(hr_prof, cgm_prof)
    return float(np.argmax(sc) - 12), sc


def shift_wrist(tab: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Move wristband data by -hours (wristband clock was `hours` ahead of the CGM clock)."""
    if hours == 0:
        return tab
    k = int(hours * 60 / STEP)
    tab = tab.copy()
    for c in WRIST_COLS:
        if c in tab:
            s = tab[c].shift(-k)
            tab[c] = s.fillna(0.0) if c.endswith("_n") or c in ("ssd", "n_pairs") else s
    return tab


def _meal_rise_score(g: pd.Series, meal_t: np.ndarray, carbs: np.ndarray, offset_min: int) -> tuple[float, int]:
    """Carb-weighted mean of [glucose 40-80 min after meal] - [glucose 20-0 min before meal]."""
    idx = g.index
    t0 = idx[0]
    gv = g.to_numpy(dtype=float)
    n = len(gv)
    num = den = 0.0
    used = 0
    for t, c in zip(meal_t, carbs):
        if c < 15:
            continue
        pos = int(round(((t - t0).total_seconds() / 60 + offset_min) / STEP))
        lo_a, hi_a = pos - 4, pos + 1           # 20 min before .. meal
        lo_b, hi_b = pos + 8, pos + 17          # 40 .. 80 min after
        if lo_a < 0 or hi_b > n:
            continue
        before = np.nanmean(gv[lo_a:hi_a]) if np.isfinite(gv[lo_a:hi_a]).any() else np.nan
        after = np.nanmean(gv[lo_b:hi_b]) if np.isfinite(gv[lo_b:hi_b]).any() else np.nan
        if np.isnan(before) or np.isnan(after):
            continue
        num += c * (after - before)
        den += c
        used += 1
    return (num / den if den else np.nan), used


def meal_offset_minutes(tab: pd.DataFrame, meals: pd.DataFrame) -> tuple[int, float, float, int]:
    """(chosen offset, score at 0, score at chosen, meals used). Offset 0 unless the evidence is clear."""
    if meals is None or not len(meals):
        return 0, np.nan, np.nan, 0
    g = tab["glucose_obs"]
    mt = pd.to_datetime(meals["time"])
    carbs = meals["net_carbs"].to_numpy(dtype=float)
    s0, used = _meal_rise_score(g, mt, carbs, 0)
    if used < config.ALIGN_MIN_MEALS or np.isnan(s0):
        return 0, s0, s0, used
    best_off, best = 0, s0
    for off in range(-config.ALIGN_MEAL_MAX_MIN, config.ALIGN_MEAL_MAX_MIN + 1, 15):
        s, u = _meal_rise_score(g, mt, carbs, off)
        if u >= config.ALIGN_MIN_MEALS and s > best:
            best_off, best = off, s
    gain = best - s0
    if best_off != 0 and gain >= config.ALIGN_MEAL_MIN_GAIN and gain >= 0.25 * max(abs(s0), 1.0):
        return best_off, s0, best, used
    return 0, s0, s0, used


def meal_coverage(tab: pd.DataFrame, meals: pd.DataFrame) -> float:
    if meals is None or not len(meals):
        return 0.0
    mt = pd.to_datetime(meals["time"])
    return float(((mt >= tab.index[0]) & (mt <= tab.index[-1])).mean())


def align_all(tables: dict, meals: dict, verbose: bool = True) -> tuple[dict, dict, pd.DataFrame]:
    rows = []
    # 1) wristband: pool the day-pattern match over everyone - a clock/export setting is shared by all
    #    devices, and pooling 16 people is far more reliable than any single person's estimate
    est = {pid: wrist_offset_hours(t) for pid, t in tables.items()}
    curves = [c for _, c in est.values() if c is not None]
    global_off = int(np.argmax(np.nanmean(curves, axis=0)) - 12) if curves else 0
    apply_wrist = abs(global_off) >= config.ALIGN_WRIST_MIN_HOURS
    median_off = float(global_off)
    new_tables, new_meals = {}, {}
    for pid in sorted(tables):
        tab = tables[pid]
        own, curve = est[pid]
        wrist_shift = 0
        if apply_wrist:
            wrist_shift = global_off
        off_h = own
        tab = shift_wrist(tab, wrist_shift)

        # 2) meal-log coverage, 3) meal clock offset
        m = meals.get(pid)
        cov = meal_coverage(tab, m)
        meal_ok = cov >= config.ALIGN_MIN_COVERAGE
        moff, s0, s1, used = (0, np.nan, np.nan, 0)
        if meal_ok:
            moff, s0, s1, used = meal_offset_minutes(tab, m)
            if moff:
                m = m.copy()
                m["time"] = pd.to_datetime(m["time"]) + pd.Timedelta(minutes=moff)
        else:
            m = m.iloc[0:0].copy() if m is not None else m
        tab = tab.copy()
        tab["meal_log_ok"] = int(meal_ok)
        new_tables[pid], new_meals[pid] = tab, m
        rows.append(dict(pid=pid, wrist_offset_est_h=off_h,
                         wrist_shift_applied_h=wrist_shift, meals_in_cgm_window=cov, meal_log_used=meal_ok,
                         meals_scored=used, meal_rise_at_logged_time=s0, meal_offset_applied_min=moff,
                         meal_rise_after_alignment=s1))
    report = pd.DataFrame(rows).set_index("pid")
    if verbose:
        print(f"  wristband vs CGM: clock offset {median_off:+.0f} h (pooled over everyone) -> "
              + ("shifting wristband data" if apply_wrist else "clocks agree, no shift"))
        for pid, r in report.iterrows():
            meal_txt = ("meal log NOT used (meals outside the glucose recording: "
                        f"{r.meals_in_cgm_window:.0%} inside)") if not r.meal_log_used else (
                f"meals: rise {r.meal_rise_at_logged_time:+.0f} mg/dL at logged time"
                + (f" -> {r.meal_rise_after_alignment:+.0f} after shifting meals {r.meal_offset_applied_min:+d} min"
                   if r.meal_offset_applied_min else " (kept as logged)"))
            print(f"  [{pid}] wristband shift {r.wrist_shift_applied_h:+d} h | {meal_txt}")
    return new_tables, new_meals, report
