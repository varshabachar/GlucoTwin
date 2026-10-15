"""Plain-language names for every feature - used in plots and in the doctor dashboard."""

FEATURE_LABELS = {
    # CGM
    "glucose": "Current glucose", "d15": "Glucose change, last 15 min", "d30": "Glucose change, last 30 min",
    "d60": "Glucose change, last 60 min", "slope30": "Glucose trend (mg/dL per min)",
    "accel15": "Glucose acceleration", "g_mean_60": "Average glucose, last hour",
    "g_std_60": "Glucose variability, last hour", "g_min_60": "Lowest glucose, last hour",
    "g_max_60": "Highest glucose, last hour", "g_mean_3h": "Average glucose, last 3 h",
    "g_above_base": "Glucose above personal baseline", "baseline_24h": "Personal 24-h baseline",
    "hour_sin": "Time of day", "hour_cos": "Time of day",
    # wearables
    "hr_15": "Heart rate, last 15 min", "hr_60": "Heart rate, last hour", "hr_std_60": "Heart-rate variability (spread)",
    "hr_rel_rest": "Heart rate above resting", "rmssd_60": "HRV (RMSSD), last hour",
    "act_15": "Activity, last 15 min", "act_60": "Activity, last hour", "act_3h": "Activity, last 3 h",
    "temp_15": "Skin temperature", "eda_15": "Skin conductance (stress)",
    "sleep_prev_min": "Sleep last night", "sleep_prev_hr": "Heart rate during sleep",
    "sleep_prev_rmssd": "HRV during sleep",
    "hr_15_dev": "Heart rate vs own normal, last 15 min", "hr_60_dev": "Heart rate vs own normal, last hour",
    "rmssd_60_rel": "HRV vs own normal", "temp_15_dev": "Skin temperature vs own normal",
    "eda_15_dev": "Skin conductance vs own normal", "sleep_prev_hr_dev": "Sleep heart rate vs own normal",
    "sleep_prev_rmssd_rel": "Sleep HRV vs own normal",
    # meals
    "mins_since_meal": "Time since last meal", "carbs_last_meal": "Net carbs in last meal",
    "sugar_last_meal": "Sugar in last meal", "fat_last_meal": "Fat in last meal",
    "protein_last_meal": "Protein in last meal", "fiber_last_meal": "Fibre in last meal",
    "carbs_3h": "Net carbs eaten, last 3 h", "calories_3h": "Calories eaten, last 3 h",
    # EHR
    "age": "Age", "sex_male": "Sex", "bmi": "BMI", "hba1c": "HbA1c", "fasting_glucose": "Fasting glucose",
    "triglycerides": "Triglycerides", "hdl": "HDL cholesterol", "ldl": "LDL cholesterol",
    "sbp": "Systolic BP", "dbp": "Diastolic BP", "dx_prediabetes": "Prediabetes diagnosis",
    "dx_hypertension": "Hypertension diagnosis", "dx_dyslipidemia": "Dyslipidaemia diagnosis",
    "family_hx_t2d": "Family history of T2D", "on_metformin": "On metformin", "on_statin": "On statin",
    "prs": "Genetic risk score",
    # twin
    "twin_peak_2h": "Twin-projected peak (next 2 h)", "twin_rise_2h": "Twin-projected rise (next 2 h)",
    "twin_g60": "Twin-projected glucose in 1 h", "twin_meal_effect_now": "Meal effect still active",
    "twin_carb_sens": "Personal carb sensitivity", "twin_time_to_peak": "Personal time-to-peak",
}

UNITS = {
    "glucose": "mg/dL", "d15": "mg/dL", "d30": "mg/dL", "d60": "mg/dL", "g_mean_60": "mg/dL",
    "g_max_60": "mg/dL", "g_min_60": "mg/dL", "g_above_base": "mg/dL", "baseline_24h": "mg/dL",
    "hr_15": "bpm", "hr_60": "bpm", "hr_rel_rest": "bpm", "hr_15_dev": "bpm", "hr_60_dev": "bpm",
    "sleep_prev_hr_dev": "bpm", "rmssd_60": "ms", "act_15": "mg", "act_60": "mg",
    "act_3h": "mg", "sleep_prev_min": "min", "mins_since_meal": "min", "carbs_last_meal": "g",
    "sugar_last_meal": "g", "carbs_3h": "g", "calories_3h": "kcal", "bmi": "kg/m²", "hba1c": "%",
    "twin_peak_2h": "mg/dL", "twin_rise_2h": "mg/dL", "twin_g60": "mg/dL", "twin_meal_effect_now": "mg/dL",
    "twin_time_to_peak": "min", "fasting_glucose": "mg/dL",
}


def label(f: str) -> str:
    return FEATURE_LABELS.get(f, f)


def fmt_value(f: str, v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v != v:  # NaN
        return "no data"
    if f in ("sex_male",):
        return "male" if v >= 0.5 else "female"
    if f.startswith(("dx_", "on_", "family_")):
        return "yes" if v >= 0.5 else "no"
    if f in ("hour_sin", "hour_cos"):
        return ""
    if f == "sleep_prev_min":
        return f"{v / 60:.1f} h"
    u = UNITS.get(f, "")
    return f"{v:.1f} {u}".strip() if abs(v) < 10 else f"{v:.0f} {u}".strip()
