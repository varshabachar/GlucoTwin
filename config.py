"""
Central settings for GlucoTwin. Change values here, not inside the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- folders
RAW_DIR = ROOT / "data" / "raw"            # unzip the PhysioNet download here
DEMO_DIR = ROOT / "data" / "demo"          # lookalike data made by scripts/make_demo_data.py
PROCESSED_DIR = ROOT / "data" / "processed"  # cached 5-minute tables (auto-created)
ARTIFACTS_DIR = ROOT / "artifacts"         # models, metrics, EHR bundles, dataset for the dashboard
FIGURES_DIR = ARTIFACTS_DIR / "figures"    # plots for README / slides / video

# ---------------------------------------------------------------- time grid
BIN_MIN = 5                 # everything is aligned to the CGM's 5-minute rhythm
MAX_FFILL_BINS = 3          # carry the last glucose reading forward for at most 15 min of gaps

# Used only when accelerometer files were not downloaded: activity is estimated from heart rate
HR_ACTIVITY_MARGIN = 5      # bpm above resting that still counts as "at rest"
HR_TO_ACTIVITY_MG = 8.0     # ~ milli-g of wrist movement per bpm of heart-rate rise (walking ~ +20-30 bpm)

# ---------------------------------------------------------------- clock alignment (src/align.py)
ALIGN_WRIST_MIN_HOURS = 2.0   # shift wristband data only if its clock is off by >= 2 h (median over people)
ALIGN_MEAL_MAX_MIN = 120      # test meal-time offsets of -2 h .. +2 h
ALIGN_MEAL_MIN_GAIN = 4.0     # mg/dL: keep a meal offset only if the post-meal rise gets clearly stronger
ALIGN_MIN_MEALS = 8           # need at least this many carb meals to judge alignment
ALIGN_MIN_COVERAGE = 0.30     # if <30% of logged meals fall inside the CGM recording, the meal log is not used

# ---------------------------------------------------------------- prediction target
SPIKE_THRESHOLD = 140       # mg/dL. 140 = 2-hour OGTT cutoff for impaired glucose tolerance
HORIZON_MIN = 120           # predict a spike within the next 2 hours
MIN_FUTURE_COVERAGE = 0.75  # need >=75% of future CGM readings to trust a label

# ---------------------------------------------------------------- alerts / evaluation
ALERT_THRESHOLD = 0.5       # fallback only - each patient's alert threshold is chosen by inner cross-validation
TARGET_SENSITIVITY = 0.80   # inner CV picks the threshold that warns >=80% of spikes with fewest false alarms
INNER_FOLDS = 3
INNER_THR_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
ALERT_GAP_MIN = 30          # alerts separated by >=30 min count as separate alerts
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# ---------------------------------------------------------------- digital twin
# Prior physiology for a "new" patient before the twin has seen their data.
# The twin starts here and moves toward each person's own values as days of data arrive.
TWIN_PRIOR = {
    "k": 1.0,       # mg/dL rise per gram of net carbs at the response peak
    "tp": 45.0,     # minutes from start of a meal to its peak glucose effect
    "a": 0.15,      # mg/dL drop per milli-g of smoothed wrist activity
    "tr": 60.0,     # minutes for an unexplained glucose deviation to fade back to baseline
}
TWIN_PRIOR_SD = {"k": 0.5, "tp": 20.0, "a": 0.10, "tr": 40.0}
TWIN_BOUNDS = {"k": (0.05, 4.0), "tp": (15.0, 150.0), "a": (0.0, 1.0), "tr": (10.0, 300.0)}
TWIN_ACTIVITY_TAU = 40.0    # minutes: how long a bout of activity keeps helping
TWIN_FIT_HORIZONS = [30, 60, 90, 120]
TWIN_OBS_SD = 35.0          # mg/dL, inflated because CGM errors are correlated in time

# ---------------------------------------------------------------- model
RANDOM_SEED = 42
# Deliberately simple and regularised: with only 16 people, a flexible model memorises individuals
# instead of learning rules that carry over to a new patient. Chosen in advance, not tuned on results.
LGBM_PARAMS = dict(
    n_estimators=250,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    random_state=RANDOM_SEED,
    verbose=-1,
)
