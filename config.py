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

# ================================================================= NEW: per-meal learning twin
# Everything below is an ADDITION reported separately from the locked fusion-model results.
# It does not touch the LightGBM model, its features, or the leave-one-subject-out evaluation.

# ---------------------------------------------------------------- per-meal learning (src/meal_twin.py)
MEAL_MIN_CARBS = 15.0        # g net carbs: below this a "meal" is a snack and its response is in the noise
MEAL_PEAK_WINDOW_MIN = 180   # look for the post-meal peak within 3 h of the first bite
MEAL_FIT_HORIZONS = [15, 30, 45, 60, 90, 120, 150, 180]   # minutes after a meal that the fit is scored on
MEAL_LEARN_MIN_HISTORY = 3   # meals needed before the twin is personalised (fewer -> population average)
MEAL_MIN_COVERAGE = 0.60     # need >=60% of CGM readings in the response window to score a meal

# ---------------------------------------------------------------- food composition (Step 2)
# Two extra per-person parameters. Both PRIORS ARE ZERO: the twin starts from "composition makes no
# difference" and only moves if that person's own meals say otherwise. Bounds are one-sided because
# fibre/fat/protein are expected to blunt and delay a carb peak, not sharpen or hasten it.
#   fb = fibre blunting : peak amplitude x 1/(1 + fb * fibre/(net_carbs+1))
#   fd = fat/protein delay : time-to-peak x (1 + fd * (protein+fat)/(net_carbs+10))
TWIN_COMP_PRIOR = {"fb": 0.0, "fd": 0.0}
TWIN_COMP_PRIOR_SD = {"fb": 1.0, "fd": 0.5}
TWIN_COMP_BOUNDS = {"fb": (0.0, 4.0), "fd": (0.0, 1.5)}

# ---------------------------------------------------------------- eating order (Step 3)
# NOT learned from this dataset: the BIG IDEAs food logs record what and when, never in what order.
# Literature-based and deliberately CONSERVATIVE. Shukla AP, Iliescu RG, Thomas CE, Aronne LJ,
# "Food Order Has a Significant Impact on Postprandial Glucose and Insulin Levels",
# Diabetes Care 2015;38(7):e98-e99, doi:10.2337/dc15-0429 - a crossover study in 11 adults with
# type 2 diabetes reporting 29-37% lower glucose at 30-60 min and a 73% lower 0-120 min incremental
# AUC when vegetables and protein were eaten before carbohydrate.
# We apply 25% to the peak RISE for vegetables-and-protein-first, well below that study's 29-37%,
# because n=11, the meal was a Western one, and our cohort is not diabetic. Treat as illustrative.
ORDER_PEAK_MULTIPLIER = {"Carbs first": 1.00, "Mixed": 0.90, "Vegetables & protein first": 0.75}
ORDER_DELAY_MULTIPLIER = {"Carbs first": 1.00, "Mixed": 1.08, "Vegetables & protein first": 1.20}
ORDER_DEFAULT = "Mixed"
ORDER_SOURCE_NOTE = ("Order effect: based on published research (Shukla et al., Diabetes Care 2015), "
                     "not learned from this dataset. With order logging, the twin would personalise this too.")

# ---------------------------------------------------------------- Indian meal menu (Step 4)
MENU_FILE = ROOT / "data" / "indian_meals.csv"

# ---------------------------------------------------------------- menstrual cycle (women's-health module)
# NOT learned from this dataset, and it cannot be: the BIG IDEAs release records no cycle, hormone or
# menstrual field at all, and each person is recorded for only 6.9-9.9 days while a cycle is ~28 days.
# You cannot observe a cycle - let alone compare phases within one person - in eight days of data.
# So, exactly like ORDER_PEAK_MULTIPLIER above, this is a literature-based constant the doctor sets by
# hand in the what-if simulator. It never touches the per-meal accuracy numbers or the LightGBM model.
#
# Evidence for lower insulin sensitivity in the luteal phase:
#  - Valdes CT & Elkind-Hirsch KE, J Clin Endocrinol Metab 1991 (n=12 healthy women): insulin sensitivity
#    5.03 +/- 0.72 (follicular) -> 2.22 +/- 0.45 (luteal), p<0.001.  https://pubmed.ncbi.nlm.nih.gov/10071420/
#  - Dexcom G6 CGM, 49 people / 149 cycles: daily median glucose highest in the luteal phase and lowest
#    in late-follicular.  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10421863/
#  - Apple Heart & Movement Study, 1,982 cycles: time-in-range 68.5% (follicular) vs 66.8% (luteal).
#
# Choosing the number: the insulin-sensitivity drop above is >50%, but that is Si, not a post-meal
# glucose peak, and n=12. The CGM studies show mean-glucose differences of only a few mg/dL. We put the
# post-meal effect deliberately between the two and at the low end: +15% on carb sensitivity and +3 mg/dL
# on baseline in the luteal phase. Treat as illustrative, not as a measured effect in this cohort.
#
# !! AND READ THIS BEFORE USING IT ON THIS COHORT !!
# The BIG IDEAs study enrolled "subjects aged 35-65 years, inclusive, INCLUDING ONLY POST-MENOPAUSAL
# FEMALES" (https://physionet.org/content/big-ideas-glycemic-wearable/). Every one of the 9 women in
# this dataset is post-menopausal and does not menstruate, so NO patient here has a cycle phase.
# The control is therefore a CAPABILITY DEMONSTRATION - it shows what the twin would do for a
# premenopausal patient - and the dashboard says so next to it. It is never a claim about a patient.
# The dataset that would let the twin actually LEARN this is mcPHASES (PhysioNet, 42 women, ~3 months
# each, daily LH/E3G/PdG assays and labelled cycle phases). See "Future work" in the README.
CYCLE_NA = "Not applicable - post-menopausal cohort"
CYCLE_PHASES = [CYCLE_NA, "Follicular (day 1-14) - demo", "Luteal (day 15-28) - demo"]
CYCLE_K_MULTIPLIER = {CYCLE_NA: 1.00, "Follicular (day 1-14) - demo": 1.00, "Luteal (day 15-28) - demo": 1.15}
CYCLE_BASELINE_SHIFT = {CYCLE_NA: 0.0, "Follicular (day 1-14) - demo": 0.0, "Luteal (day 15-28) - demo": 3.0}
CYCLE_DEFAULT = CYCLE_NA
CYCLE_COHORT_NOTE = (
    "Every woman in this cohort is **post-menopausal** - the BIG IDEAs study enrolled only "
    "post-menopausal females - so none of them has a menstrual cycle. The control below is a "
    "**capability demonstration**: it shows how this twin would adjust for a premenopausal patient. "
    "It is not a claim about this patient.")
CYCLE_SOURCE_NOTE = (
    "Cycle effect: based on published research, not learned from this dataset - and it could not be. "
    "These recordings are 7-10 days against a ~28-day cycle, no file records cycle information, and "
    "the cohort is post-menopausal by design. mcPHASES (PhysioNet: 42 women, ~3 months each, daily "
    "hormone assays and labelled phases) is the dataset that would let the twin learn this per patient.")
