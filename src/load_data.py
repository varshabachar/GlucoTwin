"""
Step 1 - Read the raw BIG IDEAs files and squeeze every sensor into 5-minute bins.

Raw layout (PhysioNet "BIG IDEAs Lab Glycemic Variability and Wearable Device Data"):
    <root>/Demographics.csv
    <root>/001/Dexcom_001.csv  HR_001.csv  IBI_001.csv  ACC_001.csv  TEMP_001.csv  EDA_001.csv
    <root>/001/Food_Log_001.csv
    ... up to 016

Bin convention (important for honesty): a bin labelled 12:05 holds data from (12:00, 12:05].
So a prediction made "at 12:05" only ever sees data up to 12:05 - never the future.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config

BIN = f"{config.BIN_MIN}min"
_ISO_MINUTE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}")


# ------------------------------------------------------------------ discovery
def find_data_root(base: Path) -> Path:
    """Return the folder that contains Demographics.csv (works however the ZIP was unpacked)."""
    base = Path(base)
    if (base / "Demographics.csv").exists():
        return base
    hits = sorted(base.rglob("Demographics.csv"))
    if not hits:
        raise FileNotFoundError(
            f"Could not find Demographics.csv under {base}. "
            "Unzip the PhysioNet download into data/raw/ (or run scripts/make_demo_data.py)."
        )
    return hits[0].parent


def list_participants(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit())


def _find_file(folder: Path, prefix: str) -> Path | None:
    """Find e.g. HR_001.csv or HR.csv, case-insensitive."""
    for f in sorted(folder.glob("*.csv")):
        stem = f.stem.lower()
        if stem == prefix.lower() or stem.startswith(prefix.lower() + "_"):
            return f
    return None


# ------------------------------------------------------------------ helpers
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _bin_labels_fast(ts_str: pd.Series) -> pd.Series:
    """Map timestamp strings to the right-closed 5-minute bin they belong to.

    Parsing tens of millions of full timestamps is slow, so we only parse the unique
    'YYYY-MM-DD HH:MM' prefixes and map back. A sample at 12:03:41 -> bin 12:05.
    """
    prefix = ts_str.str.slice(0, 16)
    codes, uniques = pd.factorize(prefix)
    minute = pd.to_datetime(pd.Series(uniques), format="mixed", errors="coerce")
    labels = minute.dt.floor(BIN) + pd.Timedelta(minutes=config.BIN_MIN)
    out = labels.to_numpy()[codes]
    out[codes < 0] = np.datetime64("NaT")
    return pd.Series(out, index=ts_str.index)


def _aggregate_signal(path: Path, chunksize: int = 2_000_000, kind: str = "mean") -> pd.DataFrame:
    """Read a large Empatica CSV in chunks and return per-bin sums/counts.

    kind="mean": single value column -> columns [sum, n]
    kind="acc" : x,y,z columns -> wrist movement (ENMO, in milli-g) -> [sum, n]
    """
    parts = []
    reader = pd.read_csv(path, chunksize=chunksize, dtype=str, skipinitialspace=True)
    acc_scale = None
    for chunk in reader:
        chunk = _clean_cols(chunk)
        tcol = chunk.columns[0]
        ts = chunk[tcol].astype(str)
        sample = ts.dropna().iloc[0] if len(ts.dropna()) else ""
        if _ISO_MINUTE.match(sample):
            bins = _bin_labels_fast(ts)
        else:  # unusual format -> slow but safe
            full = pd.to_datetime(ts, format="mixed", errors="coerce")
            bins = full.dt.floor(BIN) + pd.Timedelta(minutes=config.BIN_MIN)
        vals = chunk.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        if kind == "acc":
            xyz = vals.iloc[:, :3].to_numpy(dtype=float)
            vm = np.sqrt((xyz ** 2).sum(axis=1))
            if acc_scale is None:
                med = np.nanmedian(vm)
                # Empatica E4 reports acceleration in 1/64 g (rest ~ 64); otherwise assume g.
                acc_scale = 64.0 if med > 10 else 1.0
            enmo_mg = np.clip(vm / acc_scale - 1.0, 0, None) * 1000.0
            v = pd.Series(enmo_mg, index=chunk.index)
        else:
            v = vals.iloc[:, 0]
        df = pd.DataFrame({"bin": bins, "v": v}).dropna()
        g = df.groupby("bin")["v"].agg(["sum", "count"])
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["sum", "count"])
    out = pd.concat(parts).groupby(level=0).sum()
    out.index.name = "time"
    return out


# ------------------------------------------------------------------ individual streams
def load_dexcom(path: Path) -> pd.Series:
    """Glucose readings (mg/dL), labelled to their 5-minute bin."""
    df = _clean_cols(pd.read_csv(path, dtype=str))
    tcol = next(c for c in df.columns if c.startswith("timestamp"))
    gcol = next(c for c in df.columns if c.startswith("glucose value") or c == "value")
    if "event type" in df.columns:
        df = df[df["event type"].astype(str).str.upper() == "EGV"]
    g = df[gcol].astype(str).str.strip().replace({"Low": "40", "High": "400", "LOW": "40", "HIGH": "400"})
    g = pd.to_numeric(g, errors="coerce")
    t = pd.to_datetime(df[tcol], format="mixed", errors="coerce")
    s = pd.Series(g.to_numpy(), index=t).dropna()
    s = s[s.index.notna()]
    bins = s.index.floor(BIN) + pd.Timedelta(minutes=config.BIN_MIN)
    # a reading exactly on a boundary belongs to that boundary's bin
    exact = s.index == s.index.floor(BIN)
    bins = bins.where(~exact, s.index)
    s = s.groupby(bins).mean()
    s.index.name = "time"
    return s


def load_ibi(path: Path) -> pd.DataFrame:
    """Heart-rate variability (RMSSD) building blocks per bin: sum of squared successive diffs + count."""
    df = _clean_cols(pd.read_csv(path, skipinitialspace=True))
    t = pd.to_datetime(df.iloc[:, 0].astype(str), format="mixed", errors="coerce")
    ibi = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    tt = t.to_numpy()
    ok = (~pd.isna(tt)) & np.isfinite(ibi) & (ibi > 0.3) & (ibi < 2.0)
    tt, ibi = tt[ok], ibi[ok]
    if len(ibi) < 3:
        return pd.DataFrame(columns=["ssd", "n_pairs"])
    gap = np.diff(tt).astype("timedelta64[ms]").astype(float) / 1000.0
    # consecutive beats: time between beats ~ the reported interval
    consecutive = np.abs(gap - ibi[1:]) < 0.25
    d = np.diff(ibi)
    valid = consecutive & (np.abs(d) < 0.3)
    # need beat i-1 and i both valid -> diff i uses pair (i-1, i)
    sq = np.where(valid, d ** 2, np.nan)
    times = pd.to_datetime(tt[1:])
    bins = times.floor(BIN) + pd.Timedelta(minutes=config.BIN_MIN)
    out = pd.DataFrame({"bin": bins, "sq": sq}).dropna()
    g = out.groupby("bin")["sq"].agg(["sum", "count"]).rename(columns={"sum": "ssd", "count": "n_pairs"})
    g.index.name = "time"
    return g


MEAL_COLUMNS = ["time", "carbs", "fiber", "sugar", "protein", "fat", "calories", "net_carbs"]
_HEADER_WORDS = ("date", "time", "calorie", "carb", "food")
# Participant 003's file in the PhysioNet release has NO header row and only 11 columns.
# Checked against the foods (e.g. 8 chicken nuggets = 393 kcal, 19 g carbs, 0.1 g sugar, 20 g protein):
_HEADERLESS_11 = ["date", "time", "time_begin", "logged_food", "amount", "unit", "searched_food",
                  "calorie", "total_carb", "sugar", "protein"]


def _read_food_csv(path: Path) -> pd.DataFrame:
    """Read a food log whatever its layout (with or without a header row)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        first = fh.readline().lower()
    if any(w in first for w in _HEADER_WORDS):
        return _clean_cols(pd.read_csv(path, dtype=str))
    df = pd.read_csv(path, dtype=str, header=None)
    if df.shape[1] == len(_HEADERLESS_11):
        df.columns = _HEADERLESS_11
    else:  # unknown headerless layout: 3rd column = start time, then the 4 last numbers
        cols = [f"c{i}" for i in range(df.shape[1])]
        cols[0], cols[1], cols[2] = "date", "time", "time_begin"
        for name, i in zip(["calorie", "total_carb", "sugar", "protein"], range(df.shape[1] - 4, df.shape[1])):
            cols[i] = name
        df.columns = cols
    return df


def _meal_times(df: pd.DataFrame) -> pd.Series:
    """Start time of each food item: 'time_begin' if present, otherwise date + time of day."""
    t = (pd.to_datetime(df["time_begin"], format="mixed", errors="coerce") if "time_begin" in df
         else pd.Series(pd.NaT, index=df.index))
    if t.isna().any() and "date" in df:
        tod = df["time"] if "time" in df else df.get("time_of_day")
        if tod is not None:
            fb = pd.to_datetime(df["date"].astype(str) + " " + tod.astype(str), format="mixed", errors="coerce")
            t = t.fillna(fb)
    return t


def load_food_log(path: Path) -> pd.DataFrame:
    """Meals = food items logged within 15 minutes of each other, nutrients summed."""
    df = _read_food_csv(path)
    t = _meal_times(df)
    nut = {}
    for name, keys in {
        "carbs": ["total_carb", "carb", "carbs"],
        "fiber": ["dietary_fiber", "fiber"],
        "sugar": ["sugar"],
        "protein": ["protein"],
        "fat": ["total_fat", "fat"],
        "calories": ["calorie", "calories"],
    }.items():
        col = next((c for c in df.columns if c in keys), None)
        nut[name] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col else 0.0
    items = pd.DataFrame({"time": t, **nut}).dropna(subset=["time"]).sort_values("time")
    if items.empty:
        return pd.DataFrame(columns=MEAL_COLUMNS)
    new_meal = items["time"].diff().gt(pd.Timedelta(minutes=15)) | items["time"].diff().isna()
    items["meal_id"] = new_meal.cumsum()
    meals = items.groupby("meal_id").agg(
        time=("time", "min"), carbs=("carbs", "sum"), fiber=("fiber", "sum"), sugar=("sugar", "sum"),
        protein=("protein", "sum"), fat=("fat", "sum"), calories=("calories", "sum"),
    ).reset_index(drop=True)
    meals["net_carbs"] = (meals["carbs"] - meals["fiber"]).clip(lower=0)
    return meals


def load_demographics(root: Path) -> pd.DataFrame:
    df = _clean_cols(pd.read_csv(root / "Demographics.csv"))
    df["pid"] = df["id"].astype(int).map(lambda x: f"{x:03d}")
    df["gender"] = df["gender"].astype(str).str.upper().str.strip()
    df["hba1c"] = pd.to_numeric(df["hba1c"], errors="coerce")
    return df[["pid", "gender", "hba1c"]].set_index("pid")


def activity_from_heart_rate(tab: pd.DataFrame) -> pd.DataFrame:
    """Fallback when the accelerometer file is not available (it is ~840 MB per person).

    Movement raises heart rate, so activity is estimated as how far heart rate sits above the person's
    resting level (10th percentile of the PAST 72 h - never future data), converted to the same
    milli-g scale the accelerometer would give (config.HR_TO_ACTIVITY_MG milli-g per bpm).
    """
    hr = tab["hr_sum"] / tab["hr_n"].replace(0, np.nan)
    rest = hr.rolling("72h", min_periods=24).quantile(0.10)
    rest = rest.fillna(hr.expanding(min_periods=1).quantile(0.10))
    proxy = (hr - rest - config.HR_ACTIVITY_MARGIN).clip(lower=0) * config.HR_TO_ACTIVITY_MG
    tab["act_sum"] = proxy
    tab["act_n"] = proxy.notna().astype(float)
    tab["act_from_hr"] = 1
    return tab


# ------------------------------------------------------------------ one participant -> 5-min table
def build_participant_table(folder: Path, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (5-minute sensor table, meals table) for one participant."""
    pid = folder.name
    say = print if verbose else (lambda *a, **k: None)

    dex = _find_file(folder, "Dexcom")
    if dex is None:
        raise FileNotFoundError(f"No Dexcom file in {folder}")
    glucose = load_dexcom(dex)
    grid = pd.date_range(glucose.index.min(), glucose.index.max(), freq=BIN, name="time")
    tab = pd.DataFrame(index=grid)
    tab["glucose_obs"] = glucose.reindex(grid)
    say(f"  [{pid}] CGM: {glucose.notna().sum()} readings over {(grid[-1] - grid[0]).days} days")

    for name, kind in [("HR", "mean"), ("TEMP", "mean"), ("EDA", "mean"), ("ACC", "acc")]:
        f = _find_file(folder, name)
        col = {"HR": "hr", "TEMP": "temp", "EDA": "eda", "ACC": "act"}[name]
        if f is None:
            if name != "ACC":
                say(f"  [{pid}] {name}: not downloaded -> left blank (the model handles missing values)")
            tab[col + "_sum"], tab[col + "_n"] = np.nan, 0.0
            continue
        say(f"  [{pid}] {name}: reading {f.name} ...")
        agg = _aggregate_signal(f, kind=kind).reindex(grid)
        tab[col + "_sum"] = agg["sum"]
        tab[col + "_n"] = agg["count"].fillna(0)

    tab["act_from_hr"] = 0
    if tab["act_n"].sum() == 0 and tab["hr_n"].sum() > 0:
        tab = activity_from_heart_rate(tab)
        say(f"  [{pid}] ACC: not downloaded -> activity estimated from heart rate")

    f = _find_file(folder, "IBI")
    if f is not None:
        ib = load_ibi(f).reindex(grid)
        tab["ssd"] = ib["ssd"].fillna(0.0)
        tab["n_pairs"] = ib["n_pairs"].fillna(0.0)
    else:
        tab["ssd"], tab["n_pairs"] = 0.0, 0.0

    fl = _find_file(folder, "Food_Log")
    meals = pd.DataFrame(columns=MEAL_COLUMNS)
    if fl is not None:
        try:
            meals = load_food_log(fl)
        except Exception as e:  # odd file layout -> keep going without meals for this person
            say(f"  [{pid}] WARNING: could not read {fl.name} ({type(e).__name__}: {e}) -> no meals for this person")
    say(f"  [{pid}] meals logged: {len(meals)}")
    tab["pid"] = pid
    return tab, meals


def load_all(base: Path, use_cache: bool = True, verbose: bool = True) -> tuple[dict, dict, pd.DataFrame]:
    """Load every participant (cached after the first run)."""
    root = find_data_root(base)
    tag = "demo" if "demo" in str(root).lower() else "real"
    cache = config.PROCESSED_DIR / tag
    cache.mkdir(parents=True, exist_ok=True)
    demo = load_demographics(root)
    tables, meals = {}, {}
    for pid in list_participants(root):
        if _find_file(root / pid, "Dexcom") is None:
            if verbose:
                print(f"  [{pid}] no CGM file yet - skipped")
            continue
        tp, mp = cache / f"{pid}_sensors.pkl", cache / f"{pid}_meals.pkl"
        if use_cache and tp.exists() and mp.exists():
            tables[pid], meals[pid] = pd.read_pickle(tp), pd.read_pickle(mp)
            if verbose:
                print(f"  [{pid}] loaded from cache")
            continue
        try:
            t, m = build_participant_table(root / pid, verbose=verbose)
        except Exception as e:
            print(f"  [{pid}] WARNING: skipped this person - {type(e).__name__}: {e}")
            continue
        t.to_pickle(tp)
        m.to_pickle(mp)
        tables[pid], meals[pid] = t, m
    return tables, meals, demo
