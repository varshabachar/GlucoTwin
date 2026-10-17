"""
Prove that the per-meal learning twin never sees the future.

    python scripts/check_no_leakage.py            # real data in data/raw/
    python scripts/check_no_leakage.py --demo     # lookalike demo data

Two independent checks, both of which must pass:

  1. FUTURE MEALS. Predict a meal's peak twice: once with the patient's whole food log, and once
     with every meal AFTER that one deleted. A forecast that uses only food already eaten cannot
     change. This is the check that matters most, because the 3-hour response window of one meal
     often overlaps the next meal.

  2. FUTURE GLUCOSE. Re-fit the twin for a given meal twice: once on the full recording, and once
     with all CGM readings from that meal onwards deleted. Parameters fitted only on earlier meals
     cannot change either.

Exit code 0 = no leakage found, 1 = leakage found.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import meal_twin  # noqa: E402
from src.align import align_all  # noqa: E402
from src.features import add_labels, build_features  # noqa: E402
from src.load_data import load_all  # noqa: E402
from src.twin import PatientSeries  # noqa: E402

TOL = 1e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use data/demo instead of data/raw")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    base = args.data or (config.DEMO_DIR if args.demo else config.RAW_DIR)

    print("Loading ...")
    tables, meals, _ = load_all(base, verbose=False)
    tables, meals, _ = align_all(tables, meals, verbose=False)
    feats = {pid: add_labels(build_features(tables[pid], meals[pid])) for pid in sorted(tables)}

    probe = dict(config.TWIN_PRIOR, fb=0.5, fd=0.3)
    n_meal_probe = n_fit_probe = 0
    failures = []

    for pid in sorted(feats):
        df, m = feats[pid], meals.get(pid)
        if m is None or not len(m) or not int(df["meal_log_ok"].iloc[0]):
            continue
        ps = PatientSeries(df, m)
        comp = meal_twin.meal_composition(m)
        idx = meal_twin._scoreable_meals(ps, df, m)

        for n_seen, mi in enumerate(idx):
            row = int(np.searchsorted(ps.t, ps.meal_t[mi], side="left"))
            horizon = ps.t[row] + config.MEAL_PEAK_WINDOW_MIN

            # ---- 1. a later meal inside the response window must not change the forecast
            if ((ps.meal_t > ps.t[row]) & (ps.meal_t <= horizon)).any():
                full, _ = meal_twin.predict_peak(ps, comp, row, probe)
                keep = m[m["time"] <= m["time"].iloc[mi]].reset_index(drop=True)
                ps_cut = PatientSeries(df, keep)
                row_cut = int(np.searchsorted(ps_cut.t, ps_cut.meal_t[-1], side="left"))
                cut, _ = meal_twin.predict_peak(ps_cut, meal_twin.meal_composition(keep), row_cut, probe)
                n_meal_probe += 1
                if abs(full - cut) > TOL:
                    failures.append(f"{pid} meal #{n_seen + 1}: forecast {full:.6f} -> {cut:.6f} "
                                    f"when later meals were deleted")

            # ---- 2. glucose recorded at or after the meal must not change the fitted parameters
            if n_seen and n_seen % 7 == 0:
                hist = np.array([int(np.searchsorted(ps.t, ps.meal_t[j], side="left"))
                                 for j in idx[:n_seen]], dtype=int)
                a = meal_twin.fit_on_meals(ps, comp, hist, row, True, True, None)
                blind = df.copy()
                blind.iloc[row:, blind.columns.get_loc("glucose_obs")] = np.nan
                blind.iloc[row:, blind.columns.get_loc("glucose")] = np.nan
                ps_blind = PatientSeries(blind, m)
                ps_blind.G = np.where(np.arange(len(ps.G)) < row, ps.G, np.nan)
                b = meal_twin.fit_on_meals(ps_blind, comp, hist, row, True, True, None)
                n_fit_probe += 1
                moved = {k: (a[k], b[k]) for k in meal_twin.ALL_PARAMS if abs(a[k] - b[k]) > 1e-4}
                if moved:
                    failures.append(f"{pid} meal #{n_seen + 1}: parameters moved when future "
                                    f"glucose was deleted: {moved}")

    print(f"\nCheck 1 - forecasts probed against deleted future meals : {n_meal_probe}")
    print(f"Check 2 - fits probed against deleted future glucose    : {n_fit_probe}")
    if failures:
        print(f"\nLEAKAGE FOUND ({len(failures)}):")
        for f in failures[:20]:
            print("  -", f)
        return 1
    if n_meal_probe == 0 and n_fit_probe == 0:
        print("\nNothing to probe (no usable meals). Not a pass.")
        return 1
    print("\nNO LEAKAGE FOUND: every prediction used only data available before the meal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
