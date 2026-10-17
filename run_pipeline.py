"""
GlucoTwin - one command runs everything:

    python run_pipeline.py            # real data in data/raw/
    python run_pipeline.py --demo     # lookalike demo data (make it first: python scripts/make_demo_data.py)

Steps: load sensors -> synthetic FHIR EHR -> features + labels -> digital twin -> LOSO training
       -> evaluation -> figures -> artifacts for the dashboard.
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

import config
from src import meal_twin
from src.align import align_all
from src.ehr import build_ehr
from src.figures import make_all, meal_learning_chart
from src.load_data import load_all
from src.train import FINAL_FEATURES, build_dataset, run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use data/demo instead of data/raw")
    ap.add_argument("--data", default=None, help="custom data folder")
    ap.add_argument("--no-cache", action="store_true", help="re-read raw sensor files")
    ap.add_argument("--women", action="store_true",
                    help="women-only cohort -> writes to artifacts_women/ (main results untouched)")
    args = ap.parse_args()
    base = args.data or (config.DEMO_DIR if args.demo else config.RAW_DIR)
    if args.women:
        # a separate output folder, so the main 16-participant artifacts and the dashboard are untouched
        config.ARTIFACTS_DIR = config.ROOT / "artifacts_women"
        config.FIGURES_DIR = config.ARTIFACTS_DIR / "figures"
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("\n[1/6] Loading wearable + CGM + food-log files (first run on real data takes a while; cached after)")
    tables, meals, demo = load_all(base, use_cache=not args.no_cache)
    if args.women:
        women = {p for p in tables if str(demo.loc[p, "gender"]).upper().startswith("F")} \
            if len(demo) else set()
        dropped = sorted(set(tables) - women)
        tables = {p: t for p, t in tables.items() if p in women}
        meals = {p: m for p, m in meals.items() if p in women}
        print(f"\n  WOMEN-ONLY COHORT: keeping {len(tables)} female participants "
              f"({', '.join(sorted(tables))})")
        print(f"  dropped {len(dropped)} male participants ({', '.join(dropped)})")
        print("  NOTE: this is a smaller cohort, so these numbers are NOT comparable to the "
              "16-participant table and are reported separately.")
    print("\n[1b] Checking that the three device clocks (CGM, wristband, meal log) line up")
    tables, meals, align_report = align_all(tables, meals)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    align_report.to_csv(config.ARTIFACTS_DIR / "alignment_report.csv")

    print("\n[2/6] Generating synthetic EHR (FHIR R4) and ingesting it")
    ehr = build_ehr(demo)
    print(f"  wrote {len(ehr)} FHIR bundles to artifacts/ehr/")

    print("\n[3/6] Building features, labels and the per-patient digital twin")
    data, twin_params, final_params, curve = build_dataset(tables, meals, ehr)
    n_lab = int(data["y"].notna().sum())
    print(f"  {len(data)} five-minute rows, {n_lab} labelled, spike-ahead rate {data['y'].mean():.1%}")

    print("\n[4/7] Training + leave-one-subject-out evaluation (ablation)")
    result = run_training(data)

    # ---------------------------------------------------------------- NEW: per-meal learning twin
    # An ADDITION, reported separately. It does not feed the model above or change any locked number.
    print("\n[5/7] Per-meal learning twin: predict each meal's peak before it happens, then learn")
    avail = meal_twin.composition_available(meals)
    print("  what the food logs actually contain (checked, not assumed):")
    print(avail.to_string().replace("\n", "\n    "))
    feats = {pid: data[data["pid"] == pid] for pid in sorted(tables)}
    meal_log = meal_twin.run_all(feats, meals)
    meal_cmp = meal_twin.compare_variants(meal_log)
    meal_final = {pid: meal_twin.fit_final(pid, feats[pid], meals.get(pid, pd.DataFrame()))
                  for pid in sorted(tables)}

    print("\n[6/7] Figures")
    make_all(result, curve, meals, FINAL_FEATURES)
    meal_learning_chart(meal_log, meal_cmp)

    print("\n[7/7] Saving artifacts for the dashboard")
    out = config.ARTIFACTS_DIR
    result["data"].to_pickle(out / "dataset.pkl.gz", compression="gzip")
    pd.to_pickle(meals, out / "meals.pkl")
    (out / "twin_params.json").write_text(json.dumps({"final": final_params, "by_day": twin_params}, indent=2))
    curve.to_csv(out / "twin_learning_curve.csv", index=False)
    summary = {
        "data_source": str(base),
        "participants": sorted(tables),
        "rows": len(data), "labelled_rows": n_lab,
        "spike_threshold_mg_dl": config.SPIKE_THRESHOLD, "horizon_min": config.HORIZON_MIN,
        "alert_threshold": config.ALERT_THRESHOLD,
        "ablation": result["ablation"].drop(columns="column").to_dict(orient="records"),
        "thresholds": result["thresholds"].to_dict(orient="records"),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    # per-meal learning artifacts (the dashboard reads these; it never touches raw data)
    meal_log.to_csv(out / "meal_learning.csv", index=False)
    meal_cmp.to_csv(out / "meal_learning_summary.csv", index=False)
    avail.to_csv(out / "meal_composition_available.csv")
    (out / "meal_twin_params.json").write_text(json.dumps(meal_final, indent=2, default=float))

    pd.set_option("display.width", 160)
    show = result["ablation"][["model", "auroc", "auprc", "sensitivity", "median_lead_min", "false_alarms_per_day"]]
    print("\n=== RESULTS (each person predicted by a model that never saw them) ===")
    print(show.round(3).to_string(index=False))
    rep = align_report
    shifted = [f"{p} ({int(m):+d} min)" for p, m in rep["meal_offset_applied_min"].items() if m]
    unused = [p for p, ok in rep["meal_log_used"].items() if not ok]
    print("\n=== CLOCK ALIGNMENT ===")
    print(f"  wristband shift: {int(rep['wrist_shift_applied_h'].iloc[0]):+d} h (same for everyone)")
    print(f"  meal times shifted: {', '.join(shifted) if shifted else 'none'}")
    print(f"  meal logs not used (outside glucose dates): {', '.join(unused) if unused else 'none'}")
    if len(meal_cmp):
        print("\n=== PER-MEAL LEARNING TWIN (separate from the table above) ===")
        print("Error in the predicted post-meal peak. Every meal is predicted from parameters")
        print("fitted only on that patient's EARLIER meals.")
        show_m = meal_cmp.rename(columns={
            "variant": "Twin", "mae_all": "mean error (mg/dL)", "mae_late": "later half of meals",
            "median_all": "median", "meals": "meals", "improvement_vs_population": "vs population"})
        print(show_m.round(3).to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Figures: artifacts/figures/  "
          f"Dashboard: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
