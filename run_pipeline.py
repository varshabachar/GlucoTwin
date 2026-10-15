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
from src.align import align_all
from src.ehr import build_ehr
from src.figures import make_all
from src.load_data import load_all
from src.train import FINAL_FEATURES, build_dataset, run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use data/demo instead of data/raw")
    ap.add_argument("--data", default=None, help="custom data folder")
    ap.add_argument("--no-cache", action="store_true", help="re-read raw sensor files")
    args = ap.parse_args()
    base = args.data or (config.DEMO_DIR if args.demo else config.RAW_DIR)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("\n[1/6] Loading wearable + CGM + food-log files (first run on real data takes a while; cached after)")
    tables, meals, demo = load_all(base, use_cache=not args.no_cache)
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

    print("\n[4/6] Training + leave-one-subject-out evaluation (ablation)")
    result = run_training(data)

    print("\n[5/6] Figures")
    make_all(result, curve, meals, FINAL_FEATURES)

    print("\n[6/6] Saving artifacts for the dashboard")
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
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Figures: artifacts/figures/  "
          f"Dashboard: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
