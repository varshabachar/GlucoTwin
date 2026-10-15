"""
GlucoTwin - Doctor dashboard (Streamlit).

    streamlit run app/dashboard.py

A doctor's view of each virtual patient: who is at risk right now, why, what the twin expects,
and what would change the outcome. A "replay clock" in the sidebar walks through the recorded week
as if it were happening live; nothing after the clock time is used for any prediction.
"""
from __future__ import annotations

import json
import sys
from datetime import time as dtime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from src.names import fmt_value, label  # noqa: E402
from src.twin import activity_effect, simulate_whatif  # noqa: E402

A = config.ARTIFACTS_DIR
BLUE, ORANGE, AQUA, VIOLET, MUTED, CRITICAL = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#898781", "#d03b3b"
THR, ALERT = config.SPIKE_THRESHOLD, config.ALERT_THRESHOLD

st.set_page_config(page_title="GlucoTwin · Doctor dashboard", page_icon="🩺", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading virtual patients ...")
def load():
    data = pd.read_pickle(A / "dataset.pkl.gz")
    meals = pd.read_pickle(A / "meals.pkl")
    ehr = pd.read_csv(A / "ehr" / "ehr_summary.csv", dtype={"pid": str}).set_index("pid")
    twin = json.loads((A / "twin_params.json").read_text())
    feats = json.loads((A / "features.json").read_text())["final"]
    metrics = json.loads((A / "metrics.json").read_text())
    thr_file = A / "alert_thresholds.json"
    thr = json.loads(thr_file.read_text())["final_model_per_patient"] if thr_file.exists() else {}
    return data, meals, ehr, twin, feats, metrics, thr


@st.cache_resource
def booster(pid: str):
    f = A / "models" / f"fold_{pid}.txt"
    return lgb.Booster(model_file=str(f)) if f.exists() else None


if not (A / "dataset.pkl.gz").exists():
    st.error("No results yet. Run `python run_pipeline.py` (or `--demo`) first, then reload this page.")
    st.stop()

data, meals, ehr, twin, FEATS, metrics, PT_THR = load()
PIDS = sorted(data["pid"].unique())
patients = {pid: data[data["pid"] == pid] for pid in PIDS}
max_days = int(max(d["rel_day"].max() for d in patients.values())) + 1


def now_row(pid: str, day: int, clock: dtime):
    d = patients[pid]
    start = d.index.min().normalize()
    now = start + pd.Timedelta(days=day - 1, hours=clock.hour, minutes=clock.minute)
    past = d[d.index <= now]
    return (past.iloc[-1], past.index[-1], now) if len(past) else (None, None, now)


def reasons(pid: str, row: pd.Series, n: int = 3, positive_only: bool = True) -> list[tuple[str, str, float]]:
    """Top SHAP contributions (from the model that never saw this patient), merged by plain-language name."""
    b = booster(pid)
    if b is None or row is None:
        return []
    x = row[FEATS].astype(float).to_frame().T
    contrib = b.predict(x, pred_contrib=True)[0][:-1]
    s = pd.DataFrame({"f": FEATS, "c": contrib})
    s["name"] = s["f"].map(label)
    agg = s.groupby("name").agg(c=("c", "sum"), f=("f", "first")).sort_values("c", ascending=False)
    if positive_only:
        agg = agg[agg["c"] > 0]
    def shown(f):
        return row.name.strftime("%H:%M") if f in ("hour_sin", "hour_cos") else fmt_value(f, row[f])
    return [(name, shown(r["f"]), float(r["c"])) for name, r in agg.head(n).iterrows()]


def alert_thr(pid: str) -> float:
    return float(PT_THR.get(pid, ALERT))


def status(row, pid: str) -> tuple[str, str]:
    if row is None or pd.isna(row["glucose"]):
        return "⚪", "No signal"
    if row["glucose"] >= THR:
        return "🩸", "In spike (≥140)"
    r = row["risk"]
    if pd.isna(r):
        return "⚪", "No prediction"
    if r >= alert_thr(pid):
        return "🔴", "Alert"
    if r >= 0.5 * alert_thr(pid):
        return "🟠", "Watch"
    return "🟢", "Stable"


def pct(r) -> str:
    """Never show 0% or 100% - a model is never certain."""
    return f"{min(max(float(r), 0.01), 0.99):.0%}"


def trend_arrow(d15) -> str:
    if pd.isna(d15):
        return ""
    return "⇈" if d15 > 15 else "↑" if d15 > 5 else "⇊" if d15 < -15 else "↓" if d15 < -5 else "→"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🩺 GlucoTwin")
st.sidebar.caption("Patient digital twins · glucose-spike early warning")
page = st.sidebar.radio("View", ["Ward overview", "Patient twin", "Model performance", "Data & compliance"])
st.sidebar.markdown("---")
st.sidebar.subheader("⏱ Replay clock")
day = st.sidebar.slider("Monitoring day", 1, max_days, min(2, max_days))
times = [dtime(h, m) for h in range(24) for m in (0, 15, 30, 45)]
clock = st.sidebar.select_slider("Time of day", options=times, value=dtime(13, 0), format_func=lambda t: t.strftime("%H:%M"))
st.sidebar.caption("Predictions only use data up to this moment.")
st.sidebar.markdown("---")
st.sidebar.caption("Proof-of-concept on open/synthetic data. Not a medical device.")


# ================================================================== WARD OVERVIEW
if page == "Ward overview":
    st.title("Ward overview")
    st.caption(f"Day {day}, {clock.strftime('%H:%M')} · risk of glucose ≥{THR} mg/dL within the next 2 hours")
    rows = []
    for pid in PIDS:
        row, t_row, _ = now_row(pid, day, clock)
        icon, stat = status(row, pid)
        e = ehr.loc[pid] if pid in ehr.index else None
        top = reasons(pid, row, 1) if row is not None and stat in ("Alert", "Watch") else []
        rows.append({
            "Status": f"{icon} {stat}",
            "Patient": f"P-{pid}",
            "Age / sex": f"{int(e['age'])} {'M' if e['sex_male'] else 'F'}" if e is not None else "",
            "HbA1c %": float(e["hba1c"]) if e is not None else np.nan,
            "Glucose": f"{row['glucose']:.0f} {trend_arrow(row['d15'])}" if row is not None and pd.notna(row["glucose"]) else "—",
            "2-h spike risk": float(row["risk"]) if row is not None and pd.notna(row["risk"]) else np.nan,
            "Last meal": (f"{row['carbs_last_meal']:.0f} g, {row['mins_since_meal']:.0f} min ago"
                          if row is not None and row["mins_since_meal"] < 720 else "none in 12 h"),
            "Main reason": f"{top[0][0]} ({top[0][1]})" if top else "",
            "_order": {"Alert": 0, "In spike (≥140)": 1, "Watch": 2}.get(stat, 3),
        })
    ward = pd.DataFrame(rows).sort_values(["_order", "2-h spike risk"], ascending=[True, False]).drop(columns="_order")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients monitored", len(PIDS), border=True)
    c2.metric("🔴 Alerts now", int(ward["Status"].str.contains("Alert").sum()), border=True)
    c3.metric("🟠 Watch", int(ward["Status"].str.contains("Watch").sum()), border=True)
    c4.metric("🩸 Already above 140", int(ward["Status"].str.contains("In spike").sum()), border=True)
    st.dataframe(
        ward, hide_index=True,
        column_config={
            "2-h spike risk": st.column_config.ProgressColumn("2-h spike risk", min_value=0, max_value=1, format="percent"),
            "HbA1c %": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption("Open a patient in **Patient twin** (sidebar) to see the reasons, the timeline and the what-if simulator.")


# ================================================================== PATIENT TWIN
elif page == "Patient twin":
    pid = st.selectbox("Patient", PIDS, format_func=lambda p: f"P-{p}")
    d = patients[pid]
    row, t_row, now = now_row(pid, day, clock)
    e = ehr.loc[pid] if pid in ehr.index else None
    st.title(f"Virtual patient P-{pid}")
    if row is None:
        st.warning("No data before this time for this patient. Move the replay clock forward.")
        st.stop()
    icon, stat = status(row, pid)
    st.caption(f"{now:%a %d %b %H:%M} (replay) · status: {icon} {stat} · "
               f"alert threshold {alert_thr(pid):.2f} (set on other patients only)")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Glucose", f"{row['glucose']:.0f} mg/dL" if pd.notna(row["glucose"]) else "—",
              f"{row['d15']:+.0f} in 15 min" if pd.notna(row["d15"]) else None, delta_color="inverse", border=True)
    m2.metric("Spike risk, next 2 h", pct(row["risk"]) if pd.notna(row["risk"]) else "—", border=True)
    m3.metric("Last meal", f"{row['carbs_last_meal']:.0f} g carbs" if row["mins_since_meal"] < 720 else "—",
              f"{row['mins_since_meal']:.0f} min ago" if row["mins_since_meal"] < 720 else None, delta_color="off", border=True)
    m4.metric("Sleep last night", f"{row['sleep_prev_min'] / 60:.1f} h" if pd.notna(row["sleep_prev_min"]) else "—", border=True)
    m5.metric("HRV (last hour)", f"{row['rmssd_60']:.0f} ms" if pd.notna(row["rmssd_60"]) else "—", border=True)

    if stat == "Alert":
        why = reasons(pid, row, 3)
        st.error(f"**🔴 Early warning: {pct(row['risk'])} risk of glucose ≥{THR} mg/dL within 2 hours** "
                 f"(this patient's alert level is {alert_thr(pid):.0%}).  \n"
                 + "Main reasons: " + "; ".join(f"{n} ({v})" for n, v, _ in why))
    elif stat == "Watch":
        st.warning(f"**🟠 Watch: {pct(row['risk'])} risk of a spike in the next 2 hours.**")
    elif stat.startswith("In spike"):
        st.error("**🩸 Glucose is already above 140 mg/dL.**")
    else:
        st.success("**🟢 Stable: low risk of a spike in the next 2 hours.**")

    left, right = st.columns([1, 2.6])
    # ---------------- EHR card
    with left:
        with st.container(border=True):
            st.subheader("Health record")
            st.caption("Synthetic FHIR R4 record · sex & HbA1c are real")
            if e is not None:
                st.markdown(
                    f"**{int(e['age'])} y, {'male' if e['sex_male'] else 'female'}** · BMI {e['bmi']:.1f}  \n"
                    f"HbA1c **{e['hba1c']:.1f} %** · fasting glucose {e['fasting_glucose']:.0f} mg/dL  \n"
                    f"BP {e['sbp']:.0f}/{e['dbp']:.0f} · TG {e['triglycerides']:.0f} · HDL {e['hdl']:.0f} · LDL {e['ldl']:.0f}")
                dx = [n for k, n in [("dx_prediabetes", "Prediabetes"), ("dx_hypertension", "Hypertension"),
                                     ("dx_dyslipidemia", "Dyslipidaemia")] if e[k]]
                meds = [n for k, n in [("on_metformin", "Metformin"), ("on_statin", "Atorvastatin")] if e[k]]
                st.markdown("**Diagnoses:** " + (", ".join(dx) if dx else "none"))
                st.markdown("**Medications:** " + (", ".join(meds) if meds else "none"))
                st.markdown("**Family history of T2D:** " + ("yes" if e["family_hx_t2d"] else "no"))
                st.markdown(f"**Genetic risk score:** {e['prs']:+.2f} SD")
                with st.expander("Genotypes"):
                    for g in str(e["genotypes"]).split("; "):
                        st.write(g)
        with st.container(border=True):
            st.subheader("Why this risk?")
            st.caption("How each factor pushes the risk up (+) or down (−)")
            allr = reasons(pid, row, 8, positive_only=False) if pd.notna(row["risk"]) else []
            if allr:
                df_r = pd.DataFrame(allr, columns=["Factor", "Value", "Effect"])
                df_r["Effect"] = df_r["Effect"].map(lambda v: f"{'+' if v > 0 else '−'}{abs(v):.2f}")
                st.dataframe(df_r, hide_index=True)
            else:
                st.write("No prediction at this moment.")

    # ---------------- timeline
    with right:
        cA, cB = st.columns([3, 1])
        span = cA.slider("Days shown", 1, 7, 2)
        reveal = cB.toggle("Show what happened next", value=False, help="For demos: reveal the next 2 hours")
        lo = now - pd.Timedelta(days=span)
        hi = now + (pd.Timedelta(hours=2) if reveal else pd.Timedelta(0))
        w = d[(d.index > lo) & (d.index <= hi)]
        past = w[w.index <= now]
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            row_heights=[0.4, 0.2, 0.2, 0.2],
                            subplot_titles=("Glucose (mg/dL)", "Model spike risk (next 2 h)",
                                            "Heart rate (bpm) and HRV (ms)", "Activity · estimated sleep shaded"))
        fig.add_trace(go.Scatter(x=past.index, y=past["glucose_obs"], name="Glucose", line=dict(color=BLUE, width=2)), 1, 1)
        if reveal:
            fut = w[w.index > now]
            fig.add_trace(go.Scatter(x=fut.index, y=fut["glucose_obs"], name="What happened next",
                                     line=dict(color=BLUE, width=2, dash="dot")), 1, 1)
        fig.add_hline(y=THR, line=dict(color=CRITICAL, dash="dash", width=1), row=1, col=1)
        m = meals[pid]
        m = m[(m["time"] > lo) & (m["time"] <= hi)]
        if len(m):
            ymeal = np.nanmax(w["glucose_obs"]) + 10 if w["glucose_obs"].notna().any() else 150
            fig.add_trace(go.Scatter(x=m["time"], y=[ymeal] * len(m), mode="markers+text", name="Meal",
                                     marker=dict(symbol="triangle-down", size=11, color=AQUA),
                                     text=[f"{c:.0f}g" for c in m["net_carbs"]], textposition="top center",
                                     hovertext=[f"{c:.0f} g net carbs, {k:.0f} kcal" for c, k in zip(m["net_carbs"], m["calories"])],
                                     hoverinfo="text"), 1, 1)
        fig.add_trace(go.Scatter(x=past.index, y=past["risk"], name="Risk", line=dict(color=ORANGE, width=2),
                                 fill="tozeroy", fillcolor="rgba(235,104,52,0.12)"), 2, 1)
        fig.add_hline(y=alert_thr(pid), line=dict(color=MUTED, dash="dash", width=1), row=2, col=1)
        fig.add_trace(go.Scatter(x=past.index, y=past["hr"], name="Heart rate", line=dict(color=VIOLET, width=1.5)), 3, 1)
        fig.add_trace(go.Scatter(x=past.index, y=past["rmssd_60"], name="HRV (RMSSD)", line=dict(color=AQUA, width=1.5)), 3, 1)
        fig.add_trace(go.Bar(x=past.index, y=past["act"], name="Activity", marker_color=MUTED), 4, 1)
        sl = past["is_sleep"].fillna(0).astype(int)
        if sl.any():
            blocks = (sl.diff().fillna(sl.iloc[0]) != 0).cumsum()
            for _, b in past[sl == 1].groupby(blocks[sl == 1]):
                fig.add_vrect(x0=b.index[0], x1=b.index[-1], fillcolor="rgba(98,80,214,0.10)", line_width=0, row=4, col=1)
        fig.add_vline(x=now, line=dict(color=MUTED, width=1.5))
        fig.update_yaxes(range=[0, 1], row=2, col=1)
        fig.update_layout(height=640, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                          legend=dict(orientation="h", y=-0.06))
        st.plotly_chart(fig)

    # ---------------- twin + what-if
    st.markdown("---")
    st.header("🧬 Digital twin: this patient's glucose physiology")
    by_day = {p["day"]: p for p in twin["by_day"][pid]}
    tp_now = by_day.get(int(row["rel_day"]), twin["final"][pid])
    prior = config.TWIN_PRIOR
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Carb sensitivity", f"{tp_now['k']:.2f} mg/dL per g", f"{tp_now['k'] / prior['k'] - 1:+.0%} vs average",
              delta_color="inverse", border=True, help="Peak glucose rise per gram of net carbs")
    t2.metric("Time to peak", f"{tp_now['tp']:.0f} min", border=True, help="How fast glucose peaks after eating")
    if tp_now.get("activity_fitted", True):
        t3.metric("Activity benefit", f"{tp_now['a']:.2f}", f"{tp_now['a'] / prior['a'] - 1:+.0%} vs average", border=True,
                  help="mg/dL lowered per milli-g of smoothed wrist activity")
    else:
        t3.metric("Activity benefit", f"{tp_now['a']:.2f}", "population average", delta_color="off", border=True,
                  help="No accelerometer data for this patient: activity is estimated from heart rate, which also "
                       "rises after meals, so the twin uses the population-average activity effect.")
    t4.metric("Recovery time", f"{tp_now['tr']:.0f} min", border=True, help="Time for unexplained highs/lows to fade")
    n_days = int(row["rel_day"])
    st.caption(f"Calibrated on {n_days} previous day(s) of this patient's data"
               + (" — still on population defaults (day 1)." if n_days == 0 else ". Re-fitted every day as new data arrives."))

    st.subheader("What-if simulator")
    w1, w2, w3, w4 = st.columns(4)
    carbs = w1.slider("Meal now: net carbs (g)", 0, 150, 60, 5)
    walk = w2.slider("Walk after the meal (min)", 0, 60, 15, 5)
    delay = w3.slider("Walk starts (min after meal)", 0, 60, 15, 5)
    pace = w4.select_slider("Walking pace", ["light", "moderate", "brisk"], value="moderate")
    intensity = {"light": 80, "moderate": 150, "brisk": 250}[pace]
    upto = d[d.index <= t_row]
    A_now = float(activity_effect(upto["act"].to_numpy())[-1])
    g_now = float(row["glucose"]) if pd.notna(row["glucose"]) else float(upto["glucose"].dropna().iloc[-1])
    base = float(row["baseline_24h"]) if pd.notna(row["baseline_24h"]) else float(upto["glucose"].quantile(0.2))
    pm = meals[pid]
    pm = pm[(pm["time"] <= t_row) & (pm["time"] > t_row - pd.Timedelta(hours=6))]
    past_meals = [((t_row - t).total_seconds() / 60, c) for t, c in zip(pm["time"], pm["net_carbs"])]
    sim = simulate_whatif(tp_now, g_now, base, A_now, past_meals, carbs, walk, delay, intensity)
    peaks = {k: (sim[k].max(), sim.loc[sim[k].idxmax(), "minute"]) for k in ["no_change", "meal", "meal_plus_walk"]}
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=sim["minute"], y=sim["no_change"], name="No new meal", line=dict(color=MUTED, dash="dash")))
    f2.add_trace(go.Scatter(x=sim["minute"], y=sim["meal"], name=f"{carbs} g meal", line=dict(color=ORANGE, width=2.5)))
    f2.add_trace(go.Scatter(x=sim["minute"], y=sim["meal_plus_walk"], name=f"{carbs} g meal + {walk}-min {pace} walk",
                            line=dict(color=BLUE, width=2.5)))
    f2.add_hline(y=THR, line=dict(color=CRITICAL, dash="dash", width=1), annotation_text=f"{THR} mg/dL")
    if walk:
        f2.add_vrect(x0=delay, x1=delay + walk, fillcolor="rgba(42,120,214,0.08)", line_width=0,
                     annotation_text="walk", annotation_position="top left")
    f2.update_layout(height=360, xaxis_title="Minutes from now", yaxis_title="Projected glucose (mg/dL)",
                     margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12))
    s1, s2 = st.columns([2.2, 1])
    with s1:
        st.plotly_chart(f2)
    with s2:
        def verdict(p):
            return "🔴 crosses 140" if p >= THR else "🟢 stays below 140"
        st.markdown("**Projected peak**")
        st.markdown(f"No new meal: **{peaks['no_change'][0]:.0f}** mg/dL")
        st.markdown(f"Meal: **{peaks['meal'][0]:.0f}** mg/dL at +{peaks['meal'][1]:.0f} min · {verdict(peaks['meal'][0])}")
        st.markdown(f"Meal + walk: **{peaks['meal_plus_walk'][0]:.0f}** mg/dL · {verdict(peaks['meal_plus_walk'][0])}")
        drop = peaks["meal"][0] - peaks["meal_plus_walk"][0]
        if walk and carbs:
            st.info(f"A {walk}-minute {pace} walk lowers this patient's projected peak by **{drop:.0f} mg/dL**.")
        st.caption("Simulation from the patient's calibrated twin. Illustrative decision support, not a prescription.")


# ================================================================== MODEL PERFORMANCE
elif page == "Model performance":
    st.title("Model performance")
    st.caption("Leave-one-subject-out: every patient is scored by a model trained only on the other patients.")
    abl = pd.DataFrame(metrics["ablation"])
    final = abl.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUROC", f"{final['auroc']:.2f}", border=True)
    c2.metric("Spikes warned in advance", f"{final['sensitivity']:.0%}", border=True)
    c3.metric("Median warning time", f"{final['median_lead_min']:.0f} min", border=True)
    c4.metric("False alarms / day", f"{final['false_alarms_per_day']:.1f}", border=True)
    st.subheader("Does fusing the streams help? (ablation)")
    show = abl[["model", "auroc", "auprc", "auroc_subject_mean", "sensitivity", "median_lead_min", "false_alarms_per_day"]]
    st.dataframe(show.rename(columns={
        "model": "Data used", "auroc": "AUROC", "auprc": "AUPRC", "auroc_subject_mean": "AUROC per patient (mean)",
        "sensitivity": "Spikes warned", "median_lead_min": "Median lead (min)", "false_alarms_per_day": "False alarms/day"}),
        hide_index=True, column_config={"Spikes warned": st.column_config.NumberColumn(format="percent")})
    fig_dir = config.FIGURES_DIR
    g1, g2 = st.columns(2)
    for col, name, cap in [(g1, "ablation.png", "Each stream added on top of the last"),
                           (g2, "roc.png", "ROC curves"),
                           (g1, "twin_learning.png", "The twin's forecast error falls as it sees more of the patient"),
                           (g2, "alert_tradeoff.png", "Sensitivity vs false alarms at different alert thresholds"),
                           (g1, "shap_importance.png", "What drives predictions"),
                           (g2, "example_patient_day.png", "One patient-day")]:
        if (fig_dir / name).exists():
            col.image(str(fig_dir / name), caption=cap)
    st.subheader("Alert threshold trade-off")
    st.dataframe(pd.DataFrame(metrics["thresholds"]).round(3), hide_index=True)


# ================================================================== DATA & COMPLIANCE
else:
    st.title("Data & compliance")
    st.markdown(f"""
**Sandbox compliance (DPDP Act / HIPAA).** No real patient records are used.

| Stream | Source | Real or synthetic |
|---|---|---|
| CGM (Dexcom G6, 5-min) | BIG IDEAs Lab dataset, PhysioNet, open access (ODC-By) | Real, de-identified |
| Smartwatch (Empatica E4): heart rate, beat-to-beat intervals → HRV; activity & sleep estimated from them | same | Real, de-identified |
| Food log | same | Real, de-identified |
| Sex, HbA1c | same (Demographics.csv) | Real, de-identified |
| Age, labs, diagnoses, medications, family history, genetic markers | Generated by `src/ehr.py` as FHIR R4 | **Synthetic** |

Data source in this build: `{metrics.get('data_source', '')}` · {len(PIDS)} participants · {metrics.get('rows', 0):,} five-minute rows.
""")
    al = A / "alignment_report.csv"
    if al.exists():
        st.subheader("Device clock alignment")
        st.caption("CGM, wristband and phone meal log keep separate clocks. Each was checked against the CGM "
                   "before modelling (src/align.py).")
        rep = pd.read_csv(al, dtype={"pid": str})
        show = pd.DataFrame({
            "Patient": "P-" + rep["pid"],
            "Wristband shift (h)": rep["wrist_shift_applied_h"],
            "Meal log used": rep["meal_log_used"].map({True: "yes", False: "no: outside CGM dates"}),
            "Meal-time shift (min)": rep["meal_offset_applied_min"],
            "Post-meal rise as logged (mg/dL)": rep["meal_rise_at_logged_time"].round(0),
            "After alignment (mg/dL)": rep["meal_rise_after_alignment"].round(0),
        })
        st.dataframe(show, hide_index=True)
    pid = st.selectbox("View a patient's FHIR R4 record (what the twin ingests)", PIDS, format_func=lambda p: f"P-{p}")
    bundle = json.loads((A / "ehr" / f"patient_{pid}.json").read_text())
    st.json(bundle, expanded=False)
