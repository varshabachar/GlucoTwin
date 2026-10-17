"""
Step 7 - Figures for the README, the slides and the video. Saved to artifacts/figures/.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import config
from src.features import spike_onsets
from src.names import label

# reference palette (light mode) - see README "Design" note
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL = "#d03b3b"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "text.color": INK,
    "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True,
})


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def ablation_chart(ablation: pd.DataFrame):
    """Two small charts (AUROC, AUPRC) - one scale each, never a dual axis."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    names = ablation["model"].tolist()
    y = np.arange(len(names))[::-1]
    for ax, metric, title in [(axes[0], "auroc", "AUROC (higher is better)"),
                              (axes[1], "auprc", "AUPRC (higher is better)")]:
        vals = ablation[metric].to_numpy()
        colors = [MUTED if n.startswith("Rule") else BLUE for n in names]
        ax.barh(y, vals, color=colors, height=0.6)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v + 0.01, yi, f"{v:.3f}", va="center", color=INK2, fontsize=9)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, 1.08)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, names)
    fig.suptitle("Each data stream added on top of the last (leave-one-subject-out)", x=0.01, ha="left",
                 color=INK2, fontsize=10, y=1.02)
    _save(fig, "ablation.png")


def roc_chart(data: pd.DataFrame, ablation: pd.DataFrame):
    pick = [(ablation.iloc[0], MUTED, "--"), (ablation.iloc[1], ORANGE, "-"), (ablation.iloc[-1], BLUE, "-")]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for row, c, ls in pick:
        ok = data["y"].notna() & data[row["column"]].notna()
        fpr, tpr, _ = roc_curve(data.loc[ok.to_numpy(), "y"], data.loc[ok.to_numpy(), row["column"]])
        ax.plot(fpr, tpr, color=c, ls=ls, label=f"{row['model']}  (AUROC {row['auroc']:.2f})")
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Spike-in-2-hours: ROC", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, "roc.png")


def twin_learning_chart(curve: pd.DataFrame):
    if curve.empty:
        return
    g = curve.groupby("days_of_data")[["mae_twin", "mae_prior", "mae_persistence"]].mean()
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(g.index, g["mae_twin"], color=BLUE, marker="o", ms=5, label="Personal twin (re-fitted daily)")
    ax.plot(g.index, g["mae_prior"], color=ORANGE, marker="o", ms=5, label="Generic population model")
    ax.plot(g.index, g["mae_persistence"], color=MUTED, ls="--", label="'Glucose stays the same'")
    ax.set_xlabel("Days of this patient's data the twin has seen")
    ax.set_ylabel("1-hour forecast error (mg/dL)")
    ax.set_title("The twin learns the patient", loc="left")
    ax.legend(fontsize=8.5)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    _save(fig, "twin_learning.png")


def threshold_chart(thresholds: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.plot(thresholds["false_alarms_per_day"], thresholds["sensitivity"] * 100, color=BLUE, marker="o", ms=6)
    for _, r in thresholds.iterrows():
        ax.annotate(f"risk ≥ {r['threshold']:.1f}", (r["false_alarms_per_day"], r["sensitivity"] * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=INK2)
    ax.set_xlabel("False alarms per patient-day")
    ax.set_ylabel("Spikes warned in advance (%)")
    ax.set_title("Alert trade-off", loc="left")
    _save(fig, "alert_tradeoff.png")


def shap_charts(model, data: pd.DataFrame, features: list[str], n: int = 3000):
    import shap
    lab = data["y"].notna().to_numpy()
    X = data.loc[lab, features].astype(float)
    X = X.sample(min(n, len(X)), random_state=config.RANDOM_SEED)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = shap.TreeExplainer(model).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    # features that share a plain-language name (e.g. the two time-of-day terms) are added together
    imp = pd.Series(np.abs(sv).mean(axis=0), index=[label(f) for f in features]).groupby(level=0).sum()
    imp = imp.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
    ax.set_xlabel("Mean |SHAP value| (impact on predicted risk)")
    ax.set_title("What drives the spike prediction", loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, "shap_importance.png")
    plt.figure()
    shap.summary_plot(sv, X.rename(columns=label), show=False, max_display=15, plot_size=(8, 6))
    plt.title("Direction of each factor's effect", loc="left")
    fig2 = plt.gcf()
    _save(fig2, "shap_beeswarm.png")
    imp.sort_values(ascending=False).to_csv(config.ARTIFACTS_DIR / "shap_importance.csv")


def example_patient_chart(data: pd.DataFrame, meals: dict, thresholds: dict):
    """One patient-day: glucose with meals + spike starts (top), model risk with alerts (bottom)."""
    best = None
    for pid, d in data.groupby("pid"):
        for day, dd in d.groupby("rel_day"):
            n_on = len(spike_onsets(dd))
            if n_on >= 2 and dd["glucose_obs"].notna().mean() > 0.9 and (best is None or n_on > best[2]):
                best = (pid, day, n_on)
    if best is None:
        return
    pid, day, _ = best
    d = data[(data["pid"] == pid) & (data["rel_day"] == day)]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    a1.plot(d.index, d["glucose_obs"], color=INK, lw=1.8, label="CGM glucose")
    a1.axhline(config.SPIKE_THRESHOLD, color=CRITICAL, lw=1.2, ls="--")
    a1.text(d.index[0], config.SPIKE_THRESHOLD + 3, f"{config.SPIKE_THRESHOLD} mg/dL spike line", color=CRITICAL, fontsize=8.5)
    m = meals[pid]
    m = m[(m["time"] >= d.index[0]) & (m["time"] <= d.index[-1])]
    top = float(np.nanmax(d["glucose_obs"]))
    a1.set_ylim(top=top + 28)
    for _, r in m.iterrows():
        a1.axvline(r["time"], color=AQUA, lw=1, alpha=0.8)
        a1.text(r["time"], top + 12, f"{r['net_carbs']:.0f} g", color=INK2, fontsize=8, ha="center",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    a1.set_ylabel("mg/dL")
    a1.set_title(f"Participant {pid}, day {day + 1}: meals (green lines, net carbs) and glucose", loc="left")
    thr = thresholds.get(pid, config.ALERT_THRESHOLD)
    a2.plot(d.index, d["risk"], color=BLUE, lw=1.8)
    a2.axhline(thr, color=MUTED, lw=1, ls="--")
    a2.text(d.index[0], thr + 0.03, f"alert threshold {thr:.2f} (chosen on other patients)", color=INK2, fontsize=8)
    a2.fill_between(d.index, 0, d["risk"].where(d["risk"] >= thr), color=BLUE, alpha=0.15)
    for o in spike_onsets(d):
        for ax in (a1, a2):
            ax.axvline(o, color=CRITICAL, lw=1.2, alpha=0.6)
    a2.set_ylim(0, 1)
    a2.set_ylabel("Spike risk (next 2 h)")
    a2.set_title("Model risk (patient unseen in training). Red lines = actual spike starts", loc="left", fontsize=10)
    fig.autofmt_xdate()
    _save(fig, "example_patient_day.png")


def make_all(result: dict, curve: pd.DataFrame, meals: dict, features: list[str]):
    data = result["data"]
    ablation_chart(result["ablation"])
    roc_chart(data, result["ablation"])
    twin_learning_chart(curve)
    threshold_chart(result["thresholds"])
    shap_charts(result["final_model"], data, features)
    example_patient_chart(data, meals, result.get("patient_thresholds", {}))


# ------------------------------------------------------------------ NEW: per-meal learning twin
def meal_learning_chart(log: pd.DataFrame, compare: pd.DataFrame):
    """Does the twin predict a meal better once it has seen more of that person's meals?

    Left: peak error against meals already seen - one thin line per patient, thick lines averaged.
    Right: mean error of the three twin variants over all scored meals.
    Reported separately from the LightGBM results: this is the twin's own accuracy, not the model's.
    """
    if log is None or log.empty:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.55, 1]})

    for _, d in log.groupby("pid"):
        d = d.sort_values("n_history_meals")
        ax.plot(d["n_history_meals"], d["err_comp"].rolling(5, min_periods=2).mean(),
                color=BLUE, lw=0.8, alpha=0.30)
    bin_size, min_patients = 5, 5
    g = log.copy()
    g["seen"] = (g["n_history_meals"] // bin_size) * bin_size
    agg = g.groupby("seen").agg(err_prior=("err_prior", "mean"), err_carbs=("err_carbs", "mean"),
                                err_comp=("err_comp", "mean"), patients=("pid", "nunique"))
    # Only a handful of people logged more than ~30 meals, so the far right of this curve would be
    # one or two patients' noise. Bins carrying fewer than `min_patients` people are not averaged.
    full = agg[agg["patients"] >= min_patients]
    ax.plot(full.index, full["err_prior"], color=ORANGE, marker="o", ms=5,
            label="Generic population twin")
    ax.plot(full.index, full["err_carbs"], color=MUTED, ls="--", marker="o", ms=4,
            label="Learns each meal (carbs only)")
    ax.plot(full.index, full["err_comp"], color=BLUE, marker="o", ms=5,
            label="Learns each meal + composition")
    if len(full) and len(full) < len(agg):
        ax.axvline(full.index.max() + bin_size / 2, color=AXIS, lw=1, ls=":")
        ax.text(full.index.max() + bin_size * 0.7, ax.get_ylim()[1] * 0.96,
                f"fewer than {min_patients}\npatients beyond here",
                fontsize=8, color=MUTED, va="top")
    ax.set_xlabel("Meals of this patient the twin had already seen")
    ax.set_ylabel("Error in predicted peak (mg/dL)")
    ax.set_title("The twin learns meal by meal", loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    names = ["Population\ntwin", "Per-meal\n(carbs only)", "Per-meal\n+ composition"]
    vals = compare["mae_all"].to_numpy() if compare is not None and len(compare) else np.array([])
    if len(vals) == 3:
        ax2.bar(names, vals, color=[ORANGE, MUTED, BLUE], width=0.6)
        for i, v in enumerate(vals):
            ax2.text(i, v + 0.3, f"{v:.1f}", ha="center", color=INK2, fontsize=9.5)
        ax2.set_ylabel("Mean error in predicted peak (mg/dL)")
        ax2.set_title(f"Averaged over {int(compare['meals'].iloc[0])} real meals", loc="left")
        ax2.grid(axis="x", visible=False)
        ax2.set_ylim(0, max(vals) * 1.22)
    _save(fig, "meal_learning.png")


def meal_order_chart(sim: pd.DataFrame, meal_name: str, orders: list[str], threshold: float = None):
    """The same meal under each eating order, for the slides. Literature-based, not learned."""
    threshold = config.SPIKE_THRESHOLD if threshold is None else threshold
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for order, c in zip(orders, [ORANGE, MUTED, BLUE]):
        if order in sim:
            ax.plot(sim["minute"], sim[order], color=c, lw=2.2,
                    label=f"{order}  (peak {sim[order].max():.0f})")
    ax.axhline(threshold, color=CRITICAL, lw=1.2, ls="--")
    ax.text(0, threshold + 2, f"{threshold:.0f} mg/dL spike line", color=CRITICAL, fontsize=8.5)
    ax.set_xlabel("Minutes after the first bite")
    ax.set_ylabel("Projected glucose (mg/dL)")
    ax.set_title(f"{meal_name}: same food, different order", loc="left")
    ax.legend(fontsize=8.5)
    fig.text(0.0, -0.06, "Order effect from published research (Shukla et al., Diabetes Care 2015), "
             "not learned from this dataset.", fontsize=8, color=MUTED)
    _save(fig, "meal_order.png")
