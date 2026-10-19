"""Draw the system architecture diagram -> docs/architecture.pdf and docs/architecture.png

    python scripts/make_architecture.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
FILL = {"static": ("#ece9fb", "#6250d6"), "dynamic": ("#e3eefb", "#2a78d6"),
        "process": ("#f1f0ec", "#898781"), "brain": ("#fde9df", "#eb6834"), "out": ("#ddf3ea", "#1baf7a")}

fig = plt.figure(figsize=(16, 9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor(SURFACE)


def box(x, y, w, h, title, body, kind):
    fill, edge = FILL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fill, ec=edge, lw=1.6))
    ax.text(x + 0.15, y + h - 0.22, title, ha="left", va="top", fontsize=11.5, weight="bold", color=INK)
    ax.text(x + 0.15, y + h - 0.58, body, ha="left", va="top", fontsize=9.3, color=INK2, linespacing=1.45)
    return (x, y, w, h)


def arrow(a, b, start="r", end="l", color=MUTED):
    def pt(bx, side):
        x, y, w, h = bx
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "b": (x + w / 2, y), "t": (x + w / 2, y + h)}[side]
    ax.add_patch(FancyArrowPatch(pt(a, start), pt(b, end), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.5, color=color, connectionstyle="arc3,rad=0.0"))


ax.text(0.4, 8.6, "GlucoTwin - system architecture", fontsize=18, weight="bold", color=INK, va="top")
ax.text(0.4, 8.12, "Static EHR + dynamic wearable streams -> per-patient digital twin -> 2-hour glucose-spike early warning "
        "for the doctor", fontsize=11, color=INK2, va="top")
for x, label in [(0.4, "1  DATA SOURCES"), (4.3, "2  INGESTION & FUSION"), (8.2, "3  INTELLIGENCE"), (12.1, "4  OUTPUTS")]:
    ax.text(x, 7.55, label, fontsize=10.5, weight="bold", color=MUTED, va="top")

cgm = box(0.4, 6.0, 3.4, 1.1, "CGM (dynamic)", "Dexcom G6 · glucose every 5 min", "dynamic")
wear = box(0.4, 4.2, 3.4, 1.55, "Smartwatch (dynamic)",
           "Empatica E4: heart rate, beat-to-beat\nintervals (HRV); activity & sleep\nestimated from them", "dynamic")
meal = box(0.4, 2.85, 3.4, 1.1, "Meal log (dynamic)", "time · carbs · sugar · fat · protein", "dynamic")
ehr = box(0.4, 0.75, 3.4, 1.85, "Health record (static)",
          "Synthetic FHIR R4 bundle\ndemographics · labs · diagnoses\nmedications · family history\n5 genetic risk variants", "static")

grid = box(4.3, 3.6, 3.4, 3.5, "Streaming fusion grid",
           "CSVs -> 5-minute bins\nclock alignment: wristband & meal log\nchecked against the CGM\nonly data up to 'now' is used\n\nfeatures: glucose dynamics, HRV,\nactivity, sleep, meal load\n\nlabel: glucose crosses 140 mg/dL\nwithin the next 2 hours", "process")
fhir = box(4.3, 1.0, 3.4, 1.35, "FHIR ingestion", "LOINC / SNOMED / RxNorm\n-> static feature vector (17)", "process")

twin = box(8.2, 5.15, 3.4, 1.95, "Digital twin (per patient)",
           "meal response + activity + recovery\nparameters k, tp, a, tr\nre-calibrated daily on own data\n"
           "learns MEAL BY MEAL: predict the peak,\ncompare with the CGM, re-fit (fb, fd)\n-> 2-h glucose forecast", "brain")
model = box(8.2, 2.95, 3.4, 1.95, "Fusion model",
            "LightGBM · ~60 features\nCGM + wearable + meals + EHR + twin\nleave-one-subject-out validated\nthreshold set by inner CV", "brain")
shap = box(8.2, 1.0, 3.4, 1.55, "Explainability", "TreeSHAP per alert\nplain-language reasons", "brain")

whatif = box(12.1, 5.25, 3.3, 1.85, "What-if / plan a meal",
             "meal size x post-meal walk\nIndian meal menu (approximate)\neating order + cycle phase:\npublished research, NOT learned\nhere (cohort is post-menopausal)\n-> projected peak", "out")
risk = box(12.1, 3.85, 3.3, 1.3, "Early warning", "2-hour spike risk + alert\npatient-specific threshold", "out")
why = box(12.1, 2.35, 3.3, 1.3, "Why", "top reasons, e.g. 'large lunch,\nshort sleep, low activity'", "out")
dash = box(12.1, 0.55, 3.3, 1.5, "Doctor dashboard",
           "ward view · patient twin timeline\nperformance · FHIR record viewer", "out")

arrow(cgm, grid)
arrow(wear, grid)
arrow(meal, grid)
arrow(ehr, fhir)
arrow(grid, twin)
arrow(grid, model)
arrow(fhir, model)
arrow(twin, model, "b", "t")
arrow(model, shap, "b", "t")
arrow(twin, whatif)
arrow(model, risk)
arrow(shap, why)
# bracket: every output feeds the dashboard
bx = 15.65
for b in (whatif, risk, why):
    x, y, w, h = b
    ax.plot([x + w, bx], [y + h / 2, y + h / 2], color=MUTED, lw=1.5)
ax.plot([bx, bx], [why[1] + why[3] / 2, whatif[1] + whatif[3] / 2], color=MUTED, lw=1.5)
ax.plot([bx, bx], [why[1] + why[3] / 2, dash[1] + dash[3] / 2], color=MUTED, lw=1.5)
ax.add_patch(FancyArrowPatch((bx, dash[1] + dash[3] / 2), (dash[0] + dash[2], dash[1] + dash[3] / 2),
                             arrowstyle="-|>", mutation_scale=14, lw=1.5, color=MUTED))

ax.text(0.4, 0.3, "Data: BIG IDEAs Lab Glycemic Variability & Wearable Device Data (PhysioNet, open access, de-identified) "
        "+ synthetic EHR. No real patient records (DPDP Act / HIPAA sandbox).", fontsize=9, color=MUTED)

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "architecture.pdf", facecolor=SURFACE)
fig.savefig(OUT / "architecture.png", dpi=150, facecolor=SURFACE)
print("wrote", OUT / "architecture.pdf")
