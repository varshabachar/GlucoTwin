# brag-plan — GlucoTwin

**Tone:** `polished`. This is a clinical research project for a health challenge; chaotic or
yc-parody would undercut it. Restraint reads as credibility here.
**Format:** landscape 1920×1080, 30fps. **Duration:** 21.06s.
**Music:** `happy-beats-business-moves-vol-9` (114.84 BPM). Every scene cut lands on a strong
beat from the bundled cue grid: 0.00 / 4.23 / 8.44 / 12.65 / 16.86 / 21.06.

## Angle

Lead with the failure of the thing people already have. A CGM is not useless — it is *late*.
The hook is a glucose curve drawing itself across the 140 line in real time, which is exactly
what a patient sees and exactly when it stops being actionable. Everything after that is the
two hours GlucoTwin buys back.

## Hook (0.00–4.23)

A believable post-meal excursion draws left to right against a dashed 140 line. The marker
turns red as it crosses. "Your glucose just crossed 140. / You found out at the moment it
happened." Then, low and in red: TOO LATE TO DO ANYTHING ABOUT IT.

## Reveal (4.23–8.44)

GlucoTwin. "Two hours earlier." One line on what it actually is — a digital twin per patient,
fusing health record, CGM, wearable and meals.

## Highlight 1 — fusion works (8.44–12.65)

Two counters animate: AUROC 0.612 → **0.728**, false alarms 6.8 → **2.0** per patient-day.
Both land green. Supporting line: 86% of spikes, median 45 minutes ahead, every number from a
model that never saw that patient.

## Highlight 2 — it learns you (12.65–16.86)

One counter: **32.5 → 25.2 mg/dL** error in the predicted post-meal peak. Then the mechanism in
two lines: predict before the meal, compare with the sensor after it, re-fit. Footer: 431 real
meals, 15 patients, 22% better, no leakage and it's tested.

## Punchline (16.86–21.06)

The order moment, which is the most human thing the project does. **146** in red (crosses 140)
against **135** in green (stays below), same plate. "Same food. Same patient. Same afternoon."
Sign-off: GlucoTwin · Team Uno · RV University, Bengaluru.

## Claims check

Every figure is from `artifacts/` on the real BIG IDEAs data, not invented:
0.612 / 0.728 / 6.8 / 2.0 / 86% / 45 min (ablation.csv) · 32.5 / 25.2 / 431 meals / 15 patients
(meal_learning_summary.csv) · 146 / 135 (dashboard, P-005 day 3 16:00, rice + dal + sabzi).
The eating-order effect is a published constant, not a result of this dataset — the clip says
"same food, same patient", which is true, and does not claim the effect was measured here.

## Poster

t=19.6s — 146 vs 135 fully settled. Attached as mp4 cover art rather than overwriting frame 0,
because the clip fades in from black and a replaced frame 0 would read as a one-frame glitch.
