# New slides + video opening: the per-meal learning twin

Material for the Phase 1 video and deck, covering the addition described in README section 8.

**Every number on this page is real** — taken from `artifacts/meal_learning.csv`,
`artifacts/meal_learning_summary.csv` and the live dashboard after
`python run_pipeline.py` on the real BIG IDEAs data. Nothing here is illustrative or rounded up.

> ⚠️ **One thing to fix before you record.** The brief asked for *"Anita, 48, prediabetic"*.
> The patient these numbers come from is **P-005**, whose synthetic record says **51**, not 48.
> Say **51**, or pick a different patient and re-read the numbers off the screen — but do not say 48
> over P-005's charts. Also remember **the age is synthetic**; only sex and HbA1c are real for
> every participant (README section 10). The wording below already handles this correctly.

---

## Slide A — "A twin that learns from every meal"

**Title:** A twin that learns how *your* body answers *your* food

**Body:**
- The twin starts as a population average: 1.0 mg/dL per gram of carbohydrate, peaking at 45 minutes.
- Before each meal it predicts the peak — using **only that patient's earlier meals**.
- After the meal, the CGM records what really happened, and the twin re-fits.
- Over 431 real meals from 15 participants, this cut the error in the predicted peak from
  **32.5 → 25.2 mg/dL**, a **22% reduction** (bootstrap 95% CI [4.1, 10.6] mg/dL).
- It helped **13 of the 15** patients.

**Speaker note:** "The generic model says every gram of rice does the same thing to everybody. After
about ten of her own meals, this twin knows it doesn't."

---

## Slide B — "What's in the meal, and in what order"

**Title:** Not just how much — what, and in what order

**Body — two columns, and the honesty line matters:**

| Learned from the patient's data | Taken from published research |
|---|---|
| Carb sensitivity `k`, time-to-peak `tp` | Eating order: carbs first / mixed / vegetables & protein first |
| Activity benefit `a`, recovery `tr` | Shukla et al., *Diabetes Care* 2015;38(7):e98–e99, n=11 |
| Fibre blunting `fb`, fat/protein delay `fd` | We apply 25%, below their reported 29–37% |

- The food logs record **what** and **when**, never **in what order** — so order cannot be learned here.
- Food composition was tested honestly: both parameters start at **zero**. Result: it changed the error
  by **0.12 mg/dL**, CI **[−0.01, +0.25]** — **no measurable gain**, and we say so.

**Speaker note:** "We kept the composition terms because they aren't worse, not because they helped.
With only 15 people, and fibre missing from several food logs, there wasn't enough signal to find."

---

## Slide C — "No leakage, and we test it"

**Title:** The twin never sees the future — and that is tested, not promised

**Body:**
- Parameters for meal *i* come from meals 1…*i*−1 only.
- Every residual used in that fit ends **before** meal *i* is served, so a long response window from the
  previous meal can never reach past the moment the next meal starts.
- `scripts/check_no_leakage.py` deletes future meals and future glucose, then checks that neither the
  forecast nor the fitted parameters move.
- On the real data: **192 forecast probes, 53 re-fit probes, all unchanged.**

**Speaker note:** "If you only remember one slide about method, make it this one. A glucose model that
peeks at the next reading can score beautifully and be worthless."

---

## Video opening (~1 minute)

**Setup before you hit record.** Dashboard → **Patient twin** → patient **P-005** →
replay clock **day 3, 16:00** → scroll to **🍽 Plan a meal** → meal **Rice + dal + sabzi** →
walk slider at **0**.

These exact strings were read back off the rendered dashboard, not calculated separately:

- top of page: **Glucose 110 mg/dL**, **Spike risk 24%**, **🟢 Stable**, **last meal 59 g carbs, 300 min ago**
- Plan a meal: **Carbs first 146 · Mixed 142 · Vegetables & protein first 135**
- green box: *"Eating the vegetables and protein first lowers the projected peak by **10** mg/dL."*

The reason this moment is worth using: the alarm is **calm — 24%, stable** — and the twin still shows
the meal crossing 140. That is the whole argument for having a twin as well as a classifier.

---

> **[Screen: ward overview]**
>
> "This is a doctor's ward list. Sixteen people wearing a glucose sensor and a smartwatch.
> GlucoTwin's job is to say which of them is about to spike — two hours before it happens."
>
> **[Click P-005, Patient twin]**
>
> "This is patient five. Her HbA1c is 5.7% — that's real, from the dataset, and it puts her in the
> prediabetes range. The rest of her record, her age of 51 and her labs, is synthetic: the challenge
> rules don't allow real patient records."
>
> **[Point at the status line: 24%, Stable]**
>
> "Right now the alarm is calm. Twenty-four percent, green, stable. She last ate five hours ago and she's
> sitting at 110. Nothing here needs a doctor."
>
> **[Scroll to Plan a meal. Choose Rice + dal + sabzi]**
>
> "But she's about to eat — rice, dal and sabzi. And her twin has learned from twenty-one of her own
> meals: 0.44 mg/dL per gram of carbohydrate, peaking at 65 minutes, slower than the population's 45.
>
> Watch what the order does.
>
> **Rice first — her projected peak is 146. That crosses the line.**
>
> **Dal and sabzi first — 135. She stays under it.**
>
> Same food. Same patient. Same afternoon. Ten milligrams per decilitre, and the difference between
> crossing 140 and not — on a patient the alarm had already called stable."
>
> **[Point at the orange warning box]**
>
> "And I want to be straight about this box. That order effect is *not* learned from this data — these
> food logs never recorded what order anyone ate in. It comes from a published crossover trial of eleven
> patients, and I've applied it at 25% when they reported 29 to 37. If order were logged, the twin would
> personalise this too."
>
> **[Scroll to "After the meal"; drag the slider to meal 7]**
>
> "And this is the part I find most convincing. Here is a real meal she ate on the 29th of February.
> Before she ate it, the generic population twin predicted a peak of **182**. Her own twin, trained on
> the six meals it had seen so far, predicted **148**.
>
> The sensor recorded **147**.
>
> Then it learned from that one too."

> **[Open the "🧬 Cycle phase — capability demo" expander. Set it to "Luteal — demo".
> The three peaks move to 153 / 148 / 141]**
>
> "One more, and I want to be careful with this one. Insulin sensitivity drops in the luteal phase, so
> the same meal peaks higher — and that matters for a tool aimed at prediabetes in India, where most of
> the people being screened are women.
>
> **But read the warning first.** This study enrolled *only post-menopausal women*. Every woman in this
> dataset is post-menopausal. None of them has a cycle. So this control is not telling you anything
> about patient five — it's a capability demonstration, showing what the twin would do for a
> premenopausal patient.
>
> And with that caveat, look at the vegetables-first number. Follicular: **135**, she stays under the
> line. Luteal: **141**. *The same advice stops working.*
>
> To actually learn this rather than assume it you need mcPHASES — forty-two women, three months each,
> daily hormone assays. That's in my future work, and it's the honest answer to 'why didn't you just
> train it?'"
>
> **[Optional closing beat, 10 seconds — only if you have room]**
>
> "One honest note. Her fibre and fat terms are both sitting at zero. Across all 431 meals, adding food
> composition changed the error by a tenth of a milligram — nothing. I kept it because it isn't worse,
> and I'm showing you a negative result rather than hiding it."

---

### The exact numbers used above

| Where | Number | Source |
|---|---|---|
| HbA1c 5.7%, female | real | `Demographics.csv` |
| Age 51, labs | **synthetic** | `src/ehr.py` |
| k = 0.44 mg/dL per g, time-to-peak 65 min, 21 meals | learned | `artifacts/meal_twin_params.json` |
| Day 3 16:00: glucose **110**, risk **24% stable**, last meal **300 min ago** | model | dashboard |
| Rice+dal+sabzi: **146 / 142 / 135**, "lowers by **10** mg/dL" | twin + research constant | dashboard |
| Meal 7 (29 Feb): generic **182**, her twin **148**, actual **147** | learned, no leakage | `artifacts/meal_learning.csv` |
| Her per-meal error: 20.8 → 11.7 mg/dL | learned | `artifacts/meal_learning.csv` |
| Her fibre blunting `fb` = **0.00**, fat/protein delay `fd` = **0.00** | learned (found nothing) | dashboard caption |

Cohort-level figures for the slides: **431 meals, 15 patients, 32.5 → 25.2 → 25.1 mg/dL**;
per-meal learning **+7.3 mg/dL, 95% CI [4.1, 10.6]**, helped **13/15**; composition **+0.12 mg/dL,
95% CI [−0.01, +0.25]** (not significant); leakage probes **192 + 53, all unchanged**.

**Don't quote the big "Digital twin" metric cards higher up the page** — those show the *daily* twin
(re-fitted per day), which has different values from the *meal-learned* twin used by Plan a meal. Both
are correct and they are different fits. Quote the Plan a meal caption, as the script above does.

If you re-run the pipeline these could shift slightly. Read them off the screen before recording.

---

## Likely judge questions

**Q1. "Your twin improves meal prediction by 22%. How do I know it isn't just fitting noise, or peeking
at the answer?"**

Two things. The improvement is 7.3 mg/dL with a bootstrap 95% confidence interval of [4.1, 10.6], so it
does not cross zero, and it holds in 13 of 15 patients independently. And every prediction uses only
meals that came earlier — `scripts/check_no_leakage.py` deletes future meals and future glucose and
confirms neither the forecast nor the parameters change. On the real data that's 192 forecast probes and
53 re-fit probes, all unchanged. I'd rather show you the test than ask you to trust the number.

**Q2. "You added fibre, fat and protein to the model. Did it actually help?"**

No, not measurably — 0.12 mg/dL, with a confidence interval of [−0.01, +0.25] that includes zero. I kept
it because it isn't worse, and I'm reporting it as a negative result. There are two honest reasons it
probably failed here: fibre is barely logged by several participants — 3% of meals for one, 11% for
another, and participant 003's food log has no fibre or fat column at all — and fifteen people is too few
to fit two extra parameters. With properly logged meals I'd expect it to matter; I can't claim that from
this data.

**Q3. "Isn't the eating-order feature just a made-up number?"**

It's a published number, applied conservatively, and labelled on screen as not learned from this dataset.
Shukla and colleagues in *Diabetes Care* 2015 reported 29 to 37 percent lower post-meal glucose when
vegetables and protein came before carbohydrate; I apply 25 percent, because their study was eleven
people with type 2 diabetes eating a Western meal, and this cohort is neither. It's one named constant in
`config.py`, so you can change it or switch it off. The point of the feature is the *mechanism* — the
twin is personal, the order multiplier is not, and the UI never blurs the two.

**Q4. "Does any of this change your headline AUROC of 0.733?"**

No. The per-meal twin is a separate addition — it doesn't feed the LightGBM model, doesn't touch its
features, and doesn't change the leave-one-subject-out evaluation. `src/twin.py` was not modified at all;
the new work lives in `src/meal_twin.py`. The section 7 table measures the alarm; section 8 measures the
twin's own accuracy. They're reported separately on purpose.

**Q5. "You've added a menstrual-cycle feature. Did you validate it on this data? And why not build the
whole thing for women?"**

No — and it's worse than that, which is why the control says so on screen. **This study enrolled only
post-menopausal females.** That's the stated inclusion criterion on PhysioNet. So every woman in the
dataset is post-menopausal and none of them has a cycle at all; the feature could not apply to a single
patient here even in principle. On top of that there's no cycle or hormone field in any of the 65 files,
and the recordings are seven to ten days against a twenty-eight-day cycle.

So I kept it as an explicitly labelled capability demonstration rather than deleting it, because the
clinical point is real and the gap is real — most glucose models ignore it. The dataset that would let
me actually learn it is mcPHASES on PhysioNet: forty-two women, three months each, daily LH, E3G and
PdG assays with labelled phases. Three months is three cycles per woman, enough to fit a per-patient
luteal parameter the way I fit carb sensitivity today. That's in my future work.

On building it women-only — I did run it, and it's in the README. Nine women, and the full-fusion AUROC
falls from 0.728 to 0.654, with the EHR actually making things *worse* rather than better. That's a
cohort-size effect, not a finding about women: nine people, one of whom has no usable food log. The
honest version is what you see — keep all sixteen for the headline numbers, and add the cycle phase as a
clearly-labelled clinical module.

**Q6 (if they press on clinical value). "What would a doctor actually do with this?"**

Two things the alarm alone can't do. The alarm says *this patient will probably spike*; the twin says
*why, and what would change it* — this meal, at this size, with or without a walk, in this order. And
because it re-fits after every meal, it gets more specific to that person over time rather than staying a
population average. It is decision support, not a prescription, and it would need prospective validation
before it went anywhere near a clinic.
