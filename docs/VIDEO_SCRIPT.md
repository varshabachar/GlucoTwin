# GlucoTwin — full video script (~24 minutes)

**For Varsha to read aloud while screen-recording.** Requirement is ≥ 20 minutes; read at a normal
pace with the pauses marked and this lands around 24.

Every number in **bold** was read off the running dashboard, not calculated separately. If you re-run
the pipeline before recording, check the three starred ⭐ numbers again.

---

## Before you hit record — 5-minute checklist

1. Start the dashboard: `.\venv\Scripts\streamlit run app\dashboard.py`
2. Open it in Chrome, press **Ctrl+Shift+J** once to check there are no red errors, then close that.
3. Set browser zoom to **80%** (Ctrl+Minus twice) so whole tables fit on screen.
4. Open a second tab with the GitHub repo.
5. Have `docs/architecture.png` open in a third tab.
6. Recording: **Win+Alt+R** starts/stops, **Win+Alt+M** mutes the mic.
7. Record in **sections** — don't try for one perfect take. Join them in Clipchamp afterwards.

**One thing to know before you speak.** The big card at the top of *Model performance* says
**AUROC 0.72**. That is the "+ Digital twin" row — the full system including the twin's forecasts.
The headline **0.733** in the README is the "+ EHR (full fusion)" row. Both are in the table on the
same screen. The script below points at the **table**, not the card, and explains the difference. Do
it that way and it reads as rigour. Don't say "0.73" while the card shows 0.72 without explaining.

---

## PART 1 — The problem (0:00 – 2:30)

**[SCREEN: title slide, or the GitHub README top]**

> Hello. I'm Varsha B S Acharya from RV University, Bengaluru, and this is GlucoTwin — my submission
> for the Happiest Health Digital Twin Challenge, Phase 1. Team name is Uno, and I'm a team of one.
>
> Let me start with the number that made me build this.
>
> India has an estimated **101 million people living with diabetes**, and **136 million with
> prediabetes**. That's from the ICMR-INDIAB national survey published in Lancet Diabetes and
> Endocrinology in 2023. A hundred and thirty-six million people who are not diabetic yet — and for
> most of them, whether they become diabetic is still modifiable.
>
> In prediabetes, the early warning sign is the post-meal glucose spike. Glucose climbs after eating,
> stays high for an hour or two, and comes down. Repeated often enough, that pattern is what
> progresses toward type 2 diabetes.
>
> Here's the gap. A doctor sees that patient maybe twice a year, and gets a single HbA1c number — an
> average over three months. And a continuous glucose monitor, a CGM, will tell you about a spike —
> but only once it's already happening. By then the patient has already eaten.
>
> **[pause]**
>
> So the question I set out to answer is: *can you warn someone two hours before the spike, early
> enough that they can actually do something about it?*
>
> And the second question, which turned out to matter more: *can you tell them what to do about it,
> specifically, for their body?*

---

## PART 2 — What a digital twin means here (2:30 – 5:00)

**[SCREEN: docs/architecture.png]**

> This is the architecture. I'll walk left to right.
>
> A digital twin, in this project, means two things fused together.
>
> On the left, the **static stream** — the health record. Demographics, labs, diagnoses, medications,
> family history, genetic markers. That's who the patient *is*.
>
> Above it, the **dynamic streams** — a Dexcom G6 CGM reading glucose every five minutes, an Empatica
> E4 wristband giving heart rate and beat-to-beat intervals for heart rate variability, and a meal log
> from a phone app. That's what the patient is *doing*, right now.
>
> **[point at the middle column]**
>
> Those get fused onto a single five-minute grid. The health record is ingested as a **FHIR R4
> bundle** — that's the HL7 interoperability standard, with LOINC codes for labs, SNOMED for
> diagnoses, RxNorm for medications. I used FHIR specifically because that's the standard India's
> ABDM — the Ayushman Bharat Digital Mission — is built on. If this were ever deployed, it would have
> to speak that language, so it speaks it now.
>
> **[point at the Intelligence column]**
>
> Then two things in parallel. A **per-patient physiological twin** — four numbers describing that
> person's glucose physiology, re-fitted as their own data arrives. And a **LightGBM fusion model**
> that predicts the spike.
>
> **[point at Outputs]**
>
> And on the right, what a doctor actually sees: a risk score, the reasons behind it in plain
> language, and a what-if simulator.
>
> That last one is the point of a twin. A classifier tells you *that* something will happen. A twin
> lets you ask *what if*.

---

## PART 3 — The data, and what's real (5:00 – 7:30)

**[SCREEN: dashboard → Data & compliance]**

> Let's be precise about the data, because this is where projects like this usually overclaim.
>
> The glucose, heart rate, beat-to-beat intervals and food logs are **real, de-identified human data**
> — the BIG IDEAs Lab Glycemic Variability and Wearable Device Data set, published on PhysioNet under
> an open-access licence. **Sixteen participants**, each monitored for about **eight to ten days**.
> Sex and HbA1c are real too, from the demographics file.
>
> **[point at the table]**
>
> Everything in the bottom row — age, the lab panel, diagnoses, medications, family history, the
> genetic markers — is **synthetic**. I generate it. The challenge sandbox rules, and the DPDP Act and
> HIPAA more generally, don't allow real patient records. So no real patient record is used anywhere
> in this project.
>
> But I generate it *consistently* with each person's real HbA1c. If someone's real HbA1c is 6.2, the
> synthetic fasting glucose and the prediabetes diagnosis follow from it, rather than being random.
>
> **[scroll to the FHIR viewer, expand one patient]**
>
> And this is what the twin actually ingests — a real FHIR R4 bundle, with the actual code systems.
> Not a CSV pretending to be a health record.
>
> Two rows further down you'll see I've also flagged the eating-order effect and the Indian meal menu
> as things that come from published research rather than from this dataset. I'll come back to both.
>
> **[pause]**
>
> In total that's **38,115 five-minute rows**, of which **31,945** are labelled. Twenty-two percent of
> them are followed by a spike within two hours. That twenty-two percent is the number to keep in your
> head, because it's what random guessing would score.

---

## PART 4 — Clock alignment: the unglamorous part (7:30 – 10:00)

**[SCREEN: Data & compliance → the device clock alignment table]**

> I want to spend two minutes on something that isn't in most projects, because it nearly broke mine.
>
> Three devices. A CGM, a wristband, and a phone. **Each keeps its own clock.** And in a de-identified
> public dataset, timestamps get shifted for privacy, people log meals late, time zones differ.
>
> If those streams are misaligned and you don't check, the model quietly learns nothing and you never
> find out. So I check explicitly, and I print the report — that's this table.
>
> Three checks. **First**, the wristband against the CGM: heart rate and glucose are both quiet at
> night and busy in the day, so I find the shift that best lines up those two 24-hour patterns, pooled
> across all sixteen people. Here, the clocks agreed — zero hours.
>
> **Second**, the meal log against the CGM: glucose should rise forty to eighty minutes after a carb
> meal. I test meal-time offsets from minus two hours to plus two, and I only keep one if it clearly
> strengthens that rise. **Participant 016's meals were being logged 105 minutes late** — you can see
> the correction in this table.
>
> **Third**, coverage. **Participant 015's entire food log falls outside their CGM recording.** So
> their meal features are marked *unavailable* — meaning "unknown", not "this person never ate".
> That distinction matters: tell a model someone ate zero carbs when you simply don't know, and you've
> taught it something false.
>
> **[pause]**
>
> I tested this step on a deliberately broken copy — wristband five hours off, meals an hour late, one
> log from a different fortnight. It recovered every offset exactly, and AUROC came back to within
> 0.01 of the clean data.

---

## PART 5 — Features, and never seeing the future (10:00 – 13:00)

**[SCREEN: GitHub → src/features.py]**

> About sixty features. I'll group them rather than list them.
>
> From the **CGM**: current glucose, the change over fifteen, thirty and sixty minutes, the trend, the
> acceleration, variability, and each person's own 24-hour baseline.
>
> From the **wearables**: heart rate, heart rate variability as RMSSD, activity over fifteen minutes,
> one hour and three hours, and last night's sleep.
>
> **[point at the comment block in the file]**
>
> Here's a decision I want to defend. Every wearable feature is expressed **relative to that person's
> own normal** — their trailing three days. Not raw values.
>
> The reason: resting heart rate varies far more *between* people than *within* a person. If you feed
> raw heart rate to a model, it mostly learns *who* someone is. That looks excellent in testing and
> transfers to a new patient as nothing. Relative features ask the right question — "is this person
> unusual *for them*, right now?"
>
> From the **meal log**: time since the last meal, and its carbs, sugar, fat, protein and fibre.
>
> From the **health record**: the seventeen static features.
>
> **[pause — then SCREEN: src/load_data.py, the bin convention comment near the top]**
>
> Now the single most important rule in this codebase.
>
> **A feature at time T may only use data from time T or earlier.** Only the label looks forward —
> that's the thing being predicted.
>
> It's enforced right down to the binning. A bin labelled 12:05 holds data from *after* 12:00 up to
> and including 12:05. So a prediction made "at 12:05" has, by construction, never seen 12:06.
>
> Every rolling statistic is past-only. Even the resting heart rate the activity estimate is built
> from is the tenth percentile of the *previous* 72 hours.
>
> This sounds pedantic. It is the difference between a glucose model that works and one that's
> cheating without its author realising.

---

## PART 6 — The twin, in one equation (13:00 – 15:30)

**[SCREEN: README section 5, the equation]**

> Now the twin itself.
>
> For each patient, glucose is modelled as four things added together.
>
> Their **personal baseline**. Plus **k** times the carbohydrate they ate, spread over a hump that
> peaks at **tp** minutes. Minus **a** times their recent movement. Plus whatever is left over,
> unexplained, fading away over a recovery time **tr**.
>
> **Those four numbers are the twin.**
>
> - **k** is carb sensitivity — how much one gram of carbohydrate raises *this person's* glucose.
> - **tp** is how fast they peak.
> - **a** is how much activity helps them.
> - **tr** is how fast they recover.
>
> Everyone starts at the population average: 1.0 milligrams per decilitre per gram, peaking at
> forty-five minutes. Then it's **re-fitted every day, on that patient's own past data only**. Day one
> uses population defaults, because on day one that's genuinely all you know.
>
> It's fitted with a prior pulling it toward the population average — with only eight to ten days per
> person, that stops one strange day from wrecking the parameters.
>
> **[pause]**
>
> One deliberate restriction. Only the twin's **time-varying forecasts** go into the machine learning
> model — projected peak, projected rise, projected glucose in an hour. The four personal parameters
> themselves do **not**. They're constant within a person, so to the model they'd work as an ID card —
> a way to recognise the patient rather than understand them.

---

## PART 7 — Results (15:30 – 19:00)

**[SCREEN: dashboard → Model performance]**

> Results. And how they're measured matters as much as what they are.
>
> **Leave-one-subject-out.** Train on fifteen people, test on the sixteenth, repeat sixteen times. So
> **every number you're about to see comes from a model that has never seen the patient it's
> scoring.** That's the honest way to evaluate a medical model. If you split rows randomly instead,
> the same person appears in both training and testing and your scores inflate.
>
> The alert thresholds are also chosen by an inner cross-validation on the **training** patients only
> — so nothing about the test patient leaks into either the model or the alarm.
>
> **[point at the ablation table — the table, not the card above it]**
>
> Read this table top to bottom; each row adds one more data stream.
>
> **Row one** is the baseline — what a simple CGM alarm does. Extrapolate the current trend forward
> two hours. AUROC **0.612**, and look at the last column: **6.8 false alarms per patient per day.**
> That's alarm fatigue. Nobody keeps listening to that.
>
> **CGM alone, with machine learning: 0.681.**
>
> **Adding wearables: 0.667** — it goes *down*, and I'll come back to that.
>
> **Adding the meal log: 0.708.** Biggest single jump in the table.
>
> **Adding the health record — full fusion: 0.728.** ⭐ And **1.98 false alarms a day**.
>
> So: from **0.612** to **0.728**, and from **6.8 false alarms a day down to about 2**, warning of
> **86%** of spikes a median of **45 minutes** ahead.
>
> **[point at the card above the table]**
>
> One note on that card at the top, which reads **0.72**. It shows the last row — the full system
> *including* the twin's forecasts, at **0.725**. Adding those forecasts didn't improve the score:
> 0.725 against 0.728. With sixteen people that difference is well inside the noise.
>
> I'm reporting that honestly rather than hiding it, because the twin's value isn't the score. It's
> the simulation — which is the next part of this video.
>
> **[pause]**
>
> Two more honest readings of this table.
>
> **Wearables going down.** In this build the accelerometer files weren't downloaded — they're about
> thirteen gigabytes — so activity is estimated from heart rate instead. Heart rate also rises with
> stress and after eating, so it's a noisy proxy. Wearables still contribute inside the full fusion;
> they just don't beat CGM on their own here.
>
> **And the AUPRC column.** Random guessing scores 0.22 on this data, because 22% of rows are followed
> by a spike. Full fusion gets **0.478** — a bit over twice baseline.

---

## PART 8 — Live: the ward (19:00 – 21:30)

**[SCREEN: dashboard → Ward overview. Sidebar: day 3, time 20:00]**

> Now what a doctor actually sees. This is a ward list for sixteen patients, and there's a replay
> clock in the sidebar walking through the recorded week as if it were live.
>
> Right now it's **day three, eight in the evening**. Two patients in red — alert. Five on watch. Two
> already above 140.
>
> **[point at the top row]**
>
> **Patient 009. Risk 80%.** And the column that makes this usable: **the reason**. "Twin-projected
> glucose in one hour — 142." He ate forty-seven grams of carbohydrate ten minutes ago, and his twin
> already expects him over the line.
>
> **Patient 016**, 53% — "net carbs eaten in the last three hours: 114 grams."
>
> These reasons aren't hand-written rules. They're **SHAP values** — from the model that never saw
> that patient — translated into plain language.
>
> **[pause]**
>
> The two patients marked "in spike" are already above 140. The system doesn't alert on them, because
> the job is predicting *onset*, not announcing something the CGM is already showing.
>
> **[click the cohort filter → Women 9]**
>
> One more thing on this page. You can split the ward by sex. **Nine women, seven men.** I'll come
> back to what that note says in a moment.
>
> **[click a patient → Patient twin]**
>
> And opening a patient gives their record, the alert with its reasons, a multi-day timeline of
> glucose, risk, heart rate, HRV, activity and sleep — and their twin's four personal numbers.

---

## PART 9 — The twin that learns from every meal (21:30 – 25:00)

**[SCREEN: Patient twin → P-005 → day 3 → 16:00 → scroll to 🍽 Plan a meal]**

> This is the part I'm most pleased with, and it's new.
>
> The twin I described re-fits once a day. But a patient doesn't live in days — they live in meals. So
> I built a version that learns **meal by meal**.
>
> Before each meal, it predicts the peak, using parameters fitted on that person's **earlier meals
> only**. After the meal, the CGM records what really happened. Then it re-fits.
>
> **[SCREEN: Model performance → scroll to the per-meal table]**
>
> Over **431 real meals from 15 participants**: a generic population twin gets the peak wrong by
> **32.5 milligrams per decilitre** on average. The twin that learns each meal: **25.2**. ⭐
>
> That's a **22% reduction**, and a bootstrap 95% confidence interval of 4.1 to 10.6 — so it isn't
> chance. It helped **13 of the 15** patients. And look at the next column — over the *later half* of
> each person's meals, the generic twin drifts to 35.2 while the learning twin holds at 24.8. The gap
> widens as it sees more of you.
>
> **[point at the third row]**
>
> Now a negative result, which I'm reporting because it's true. I also let the twin use what *else*
> was in the meal — fibre, fat, protein. The improvement was **0.12 milligrams per decilitre**, with a
> confidence interval that includes zero. **It did not measurably help.**
>
> I kept it because it isn't worse. Two honest reasons it probably failed: fibre is barely logged by
> several participants — three percent of meals for one person — and one participant's food log has no
> fibre or fat column at all. And fifteen people is too few to fit two extra parameters.
>
> **[pause]**
>
> And the claim that all of this rests on is that the twin never sees the future. So I don't just
> assert it — there's a script that **deletes** future meals and future glucose and checks that
> neither the forecast nor the fitted parameters move. On the real data that's **192 forecast probes
> and 53 re-fit probes, all unchanged.**

---

## PART 10 — Plan a meal (25:00 – 28:00)

**[SCREEN: back to Patient twin → P-005 → day 3, 16:00 → Plan a meal → "Rice + dal + sabzi"]**

> Here's where the twin earns its place.
>
> Patient five. Her HbA1c is 5.7, which is real, and in the prediabetes range. Her age and labs are
> synthetic. Her twin has learned from **21 of her own meals**: carb sensitivity **0.44**, peaking at
> **65 minutes** — slower than the population's forty-five.
>
> **[point at the status line]**
>
> Note what the alarm says right now: **24%, green, stable.** Nothing here needs a doctor. She last
> ate five hours ago.
>
> **[select Rice + dal + sabzi]**
>
> But she's about to eat. Rice, dal and sabzi. Watch the three lines.
>
> **Rice first — projected peak 146. That crosses 140.**
>
> **Dal and sabzi first — 135. She stays under.**
>
> Same food. Same patient. Same afternoon. About ten milligrams per decilitre, and the difference
> between crossing the line and not — **on a patient the alarm had already called stable.** That is
> something a classifier alone cannot give you.
>
> **[point at the orange warning box]**
>
> And I want to be straight about that box. The order effect is **not learned from this data** — these
> food logs record what and when, never in what order. It comes from a published crossover trial by
> Shukla and colleagues in Diabetes Care, 2015 — eleven patients with type 2 diabetes. They reported
> 29 to 37 percent lower glucose. **I apply 25 percent**, deliberately below their figure, because
> eleven people eating a Western meal is not this cohort. If order were logged, the twin would
> personalise this the way it personalises carb sensitivity.
>
> **[point at "Largest safe portion"]**
>
> And rather than just "this meal is risky", it says how much is fine: about **115% of this portion**,
> roughly 100 grams of net carbs, eaten vegetables-first.
>
> **[switch to "Build your own", then "Describe it in words" if you have the key set]**
>
> It isn't limited to the ten meals in my menu. You can enter any food's numbers directly — or
> describe it in plain English and have it estimated, then edit it. That matters for Indian food,
> where nobody knows the gram values of two rotis and a katori of dal.
>
> **[SCROLL DOWN to "After the meal"]**
>
> And finally — the loop closing. This is a real meal she ate on the 29th of February. Before she ate
> it, the generic population twin predicted a peak of **182**. Her own twin, trained on the six meals
> it had seen so far, predicted **148**.
>
> The sensor recorded **147**.
>
> Then it learned from that one too.

---

## PART 11 — Women's health, and being honest about a cohort (28:00 – 30:00)

**[SCREEN: expand the 🧬 Cycle phase panel]**

> One more module, and it comes with an important caveat I want to say out loud.
>
> Insulin sensitivity falls in the luteal phase — the second half of the menstrual cycle — so the same
> meal peaks higher. For a tool aimed at prediabetes in India, where a great many of the people being
> screened are women, that's a real clinical factor most glucose models ignore.
>
> **[point at the warning]**
>
> **But read this.** The BIG IDEAs study enrolled **only post-menopausal females.** Every woman in this
> dataset is post-menopausal. None of them has a cycle. So this control is **not** telling you anything
> about this patient — it's a capability demonstration, showing what the twin would do for a
> premenopausal patient.
>
> **[set it to Luteal — demo]**
>
> With that caveat: watch the vegetables-first number. Follicular, **135** — she stays under. Luteal,
> **141**. The same advice stops working.
>
> **[SCREEN: Model performance → women-only table]**
>
> I also re-ran the whole pipeline on the nine women alone. Full fusion falls from **0.728 to 0.654**,
> and the health record actually makes it *worse* rather than better. That is a **cohort-size effect,
> not a finding about women** — nine people, one of whom has no usable food log. I'm showing it because
> running it was the honest way to answer "should this be a women's tool?", and the answer is that this
> dataset is too small to support one.
>
> The dataset that *would* let the twin learn cycle effects properly is mcPHASES on PhysioNet —
> forty-two women, three months each, daily hormone assays. That's in my future work.

---

## PART 12 — Limitations and close (30:00 – 32:00)

**[SCREEN: README section 12, Honest limitations]**

> Let me close by arguing against my own project, because I'd rather you hear these from me.
>
> **Sixteen people.** This shows feasibility, not clinical validity. Nothing here is evidence about a
> population.
>
> **The health record is synthetic** apart from sex and HbA1c. So the EHR's contribution demonstrates
> that the fusion *pipeline* works. It is not evidence about real records.
>
> **Activity and sleep are estimated** from heart rate, not measured with an accelerometer or
> polysomnography.
>
> **Unlogged meals can't be anticipated.** If someone doesn't log it, the model can only react.
>
> **The eating order and the cycle effect are published constants, not learned here** — and the
> dashboard says so on screen, every time.
>
> **And food composition didn't help**, which I've reported as the negative result it is.
>
> **[pause]**
>
> What I think this does show: that fusing a health record with wearable streams beats glucose alone —
> **0.612 to 0.728** — and cuts a basic CGM alarm's **6.8 false alarms a day down to about 2**. That a
> physiological twin can learn a person meal by meal and get **22% better** at predicting their peaks.
> And that the result is a tool which doesn't just say *"she will spike"*, but *"if she eats the dal
> before the rice, she won't."*
>
> Deploying it would need prospective validation, regulatory review, and ABDM-compliant consent flows.
> It is decision support, not a prescription, and it is not a medical device.
>
> Thank you for watching.

---

## If you need to reach 20 minutes and you're short

Expand these, in this order — they're the ones with the most real material behind them:
1. **Part 7 (results)** — read every row of the ablation table aloud, with its AUPRC.
2. **Part 4 (clock alignment)** — walk the table patient by patient.
3. **Part 9** — open `artifacts/meal_learning.csv` and talk through a few real meals.
4. **Part 5** — scroll `src/features.py` and name features group by group.

## ⭐ Re-check these three if you re-run the pipeline

- Full fusion AUROC **0.728** (Model performance table, "+ EHR (full fusion)")
- Per-meal learning **32.5 → 25.2** mg/dL (per-meal table)
- P-005 Plan-a-meal peaks **146 / 142 / 135** (day 3, 16:00, Rice + dal + sabzi)
