# GlucoTwin — video run sheet (target 18–19 min)

**For Varsha, recording the submission video.**

> ### ⚠️ The limit is **15–20 minutes**, not "at least 20"
> The Unstop form says *"15-20 minute Demo Video"*. That is a **maximum of 20**. An earlier draft of
> this script ran 24 minutes, which would have been over. This run sheet is built to land at
> **18–19 minutes**, with a trim list at the end if you run long.

## Shape

| Part | What | Time |
|---|---|---|
| **A** | Present the deck — slides 1 to 20 | ~13 min |
| **B** | Play the 21-second clip (slide 21) | ~1 min |
| **C** | Live demo in the dashboard (slide 22) | ~4 min |
| **D** | Closing slide (23) | ~20 s |
| | | **≈ 18.5 min** |

---

## Before you record

1. `.\venv\Scripts\streamlit run app\dashboard.py` — leave it running in Chrome, zoom **80%**
2. Second Chrome window: `docs\presentation.html`, press **F** for fullscreen, **N** for notes
3. Third window: `brag-output\brag.mp4` ready to play
4. Mic tested. Notifications off (**Win+N → Focus**).
5. **Record in parts: A, B, C separately.** Join in Clipchamp. A fumble then costs one short retake.

---

# PART A — the deck (~13 min)

**Every slide has its spoken words built in — press `N` to show them.** Don't read from this file
during Part A; read from the slide notes, which sit under each slide as you present.

Pacing, so you land on 13 minutes:

| Slides | Content | Target |
|---|---|---|
| 1–3 | Title, the 136 M problem, the gap | 1:20 |
| 4–5 | Architecture, what's real vs synthetic | 1:30 |
| 6–8 | Clock alignment, features, the twin equation | 1:40 |
| 9–11 | Leave-one-subject-out, results table, ablation figure | 1:40 |
| **12** | **The ward — real screenshot** | 1:00 |
| 13–14 | The per-meal twin, and its results | 1:30 |
| **15** | **Plan a meal — real screenshot, 146 vs 135** | 0:55 |
| 16 | Eating order: a published constant, labelled | 0:40 |
| **17** | **Women's module — real screenshot, the caveat** | 0:40 |
| 18 | The women-only cohort | 0:40 |
| 19–20 | Limitations, what it does show | 1:25 |

**Two things not to fumble:**

- **Slide 10.** Read from the **table**, not the card. The table's highlighted row is *+ EHR (full
  fusion) = 0.728*. If you ever quote 0.733, say it is the original locked run and that current
  library versions reproduce 0.728 — that line is already on the slide.
- **Slide 17.** Say the post-menopausal caveat *before* you say the 135 → 141 numbers. The caveat is
  what makes the number credible, not a disclaimer bolted after it.

---

# PART B — the clip (~1 min)

**On slide 21.**

> "Before I show you the system running, here it is in twenty-one seconds."

**[Alt-tab. Play `brag-output\brag.mp4` full screen, with sound. 21 s.]**

> "Everything in that clip is the real dashboard, and every number in it comes from the artifacts in
> the repository. Let me show you the same thing live, so you can see it isn't a video trick."

---

# PART C — the live demo (~4 min)

**On slide 22, then alt-tab to Chrome.** This is the one part with no slide notes, so read it here.
Core beats first; the ones marked *optional* are what you drop if you're running long.

### C1 · The ward (45 s)

**[Ward overview. Sidebar: day 3, time 20:00.]**

> "This is the dashboard running on my machine. A ward list of sixteen patients, and a replay clock in
> the sidebar — I can walk through the recorded week as if it were happening live. Nothing after this
> moment is used for any prediction.
>
> It's day three, eight in the evening. Two patients in red. Five on watch. Two already above 140.
>
> Patient 009 — eighty percent. And the reason, in plain language: his twin projects him at 142
> within the hour, and he ate forty-seven grams of carbohydrate ten minutes ago."

### C2 · Split the ward (20 s) — *optional*

**[Click the Cohort filter → Women 9.]**

> "I can also split the ward by sex — nine women, seven men. And this note is important: every woman
> in this cohort is post-menopausal, because that was the study's inclusion criterion. I'll come back
> to why that matters."

### C3 · One patient (40 s)

**[Click Patient twin → P-005. Sidebar: day 3, 16:00. Scroll the timeline.]**

> "Opening a patient gives her health record, the alert with its reasons, and a multi-day timeline —
> glucose, risk, heart rate, HRV, activity, and the shaded blocks are detected sleep.
>
> And here are her twin's four personal numbers, re-fitted every day on her own past data."

### C4 · Plan a meal — the moment (70 s)

**[Scroll to 🍽 Plan a meal. Meal: Rice + dal + sabzi.]**

> "Now the part I most want you to see. Look at her status first: twenty-four percent, green, stable.
> The alarm says nothing here needs a doctor. She last ate five hours ago.
>
> But she's about to eat — rice, dal and sabzi. Her twin has learned this from twenty-one of her own
> meals.
>
> **Rice first: a projected peak of 146. That crosses 140.**
>
> **Dal and sabzi first: 135. She stays under it.**
>
> Same plate, same patient, same afternoon — on someone the alarm had already called stable."

**[Point at the orange box.]**

> "And that box is there deliberately. The order effect is not learned from this data — these food
> logs never recorded what order anyone ate in. It's a published constant from a trial of eleven
> patients, applied at twenty-five percent when they reported twenty-nine to thirty-seven."

**[Point at Largest safe portion.]**

> "It also doesn't just say 'risky' — it says how much is fine. About a hundred grams of net carbs,
> eaten vegetables-first."

### C5 · Any food, not just my menu (35 s) — *optional*

**[Click "Build your own", change Carbs to 120. Then "Describe it in words", type *2 rotis, dal and a bowl of curd*, click Estimate.]**

> "It isn't limited to the ten meals in my file. You can enter any food's numbers — or describe it in
> plain English and have it estimated, then edit it. That matters for Indian food, where nobody knows
> the gram values of two rotis and a katori of dal."

### C6 · The twin learning (40 s)

**[Scroll to "After the meal". Drag the slider to meal 7.]**

> "And this is the loop closing. A real meal she ate on the twenty-ninth of February. Before she ate
> it, the generic population twin predicted a peak of 182. Her own twin, trained on the six meals it
> had seen so far, predicted 148.
>
> The sensor recorded 147.
>
> Then it learned from that one too."

### C7 · The women's module (45 s)

**[Expand 🧬 Cycle phase. Read the warning. Set Luteal — demo.]**

> "Last thing. Insulin sensitivity falls in the luteal phase, so the same meal peaks higher — and for
> a tool aimed at prediabetes in India, where most of the people being screened are women, that
> matters.
>
> But read this warning first. Every woman in this study is post-menopausal. None of them has a cycle.
> So this is a labelled capability demonstration, not a claim about this patient.
>
> With that said — watch the vegetables-first number. It was 135, under the line. In the luteal phase
> it's 141. The same advice stops working."

### C8 · The numbers behind it (25 s)

**[Model performance. Scroll the ablation, then the per-meal table.]**

> "And everything I've shown is backed here — the ablation, leave-one-subject-out, and the per-meal
> results: 32.5 down to 25.2 milligrams per decilitre across 431 real meals."

**[Alt-tab back to slide 23.]**

---

# PART D — close (~20 s)

**Slide 23.**

> "Thank you for watching. The code, the data provenance, every result, and the leakage test are all
> in the repository."

---

## If you're running long, cut in this order

1. **C5** (Build your own / Describe it in words) — 35 s
2. **C2** (cohort filter) — 20 s, it's also covered on slide 17
3. **Slide 6** (clock alignment) — trim to two sentences, saves ~25 s
4. **Slide 11** (ablation figure) — the table on slide 10 already made the point, saves ~35 s

Cutting all four brings you to about **16:30**, still comfortably inside the window.

## ⭐ Re-check these if you re-run the pipeline

- Full fusion **0.728** (slide 10, "+ EHR (full fusion)" row)
- Per-meal **32.5 → 25.2** mg/dL (slide 14)
- P-005 Plan-a-meal **146 / 142 / 135**, luteal **153 / 148 / 141** (slides 15, 17 and the live demo)
