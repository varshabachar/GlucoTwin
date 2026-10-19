# GlucoTwin — explainer script

**61 scenes · 15.7 minutes · 2113 words**

Timing: `words / 2.4 + 1s`. Replace a scene's estimate by dropping `public/vo/<id>.mp3` in and letting `calculateMetadata` size it to the audio.

## Runtime by section

| Section | Scenes | Minutes |
|---|---|---|
| hook | 2 | 0.5 |
| problem | 6 | 1.6 |
| solution | 5 | 1.2 |
| architecture | 8 | 2.1 |
| features | 24 | 6.1 |
| decisions | 6 | 1.9 |
| results | 5 | 1.2 |
| next | 4 | 1.0 |
| outro | 1 | 0.1 |
| **total** | **61** | **15.7** |


## hook

**`hook-01`** · 16.4s · glimpse: `—` · starts 0:00  
> This is a glucose reading from a real person, recorded every five minutes for ten days. Watch what happens after lunch. It climbs, and it crosses 140 - the line where a post-meal spike starts doing damage.

*Visual:* Sam stands left. A hand-drawn CGM trace draws on left to right, then crosses a dashed red 140 line. ScribbleCircle on the crossing.

**`hook-02`** · 15.6s · glimpse: `—` · starts 0:16  
> Her monitor tells her at the exact moment it happens. By then she has already eaten. So I built something that says it two hours earlier, and then tells her what would have changed it.

*Visual:* Sam shrugs at the late alarm bubble, then points right. Title card draws on: GlucoTwin.


## problem

**`prob-01`** · 14.8s · glimpse: `—` · starts 0:32  
> India has about 101 million people living with diabetes. It has 136 million more with prediabetes. That second number is the interesting one, because for most of those people it is still reversible.

*Visual:* Two hand-drawn bars grow: 101M and 136M. ScribbleCircle around the 136M bar; Sam taps it.

**`prob-02`** · 18.5s · glimpse: `—` · starts 0:46  
> The early warning sign is the post-meal spike. Glucose climbs after eating, stays high for an hour or two, and comes back down. Repeat that often enough and it progresses toward type 2 diabetes. So the spikes are the thing worth catching.

*Visual:* The trace from the hook replays slower with a bracket over the peak; several small traces stack up.

**`prob-03`** · 13.9s · glimpse: `—` · starts 1:05  
> Here is the gap. A doctor sees that patient maybe twice a year, and gets a single HbA1c, one number averaging three months. Everything that happened between those visits is invisible.

*Visual:* Calendar with two circled days; one number floats beside it. The rest of the year greys out.

**`prob-04`** · 15.2s · glimpse: `—` · starts 1:19  
> A continuous glucose monitor is better. It reads every five minutes. But it reports the spike once it is already happening, which is the one moment you can no longer do anything about it.

*Visual:* Clock sweeping. The trace crosses 140 and an alarm bubble appears visibly late.

**`prob-05`** · 17.2s · glimpse: `—` · starts 1:34  
> You could just extrapolate the trend and warn early. I tried that, and it is the baseline in this project. It catches almost every spike, at about seven false alarms per patient per day. Nobody keeps listening to that.

*Visual:* Seven alarm bubbles pop in fast; Sam covers ears. Boil on the bubbles.

**`prob-06`** · 14.3s · glimpse: `—` · starts 1:51  
> So the real question has two halves. Can you warn two hours ahead without crying wolf? And once you have warned someone, can you tell them what to actually do about it?

*Visual:* Sam in thinking pose. The two questions hand-letter on, one above the other.


## solution

**`sol-01`** · 15.2s · glimpse: `—` · starts 2:05  
> GlucoTwin builds a digital twin of each patient, and two streams feed it. The static stream is the health record: demographics, labs, diagnoses, medications, family history, genetic markers. That is who the patient is.

*Visual:* Sam points at two incoming arrows. A hand-drawn paper file draws on with the items listed inside.

**`sol-02`** · 15.2s · glimpse: `—` · starts 2:21  
> The dynamic stream is what they are doing right now: a Dexcom G6 glucose monitor reading every five minutes, a wristband giving heart rate and heart-rate variability, and a meal log from a phone.

*Visual:* Three devices draw on in turn: sensor patch, wristband, phone.

**`sol-03`** · 15.6s · glimpse: `—` · starts 2:36  
> All of it is fused onto a single five-minute grid, and two different things read that grid. The first is a physiological twin, four numbers that describe this person's own glucose behaviour and nobody else's.

*Visual:* Arrows converge into a grid. Four labelled boxes appear: k, tp, a, tr.

**`sol-04`** · 9.8s · glimpse: `—` · starts 2:51  
> The second is a machine learning model that answers one question: will this person cross 140 within the next two hours?

*Visual:* A box labelled LightGBM with an output arrow to a risk dial that sweeps up.

**`sol-05`** · 16.8s · glimpse: `—` · starts 3:01  
> The model tells you that something will happen. The twin lets you ask what if. That difference is the entire point of the project, and it is why a twin is worth building rather than just a classifier.

*Visual:* Sam weighs both hands. HandText: THAT - versus - WHAT IF.


## architecture

**`arch-01`** · 17.2s · glimpse: `—` · starts 3:18  
> Here is the whole system, left to right. On the left, the data sources. The glucose, heart rate, beat-to-beat intervals and food logs are real, de-identified human data from PhysioNet. Sixteen participants, each recorded for eight to ten days.

*Visual:* The architecture diagram draws on, four columns. Column one highlights; sixteen tiny stick figures appear.

**`arch-02`** · 15.6s · glimpse: `—` · starts 3:35  
> That comes to thirty-eight thousand five-minute rows, and about twenty-two percent of them are followed by a spike within two hours. Keep that twenty-two percent in mind, because it is what random guessing would score.

*Visual:* Counter ticks to 38,115. A pie fills to 22%.

**`arch-03`** · 17.2s · glimpse: `—` · starts 3:51  
> The health record is synthetic, because the challenge rules forbid real patient records. But it is generated consistently with each person's real HbA1c, so someone with a higher reading gets a matching lab panel rather than a random one.

*Visual:* The paper file gets a hand-drawn SYNTHETIC stamp; values flow from an HbA1c number into the file.

**`arch-04`** · 21.4s · glimpse: `—` · starts 4:08  
> It is written as a FHIR R4 bundle, with LOINC codes for labs, SNOMED for diagnoses and RxNorm for medications. That matters because FHIR is the standard India's ABDM is built on. If this were ever deployed it would have to speak that language, so it speaks it now.

*Visual:* Code chips draw on beside the file; an ABDM label connects by arrow.

**`arch-05`** · 11.4s · glimpse: `—` · starts 4:29  
> Second column, ingestion. Multi-gigabyte sensor files are streamed into five-minute bins, and the three device clocks get checked against each other before anything is modelled.

*Visual:* Column two draws on: streaming arrows into a grid, three small clocks above it.

**`arch-06`** · 11.4s · glimpse: `—` · starts 4:41  
> Third column is the intelligence: the per-patient twin, the LightGBM model, and SHAP, which turns each prediction into a reason a doctor can actually read.

*Visual:* Column three draws on with three boxes.

**`arch-07`** · 11.0s · glimpse: `—` · starts 4:52  
> Fourth, the outputs: a risk score, the reason behind it in plain language, and a what-if simulator. All of it lands in one dashboard.

*Visual:* Column four draws on; arrows converge into a dashboard shape.

**`arch-08`** · 18.5s · glimpse: `—` · starts 5:03  
> The stack is deliberately boring. Python, pandas, SciPy for fitting the twin, LightGBM for the model, SHAP for explanations, Streamlit and Plotly for the dashboard. Nothing here needs a GPU, and the whole pipeline runs in a few minutes on a laptop.

*Visual:* The stack hand-letters in a row; a small laptop draws on with a timer.


## features

**`feat-01`** · 18.5s · glimpse: `ward-overview` · starts 5:22  
> Let me show you it running. This is the real dashboard on my machine, not a mockup. A ward list of sixteen patients, and a replay clock in the sidebar that walks through the recorded week as if it were happening live.

*Visual:* Hand-drawn browser frame draws on, the clip starts inside it. Arrow points at the sidebar replay clock.

**`feat-02`** · 17.2s · glimpse: `ward-overview` · starts 5:40  
> Nothing after that clock time is used for any prediction, a rule that holds everywhere in this project. Right now it is day three, eight in the evening: two patients in red, five on watch, two already above 140.

*Visual:* ScribbleCircle on the caption under the clock, then Camera zooms to the four metric cards.

**`feat-03`** · 16.8s · glimpse: `ward-overview` · starts 5:57  
> Patient 009 is at eighty percent. And this last column is what makes the whole thing usable: the reason. His twin projects him at 142 within the hour, and he ate forty-seven grams of carbohydrate ten minutes ago.

*Visual:* Zoom to the P-009 row; arrow to the Main reason cell; HandText repeats the reason.

**`feat-04`** · 11.4s · glimpse: `ward-overview` · starts 6:14  
> Those reasons are not hand-written rules. They are SHAP values from the model, the same model that never saw this patient, translated into plain language.

*Visual:* Sam appears bottom-left pointing up at the frame.

**`feat-05`** · 12.7s · glimpse: `ward-overview` · starts 6:26  
> The two marked in spike are already above 140, and they deliberately get no alert. The job is predicting onset, not announcing something the sensor is already showing.

*Visual:* Arrow to the two In spike rows, then a crossed-out alarm bubble.

**`feat-06`** · 11.4s · glimpse: `ward-cohort` · starts 6:38  
> You can also split the ward by sex, nine women and seven men, which will matter later when I come back to the women's-health module.

*Visual:* Clip of clicking the cohort filter; the counts circle.

**`feat-07`** · 16.0s · glimpse: `patient-timeline` · starts 6:50  
> Open a patient and you get her record, the alert with its reasons, and a multi-day timeline: glucose, risk, heart rate, variability, activity, and the shaded blocks are sleep, estimated from the wristband rather than measured.

*Visual:* Clip scrolling the patient page; arrows label each timeline band in turn.

**`feat-08`** · 15.2s · glimpse: `patient-twin-params` · starts 7:06  
> And here are her twin's four personal numbers. Carb sensitivity, time to peak, how much activity helps her, and how fast she recovers. Every one of them re-fitted daily on her own past data.

*Visual:* Zoom to the four metric cards for k, tp, a and tr; each label hand-letters beside it.

**`feat-09`** · 14.8s · glimpse: `plan-a-meal` · starts 7:21  
> Now the part I most want to show you. Look at her status first. Twenty-four percent, green, stable. The alarm says nothing here needs a doctor, and she last ate five hours ago.

*Visual:* Clip: Plan a meal panel. Arrow to the green stable status.

**`feat-10`** · 13.9s · glimpse: `plan-a-meal` · starts 7:36  
> But she is about to eat rice, dal and sabzi. Her twin has learned this patient from twenty-one of her own meals, so these are her numbers, not a population average.

*Visual:* Zoom to the meal selector and the twin caption underneath it.

**`feat-11`** · 9.3s · glimpse: `plan-a-meal` · starts 7:50  
> Rice first: a projected peak of 146. That crosses the line. Dal and sabzi first: 135. She stays under it.

*Visual:* ScribbleCircle on 146 in red, then on 135 in green; both held side by side.

**`feat-12`** · 14.3s · glimpse: `plan-a-meal` · starts 7:59  
> Same plate, same patient, same afternoon, and the difference between crossing 140 and not, on someone the alarm had already called stable. That is the thing a classifier alone cannot give you.

*Visual:* Sam cheers. HandText: SAME FOOD, DIFFERENT ORDER.

**`feat-13`** · 19.3s · glimpse: `plan-a-meal` · starts 8:13  
> Now the honest part, and it is printed in the app itself. The eating-order effect is not learned from this data. These food logs recorded what was eaten and when, but never in what order, so there is nothing here to learn it from.

*Visual:* Camera pans to the orange warning box; arrow to the text; Sam holds palms up.

**`feat-14`** · 16.8s · glimpse: `—` · starts 8:32  
> It comes from a published crossover trial of eleven patients with type 2 diabetes. They reported twenty-nine to thirty-seven percent lower glucose. I apply twenty-five, deliberately below their figure, and the app says so on screen every time.

*Visual:* Hand-drawn citation card: Shukla et al., Diabetes Care 2015. The 25% circles.

**`feat-15`** · 12.7s · glimpse: `plan-a-meal` · starts 8:49  
> It also does not just say risky. It says how much is fine: about a hundred grams of net carbs for this patient right now, eaten vegetables first.

*Visual:* Zoom to the Largest safe portion block; the gram figure circles.

**`feat-16`** · 13.1s · glimpse: `build-your-own` · starts 9:02  
> And it is not limited to the ten meals in my menu file. You can type any food's carbs, fibre, protein and fat directly, and get the same projection.

*Visual:* Clip: switching to Build your own and typing numbers; the curves redraw.

**`feat-17`** · 15.6s · glimpse: `describe-in-words` · starts 9:15  
> Or describe it in plain English and have it estimated, then edit whatever looks wrong. That matters for Indian food, where almost nobody knows the gram values of two rotis and a katori of dal.

*Visual:* Clip: typing a meal description; the four numbers fill in.

**`feat-18`** · 14.8s · glimpse: `—` · starts 9:31  
> Next, the twin that learns from every meal, the newest part of the project. The daily twin re-fits once a day. But a patient does not live in days. They live in meals.

*Visual:* Sam wipes the frame. A calendar morphs into a row of plates.

**`feat-19`** · 17.7s · glimpse: `—` · starts 9:45  
> So before each meal it predicts the peak, using parameters fitted only on that person's earlier meals. After the meal, the sensor records what really happened. Then it re-fits, and the next meal is predicted by a slightly better twin.

*Visual:* A timeline of plates with only the left ones shaded; a loop arrow draws: predict, compare, re-fit.

**`feat-20`** · 17.7s · glimpse: `after-the-meal` · starts 10:03  
> Here is a real meal she ate on the twenty-ninth of February. The generic population twin predicted a peak of 182. Her own twin, trained on the six meals it had seen so far, predicted 148. The sensor recorded 147.

*Visual:* Clip: the After the meal panel at meal 7. Three numbers circle in sequence; 147 gets a tick.

**`feat-21`** · 15.6s · glimpse: `—` · starts 10:21  
> One more feature, and it needs its caveat said out loud rather than buried. Insulin sensitivity falls in the luteal phase of the menstrual cycle, so the same meal peaks higher in the second half.

*Visual:* Sam holds up a hand, serious. A hand-drawn cycle wheel draws on with the second half shaded.

**`feat-22`** · 13.5s · glimpse: `—` · starts 10:37  
> For a tool aimed at prediabetes in India, where most of the people being screened are women, that is a real clinical factor, and most glucose models ignore it entirely.

*Visual:* Sam points at the wheel; a row of female stick figures draws on.

**`feat-23`** · 18.5s · glimpse: `cycle-module` · starts 10:50  
> But this study enrolled only post-menopausal women. Every woman in this dataset is post-menopausal. None of them has a cycle at all. So it ships as a clearly labelled capability demonstration, defaulting to not applicable, never a claim about a patient here.

*Visual:* Clip: the yellow warning box in the cycle panel; arrow to the Not applicable option.

**`feat-24`** · 19.3s · glimpse: `cycle-luteal` · starts 11:09  
> With that said, here is what it shows. Vegetables first was 135, comfortably under the line. In the luteal phase it is 141. The same advice stops working, which is exactly the kind of thing a personalised twin should be able to tell you.

*Visual:* Clip: toggling to Luteal; the three numbers change together.


## decisions

**`dec-01`** · 19.3s · glimpse: `—` · starts 11:28  
> Four technical decisions did most of the work, and none of them are glamorous. The first: three devices, three clocks. The glucose monitor, the wristband and the phone each keep their own time, and in a de-identified dataset those times are shifted for privacy.

*Visual:* Sam at a whiteboard with four empty slots. Three clocks draw on showing different times.

**`dec-02`** · 19.3s · glimpse: `—` · starts 11:47  
> If the streams are misaligned and you do not check, the model quietly learns nothing and you never find out. So it checks, and prints a report. One participant was logging meals 105 minutes late. Another's food log fell entirely outside their glucose recording.

*Visual:* Two waveforms slide out of phase, then snap together. The alignment report draws on, two rows circled.

**`dec-03`** · 13.9s · glimpse: `—` · starts 12:06  
> That second one gets marked unavailable, meaning unknown, not never ate. Tell a model someone ate zero carbohydrate when you simply do not know, and you have taught it something false.

*Visual:* A cell fills with a question mark rather than a zero; a crossed-out zero beside it.

**`dec-04`** · 18.9s · glimpse: `—` · starts 12:20  
> Second decision: every wearable feature is measured against that person's own normal, never raw. Resting heart rate varies far more between people than within one, so raw values mostly teach a model who someone is, which transfers to a new patient as nothing.

*Visual:* Two stick figures with different resting rates; an arrow between them is crossed out.

**`dec-05`** · 14.8s · glimpse: `—` · starts 12:39  
> Third: validation is leave-one-subject-out. Train on fifteen people, test on the sixteenth, repeat sixteen times. Every number in this video comes from a model that has never seen the patient it is scoring.

*Visual:* Sixteen figures; one lifts out and is tested, sixteen times in fast succession.

**`dec-06`** · 26.0s · glimpse: `leakage-check` · starts 12:54  
> Fourth, and the one everything rests on: the twin must never see the future. A bin labelled twelve-oh-five holds data up to twelve-oh-five, so a prediction made then has never seen twelve-oh-six. And that is tested, not asserted. A script deletes future meals and future glucose and checks nothing moves. On the real data, 192 and 53 probes, all unchanged.

*Visual:* A time axis with a shaded past and a blocked future. Then a paper card: NO LEAKAGE FOUND.


## results

**`res-01`** · 13.9s · glimpse: `—` · starts 13:20  
> So does any of it work. Here is the ablation, each row adding one more data stream. The simple trend alarm gets 0.612, at 6.8 false alarms per patient per day.

*Visual:* The results table draws on row by row; row one highlights with seven alarm bubbles.

**`res-02`** · 13.5s · glimpse: `—` · starts 13:34  
> Glucose alone with machine learning, 0.681. Adding the meal log, 0.708, the biggest single jump in the table. Adding the health record, 0.728, at about two false alarms a day.

*Visual:* Rows highlight in turn; a bar grows alongside each.

**`res-03`** · 10.6s · glimpse: `—` · starts 13:47  
> That is eighty-six percent of spikes caught, a median forty-five minutes ahead, with false alarms cut from nearly seven a day to two.

*Visual:* Three hand-drawn stat cards land in a row.

**`res-04`** · 15.6s · glimpse: `—` · starts 13:58  
> And the per-meal twin: across 431 real meals from fifteen patients, the error in the predicted peak fell from 32.5 to 25.2 milligrams per decilitre. Twenty-two percent better, and it helped thirteen of the fifteen.

*Visual:* Two bars shrink side by side; thirteen of fifteen figures get ticks.

**`res-05`** · 18.9s · glimpse: `—` · starts 14:14  
> I also tried letting the twin use fibre, fat and protein. That changed the error by 0.12, with a confidence interval that includes zero. It did not help, and I report it as the negative result it is rather than quietly dropping it.

*Visual:* Sam shrugs at a flat bar. HandText: NO MEASURABLE EFFECT.


## next

**`next-01`** · 19.8s · glimpse: `—` · starts 14:32  
> What I would do next, in order. First, a cycle-aware twin learned from data instead of assumed. There is a dataset for exactly this: forty-two women, three months each, with daily hormone assays. Three full cycles per person, which is enough to fit it properly.

*Visual:* A numbered list begins. The cycle wheel returns with a tick.

**`next-02`** · 11.4s · glimpse: `—` · starts 14:52  
> Second, real health records through ABDM consent flows instead of synthetic ones. Third, federated or on-device training, so the data never leaves the patient's phone.

*Visual:* The SYNTHETIC stamp crosses out. A phone with a lock, arrows staying inside it.

**`next-03`** · 14.3s · glimpse: `—` · starts 15:04  
> And the honest limits. Sixteen people shows feasibility, not clinical validity. The health record is synthetic apart from sex and HbA1c. Activity and sleep are estimated from heart rate, not measured properly.

*Visual:* Sam beside a list of three caveats, each ticking on.

**`next-04`** · 13.9s · glimpse: `—` · starts 15:18  
> This is decision support, not a prescription, and it is not a medical device. Deploying it would need prospective validation and regulatory review before it went anywhere near a real clinic.

*Visual:* A hand-drawn stamp lands: NOT A MEDICAL DEVICE.


## outro

**`outro-01`** · 8.9s · glimpse: `—` · starts 15:32  
> All of it is open: the code, the data provenance, every result, and the leakage test. Thanks for watching.

*Visual:* The repo URL hand-letters on. Sam waves; the title card redraws and settles.
