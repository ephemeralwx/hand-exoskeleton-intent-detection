# results

ran `python online_learning_cnn_lstm.py --mode both` on 2026-03-30. total time ~16 min on cpu.

loso = leave-one-subject-out. each subject gets held out as the test set, model trains on the other 3, then online learning adapts during inference on the held-out subject. 3 seeds for randomized, 1 for chrono (deterministic so more seeds would be redundant).

## summary — randomized mode (3 seeds)

this is the main result table. "base" is the frozen loso model with no adaptation. "online" is after the online learner runs through the test set adapting as it goes. q4 = last 25% of the data, where the model has had the most updates — closer to what real deployment performance would look like.

| subject | base acc | base f1 | online acc | online f1 | q4 f1 | base det% | online det% | df1 |
|---|---|---|---|---|---|---|---|---|
| Subject 1 | 0.7154 | 0.5807 | 0.8445 | 0.7265 | 0.7818 | 78.0% | 92.9% | +0.2011 |
| Subject 2 | 0.6104 | 0.5450 | 0.7689 | 0.6660 | 0.7245 | 85.1% | 94.2% | +0.1795 |
| Subject 3 | 0.8069 | 0.6515 | 0.8601 | 0.7569 | 0.8045 | 78.2% | 96.0% | +0.1530 |
| Subject 4 (stroke) | 0.7829 | 0.4945 | 0.8511 | 0.6553 | 0.7036 | 89.5% | 97.8% | +0.2091 |
| **average** | **0.7289** | **0.5679** | **0.8311** | **0.7012** | **0.7536** | **82.7%** | **95.2%** | **+0.1857** |

det% = fraction of hand opening instances where at least 1 window was correctly predicted as positive. this is the clinically relevant metric — we care about catching the intent, not about every single window being right.

df1 is q4 f1 minus baseline f1. positive = online learning helped. every subject improved, stroke patient (subject 4) had the biggest f1 gain which is nice since thats the target population.

## chrono vs randomized

chrono processes windows in the order they were actually recorded. randomized shuffles them. randomized consistently does better — probably because shuffling gives the model a more diverse training buffer early on instead of seeing a bunch of similar consecutive windows. but chrono is more realistic for deployment since you obviously cant shuffle real-time data.

| subject | chrono f1 | random f1 | chrono det% | random det% | df1 (c-r) |
|---|---|---|---|---|---|
| Subject 1 | 0.6933 | 0.7265 | 89.0% | 92.9% | -0.0332 |
| Subject 2 | 0.6207 | 0.6660 | 85.9% | 94.2% | -0.0453 |
| Subject 3 | 0.6797 | 0.7569 | 87.6% | 96.0% | -0.0772 |
| Subject 4 (stroke) | 0.6079 | 0.6553 | 96.1% | 97.8% | -0.0475 |

randomized wins across the board on f1 by ~3-8%. detection rate gap is smaller though, chrono still catches most openings.

## threshold sweep — subject 4 (stroke), randomized, best seed by f1

the sweep shows how changing the classification threshold trades off detection rate vs false positives. lower threshold = catch more openings but more false alarms. the adapted threshold (0.83) landed on the aggressive side, probably because the optimizer is maximizing f1 on the buffer which skews toward high precision when the class balance shifts during adaptation.

### full run

covers all windows from start to finish, including early ones where the model hasnt adapted yet. numbers here are dragged down by the rough first quarter.

| thresh | det rate | detected | fp windows | window fpr | f1 |
|---|---|---|---|---|---|
| 0.30 | 100.0% | 76/76 | 432 | 23.3% | 0.630 |
| 0.35 | 100.0% | 76/76 | 417 | 22.5% | 0.627 |
| 0.40 | 100.0% | 76/76 | 380 | 20.5% | 0.635 |
| 0.45 | 100.0% | 76/76 | 347 | 18.7% | 0.640 |
| 0.50 | 100.0% | 76/76 | 311 | 16.8% | 0.648 |
| 0.55 | 98.7% | 75/76 | 272 | 14.7% | 0.664 |
| 0.60 | 98.7% | 75/76 | 247 | 13.3% | 0.661 |
| 0.65 | 97.4% | 74/76 | 224 | 12.1% | 0.664 |
| 0.70 | 97.4% | 74/76 | 199 | 10.7% | 0.665 |
| 0.75 | 97.4% | 74/76 | 173 | 9.3% | 0.660 |
| 0.80 | 97.4% | 74/76 | 137 | 7.4% | 0.646 |
| 0.85 | 96.1% | 73/76 | 98 | 5.3% | 0.587 |
| 0.90 | 80.3% | 61/76 | 44 | 2.4% | 0.498 |

### Q4 only

last 25% of the simulation — model is fully warmed up by this point. this is what we'd expect in steady-state deployment after the initial adaptation period. f1 peaks higher here (0.746 at thresh=0.80) vs full run peak of 0.665, because the early noisy predictions arent dragging it down. detection rate is lower than full run though — just fewer opening instances in the last quarter (60 vs 76 total).

| thresh | det rate | detected | fp windows | window fpr | f1 |
|---|---|---|---|---|---|
| 0.30 | 90.0% | 54/60 | 93 | 20.0% | 0.656 |
| 0.35 | 90.0% | 54/60 | 87 | 18.7% | 0.669 |
| 0.40 | 88.3% | 53/60 | 78 | 16.8% | 0.671 |
| 0.45 | 85.0% | 51/60 | 71 | 15.3% | 0.679 |
| 0.50 | 85.0% | 51/60 | 63 | 13.5% | 0.694 |
| 0.55 | 85.0% | 51/60 | 59 | 12.7% | 0.705 |
| 0.60 | 83.3% | 50/60 | 57 | 12.3% | 0.700 |
| 0.65 | 83.3% | 50/60 | 54 | 11.6% | 0.708 |
| 0.70 | 81.7% | 49/60 | 44 | 9.5% | 0.732 |
| 0.75 | 80.0% | 48/60 | 40 | 8.6% | 0.733 |
| 0.80 | 78.3% | 47/60 | 31 | 6.7% | 0.746 |
| 0.85 | 78.3% | 47/60 | 26 | 5.6% | 0.734 |
| 0.90 | 71.7% | 43/60 | 16 | 3.4% | 0.694 |

## generated plots

- `online_learning_curves_{chrono,random}.png` — running accuracy and f1 over time (100-window sliding average) for each subject, one panel per subject. vertical dashed lines = where model updates happened. dotted horizontal lines = frozen baseline for comparison. if a subject has opening id data it also plots running detection rate. this uses data from the full run, not just Q4.
- `online_learning_report_{chrono,random}.png` — same summary table and mini adaptation curves combined into one figure. meant for quick visual overview, same data as above.
- `threshold_tradeoff_{chrono,random}.png` — plots detection rate vs window false positive rate across thresholds, one panel per subject that has opening data. red star marks where the adaptive threshold ended up. orange dashed line on right axis = f1 at each threshold. this uses full run data, not Q4-only.
