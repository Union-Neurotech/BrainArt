# 4-Channel Muse VA Regression — What Actually Moves the Needle

Empirical test on FACED (123 subjects, real per-trial valence/arousal, `subject_z`
normalized), 4 Muse-proxy channels (FP1/FP2/A1/A2), subject-independent 5-fold CV,
pooled Pearson r. Script: `faced_window_experiment.py`.

## Starting point
Per-1-second window MLP: **r_val ≈ 0.03, r_aro ≈ 0.00** — essentially no arousal signal.

## Result 1 — Longer windows help, a lot (your intuition was right)
Window length sweep (engineered features, Ridge):

| Window | n samples | r_valence | r_arousal |
|---|---|---|---|
| 1 s  | 103,320 | +0.032 | +0.031 |
| 5 s  | 20,664  | +0.053 | +0.037 |
| 10 s | 10,332  | +0.064 | +0.041 |
| 15 s | 6,888   | +0.057 | +0.051 |
| **30 s** | **3,444** | **+0.104** | **+0.105** |

Aggregating to the full 30 s trial **~3×'d valence and turned arousal from noise into
real signal.** The per-window approach was throwing away the temporal evolution that
carries most of the decodable information. (Note: sub-30s rows share a label across
chunks of the same trial, so their r is mildly optimistic; 30s is the clean, best case.)

## Result 2 — Engineered features roughly double r
At 30 s, Ridge:

| Features | dim | r_valence | r_arousal |
|---|---|---|---|
| band-power mean only | 20 | +0.062 | +0.050 |
| mean + std | 40 | +0.064 | +0.048 |
| **mean + std + slope + asymmetry/ratios** | 65 | **+0.104** | **+0.105** |

The temporal **slope** across the trial, **frontal/temporal alpha asymmetry**, and
**beta–alpha ratio** are what unlock arousal. These are the classic affect correlates.

## Result 3 — Model architecture is NOT the lever
- Ridge alpha 1→300: flat at r ≈ 0.10 (signal is linear, not under-regularized).
- HistGradientBoosting: r_val 0.103, **r_aro 0.057** (worse — overfits).
- Confirms the earlier diagnosis: LSTM / deeper nets won't help. The bottleneck is
  information content of 4 channels, not capacity. Keep the model linear/tiny.

## Result 4 — PCA: don't
Ridge on top-k PCs of the 65 features:

| PCs | cum. variance | r_valence | r_arousal |
|---|---|---|---|
| 5  | 85% | +0.017 | +0.028 |
| 10 | 94% | +0.063 | +0.045 |
| 20 | 98% | +0.093 | +0.088 |
| (all 65) | 100% | +0.104 | +0.105 |

The VA-predictive signal lives in **low-variance directions**. PCA keeps the
high-variance ones, so it actively discards signal — k=5 captures 85% of variance but
almost none of the predictive content. PCA is fine as a redundancy *diagnostic*, useless
as a performance lever here.

## Result 5 — K-means → VA quadrants: dead end
Fraction of VA variance explained by cluster identity (η²):

| clusters | η²_valence | η²_arousal |
|---|---|---|
| 2 | 0.000 | 0.000 |
| 4 | 0.001 | 0.002 |
| 8 | 0.003 | 0.005 |

Unsupervised clusters track dominant variance (subject identity, overall power), not
emotion. They are ~0% aligned with valence/arousal. Don't build the pipeline on this.

## Bottom line / recommended approach
1. **Train on 20–30 s windows, not 1 s.** Change the sample unit; this is the single
   biggest win.
2. **Use engineered features** (per channel-band mean + std + slope; frontal & temporal
   alpha asymmetry; beta–alpha ratio). ~65 dims for 4 channels.
3. **Keep the model a simple linear/Ridge regressor** (or tiny MLP). Skip LSTM/boosting.
4. **Skip PCA and K-means** for the regression objective.

Realistic cross-subject ceiling with 4 channels + subject_z labels is **r ≈ 0.10–0.11**
(both dims) — real but weak (~1% variance). To get a genuinely usable live signal:
- **Per-user calibration / fine-tune** on a short Muse rest+task session. Within-subject
  decoding is far easier than cross-subject and is the real path to a usable app.
- Output **smoothed continuous estimates or coarse quadrants**, not precise values.
- For arousal especially, **autonomic signals (HR/GSR)** add more than any EEG model tweak.

### Live-inference caveat on the `[-1, 1]` output
`subject_z` uses each subject's own 28-trial mean/std — not available for a brand-new
live user. For deployment you need a **per-user baseline window** (e.g. 60 s rest) to
z-score against, or use the `global` raw `[-1,1]` mapping. Decide this before wiring the
model into BrainArt, because the training normalization has to match what you can compute
live.
