# BrainArt — Live Muse → Valence/Arousal → Art (pipeline overview)

Semi-real-time valence/arousal from a 4-channel Muse, streamed to the web art
client (`../web/`) over WebSocket. This file is the index for the live pipeline;
see `research/window_experiment_findings.md` for the empirical justification.

Layout:
```
emotion_pipeline/
├── brainart_muse_server.py   # live server (run this)
├── train_muse_va_live.py     # (re)train the model
├── models/                   # muse_va_live.npz/.json, muse_va_calib.json (created on calibrate)
└── research/                 # faced_window_experiment.py, window_experiment_findings.md
```
All commands below assume you are inside `emotion_pipeline/`.

## Why it's built this way (one paragraph)
Per-1-second-window decoding had almost no signal (r_val≈0.03, r_aro≈0.00).
Empirically, the levers that worked were **longer windows** (30 s ≈ 3× the
signal) and **engineered features** (temporal slope, frontal/temporal alpha
asymmetry, beta–alpha ratio), with a **linear model**. PCA hurt (signal lives in
low-variance directions); K-means clusters were unrelated to VA. Per-subject
feature normalization — the training-time analogue of a live per-user baseline —
lifted cross-subject CV to **r_val≈0.15, r_aro≈0.12**. That's real but weak, so
the honest target is a *relative-to-your-own-baseline* readout, not absolute VA.

## Files
| File | Role |
|---|---|
| `brainart_muse_server.py` | Muse → baseline → sliding-window inference → calibration → WebSocket |
| `train_muse_va_live.py` | trains the 30 s engineered-feature Ridge model → `models/muse_va_live.npz` (+`.json`) |
| `models/muse_va_live.npz` / `.json` | trained model (numpy-only) + metadata |
| `models/muse_va_calib.json` | per-user affine rescale, created when you calibrate |
| `research/faced_window_experiment.py` | the experiment that produced the findings |
| `research/window_experiment_findings.md` | full results table + recommendation |
| `../web/brainart_emotion.js` | web art client (has **go live** + **Calibration** UI) |

## Run
```bash
pip install brainflow websockets numpy scikit-learn pandas
# from inside emotion_pipeline/ :
python train_muse_va_live.py                 # optional: retrain (needs FACED dataset; CV r_val≈.15 r_aro≈.12)
python brainart_muse_server.py --board muse_2 # real Muse  (muse_s / muse_2016)
python brainart_muse_server.py --sim          # no hardware: test the web wiring
```
The server resolves `models/` relative to itself, so it runs from any working
directory. Open `../web/brainart_emotion.html` → **go live** (top-right). Status
dot: grey=off, amber=baseline (sit still ~90 s), green=live.

## Timing
Sliding 30 s window, new estimate every 1 s, broadcast at 4 Hz; the browser
lerps every frame → continuous glide. One-time ~90 s baseline at startup; a full
mood swing then trails by ~10–15 s (partial movement is immediate). Tune via
`WINDOW_SEC`/`EMA` (server) and `LIVE.smooth` (client).

## Per-user calibration (fills the −1…1 range)
Raw output clusters near zero (regression to the mean at modest r, plus
population-vs-user scaling). Calibration measures where *this* user actually
lands and stretches it:
1. Go live; in the **Calibration** panel, play a clip for each condition and
   click its button (records the raw output for `CALIB.secs`, default 90 s).
   Conditions span the VA corners: joy / fear-anger / calm-tender / sad / neutral.
2. Click **fit & save** (≥2 conditions). The server least-squares-fits a per-axis
   affine `out = clip(a·raw + b, −1, 1)`, saves `muse_va_calib.json`, and
   auto-loads it next run. **reset** clears it.

Control messages (client→server, same socket):
`{"cmd":"calib_clip","label":"joy","v":0.8,"a":0.8,"secs":90}`,
`{"cmd":"calib_finish"}`, `{"cmd":"calib_reset"}`.

Calibration fixes **range/centering, not accuracy** — at r≈0.15 the ordering is
still soft, so it makes the piece expressive but a happy vs sad moment will
sometimes land wrong. Arousal anchors more cleanly than valence.

## Invariants (don't break)
- `feat_full()` + band defs are **identical** in `train_muse_va_live.py` and
  `brainart_muse_server.py`. Change one → change both.
- Live Muse channels are reordered TP9,AF7,AF8,TP10 → model order
  AF7,AF8,TP9,TP10 (`reorder` in the server).
- Live per-user baseline z-score mirrors the per-subject feature z-score used in
  training — keep them consistent.

## Honest caveats
- DE ≈ `log(bandpower)`; FACED's exact DE has a constant offset absorbed by the
  baseline z-score. Fine for a relative readout.
- A 60–90 s rest baseline is narrower than across-emotion variance; calibration
  (or the `tanh`/`pred_std` fallback) compensates but the range is compressible.
- Output is valence/arousal **relative to the user's own neutral**, in [−1,1].
