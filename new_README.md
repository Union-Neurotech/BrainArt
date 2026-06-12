# Live Muse → BrainArt (semi-real-time valence/arousal)

End-to-end: Muse EEG → Python inference server → WebSocket → web art that
updates continuously.

## Pieces
| File | Role |
|---|---|
| `train_muse_va_live.py` | trains/saves the 30 s engineered-feature Ridge model → `muse_va_live.npz` |
| `muse_va_live.npz` / `.json` | the trained model + metadata (already built) |
| `brainart_muse_server.py` | Muse → baseline → sliding-window inference → WebSocket broadcast |
| `new_web_variant/brainart_emotion.js` | now has a **go live** button + WebSocket client |

## Run it

```bash
pip install brainflow websockets numpy scikit-learn pandas

# (optional) retrain the model
python train_muse_va_live.py            # subject-indep CV: r_val≈0.15, r_aro≈0.12

# start the server with a real Muse
python brainart_muse_server.py --board muse_2     # or muse_s / muse_2016
# …or test the whole web wiring with NO hardware:
python brainart_muse_server.py --sim
```

Then open `new_web_variant/brainart_emotion.html` and click **go live**
(top-right). The status dot: grey=off, amber=baseline (sit still ~90 s),
green=live. With `--sim` it jumps straight to green with synthetic motion.

## How the timing works
- **Sliding window**: each estimate uses the trailing 30 s, but a new one is
  emitted every 1 s, broadcast at 4 Hz. The browser lerps `E` toward each value
  every frame → continuous glide, not steps.
- **Latency**: ~90 s one-time baseline at startup, then the dial trails real
  mood shifts by ~10–15 s for a full swing (partial movement is immediate).
  Lower `WINDOW_SEC` in the server (e.g. 20) for snappier/jumpier; raise `EMA`
  for calmer. In the browser, `LIVE.smooth` (default 0.06) sets the glide speed.

## What it actually outputs
`valence` and `arousal` in **[-1, 1]**, **relative to the user's own rest
baseline** (this is the honest target for 4-channel dry EEG — absolute
cross-person VA isn't reliable). The server also streams the 5 EEG band powers
(also baseline-normalized) which drive the art's texture/speed/chaos.

## Important matches (don't break these)
- `feat_full()` and the band definitions in `brainart_muse_server.py` are
  **identical** to `train_muse_va_live.py`. If you change features in one,
  change both.
- Live Muse channel order is reordered TP9,AF7,AF8,TP10 → model order
  AF7,AF8,TP9,TP10 in the server (`reorder`).
- Baseline normalization (per-user μ/σ) is the live analogue of the per-subject
  feature z-scoring used in training — keep them consistent.

## Caveats
- DE here = `log(bandpower)`; FACED's exact DE has a constant offset that the
  per-user baseline z-score absorbs. Fine for a *relative* readout.
- A 60–90 s rest baseline is narrower than across-emotion variance; the `tanh`
  output scaling (`pred_std`) compensates, but expect a compressed range. A
  short 2–3 clip calibration could be added later to use the full −1…1 range.