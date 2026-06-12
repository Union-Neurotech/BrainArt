# BrainArt — Tauri desktop app

A native desktop window that runs the live BrainArt experience: it launches the
Python inference server (`../emotion_pipeline/brainart_muse_server.py`) as a
child process and renders the CPPN artwork (React/WebGL) driven by the live
valence/arousal stream over WebSocket. Same pipeline as the `web/` page — just
inside a Tauri window instead of a browser.

## How it works
```
Tauri window (React)
  │  Connect  ──► invoke("start_server", { boardId })   [Rust: src-tauri/src/main.rs]
  │                   └─► spawns:  python brainart_muse_server.py --board <id>   (cwd = ../emotion_pipeline)
  │  WebSocket ws://localhost:8765  ◄── server broadcasts {valence,arousal,bands,status,calibrated}
  │  BrainArtCanvas eases toward the live values every frame (continuous art)
  └  Disconnect / window close ──► invoke("stop_server")  (kills the Python child)
```
The Rust backend spawns the server with `std::process::Command` (no extra Tauri
permission needed). Board selection maps to the server flags: **Synthetic** →
`--sim` (no hardware), **Muse 2 / 2016 / S** → `--board muse_2|muse_2016|muse_s`.

## Prerequisites
- **Rust + Tauri CLI** (`cargo`, and Tauri 2 prereqs for your OS).
- **Node** (`npm`).
- **Python** with the pipeline deps:
  `pip install brainflow websockets numpy scikit-learn pandas`
  (the trained model already lives in `../emotion_pipeline/models/`).
- `python` must be on PATH. If your interpreter is named differently, set
  `BRAINART_PYTHON` (e.g. `set BRAINART_PYTHON=py`).

## Run (dev)
```bash
cd brainart-tauri
npm install
npm run tauri dev
```
Then in the window: pick a **Board** → **Connect**. With a real Muse, the status
dot goes amber (**baseline — sit still ~90 s**), then green (**live**) and the
art starts reacting. Pick **Synthetic** to test everything with no headset.

### Calibrate (optional, fills the −1…1 range)
While live, use the **Calibration** panel in the sidebar: play a clip for each
condition (joy / fear-anger / calm-tender / sad / neutral), click its button to
record (~90 s; ~8 s in Synthetic), then **fit & save**. The server saves
`../emotion_pipeline/models/muse_va_calib.json` and auto-loads it next time.

## Notes
- The server's model path resolves relative to the script, so the spawn works
  regardless of working directory. Override the pipeline location with
  `BRAINART_PIPELINE` if you move things.
- The collapsible **Raw EEG** panel is a placeholder — the server doesn't send
  `raw_waves` yet; add it to the broadcast in `brainart_muse_server.py` if you
  want a live trace.
- For a distributable bundle, switch the dev-time process spawn to a Tauri
  **sidecar** (bundle a packaged Python or a PyInstaller exe) — the current
  `Command::new("python")` approach targets `tauri dev` / a machine with Python.

## Build
```bash
npm run tauri build
```
(Bundles the React frontend; remember the Python sidecar caveat above for a
self-contained installer.)
