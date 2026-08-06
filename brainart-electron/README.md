# BrainArt (Electron + Python)

Desktop BrainArt: an Electron front-end (WebGL CPPN viewport + status/control bar
+ debug console) driven by a Python backend that handles BrainFlow devices and
talks to the renderer over a local WebSocket.

**You do not need an EEG headset to run the demo** — the Synthetic board produces
fake but realistic data and drives the artwork exactly like a real device.

## Quick start

```sh
# from the project root
python -m venv .esp
.esp\Scripts\python -m pip install -r requirements-backend.txt   # Windows
# .esp/bin/python  -m pip install -r requirements-backend.txt    # macOS/Linux

cd brainart-electron
npm install
npm run dev
```

Then in the window: **Synthetic** → **Connect** → **Start**. The artwork begins
reacting within a few seconds. **Stop** when done.

## Prerequisites

| | Version | Notes |
|---|---|---|
| Node.js | 18+ | Electron 33 / electron-vite 2. Verified on 22.17.1 |
| Python | 3.9+ | Verified on 3.14.6 |
| OS | Windows / macOS / Linux | Printing uses `os.startfile` on Windows, `lpr` elsewhere |

Python deps come from **`requirements-backend.txt`** (brainflow, numpy, pandas,
websockets), *not* the root `requirements.txt` — that one pins Streamlit-era
versions for the legacy UI and won't install cleanly on modern Python.

## One-time setup

**Python backend env** — from the **project root**, one level up from here:

```sh
python -m venv .esp
.esp\Scripts\python -m pip install -r requirements-backend.txt   # Windows
# .esp/bin/python  -m pip install -r requirements-backend.txt    # macOS/Linux
```

The Electron main process auto-discovers `.esp` at the project root; override the
interpreter with `BRAINART_PYTHON` if you keep your env elsewhere.

**Front-end deps:**

```sh
cd brainart-electron
npm install
```

## Running

```sh
npm run dev     # dev mode, hot reload — use this
npm run build   # produce out/ for a packaged/preview run
npm run preview # run the built output
```

`npm run dev` launches Electron, which spawns the Python backend itself and opens
the window. Backend output is prefixed `[py]` in the same terminal — that's where
save confirmations and device errors appear.

> **Use `npm run dev` while developing.** `preview`, a bare `electron .`, or a
> packaged build load the prebuilt `out/renderer/`, which is only as fresh as your
> last `npm run build`. A stale bundle renders old UI and looks like your change
> did nothing.

To run the backend by itself (useful for debugging it in isolation):

```sh
.esp\Scripts\python src\server.py --port 17321
```

## Demo walkthrough

### Without hardware (Synthetic)

1. Leave the device dropdown on **Synthetic** (it's selected by default).
2. **Connect** → the console logs `Connected to Synthetic (N EEG channels)`.
3. **Start** → raw traces appear in the debug console's wave preview immediately.
4. After a ~4 s warm-up (`METRICS_WINDOW_SEC`), live metrics begin flowing at
   ~10 Hz: the EEG band and ML-metric indicators fill in, and the artwork starts
   responding to them.
5. **Stop** → streaming ends and the backend computes one averaged feature vector
   over the *entire* session, logs it, and applies it as a final visual state.

### With a Muse 2

1. Power on the headset and make sure it isn't already paired to another app.
2. Pick **Muse2** in the dropdown → **Connect** → **Start**.

BrainFlow talks to the Muse 2 over **native BLE — no BLED112 dongle required**.
That's why no serial-port field appears for it: the port input is only rendered
for boards flagged `using_port` in [`src/assets.py`](../src/assets.py), which
among the enabled boards means only the OpenBCI Cyton.

Connecting scans for the device and can take several seconds. On failure the
console logs `Failed to connect to Muse2. Check device/port.`

### Driving the visuals by hand

The **Emotion** sliders write to the same visual state the backend patches, so
they work with nothing connected — handy for exploring the art or reproducing
rendering issues:

- **Arousal** — zoom and structural complexity
- **Valence** — warm ↔ cool, and saturation
- **Depth** — CPPN layer count (1–12); the strongest lever on visual detail

The **EEG Bands** and **ML Metrics** rows above them are read-only indicators —
they display backend values and can't be dragged.

### Saving

**Save Image** and **Print Image** capture the WebGL viewport only (not the UI
chrome) at a fixed 1600×900, and hand the PNG to the backend, which writes the
file and logs the full path. Filenames are timestamped and suffixed (`_1`, `_2`,
…) so rapid saves don't overwrite each other.

**Images are saved to your Downloads folder by default.** Electron resolves it
with `app.getPath('downloads')`, which asks the OS for the real location —
correct even if you've moved Downloads to another drive.

### Saving somewhere else

| Goal | How |
|---|---|
| Downloads | nothing to do — this is the default |
| Back to `../generated/images` | `BRAINART_SAVE_TO_PROJECT=1` |
| A specific folder | `BRAINART_IMAGE_DIR=D:\brainart-output` |

```powershell
$env:BRAINART_SAVE_TO_PROJECT = 1 ; npm run dev    # PowerShell
```
```sh
BRAINART_SAVE_TO_PROJECT=1 npm run dev             # bash/zsh
```

Running the backend directly, the same choices are CLI flags:

```sh
.esp\Scripts\python src\server.py                  # Downloads (default)
.esp\Scripts\python src\server.py --project-images  # generated/images
.esp\Scripts\python src\server.py --image-dir D:\brainart-output
```

The backend prints its destination on startup (`Saving images to: …`), so you can
always confirm where saves will land before clicking anything.

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `BRAINART_PYTHON` | auto-detected `.esp` | Interpreter used for the backend |
| `BRAINART_WS_PORT` | `17321` | WebSocket port (main process and renderer both read it) |
| `BRAINART_SAVE_TO_PROJECT` | unset | If set, save images to `../generated/images` instead of Downloads |
| `BRAINART_IMAGE_DIR` | unset | Explicit output directory; wins over `BRAINART_SAVE_TO_PROJECT` |

Image-destination precedence: CLI flag (`--downloads` / `--project-images` /
`--image-dir`) → `$BRAINART_IMAGE_DIR` → **Downloads** (default).

> These env vars are presence-checked, not value-parsed — `BRAINART_SAVE_TO_PROJECT=0`
> still counts as *set*. Unset the variable to turn it off.

## Architecture

```
Electron main (src/main/index.ts)
  ├─ spawns:  python ../src/server.py --port 17321
  └─ opens:   BrowserWindow (React + Ionic renderer)

renderer  ⇄  ws://127.0.0.1:17321  ⇄  Python backend (../src/server.py)
```

The backend reuses the existing project code: `communications.Comms`,
`preprocessing.get_simple_feature_vector`, `assets.board_id_pairs`.

There is one window and one page. The layout is a grid: WebGL viewport (top-left),
debug console + raw wave preview (bottom-left), status/control rail (right).

### Protocol (JSON over WebSocket)

- renderer → backend: `list_boards`, `connect`, `disconnect`, `start`, `stop`,
  `save_image`, `print_image`
- backend → renderer: `boards`, `log`, `status`, `waves`, `state`

A `state` patch sets the viewport's visual parameters — during streaming these
arrive ~10 Hz (EMA-smoothed), plus one final averaged patch on **Stop**.

## Troubleshooting

**Nothing happens when I click Save Image.**
The backend refuses oversized WebSocket frames. `server.py` passes
`max_size=None` for exactly this reason — screenshots run to several MB and the
library's 1 MiB default silently killed the connection (close 1009) without
delivering the message. If you see `Renderer socket closed (sent): 1009` in the
`[py]` output, that cap is back.

**The UI doesn't reflect my code change.**
You're loading a stale `out/renderer/`. Use `npm run dev`, or `npm run build`
before `preview`.

**`electron.app` is undefined / Electron runs as plain Node.**
A host terminal leaked `ELECTRON_RUN_AS_NODE=1`. `npm run dev` strips it (see
`scripts/dev.mjs`) — launch through that script rather than `npx electron-vite dev`.

**Backend never comes up; the rail reads `connecting to backend…`.**
Check the `[py]` lines in the terminal. Usually the `.esp` venv is missing or
lacks the backend deps — re-run the pip install above, or set `BRAINART_PYTHON`.
If port 17321 is taken, set `BRAINART_WS_PORT`.

**Muse 2 won't connect.**
Make sure it's on, not connected to another app or the Muse mobile app, and that
Bluetooth is enabled. On Linux, BrainFlow's BLE backend may need elevated
permissions.

**The viewport flickers.**
The WebGL loop ends each frame with a 1px `gl.readPixels` (see
`src/renderer/src/gl/renderer.js`). This forces the GPU to resolve the frame
before Chromium's compositor presents it, eliminating a present-race.
`gl.finish()` alone was not enough on this driver — keep the `readPixels` line.
Relatedly, `preserveDrawingBuffer` is intentionally **off**, which is why
`screenshot()` draws a fresh frame immediately before `toDataURL()`.
