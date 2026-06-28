# BrainArt (Electron + Python)

Desktop BrainArt: an Electron front-end (WebGL CPPN viewport + status/control bar
+ debug console) driven by a Python backend that handles BrainFlow devices and
talks to the renderer over a local WebSocket.

## Architecture

```
Electron main (src/main/index.ts)
  ├─ spawns:  python ../src/server.py --port 17321
  └─ opens:   BrowserWindow (React + Ionic renderer)

renderer  ⇄  ws://127.0.0.1:17321  ⇄  Python backend (../src/server.py)
```

The backend reuses the existing project code: `communications.Comms`,
`preprocessing.get_simple_feature_vector`, `assets.board_id_pairs`.

## One-time setup

Python backend env (from the **project root**, one level up):

```sh
python -m venv .esp
.esp\Scripts\python -m pip install -r requirements-backend.txt   # Windows
# .esp/bin/python  -m pip install -r requirements-backend.txt    # macOS/Linux
```

The Electron main process auto-discovers `.esp` at the project root. Override the
interpreter with the `BRAINART_PYTHON` env var if needed.

Front-end deps:

```sh
cd brainart-electron
npm install
```

## Run (dev)

```sh
npm run dev
```

This launches Electron, which spawns the Python backend and opens the window.
Pick **Synthetic** → **Connect** → **Start** … **Stop** to drive the visuals
with no hardware. Save/Print write to `../generated/images/`.

> `npm run dev` strips a leaked `ELECTRON_RUN_AS_NODE` (see `scripts/dev.mjs`);
> without that, Electron would run as plain Node and `electron.app` would be
> undefined.

> **Viewport flicker fix:** the WebGL loop ends each frame with a 1px
> `gl.readPixels` (see `src/renderer/src/gl/renderer.js`). This forces the GPU to
> resolve the frame before Chromium's compositor presents it, eliminating a
> present-race that made the viewport flicker. `gl.finish()` alone was not enough
> on this driver — keep the `readPixels` line.

## Protocol (JSON over WebSocket)

- renderer → backend: `list_boards`, `connect`, `disconnect`, `start`, `stop`,
  `save_image`, `print_image`
- backend → renderer: `boards`, `log`, `status`, `waves`, `state`

A `state` patch (e.g. the averaged metrics computed on **Stop**) sets the
viewport's visual parameters — this is also how the backend can drive the sliders.
