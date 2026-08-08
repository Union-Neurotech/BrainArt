# BrainArt → Electron + Python (WebSocket) migration

## Context

BrainArt today is a Streamlit web app (`src/browser_ui.py`) that connects to EEG
hardware via BrainFlow, computes an averaged feature vector, and generates a
NumPy CPPN image. Separately, `new_web_variant/` is a newer, much nicer **WebGL
CPPN viewport** (vanilla JS + GLSL) driven by a live state object. The goal is to
retire the Streamlit UI and ship a desktop app: an **Electron** front-end (the
WebGL viewport + a status/control bar + a debug console) that drives a **Python
backend** for all BrainFlow device handling.

Decisions confirmed with the user:
- **Transport:** Python runs a **WebSocket** server; the renderer connects with the
  native `WebSocket` API. One bidirectional JSON channel. (A renderer cannot speak
  raw UDP, so WebSocket is the "TCP" option that satisfies "backend can control the
  sliders.")
- **Emotion:** arousal/valence are **manual sliders** this iteration (no emotion
  model exists in Python yet), with a clean extension point for a future estimator.
- **Packaging:** **dev-run first** - `npm run dev` launches Electron, which spawns
  the existing system/venv Python backend. One-file installer is a later step.

Keep it simple: three basic React components (Viewport, StatusBar, DebugConsole)
plus a small WS client. No heavy abstractions.

## Architecture

```
+-----------------------------+----------+
|        Viewport (16:9)      |  Status  |   Electron main (src/main/index.ts):
|     WebGL CPPN canvas       |  /control|     - creates BrowserWindow
+-----------------------------+   bar    |     - spawns `python src/server.py --port 17321`
|        Debug console        | (full ht)|     - kills python on quit
|   logs  |  raw-wave preview |          |
+-----------------------------+----------+

Renderer  <--- ws://127.0.0.1:17321 --->  Python server.py
   (native WebSocket)                        (websockets lib; wraps Comms + preprocessing)
```

State flow mirrors the proven `brainart-tauri/src/App.tsx` pattern: a mutable
`stateRef` the GL loop reads every frame (no re-render), plus a `useState` copy for
UI. Both the user (dragging sliders) and Python (pushing a `state` patch) write to
this single source of truth, which feeds the shader uniforms.

## WebSocket protocol (JSON messages)

Renderer → Python:
- `{type:"list_boards"}`
- `{type:"connect", board:"Synthetic", port:"COM3"|null}`
- `{type:"disconnect"}`
- `{type:"start"}` / `{type:"stop"}`
- `{type:"save_image", png:"<base64>"}`
- `{type:"print_image", png:"<base64>"}`

Python → Renderer:
- `{type:"boards", boards:[{name,id,using_port}]}` - from `board_id_pairs`
- `{type:"log", level:"info|error", message}` - drives the debug console
- `{type:"status", connected, streaming, device}`
- `{type:"waves", channels:[[..],[..]]}` - live preview chunk (~15 Hz, last N
  samples per EEG channel, non-draining)
- `{type:"state", patch:{alpha,beta,delta,theta,gamma,concentration,mindfulness,
  relaxation,...}}` - **generic external slider/visual setter**; the averaged
  metrics after Stop arrive as one of these.

Port is a shared constant `17321` (main passes `--port`, renderer dials the same;
overridable via env). Renderer auto-reconnects if the socket drops.

## Python backend - `src/server.py` (new)

Reuses existing code directly; no rewrite of device logic:
- `src/communications.py` → `Comms` (connect/disconnect/start_stream/stop_stream).
  Use `board.get_current_board_data(n)` for the **non-draining** live preview, and
  `board.get_board_data()` once on Stop to drain the full window.
- `src/preprocessing.py` → `get_simple_feature_vector(data, boardID)` returns
  `[alpha, beta, delta, theta, gamma, concentration, mindfulness, relaxation]`
  (bands normalized to sum 1, ML metrics 0..1) - maps 1:1 onto the shader's 0..1
  slider state. Computed once over the whole Start→Stop window (the requested
  averaging) and emitted as a `state` patch.
- `src/assets.py` → `board_id_pairs` for the boards list and port requirement.

Server uses the `websockets` library (asyncio): one connection handler, a command
dispatch, and a background task that, while streaming, peeks recent samples and
emits `waves` on a timer. Save writes to `generated/images/brainart_<ts>.png`;
print saves then calls OS print (`os.startfile(path, "print")` on Windows) - a
clearly isolated `print_image()` function as the extension point.

Add `websockets` to `requirements.txt`.

## Electron app - scaffold in `brainart-electron/` (electron-vite, React + TS, Ionic)

UI is built from the **Ionic React** component framework (`@ionic/react`,
`ionicons`) - standard elements rather than hand-rolled CSS: `IonSelect`/
`IonSelectOption` (device dropdown), `IonButton` (Connect/Disconnect/Start/Stop/
Save/Print), `IonRange` (sliders), `IonItem`/`IonLabel`/`IonText`/`IonList`
(indicators + grouping), `IonProgressBar` (band/metric bars). Use Ionic's built-in
**dark palette** (`@ionic/react/css/palettes/dark.always.css`) for theming instead
of a custom theme. Only the canvas elements (WebGL viewport, raw-wave preview) and
the structural app-shell grid stay as plain elements.

```
brainart-electron/
  package.json, electron.vite.config.ts, tsconfig.json
  src/
    main/index.ts        spawn python child proc; window lifecycle; kill on quit
    preload/index.ts     minimal (no nodeIntegration in renderer)
    renderer/
      index.html
      src/
        main.tsx, App.tsx          IonApp shell + app-shell grid (see below); imports
                                   Ionic core CSS + dark palette
        state.ts                   BrainState type + defaults (mirror of E)
        ws.ts                      useBackend() WS hook: send() + event dispatch
        gl/cppn.js, gl/vertex.js   copied verbatim from new_web_variant/
        gl/renderer.js             createRenderer(canvas, stateRef): adapted from
                                   new_web_variant/brainart_emotion.js - keeps the
                                   exact uniform set + RAF loop, but reads
                                   stateRef.current instead of DOM; exposes
                                   setSize() and screenshot(); fixed 16:9 backing
                                   store (e.g. 1600x900) so screenshots are 16:9.
        components/
          Viewport.tsx             mounts canvas, runs renderer, exposes screenshot
          StatusBar.tsx            IonSelect device dropdown + IonButton
                                   Connect/Disconnect (color="danger" when connected)
                                   + Start/Stop; IonItem/IonProgressBar indicators
                                   (device, concentration, relaxation, meditative=
                                   mindfulness, 5 bands); IonRange manual sliders
                                   (arousal, valence, depth); IonButton Save/Print
          DebugConsole.tsx         left: scrolling IonList of log lines (from `log`);
                                   right: <RawWavePreview/>
          RawWavePreview.tsx       small canvas; each channel drawn as a polyline
                                   with a vertical offset (rolling buffer idea from
                                   brainart-tauri/src/components/RawEEGPlot.tsx)
```

App layout: `<IonApp>` wrapping a structural CSS grid
(`columns: 1fr 300px; rows: 1fr 160px;`) - Viewport at (1,1), DebugConsole at (2,1),
StatusBar spans both rows in column 2 (full height), matching the mockup. Theming is
Ionic's dark palette; only this grid + the two canvases use bespoke CSS.

Save Image: renderer `canvas.toDataURL('image/png')` → strip header → send
`save_image`. Print Image: same payload via `print_image`.

## Reuse summary
- Copy verbatim: `new_web_variant/cppn.js`, `new_web_variant/vertex.js`.
- Adapt (decouple from DOM): `new_web_variant/brainart_emotion.js` → `gl/renderer.js`.
- UI elements: Ionic React components (`@ionic/react`) + Ionic dark palette - no
  custom theme port from `brainart_emotion.html`.
- Python, used as-is: `Comms` (communications.py), `get_simple_feature_vector`
  (preprocessing.py), `board_id_pairs` (assets.py).
- React patterns to mirror: `brainart-tauri/src/App.tsx` (stateRef+uiState,
  SLIDER_DEFS/sections), `brainart-tauri/src/components/RawEEGPlot.tsx` (rolling buffer).
- Not needed: the `brainart-tauri/brainflow/` C++ clone (Python uses pip `brainflow`);
  `brainart-tauri/` itself is reference-only and stays untouched.

## Verification (Synthetic board - no hardware needed)
1. `pip install -r requirements.txt` (now includes `websockets`).
2. Backend alone: `python src/server.py --port 17321`; with a tiny WS test script,
   send `list_boards` → `connect Synthetic` → `start` (see `waves`) → `stop` (see a
   `state` patch with the 8 metrics).
3. `cd brainart-electron && npm install && npm run dev` - Electron opens, main
   spawns python, renderer reports connected in the console.
4. End-to-end in the app: select **Synthetic** → **Connect** (console logs success,
   button turns red "Disconnect") → **Start** (preview traces animate, visual
   responds) → **Stop** (indicators/sliders snap to averaged metrics, visual
   updates) → **Save Image** (PNG in `generated/images/`) → **Print Image**.
5. Confirm the python child process is terminated when the window is closed.

## Out of scope (future, clearly separable)
- One-file packaging: PyInstaller (backend) + electron-builder (installer).
- Real arousal/valence estimation (manual sliders for now).
- Real-hardware port selection UX beyond a text field for boards that need a port.
```
