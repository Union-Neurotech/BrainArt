// src/App.tsx
import { useEffect, useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import BrainArtCanvas from "./components/BrainArtCanvas";
import RawEEGPlot from "./components/RawEEGPlot";
import "./App.css";

// Match the Rust struct
export interface EegState {
  valence: number;
  arousal: number;
  alpha: number;
  beta: number;
  theta: number;
  delta: number;
  gamma: number;
  mindfulness: number;
  concentration: number;
  relaxation: number;
  raw_waves: number[];
}

const BOARDS: {
  id: string;
  name: string;
  needsSerial?: boolean;
  needsMac?: boolean;
  serialHint?: string;
  macHint?: string;
}[] = [
  { id: "synthetic", name: "Synthetic Board" },
  {
    id: "muse_2016",
    name: "Muse 2016",
    needsMac: true,
    macHint: "XX:XX:XX:XX:XX:XX — leave blank for auto-discovery",
  },
  {
    id: "muse_2",
    name: "Muse 2 / Muse S",
    needsMac: true,
    macHint: "XX:XX:XX:XX:XX:XX — leave blank for auto-discovery",
  },
  {
    id: "cyton",
    name: "OpenBCI Cyton",
    needsSerial: true,
    serialHint: "e.g. COM3 (Windows) or /dev/ttyUSB0 (Linux)",
  },
];

const SLIDER_DEFS: {
  id: keyof EegState;
  label: string;
  hint: string;
  min: number;
  max: number;
  color: string;
  section: "Emotion" | "EEG Bands" | "ML Metrics";
}[] = [
  { id: "valence",       label: "Valence",      hint: "warm ↔ cool hue",    min: -1, max: 1, color: "#ff7846", section: "Emotion"    },
  { id: "arousal",       label: "Arousal",       hint: "zoom + structure",   min: -1, max: 1, color: "#ff7846", section: "Emotion"    },
  { id: "alpha",         label: "Alpha (α)",     hint: "radial smoothing",   min:  0, max: 1, color: "#50a0ff", section: "EEG Bands"  },
  { id: "beta",          label: "Beta (β)",      hint: "complexity + zoom",  min:  0, max: 1, color: "#50a0ff", section: "EEG Bands"  },
  { id: "theta",         label: "Theta (θ)",     hint: "spiral chaos",       min:  0, max: 1, color: "#50a0ff", section: "EEG Bands"  },
  { id: "delta",         label: "Delta (δ)",     hint: "animation speed",    min:  0, max: 1, color: "#50a0ff", section: "EEG Bands"  },
  { id: "gamma",         label: "Gamma (γ)",     hint: "saturation + warp",  min:  0, max: 1, color: "#50a0ff", section: "EEG Bands"  },
  { id: "mindfulness",   label: "Mindfulness",   hint: "damps speed/chaos",  min:  0, max: 1, color: "#64e696", section: "ML Metrics" },
  { id: "concentration", label: "Concentration", hint: "sharpens structure", min:  0, max: 1, color: "#64e696", section: "ML Metrics" },
  { id: "relaxation",    label: "Relaxation",    hint: "softens zoom",       min:  0, max: 1, color: "#64e696", section: "ML Metrics" },
];

const SECTIONS = ["Emotion", "EEG Bands", "ML Metrics"] as const;

// ── Slider row ────────────────────────────────────────────────────────────────
function SliderRow({
  id, label, hint, min, max, color, value, onChange, disabled,
}: {
  id: keyof EegState;
  label: string;
  hint: string;
  min: number;
  max: number;
  color: string;
  value: number;
  onChange: (key: keyof EegState, val: number) => void;
  disabled?: boolean;
}) {
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex justify-between items-baseline mb-0.5">
        <span className={`text-[10px] ${disabled ? "text-neutral-500" : "text-neutral-300"}`}>{label}</span>
        <span className="text-[10px] text-neutral-500 font-mono tabular-nums">{value.toFixed(2)}</span>
      </div>
      <div className="text-[9px] text-neutral-700 mb-1">{hint}</div>
      <input
        type="range"
        aria-label={label}
        disabled={disabled}
        min={min}
        max={max}
        step={0.01}
        value={value}
        onChange={(e) => onChange(id, parseFloat(e.target.value))}
        className={`eeg-slider eeg-slider-${color === "#ff7846" ? "emotion" : color === "#50a0ff" ? "eeg" : "ml"}`}
      />
    </div>
  );
}

// ── App ────────────────────────────────────────────────────────────────────────
function App() {
  const eegStateRef = useRef<EegState>({
    valence: 0, arousal: 0, alpha: 0, beta: 0, theta: 0,
    delta: 0, gamma: 0, mindfulness: 0, concentration: 0, relaxation: 0, raw_waves: [],
  });

  const [uiState, setUiState] = useState<EegState>(eegStateRef.current);
  const [selectedBoard, setSelectedBoard] = useState("synthetic");
  const [serialPort, setSerialPort] = useState("");
  const [macAddress, setMacAddress] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [showEEG, setShowEEG] = useState(false);
  const lastUiUpdateRef = useRef(0);
  const [isManual, setIsManual] = useState(false);
  const isManualRef = useRef(false);
  isManualRef.current = isManual; // keep ref in sync so the event closure sees the latest value

  useEffect(() => {
    if (!isConnected) return;

    invoke("connect_board", {
      request: {
        boardId: selectedBoard,
        serialPort: serialPort || null,
        macAddress: macAddress || null,
      },
    });

    const unlisten = listen<EegState>("eeg-data", (event) => {
      if (isManualRef.current) return; // manual mode: ignore incoming data
      eegStateRef.current = event.payload;
      // Throttle slider UI updates to ~10 FPS; canvas reads the ref directly so it's unaffected
      const now = performance.now();
      if (now - lastUiUpdateRef.current >= 100) {
        lastUiUpdateRef.current = now;
        setUiState(event.payload);
      }
    });

    return () => { unlisten.then((f) => f()); };
  }, [isConnected, selectedBoard]);

  const handleSlider = (key: keyof EegState, value: number) => {
    eegStateRef.current = { ...eegStateRef.current, [key]: value };
    setUiState((prev) => ({ ...prev, [key]: value }));
  };

  // ── Connect screen ───────────────────────────────────────────────────────────
  if (!isConnected) {
    const boardDef = BOARDS.find((b) => b.id === selectedBoard);
    return (
      <div className="h-screen flex items-center justify-center bg-black">
        <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-10 flex flex-col gap-6 w-96">
          <div>
            <h1 className="text-2xl font-bold tracking-widest uppercase text-white">BrainArt</h1>
            <p className="text-neutral-500 text-sm mt-1">emotion + EEG → realtime visual</p>
          </div>

          {/* Board selector */}
          <div className="flex flex-col gap-2">
            <label htmlFor="board-select" className="text-xs tracking-widest uppercase text-neutral-400">
              Board Type
            </label>
            <select
              id="board-select"
              value={selectedBoard}
              onChange={(e) => {
                setSelectedBoard(e.target.value);
                setSerialPort("");
                setMacAddress("");
              }}
              className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2.5 text-white text-sm outline-none focus:border-neutral-500 cursor-pointer"
            >
              {BOARDS.map((b) => (
                <option key={b.id} value={b.id}>{b.name}</option>
              ))}
            </select>
          </div>

          {/* Serial port — Cyton */}
          {boardDef?.needsSerial && (
            <div className="flex flex-col gap-2">
              <label htmlFor="serial-port" className="text-xs tracking-widest uppercase text-neutral-400">
                Serial Port
              </label>
              <input
                id="serial-port"
                type="text"
                placeholder={boardDef.serialHint}
                value={serialPort}
                onChange={(e) => setSerialPort(e.target.value)}
                className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2.5 text-white text-sm outline-none focus:border-neutral-500 placeholder:text-neutral-600 font-mono"
              />
            </div>
          )}

          {/* MAC address — Muse boards */}
          {boardDef?.needsMac && (
            <div className="flex flex-col gap-2">
              <label htmlFor="mac-address" className="text-xs tracking-widest uppercase text-neutral-400">
                MAC Address <span className="normal-case text-neutral-600">(optional)</span>
              </label>
              <input
                id="mac-address"
                type="text"
                placeholder={boardDef.macHint}
                value={macAddress}
                onChange={(e) => setMacAddress(e.target.value)}
                className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2.5 text-white text-sm outline-none focus:border-neutral-500 placeholder:text-neutral-600 font-mono"
              />
            </div>
          )}

          <button
            onClick={() => setIsConnected(true)}
            className="bg-white text-black font-semibold py-2.5 rounded-lg text-sm tracking-widest uppercase hover:bg-neutral-200 active:scale-95 transition-all"
          >
            Connect
          </button>
        </div>
      </div>
    );
  }

  // ── Main view ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-screen bg-black text-white overflow-hidden">
      {/* Main row: art canvas + sidebar */}
      <div className="flex flex-1 overflow-hidden min-h-0">

        {/* Art canvas — 16:9 aspect ratio, centered */}
        <div className="flex-1 flex items-center justify-center overflow-hidden bg-black">
          <div className="canvas-16-9">
            <BrainArtCanvas eegStateRef={eegStateRef} />
          </div>
        </div>

        {/* Right sidebar */}
        <div className="w-64 shrink-0 border-l border-neutral-800 flex flex-col bg-neutral-950">
          {/* Header */}
          <div className="px-4 py-3 border-b border-neutral-800 flex items-center justify-between shrink-0">
            <div className="flex flex-col">
              <span className="text-[11px] font-bold tracking-widest uppercase text-neutral-300">BrainArt</span>
              <span className="text-[9px] text-neutral-600 tracking-wide mt-0.5">
                {BOARDS.find((b) => b.id === selectedBoard)?.name}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setIsManual((v) => !v)}
                className={`text-[10px] tracking-widest uppercase transition-colors ${
                  isManual ? "text-amber-400 hover:text-amber-300" : "text-neutral-500 hover:text-neutral-300"
                }`}
              >
                {isManual ? "Manual" : "Auto"}
              </button>
              <button
                onClick={() => setIsConnected(false)}
                className="text-[10px] tracking-widest uppercase text-neutral-600 hover:text-neutral-300 transition-colors"
              >
                Disconnect
              </button>
            </div>
          </div>

          {/* Slider scroll area */}
          <div className="flex-1 overflow-y-auto px-4 py-3 flex flex-col gap-5">
            {SECTIONS.map((section) => (
              <div key={section}>
                <div className="text-[8px] tracking-widest uppercase text-neutral-600 mb-2">{section}</div>
                {SLIDER_DEFS.filter((s) => s.section === section).map((s) => (
                  <SliderRow
                    key={s.id}
                    id={s.id}
                    label={s.label}
                    hint={s.hint}
                    min={s.min}
                    max={s.max}
                    color={s.color}
                    value={uiState[s.id] as number}
                    onChange={handleSlider}
                    disabled={!isManual}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom EEG bar */}
      <div className="shrink-0 border-t border-neutral-800 bg-neutral-950">
        <button
          onClick={() => setShowEEG((v) => !v)}
          aria-label="Toggle raw EEG plot"
          className="w-full px-4 py-2 text-[10px] tracking-widest uppercase text-neutral-500 hover:text-neutral-300 transition-colors flex items-center justify-between"
        >
          <span>Raw EEG</span>
          <span>{showEEG ? "▼" : "▲"}</span>
        </button>
        {showEEG && (
          <div className="h-40">
            <RawEEGPlot eegStateRef={eegStateRef} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
