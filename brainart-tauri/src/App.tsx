// src/App.tsx
import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import BrainArtCanvas from "./components/BrainArtCanvas";
import RawEEGPlot from "./components/RawEEGPlot";
import PSDPlot from "./components/PSDPlot";
import "./App.css";

// Muse electrode order as broadcast by brainart_muse_server.py (matches its reorder)
export const CHANNEL_LABELS = ["AF7", "AF8", "TP9", "TP10"] as const;
const RAW_BUFFER = 256 * 5; // 5 s rolling window of raw samples per channel

export interface PsdData {
  freqs: number[];        // Hz, 0..45
  chans: number[][];      // dB power, one array per CHANNEL_LABELS entry
}

// Matches the JSON broadcast by emotion_pipeline/brainart_muse_server.py
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
  channels: number[][];   // rolling raw EEG, one array per CHANNEL_LABELS entry (µV)
  psd: PsdData | null;     // latest power spectrum
}

const WS_URL = "ws://localhost:8765";

// Keys the live server drives (the rest stay manual sliders).
const LIVE_KEYS = ["valence", "arousal", "alpha", "beta", "theta", "delta", "gamma"] as const;

const BOARDS = [
  { id: "synthetic", name: "Synthetic (no hardware)" },
  { id: "muse_2", name: "Muse 2" },
  { id: "muse_2016", name: "Muse 2016" },
  { id: "muse_s", name: "Muse S" },
];

// Calibration conditions span the valence/arousal corners.
const CALIB_CONDITIONS = [
  { label: "exciting / joy", v: 0.8, a: 0.8 },
  { label: "fear / anger", v: -0.8, a: 0.8 },
  { label: "calm / tender", v: 0.7, a: -0.6 },
  { label: "sad", v: -0.7, a: -0.5 },
  { label: "neutral", v: 0.0, a: -0.2 },
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
  { id: "valence", label: "Valence", hint: "warm ↔ cool hue", min: -1, max: 1, color: "#ff7846", section: "Emotion" },
  { id: "arousal", label: "Arousal", hint: "zoom + structure", min: -1, max: 1, color: "#ff7846", section: "Emotion" },
  { id: "alpha", label: "Alpha (α)", hint: "radial smoothing", min: 0, max: 1, color: "#50a0ff", section: "EEG Bands" },
  { id: "beta", label: "Beta (β)", hint: "complexity + zoom", min: 0, max: 1, color: "#50a0ff", section: "EEG Bands" },
  { id: "theta", label: "Theta (θ)", hint: "spiral chaos", min: 0, max: 1, color: "#50a0ff", section: "EEG Bands" },
  { id: "delta", label: "Delta (δ)", hint: "animation speed", min: 0, max: 1, color: "#50a0ff", section: "EEG Bands" },
  { id: "gamma", label: "Gamma (γ)", hint: "saturation + warp", min: 0, max: 1, color: "#50a0ff", section: "EEG Bands" },
  { id: "mindfulness", label: "Mindfulness", hint: "damps speed/chaos", min: 0, max: 1, color: "#64e696", section: "ML Metrics" },
  { id: "concentration", label: "Concentration", hint: "sharpens structure", min: 0, max: 1, color: "#64e696", section: "ML Metrics" },
  { id: "relaxation", label: "Relaxation", hint: "softens zoom", min: 0, max: 1, color: "#64e696", section: "ML Metrics" },
];

const SECTIONS = ["Emotion", "EEG Bands", "ML Metrics"] as const;

const STATUS_COLOR: Record<string, string> = {
  off: "#555", connecting: "#888", baseline: "#ffb454",
  live: "#64e696", sim: "#50a0ff", error: "#f55",
};

// ── Slider row ──────────────────────────────────────────────────────────────
function SliderRow({
  id, label, hint, min, max, color, value, disabled, onChange,
}: {
  id: keyof EegState; label: string; hint: string; min: number; max: number;
  color: string; value: number; disabled: boolean;
  onChange: (key: keyof EegState, val: number) => void;
}) {
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex justify-between items-baseline mb-0.5">
        <span className="text-[10px] text-neutral-300">{label}</span>
        <span className="text-[10px] text-neutral-500 font-mono tabular-nums">{value.toFixed(2)}</span>
      </div>
      <div className="text-[9px] text-neutral-700 mb-1">{hint}</div>
      <input
        type="range" aria-label={label} min={min} max={max} step={0.01}
        value={value} disabled={disabled}
        onChange={(e) => onChange(id, parseFloat(e.target.value))}
        className={`eeg-slider eeg-slider-${color === "#ff7846" ? "emotion" : color === "#50a0ff" ? "eeg" : "ml"} ${disabled ? "opacity-40" : ""}`}
      />
    </div>
  );
}

// ── App ─────────────────────────────────────────────────────────────────────
function App() {
  const eegStateRef = useRef<EegState>({
    valence: 0, arousal: 0, alpha: 0, beta: 0, theta: 0,
    delta: 0, gamma: 0, mindfulness: 0, concentration: 0, relaxation: 0,
    channels: CHANNEL_LABELS.map(() => [] as number[]), psd: null,
  });
  const targetRef = useRef<EegState>({ ...eegStateRef.current });
  const wsRef = useRef<WebSocket | null>(null);
  const lastSeqRef = useRef(-1); // dedupe raw chunks by eeg_seq

  const [uiState, setUiState] = useState<EegState>(eegStateRef.current);
  const [selectedBoard, setSelectedBoard] = useState("synthetic");
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus] = useState("off");
  const [calibrated, setCalibrated] = useState(false);
  const [showEEG, setShowEEG] = useState(false);

  // calibration UI
  const [calibBusy, setCalibBusy] = useState(false);
  const [calibDone, setCalibDone] = useState<Set<string>>(new Set());
  const [calibMsg, setCalibMsg] = useState("go live, then record each condition while playing a clip");

  const live = isConnected && (status === "live" || status === "baseline" || status === "sim");

  // ── start server + connect WebSocket ───────────────────────────────────────
  useEffect(() => {
    if (!isConnected) return;
    let mounted = true;
    setStatus("connecting");

    invoke("start_server", { boardId: selectedBoard }).catch((e) => {
      console.error(e); if (mounted) setStatus("error");
    });

    const connect = () => {
      if (!mounted) return;
      let ws: WebSocket;
      try { ws = new WebSocket(WS_URL); } catch { setTimeout(connect, 1200); return; }
      wsRef.current = ws;
      ws.onmessage = (ev) => {
        let d: Record<string, unknown>;
        try { d = JSON.parse(ev.data as string); } catch { return; }
        for (const k of LIVE_KEYS) {
          if (typeof d[k] === "number") (targetRef.current as Record<string, number>)[k] = d[k] as number;
        }
        if (typeof d.status === "string") setStatus(d.status as string);
        if (typeof d.calibrated === "boolean") setCalibrated(d.calibrated as boolean);

        // raw 4-channel EEG: append each chunk once (deduped by eeg_seq) into a rolling buffer
        if (d.eeg && typeof d.eeg_seq === "number" && d.eeg_seq !== lastSeqRef.current) {
          lastSeqRef.current = d.eeg_seq;
          const eeg = d.eeg as Record<string, number[]>;
          const buf = eegStateRef.current.channels;
          CHANNEL_LABELS.forEach((lab, i) => {
            const chunk = eeg[lab];
            if (!Array.isArray(chunk)) return;
            const b = buf[i];
            for (let j = 0; j < chunk.length; j++) b.push(chunk[j]);
            if (b.length > RAW_BUFFER) b.splice(0, b.length - RAW_BUFFER);
          });
        }

        // power spectral density (per channel, dB)
        if (d.psd && typeof d.psd === "object") {
          const p = d.psd as Record<string, number[]>;
          if (Array.isArray(p.freqs)) {
            eegStateRef.current.psd = {
              freqs: p.freqs,
              chans: CHANNEL_LABELS.map((lab) => (Array.isArray(p[lab]) ? p[lab] : [])),
            };
          }
        }
      };
      ws.onclose = () => { if (mounted && isConnected) setTimeout(connect, 1200); };
      ws.onerror = () => ws.close();
    };
    // give the Python server a moment to boot before the first attempt
    const boot = setTimeout(connect, 600);

    return () => {
      mounted = false;
      clearTimeout(boot);
      if (wsRef.current) { wsRef.current.onclose = null; wsRef.current.close(); wsRef.current = null; }
      invoke("stop_server").catch(() => {});
      setStatus("off");
    };
  }, [isConnected, selectedBoard]);

  // ── per-frame easing of display state toward live targets ──────────────────
  useEffect(() => {
    let raf = 0; let frame = 0; const SMOOTH = 0.06;
    const tick = () => {
      const cur = eegStateRef.current as Record<string, number>;
      const tgt = targetRef.current as Record<string, number>;
      for (const k of LIVE_KEYS) cur[k] += (tgt[k] - cur[k]) * SMOOTH;
      if (++frame % 6 === 0) setUiState({ ...eegStateRef.current }); // ~10 Hz UI refresh
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // manual sliders (drive both refs so the ease loop is a no-op when not live)
  const handleSlider = (key: keyof EegState, value: number) => {
    (eegStateRef.current as Record<string, number>)[key] = value;
    (targetRef.current as Record<string, number>)[key] = value;
    setUiState((prev) => ({ ...prev, [key]: value }));
  };

  // ── calibration ────────────────────────────────────────────────────────────
  const sendCmd = (obj: unknown) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify(obj)); return true; }
    return false;
  };

  const recordCondition = (cond: (typeof CALIB_CONDITIONS)[number]) => {
    if (calibBusy) return;
    const secs = selectedBoard === "synthetic" ? 8 : 90;
    if (!sendCmd({ cmd: "calib_clip", label: cond.label, v: cond.v, a: cond.a, secs })) {
      setCalibMsg("not connected — go live first"); return;
    }
    setCalibBusy(true);
    let left = secs;
    setCalibMsg(`recording "${cond.label}" … ${left}s (play the clip now)`);
    const iv = setInterval(() => {
      left -= 1;
      if (left > 0) { setCalibMsg(`recording "${cond.label}" … ${left}s (play the clip now)`); return; }
      clearInterval(iv);
      setCalibBusy(false);
      setCalibDone((prev) => {
        const n = new Set(prev); n.add(cond.label);
        setCalibMsg(`captured ${n.size}/${CALIB_CONDITIONS.length}. Fit & save when ≥2 done.`);
        return n;
      });
    }, 1000);
  };

  const fitCalib = () => {
    if (calibDone.size < 2) { setCalibMsg("record at least 2 conditions first"); return; }
    if (sendCmd({ cmd: "calib_finish" })) setCalibMsg("fitted & saved — output now calibrated");
  };
  const resetCalib = () => {
    sendCmd({ cmd: "calib_reset" });
    setCalibDone(new Set());
    setCalibMsg("calibration reset");
  };

  // ── Connect screen ─────────────────────────────────────────────────────────
  if (!isConnected) {
    return (
      <div className="h-screen flex items-center justify-center bg-black">
        <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-10 flex flex-col gap-6 w-96">
          <div>
            <h1 className="text-2xl font-bold tracking-widest uppercase text-white">BrainArt</h1>
            <p className="text-neutral-500 text-sm mt-1">live Muse → valence/arousal → visual</p>
          </div>
          <div className="flex flex-col gap-2">
            <label htmlFor="board-select" className="text-xs tracking-widest uppercase text-neutral-400">Board</label>
            <select
              id="board-select" value={selectedBoard}
              onChange={(e) => setSelectedBoard(e.target.value)}
              className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2.5 text-white text-sm outline-none focus:border-neutral-500 cursor-pointer"
            >
              {BOARDS.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}
            </select>
          </div>
          <button
            onClick={() => setIsConnected(true)}
            className="bg-white text-black font-semibold py-2.5 rounded-lg text-sm tracking-widest uppercase hover:bg-neutral-200 active:scale-95 transition-all"
          >
            Connect
          </button>
          <p className="text-[10px] text-neutral-600 leading-relaxed">
            Launches the Python inference server for the chosen board. A real Muse
            runs a ~90 s rest baseline first (sit still). Synthetic needs no hardware.
          </p>
        </div>
      </div>
    );
  }

  // ── Main view ──────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-screen bg-black text-white overflow-hidden">
      <div className="flex flex-1 overflow-hidden min-h-0">
        <div className="flex-1 flex items-center justify-center overflow-hidden bg-black">
          <div className="canvas-16-9">
            <BrainArtCanvas eegStateRef={eegStateRef} />
          </div>
        </div>

        {/* Right sidebar */}
        <div className="w-64 shrink-0 border-l border-neutral-800 flex flex-col bg-neutral-950">
          <div className="px-4 py-3 border-b border-neutral-800 flex items-center justify-between shrink-0">
            <div className="flex flex-col">
              <span className="text-[11px] font-bold tracking-widest uppercase text-neutral-300">BrainArt</span>
              <span className="text-[9px] text-neutral-600 tracking-wide mt-0.5 flex items-center gap-1.5">
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: STATUS_COLOR[status] || "#555", display: "inline-block" }} />
                {status === "baseline" ? "baseline… sit still" : status}{calibrated ? " · calibrated" : ""}
              </span>
            </div>
            <button
              onClick={() => setIsConnected(false)}
              className="text-[10px] tracking-widest uppercase text-neutral-600 hover:text-neutral-300 transition-colors"
            >
              Disconnect
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-3 flex flex-col gap-5">
            {SECTIONS.map((section) => (
              <div key={section}>
                <div className="text-[8px] tracking-widest uppercase text-neutral-600 mb-2">{section}</div>
                {SLIDER_DEFS.filter((s) => s.section === section).map((s) => (
                  <SliderRow
                    key={s.id} id={s.id} label={s.label} hint={s.hint}
                    min={s.min} max={s.max} color={s.color}
                    value={uiState[s.id] as number}
                    disabled={live && (LIVE_KEYS as readonly string[]).includes(s.id as string)}
                    onChange={handleSlider}
                  />
                ))}
              </div>
            ))}

            {/* Calibration */}
            <div>
              <div className="text-[8px] tracking-widest uppercase text-neutral-600 mb-2">
                Calibration <span className="text-neutral-700">(fills −1…1)</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {CALIB_CONDITIONS.map((c) => (
                  <button
                    key={c.label} onClick={() => recordCondition(c)} disabled={!live || calibBusy}
                    title={`target valence ${c.v}, arousal ${c.a}`}
                    className={`text-[9px] px-2 py-1 rounded-full border transition-colors ${
                      calibDone.has(c.label)
                        ? "border-emerald-500/50 bg-emerald-500/15 text-emerald-300"
                        : "border-neutral-700 bg-neutral-900 text-neutral-400 hover:text-white"
                    } ${(!live || calibBusy) ? "opacity-40" : ""}`}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
              <div className="flex gap-1.5 mt-2">
                <button
                  onClick={fitCalib} disabled={!live || calibDone.size < 2}
                  className="flex-1 text-[10px] py-1.5 rounded border border-neutral-700 bg-neutral-900 text-neutral-300 hover:text-white disabled:opacity-40"
                >
                  fit &amp; save
                </button>
                <button
                  onClick={resetCalib} disabled={!live}
                  className="text-[10px] px-2.5 py-1.5 rounded border border-neutral-700 bg-neutral-900 text-neutral-400 hover:text-white disabled:opacity-40"
                >
                  reset
                </button>
              </div>
              <div className="text-[9px] text-neutral-600 mt-2 leading-relaxed">{calibMsg}</div>
            </div>
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
          <div className="flex h-44">
            <div className="flex-1 min-w-0 border-r border-neutral-800">
              <RawEEGPlot eegStateRef={eegStateRef} />
            </div>
            <div className="flex-1 min-w-0">
              <PSDPlot eegStateRef={eegStateRef} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
