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

function App() {
  // We use a ref for the WebGL canvas to avoid triggering React re-renders 30 times a second
  const eegStateRef = useRef<EegState>({
    valence: 0, arousal: 0, alpha: 0, beta: 0, theta: 0, delta: 0, gamma: 0, mindfulness: 0, concentration: 0, relaxation: 0, raw_waves: []
  });

  // State for UI updates (e.g., Sliders or Text displays)
  const [uiState, setUiState] = useState<EegState>(eegStateRef.current);

  useEffect(() => {
    // 1. Tell Rust to start the mock stream
    invoke("start_eeg_stream");

    // 2. Listen to the stream
    const unlisten = listen<EegState>("eeg-data", (event) => {
      // Update ref for high-speed WebGL reads
      eegStateRef.current = event.payload; 
      
      // Update state for UI components (throttle this in production if it causes lag)
      setUiState(event.payload);
    });

    return () => {
      unlisten.then((f) => f());
    };
  }, []);

  return (
    <div className="flex h-screen bg-neutral-900 text-white">
      {/* LEFT: Generative Art Canvas */}
      <div className="w-2/3 h-full relative">
        <BrainArtCanvas eegStateRef={eegStateRef} />
      </div>

      {/* RIGHT: UI Controls & Plot */}
      <div className="w-1/3 h-full border-l border-neutral-700 flex flex-col p-4">
        <h1 className="text-xl font-bold tracking-widest uppercase mb-4 text-neutral-300">
          BrainArt Control
        </h1>
        
        <div className="flex flex-col gap-2 mb-8">
          <div>Valence: {uiState.valence.toFixed(2)}</div>
          <div>Arousal: {uiState.arousal.toFixed(2)}</div>
        </div>

        <div className="flex-1 mt-auto bg-black rounded p-2">
           <RawEEGPlot eegStateRef={eegStateRef} />
        </div>
      </div>
    </div>
  );
}

export default App;