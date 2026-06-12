// Mirror of the new exports added to App.tsx, for isolated type-checking.
export const CHANNEL_LABELS = ["AF7", "AF8", "TP9", "TP10"] as const;

export interface PsdData {
  freqs: number[];
  chans: number[][];
}

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
  channels: number[][];
  psd: PsdData | null;
}
