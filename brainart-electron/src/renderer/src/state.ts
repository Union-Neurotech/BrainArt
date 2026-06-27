// Single source of truth for the visual params. Mirrors the `E` object from
// new_web_variant/brainart_emotion.js. Both the user (sliders) and the Python
// backend (state patches) write to this; the GL loop reads it every frame.

export interface BrainState {
  valence: number
  arousal: number
  alpha: number
  beta: number
  theta: number
  delta: number
  gamma: number
  mindfulness: number
  concentration: number
  relaxation: number
  layers: number
}

export const DEFAULT_STATE: BrainState = {
  valence: 0,
  arousal: 0,
  alpha: 0.5,
  beta: 0.4,
  theta: 0.3,
  delta: 0.2,
  gamma: 0.3,
  mindfulness: 0.5,
  concentration: 0.5,
  relaxation: 0.5,
  layers: 4
}
