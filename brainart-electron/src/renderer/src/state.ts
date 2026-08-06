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
  // `mindfulness` is retired: concentration is now backed by the same BrainFlow
  // MINDFULNESS metric, so the two were duplicates. Re-add it here (and in
  // METRIC_KEYS + get_simple_feature_vector) to bring the reading back.
  concentration: number
  relaxation: number
  layers: number
  /**
   * Which metric drives the "focus" effect (sharpens structure, damps chaos and
   * speed). BrainFlow's RESTFULNESS is the exact complement of its MINDFULNESS
   * — verified bit-exact over random band-power vectors — so concentration and
   * relaxation are one signal, and this selects which end of it reads as
   * "focused". Not a metric; the backend never patches it.
   */
  focusDriver: FocusDriver
}

export type FocusDriver = 'concentration' | 'relaxation'

/**
 * Keys of BrainState that hold numbers — i.e. everything the sliders and the
 * backend's metric patches may write. Excludes `focusDriver`, so a slider can't
 * be pointed at it by mistake.
 */
export type NumericStateKey = {
  [K in keyof BrainState]: BrainState[K] extends number ? K : never
}[keyof BrainState]

export const DEFAULT_STATE: BrainState = {
  valence: 0,
  arousal: 0,
  alpha: 0.5,
  beta: 0.4,
  theta: 0.3,
  delta: 0.2,
  gamma: 0.3,
  concentration: 0.5,
  relaxation: 0.5,
  layers: 4,
  focusDriver: 'concentration'
}
