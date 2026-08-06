// Adapted from new_web_variant/brainart_emotion.js — the exact CPPN uniform set
// and render loop, but decoupled from the DOM: it reads its visual parameters
// from `stateRef.current` each frame, renders into a fixed 16:9 backing store
// (so screenshots are always 16:9), and exposes a tiny imperative API.

import { VERTEX_SHADER as VS } from './vertex.js'
import { FS_CPPN } from './cppn.js'

const NEUTRAL_MODE = true        // false => legacy hueRot/warm-cool look
const BACKING_W = 1600
const BACKING_H = 900            // 16:9

// Seed persists across re-inits (HMR, an extra effect run, remounts) so the
// artwork stays stable and any transient double-init renders the *same* image.
// Only newSeed() changes it.
let persistedSeed = Math.random() * 20

export function createRenderer(canvas, stateRef) {
  // Guarantee exactly one renderer / RAF loop per canvas. A second init that
  // left the previous loop running (each with its own seed) is what makes the
  // canvas strobe between two entirely different images. Tear down any prior
  // renderer bound to this canvas before starting a new one.
  canvas.__brainartRenderer?.destroy()

  // preserveDrawingBuffer is intentionally OFF: with it on, Chromium/Electron uses
  // a blit-present path that races the compositor and flickers. We redraw the whole
  // frame every tick anyway, so the default (flip) path is stable; screenshots draw
  // one fresh frame synchronously right before toDataURL (see screenshot()).
  const gl =
    canvas.getContext('webgl', { antialias: false }) ||
    canvas.getContext('experimental-webgl')
  if (!gl) throw new Error('WebGL not supported')

  function compileShader(type, src) {
    const sh = gl.createShader(type)
    gl.shaderSource(sh, src)
    gl.compileShader(sh)
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(sh))
    }
    return sh
  }

  const prog = gl.createProgram()
  gl.attachShader(prog, compileShader(gl.VERTEX_SHADER, VS))
  gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, FS_CPPN))
  gl.linkProgram(prog)
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error('LINK:', gl.getProgramInfoLog(prog))
  }
  gl.useProgram(prog)

  const quadBuf = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf)
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW)
  const aLoc = gl.getAttribLocation(prog, 'a')
  gl.enableVertexAttribArray(aLoc)
  gl.vertexAttribPointer(aLoc, 2, gl.FLOAT, false, 0, 0)

  const UL = {}
  ;[
    'res', 't', 'seed', 'u_zoom', 'u_warp', 'u_speed', 'u_sat',
    'u_hue_shift', 'u_warm', 'u_complexity', 'u_chaos', 'u_radial',
    'u_offset_x', 'u_offset_y', 'u_mouse', 'u_mouse_str', 'u_neutral',
    'u_valence', 'u_arousal', 'u_layers'
  ].forEach((n) => (UL[n] = gl.getUniformLocation(prog, n)))

  // Fixed 16:9 backing store. CSS scales the canvas to fit; output stays 16:9.
  canvas.width = BACKING_W
  canvas.height = BACKING_H
  gl.viewport(0, 0, BACKING_W, BACKING_H)
  gl.uniform2f(UL.res, BACKING_W, BACKING_H)

  // Maps the live brain/emotion state to shader-friendly visual params.
  function emotionToVisuals() {
    const E = stateRef.current
    const v = E.valence, a = E.arousal, absV = Math.abs(v)
    // The "focus" signal sharpens structure and calms motion. Which metric plays
    // that role is selectable (E.focusDriver) because BrainFlow's RESTFULNESS is
    // the exact complement of its MINDFULNESS -- relaxation === 1 - concentration
    // -- so the choice flips which end of that single axis reads as "focused".
    const focus = E.focusDriver === 'relaxation' ? E.relaxation : E.concentration
    // Ceiling is 0.85 rather than 1.0 so that even at zero focus the motion stays
    // watchable instead of frantic -- raise toward 1.0 for a livelier idle.
    const damp = 0.85 - focus * 0.45
    const sharp = focus * 0.5
    const soft = E.relaxation * 0.4
    const shift_damper = 0.5
    const ap = (a + 1) * 0.5
    return {
      hue_shift: v * shift_damper,
      warm: v,
      sat: 1.0 + absV * 1.2 + E.gamma * 1.5,
      zoom: 0.7 + (1 - ap) * 0.6 + (1 - E.beta) * 0.4 + soft,
      complexity: 0.15 + ap * 1.1 + E.beta * 0.3 + sharp,
      // chaos/speed coefficients trimmed (were 1.8 and 0.02/0.12/0.06) so the
      // low-concentration end reads as calm rather than frantic.
      chaos: E.theta * 1.25 * damp,
      radial: E.alpha * 0.85,
      speed: (0.015 + E.delta * 0.08 + Math.max(a, 0) * 0.05) * damp,
      warp: 0.3 + E.gamma * 0.6 + ap * 0.4,
      offset_x: 0,
      offset_y: 0
    }
  }

  let animOn = true
  let t0 = performance.now()
  let tOffset = 0
  let seed = persistedSeed
  let raf = 0
  const mouse = { x: 0, y: 0, str: 0, down: false }
  const syncPixel = new Uint8Array(4) // reused readback target (see drawFrame)

  // Draw a single frame at the given elapsed time. No RAF scheduling here, so it
  // can also be called synchronously by screenshot() (needed now that
  // preserveDrawingBuffer is off — the buffer is only valid right after a draw).
  function drawFrame(elapsed) {
    if (!mouse.down) mouse.str = Math.max(0, mouse.str - 0.025)

    const E = stateRef.current
    const V = emotionToVisuals()
    gl.uniform1f(UL.t, elapsed)
    gl.uniform1f(UL.seed, seed)
    gl.uniform1f(UL.u_zoom, V.zoom)
    gl.uniform1f(UL.u_warp, V.warp)
    gl.uniform1f(UL.u_speed, V.speed)
    gl.uniform1f(UL.u_sat, V.sat)
    gl.uniform1f(UL.u_hue_shift, V.hue_shift)
    gl.uniform1f(UL.u_warm, V.warm)
    gl.uniform1f(UL.u_complexity, V.complexity)
    gl.uniform1f(UL.u_chaos, V.chaos)
    gl.uniform1f(UL.u_radial, V.radial)
    gl.uniform1f(UL.u_offset_x, V.offset_x)
    gl.uniform1f(UL.u_offset_y, V.offset_y)
    gl.uniform2f(UL.u_mouse, mouse.x, mouse.y)
    gl.uniform1f(UL.u_mouse_str, mouse.str)
    gl.uniform1f(UL.u_neutral, NEUTRAL_MODE ? 1.0 : 0.0)
    gl.uniform1f(UL.u_valence, E.valence)
    gl.uniform1f(UL.u_arousal, E.arousal)
    gl.uniform1f(UL.u_layers, E.layers)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)

    // Force the GPU to finish AND resolve this frame before the RAF callback yields
    // and the compositor presents the canvas. Otherwise Electron/Chromium presents
    // the texture before the draw completes — a present race that shows as the
    // viewport flickering between partial/garbled frames. On this GPU/driver,
    // gl.finish() alone does NOT prevent it; a 1px readPixels (which forces a real
    // buffer resolve/readback) is what reliably does. Keep this line.
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, syncPixel)
  }

  function elapsedNow(ts) {
    return animOn ? (ts - t0) / 1000 : tOffset
  }

  function render(ts) {
    drawFrame(elapsedNow(ts))
    raf = requestAnimationFrame(render)
  }
  raf = requestAnimationFrame(render)

  // --- mouse distortion (optional flourish, scoped to the canvas) ---
  const ac = new AbortController()
  const sig = { signal: ac.signal }
  function toNDC(cx, cy) {
    const r = canvas.getBoundingClientRect()
    return [
      ((cx - r.left) / r.width * 2 - 1) * (BACKING_W / BACKING_H),
      -((cy - r.top) / r.height * 2 - 1)
    ]
  }
  canvas.addEventListener('mousemove', (e) => {
    ;[mouse.x, mouse.y] = toNDC(e.clientX, e.clientY)
    if (mouse.down) mouse.str = Math.min(mouse.str + 0.07, 1.2)
  }, sig)
  canvas.addEventListener('mousedown', () => { mouse.down = true; mouse.str = 0.35 }, sig)
  canvas.addEventListener('mouseup', () => { mouse.down = false }, sig)
  canvas.addEventListener('mouseleave', () => { mouse.down = false }, sig)

  function newSeed() {
    persistedSeed = Math.random() * 20
    seed = persistedSeed
    t0 = performance.now()
    tOffset = 0
    return seed
  }

  const api = {
    screenshot: () => {
      drawFrame(elapsedNow(performance.now())) // fresh frame, then read it immediately
      return canvas.toDataURL('image/png')
    },
    newSeed,
    setAnim(on) {
      animOn = on
      if (on) t0 = performance.now() - tOffset * 1000
      else tOffset = (performance.now() - t0) / 1000
    },
    destroy() {
      cancelAnimationFrame(raf)
      raf = 0
      ac.abort()
      if (canvas.__brainartRenderer === api) canvas.__brainartRenderer = null
    }
  }

  canvas.__brainartRenderer = api

  // When Vite hot-swaps this module, cancel this loop so loops never stack.
  if (import.meta.hot) import.meta.hot.dispose(() => api.destroy())

  return api
}
