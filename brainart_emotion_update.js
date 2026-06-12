let neutralMode = true;   // set false to allow full CPPN color bias

const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl', { preserveDrawingBuffer: true, antialias: false })
         || canvas.getContext('experimental-webgl', { preserveDrawingBuffer: true });

if (!gl) {
  document.body.innerHTML = '<p style="color:#f55;padding:2rem">WebGL required</p>';
  throw new Error('WebGL not supported');
}

import { VERTEX_SHADER as VS } from './vertex.js';
import { FS_CPPN } from './cppn.js';

// ------- WebGL setup -------
function compileShader(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    document.body.insertAdjacentHTML('afterbegin',
      `<pre style="position:fixed;z-index:99;top:0;left:0;max-width:60vw;color:#f55;background:#000;padding:1rem;white-space:pre-wrap;font:11px monospace">${log}</pre>`);
    console.error(log);
  }
  return sh;
}

const prog = gl.createProgram();
gl.attachShader(prog, compileShader(gl.VERTEX_SHADER,   VS));
gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, FS_CPPN));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
  console.error('LINK:', gl.getProgramInfoLog(prog));
gl.useProgram(prog);

const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

const aLoc = gl.getAttribLocation(prog, 'a');
gl.enableVertexAttribArray(aLoc);
gl.vertexAttribPointer(aLoc, 2, gl.FLOAT, false, 0, 0);

const UL = {};
['res','t','seed','u_zoom','u_warp','u_speed','u_sat',
 'u_hue_shift','u_warm','u_complexity','u_chaos','u_radial',
 'u_offset_x','u_offset_y','u_mouse','u_mouse_str','u_neutral',
 'u_valence','u_arousal','u_layers']
  .forEach(n => UL[n] = gl.getUniformLocation(prog, n));

// ------- State -------
const E = {
  valence: 0, arousal: 0,
  alpha: 0.5, beta: 0.4, theta: 0.3, delta: 0.2, gamma: 0.3,
  mindfulness: 0.5, concentration: 0.5, relaxation: 0.5,
  layers: 4,
};

function emotionToVisuals() {
  const v     = E.valence, a = E.arousal, absV = Math.abs(v);
  const damp  = 1 - E.mindfulness * 0.6;
  const sharp = E.concentration * 0.5;
  const soft  = E.relaxation * 0.4;
  const shift_damper = 0.5;
  const ap = (a + 1) * 0.5;   // arousal 0..1
  return {
    hue_shift:  v * shift_damper,
    warm:       v,
    sat:        1.0 + absV * 1.2 + E.gamma * 1.5,
    zoom:       0.7 + (1 - ap) * 0.6 + (1 - E.beta) * 0.4 + soft,
    complexity: 0.15 + ap * 1.1 + E.beta * 0.3 + sharp,
    chaos:      E.theta * 1.8 * damp,
    radial:     E.alpha * 0.85,
    speed:      (0.02 + E.delta * 0.12 + Math.max(a, 0) * 0.06) * damp,
    warp:       0.3 + E.gamma * 0.6 + ap * 0.4,
    offset_x:   0,
    offset_y:   0,
  };
}

// ------- Resize -------
let canvasW = 0, canvasH = 0;

function resize() {
  const wrap = document.getElementById('canvas-wrap');
  canvasW = wrap.clientWidth;
  canvasH = wrap.clientHeight;
  canvas.width  = canvasW;
  canvas.height = canvasH;
  gl.viewport(0, 0, canvasW, canvasH);
  gl.uniform2f(UL.res, canvasW, canvasH);
}
window.addEventListener('resize', resize);
resize();

// ------- Render loop -------
let animOn = true, t0 = performance.now(), tOffset = 0, seed = 5;
const mouse = { x: 0, y: 0, str: 0, down: false };

function render(ts) {
  const elapsed = animOn ? (ts - t0) / 1000 : tOffset;
  if (!mouse.down) mouse.str = Math.max(0, mouse.str - .025);

  const V = emotionToVisuals();
  gl.uniform1f(UL.t,            elapsed);
  gl.uniform1f(UL.seed,         seed);
  gl.uniform1f(UL.u_zoom,       V.zoom);
  gl.uniform1f(UL.u_warp,       V.warp);
  gl.uniform1f(UL.u_speed,      V.speed);
  gl.uniform1f(UL.u_sat,        V.sat);
  gl.uniform1f(UL.u_hue_shift,  V.hue_shift);
  gl.uniform1f(UL.u_warm,       V.warm);
  gl.uniform1f(UL.u_complexity, V.complexity);
  gl.uniform1f(UL.u_chaos,      V.chaos);
  gl.uniform1f(UL.u_radial,     V.radial);
  gl.uniform1f(UL.u_offset_x,   V.offset_x);
  gl.uniform1f(UL.u_offset_y,   V.offset_y);
  gl.uniform2f(UL.u_mouse,      mouse.x, mouse.y);
  gl.uniform1f(UL.u_mouse_str,  mouse.str);
  gl.uniform1f(UL.u_neutral,    neutralMode ? 1.0 : 0.0);
  gl.uniform1f(UL.u_valence,    E.valence);
  gl.uniform1f(UL.u_arousal,    E.arousal);
  gl.uniform1f(UL.u_layers,     E.layers);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

// ------- Mouse / touch -------
function toNDC(cx, cy) {
  return [((cx / canvasW) * 2 - 1) * (canvasW / canvasH), -((cy / canvasH) * 2 - 1)];
}
canvas.addEventListener('mousemove',  e => { [mouse.x, mouse.y] = toNDC(e.clientX, e.clientY); if (mouse.down) mouse.str = Math.min(mouse.str + .07, 1.2); });
canvas.addEventListener('mousedown',  () => { mouse.down = true;  mouse.str = .35; });
canvas.addEventListener('mouseup',    () => mouse.down = false);
canvas.addEventListener('mouseleave', () => mouse.down = false);
canvas.addEventListener('touchstart', e => { e.preventDefault(); const t = e.touches[0]; [mouse.x, mouse.y] = toNDC(t.clientX, t.clientY); mouse.down = true;  mouse.str = .4; },          { passive: false });
canvas.addEventListener('touchmove',  e => { e.preventDefault(); const t = e.touches[0]; [mouse.x, mouse.y] = toNDC(t.clientX, t.clientY); mouse.str = Math.min(mouse.str + .05, 1.2); }, { passive: false });
canvas.addEventListener('touchend',   () => mouse.down = false);

// ------- Slider builder -------
function fmtVal(step, n) {
  if (step >= 1)   return n.toFixed(0);
  if (step < 0.1)  return n.toFixed(2);
  return n.toFixed(1);
}

function mkSlider(parent, { key, label, hint, min, max, step, cls }) {
  const row = document.createElement('div');
  row.className = 'sl-row';
  row.innerHTML = `
    <div class="sl-label">${label}<small>${hint}</small></div>
    <input type="range" class="${cls}" id="sl_${key}" min="${min}" max="${max}" step="${step}" value="${E[key]}">
    <div class="sl-val" id="sv_${key}">${fmtVal(step, E[key])}</div>
  `;
  parent.appendChild(row);
  row.querySelector('input').addEventListener('input', function () {
    E[key] = +this.value;
    document.getElementById('sv_' + key).textContent = fmtVal(step, +this.value);
  });
}

const emotionCont = document.getElementById('sl-emotion');
const eegCont     = document.getElementById('sl-eeg');
const mlCont      = document.getElementById('sl-ml');

[
  { key: 'valence', label: 'Valence', hint: 'warm \u2194 cool hue',  min: -1, max: 1, step: .01, cls: 'emotion-sl' },
  { key: 'arousal', label: 'Arousal', hint: 'zoom + structure',      min: -1, max: 1, step: .01, cls: 'emotion-sl' },
].forEach(d => mkSlider(emotionCont, d));

[
  { key: 'alpha', label: 'Alpha (\u03b1)', hint: 'radial smoothing',  min: 0, max: 1, step: .01, cls: 'eeg-sl' },
  { key: 'beta',  label: 'Beta (\u03b2)',  hint: 'complexity + zoom', min: 0, max: 1, step: .01, cls: 'eeg-sl' },
  { key: 'theta', label: 'Theta (\u03b8)', hint: 'spiral chaos',      min: 0, max: 1, step: .01, cls: 'eeg-sl' },
  { key: 'delta', label: 'Delta (\u03b4)', hint: 'animation speed',   min: 0, max: 1, step: .01, cls: 'eeg-sl' },
  { key: 'gamma', label: 'Gamma (\u03b3)', hint: 'saturation + warp', min: 0, max: 1, step: .01, cls: 'eeg-sl' },
].forEach(d => mkSlider(eegCont, d));

[
  { key: 'mindfulness',   label: 'Mindfulness',   hint: 'damps speed + chaos', min: 0, max: 1,  step: .01, cls: 'ml-sl' },
  { key: 'concentration', label: 'Concentration', hint: 'sharpens structure',  min: 0, max: 1,  step: .01, cls: 'ml-sl' },
  { key: 'relaxation',    label: 'Relaxation',    hint: 'softens zoom',        min: 0, max: 1,  step: .01, cls: 'ml-sl' },
  { key: 'layers',        label: 'Depth',         hint: 'network layers',      min: 1, max: 12, step: 1,   cls: 'ml-sl' },
].forEach(d => mkSlider(mlCont, d));

// ------- Presets -------
const PRESETS = [
  { l: 'calm',     p: { valence:  .4, arousal: -.5, alpha: .8,  beta: .2,  theta: .15, delta: .15, gamma: .35, mindfulness: .8,  concentration: .3,  relaxation: .8  } },
  { l: 'euphoric', p: { valence:  .9, arousal:  .8, alpha: .3,  beta: .7,  theta: .4,  delta: .4,  gamma: .85, mindfulness: .3,  concentration: .8,  relaxation: .2  } },
  { l: 'anxious',  p: { valence: -.7, arousal:  .85,alpha: .1,  beta: .75, theta: .8,  delta: .35, gamma: .5,  mindfulness: .1,  concentration: .6,  relaxation: .1  } },
  { l: 'sad',      p: { valence: -.6, arousal: -.6, alpha: .6,  beta: .2,  theta: .2,  delta: .1,  gamma: .2,  mindfulness: .4,  concentration: .2,  relaxation: .5  } },
  { l: 'intense',  p: { valence:  .5, arousal:  .95,alpha: .2,  beta: .9,  theta: .7,  delta: .6,  gamma: .9,  mindfulness: .05, concentration: .95, relaxation: .05 } },
  { l: 'angry',    p: { valence: -.8, arousal:  .9, alpha: .05, beta: .85, theta: .9,  delta: .5,  gamma: .6,  mindfulness: .05, concentration: .7,  relaxation: .05 } },
  { l: 'awe',      p: { valence:  .7, arousal:  .5, alpha: .5,  beta: .5,  theta: .35, delta: .2,  gamma: .7,  mindfulness: .6,  concentration: .5,  relaxation: .5  } },
];

function applyPreset(p) {
  Object.assign(E, p);
  Object.keys(p).forEach(k => {
    const inp = document.getElementById('sl_' + k);
    const val = document.getElementById('sv_' + k);
    if (!inp) return;
    inp.value = E[k];
    const step = parseFloat(inp.step) || .01;
    val.textContent = fmtVal(step, E[k]);
  });
}

const presetsCont = document.getElementById('presets');
PRESETS.forEach(({ l, p }) => {
  const btn = document.createElement('div');
  btn.className = 'chip';
  btn.textContent = l;
  btn.addEventListener('click', () => applyPreset(p));
  presetsCont.appendChild(btn);
});

// ------- Controls -------
function newSeed() {
  seed    = Math.random() * 20;
  t0      = performance.now();
  tOffset = 0;
  document.getElementById('dSeed').textContent = seed.toFixed(3);
}
newSeed();

const bAnim = document.getElementById('bAnim');
bAnim.addEventListener('click', () => {
  animOn = !animOn;
  if (animOn) t0 = performance.now() - tOffset * 1000;
  else        tOffset = (performance.now() - t0) / 1000;
  bAnim.textContent = 'anim: ' + (animOn ? 'on' : 'off');
  bAnim.classList.toggle('lit', animOn);
});

document.getElementById('bNew').addEventListener('click', newSeed);

document.getElementById('bSnap').addEventListener('click', () => {
  requestAnimationFrame(() => {
    const a = document.createElement('a');
    a.download = `brainart_${Date.now()}.png`;
    a.href = canvas.toDataURL('image/png');
    a.click();
  });
});

// ===================================================================
// LIVE EEG  (Muse -> brainart_muse_server.py -> WebSocket -> this app)
// -------------------------------------------------------------------
// A parallel rAF loop eases the art's E-state toward the latest values
// the server pushes (~4 Hz). Per-frame lerp turns the stepped updates
// into continuous motion; the existing render() just reads E as usual.
// ===================================================================
const LIVE = {
  on: false,
  status: 'off',
  url: 'ws://localhost:8765',
  ws: null,
  smooth: 0.06,                 // 0..1 per frame; lower = slower/glassier glide
  keys: ['valence','arousal','alpha','beta','theta','delta','gamma'],
  target: {},
};
LIVE.keys.forEach(k => LIVE.target[k] = E[k]);

// --- status chip + toggle injected into the panel header ---
(function addLiveUI() {
  const hdr = document.getElementById('panel-header');
  const bar = document.createElement('div');
  bar.style.cssText = 'display:flex;align-items:center;gap:8px;margin-top:7px';
  bar.innerHTML = `
    <button id="bLive" class="btn" style="flex:0 0 auto;padding:5px 11px">go live</button>
    <span id="liveDot" style="width:8px;height:8px;border-radius:50%;background:#555;display:inline-block"></span>
    <span id="liveStat" style="font-size:9px;letter-spacing:.06em;color:rgba(255,255,255,.4)">offline</span>`;
  hdr.appendChild(bar);
  document.getElementById('bLive').addEventListener('click', toggleLive);
})();

function setLiveStatus(s) {
  LIVE.status = s;
  const dot = document.getElementById('liveDot');
  const txt = document.getElementById('liveStat');
  const map = { live:'#64e696', baseline:'#ffb454', sim:'#50a0ff',
                connecting:'#888', off:'#555', error:'#f55' };
  if (dot) dot.style.background = map[s] || '#555';
  if (txt) txt.textContent = s === 'baseline' ? 'baseline… sit still'
                          : s === 'live' ? 'live' : s;
}

function connectLive() {
  setLiveStatus('connecting');
  let ws;
  try { ws = new WebSocket(LIVE.url); }
  catch (e) { setLiveStatus('error'); return; }
  LIVE.ws = ws;
  ws.onmessage = ev => {
    let d; try { d = JSON.parse(ev.data); } catch { return; }
    LIVE.keys.forEach(k => { if (typeof d[k] === 'number') LIVE.target[k] = d[k]; });
    if (d.status) setLiveStatus(d.status);
  };
  ws.onclose = () => { if (LIVE.on) { setLiveStatus('connecting'); setTimeout(connectLive, 1500); } };
  ws.onerror = () => setLiveStatus('error');
}

function toggleLive() {
  LIVE.on = !LIVE.on;
  const b = document.getElementById('bLive');
  b.textContent = LIVE.on ? 'stop live' : 'go live';
  b.classList.toggle('lit', LIVE.on);
  // disable manual sliders while live so they don't fight the stream
  document.querySelectorAll('#sl-emotion input, #sl-eeg input')
    .forEach(i => i.disabled = LIVE.on);
  if (LIVE.on) connectLive();
  else { if (LIVE.ws) LIVE.ws.close(); setLiveStatus('off'); }
}

// --- per-frame easing of E toward the live targets + slider sync ---
function liveTick() {
  if (LIVE.on && LIVE.status !== 'connecting' && LIVE.status !== 'off') {
    LIVE.keys.forEach(k => {
      E[k] += (LIVE.target[k] - E[k]) * LIVE.smooth;
      const sl = document.getElementById('sl_' + k);
      const sv = document.getElementById('sv_' + k);
      if (sl) { sl.value = E[k]; if (sv) sv.textContent = E[k].toFixed(2); }
    });
  }
  requestAnimationFrame(liveTick);
}
requestAnimationFrame(liveTick);