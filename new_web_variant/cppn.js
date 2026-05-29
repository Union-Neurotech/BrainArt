export const FS_CPPN = /* glsl */`
precision highp float;

uniform vec2  res;
uniform float t, seed;
uniform float u_zoom, u_warp, u_speed, u_sat;
uniform float u_hue_shift, u_warm, u_complexity, u_chaos, u_radial;
uniform float u_offset_x, u_offset_y;
uniform vec2  u_mouse;
uniform float u_mouse_str;
uniform float u_neutral;
uniform float u_valence, u_arousal;
uniform float u_layers;          // active hidden layers, 1..MAXL

#define MAXL 12
#define WID  6

// ---- Emotion temperature tint (circumplex corners) ----
// Leans the whole image warm/cool while preserving the CPPN's own
// relative hue differences, so greens stay greener than blues, etc.
vec3 emotionTint(){
  float vp = clamp(u_valence*0.5 + 0.5, 0.0, 1.0);   // 0 neg -> 1 pos
  float ap = clamp(u_arousal*0.5 + 0.5, 0.0, 1.0);   // 0 low -> 1 high
  float wSad   = (1.0-vp)*(1.0-ap);
  float wCalm  =      vp *(1.0-ap);
  float wAnger = (1.0-vp)*     ap;
  float wJoy   =      vp *     ap;
  vec3 cAnger = vec3(1.00, 0.40, 0.08);   // fiery orange
  vec3 cJoy   = vec3(1.00, 0.78, 0.22);   // warm gold
  vec3 cSad   = vec3(0.30, 0.45, 0.95);   // cool blue
  vec3 cCalm  = vec3(0.28, 0.85, 0.70);   // teal/green
  return cAnger*wAnger + cJoy*wJoy + cSad*wSad + cCalm*wCalm;
}

// Smooth tanh approximation (avoids true exp overflow)
float th(float x) {
  float c = clamp(x, -8., 8.);
  float e = exp(-2. * c);
  return (1. - e) / (1. + e);
}

// Pseudo-random helpers
float h1(float n) { return fract(sin(n) * 43758.5453123); }
float h2(float a, float b) { return h1(a + b * 127.1); }

// Seeded network weight / bias
float W(float l, float i, float o) {
  return th(h2(l*200. + i*13. + seed, o*7. + seed*3.) * 4. - 2.);
}
float Wb(float l, float o) {
  return th(h2(l*300. + o*17., seed*5.) * 3. - 1.5);
}

// Hue rotation matrix
vec3 hueRot(vec3 col, float angle) {
  float c = cos(angle), s = sin(angle);
  mat3 m = mat3(
    .299 + .701*c + .168*s,  .299 - .299*c + .328*s,  .299 - .300*c - .328*s,
    .587 - .587*c - .330*s,  .587 + .413*c + .035*s,  .587 - .588*c + .292*s,
    .114 - .114*c + .497*s,  .114 - .114*c - .353*s,  .114 + .886*c + .023*s
  );
  return clamp(m * col, 0., 1.);
}

void main() {
  // Map fragment to [-aspect,aspect] x [-1,1], then apply zoom and pan
  vec2 uv = (gl_FragCoord.xy / res - .5) * 2.;
  uv.x *= res.x / res.y;
  uv   *= u_zoom;
  uv   += vec2(u_offset_x, u_offset_y) * 0.6;

  float ts = t * u_speed;

  // Mouse distortion
  vec2  md   = uv - u_mouse;
  float mr   = length(md);
  float mw   = u_mouse_str * exp(-mr * mr * 1.8);
  float mang = atan(md.y, md.x);
  uv += vec2(cos(mang + 1.5708), sin(mang + 1.5708)) * mw * .4;
  uv += md * (-mw * .25 / (mr + .001));

  // Polar chaos warp
  float r   = length(uv);
  float ang = atan(uv.y, uv.x);
  float sr  = r   + u_chaos * 0.4 * sin(ang*3. + ts*0.7 + r*2.);
  float sa  = ang + u_chaos * 0.5 * cos(r*4.  + ts*0.5) * exp(-r*0.4);
  vec2  wv  = vec2(sr * cos(sa), sr * sin(sa));
  float wx  = wv.x, wy = wv.y, wr = length(wv);
  wx = mix(wx, wr * cos(ang), u_radial);
  wy = mix(wy, wr * sin(ang), u_radial);

  float cplx = 0.7 + u_complexity * 1.3;

  // Seed the hidden vector from coordinate features (WID=6 slots)
  float hprev[WID];
  float hcur[WID];
  hprev[0] = wx;
  hprev[1] = wy;
  hprev[2] = wr;
  hprev[3] = sin(ang);
  hprev[4] = cos(ang);
  hprev[5] = sin(wr * 3.14159 + ts);

  int L = int(u_layers + 0.5);
  L = int(clamp(float(L), 1.0, float(MAXL)));

  // Variable-depth feedforward. Constant loop bound (MAXL) keeps the
  // compiler happy; the break makes cost scale with the active depth L.
  for (int layer = 0; layer < MAXL; layer++) {
    if (layer >= L) break;
    for (int o = 0; o < WID; o++) {
      float acc = Wb(float(layer), float(o));
      for (int i = 0; i < WID; i++) {
        acc += W(float(layer), float(i), float(o)) * hprev[i];
      }
      // depth-independent warp injection so u_warp stays influential at any depth
      acc += sin(wr * 6.28318 * u_warp + float(o)) * 0.15;
      hcur[o] = th(acc * cplx);
    }
    for (int k = 0; k < WID; k++) hprev[k] = hcur[k];
  }

  // Output projection (-> RGB). Weight layer index L differs from hidden layers.
  float cr = th(W(float(L),0.,0.)*hprev[0] + W(float(L),1.,0.)*hprev[1] + W(float(L),2.,0.)*hprev[2] + W(float(L),3.,0.)*hprev[3] + W(float(L),4.,0.)*hprev[4] + W(float(L),5.,0.)*hprev[5] + Wb(float(L),0.));
  float cg = th(W(float(L),0.,1.)*hprev[0] + W(float(L),1.,1.)*hprev[1] + W(float(L),2.,1.)*hprev[2] + W(float(L),3.,1.)*hprev[3] + W(float(L),4.,1.)*hprev[4] + W(float(L),5.,1.)*hprev[5] + Wb(float(L),1.));
  float cb = th(W(float(L),0.,2.)*hprev[0] + W(float(L),1.,2.)*hprev[1] + W(float(L),2.,2.)*hprev[2] + W(float(L),3.,2.)*hprev[3] + W(float(L),4.,2.)*hprev[4] + W(float(L),5.,2.)*hprev[5] + Wb(float(L),2.));

  // Full CPPN colour = the variety. Keep it.
  vec3 base = vec3(cr, cg, cb) * .5 + .5;
  float lum = dot(base, vec3(0.2126, 0.7152, 0.0722));

  // Emotion grade: multiply-tint leans the whole image toward an emotion
  // temperature while preserving relative hue differences (greens vs blues survive).
  vec3  tint      = emotionTint();
  float intensity = clamp(length(vec2(u_valence, u_arousal)) / 1.41421, 0.0, 1.0);
  vec3  graded    = base * mix(vec3(1.0), tint * 1.7, intensity * 0.8);

  float glum = dot(graded, vec3(0.2126, 0.7152, 0.0722));
  graded = mix(vec3(glum), graded, clamp(u_sat, 0.0, 2.2));

  // Legacy look (neutral mode off): original hueRot + warm/cool tint
  vec3 raw = base;
  raw = mix(vec3(lum), raw, u_sat);
  raw = hueRot(raw, u_hue_shift * 1.0472);
  raw.r = mix(raw.r, min(raw.r * 1.18, 1.), max( u_warm, 0.) * .35);
  raw.b = mix(raw.b, min(raw.b * 1.18, 1.), max(-u_warm, 0.) * .35);

  vec3 col = mix(raw, graded, u_neutral);
  col = clamp(col, 0., 1.);
  gl_FragColor = vec4(col, 1.);
}
`;