export const FS_CPPN = /* glsl */`
precision highp float;

uniform vec2  res;
uniform float t, seed;
uniform float u_zoom, u_warp, u_speed, u_sat;
uniform float u_hue_shift, u_warm, u_complexity, u_chaos, u_radial;
uniform float u_offset_x, u_offset_y;
uniform vec2  u_mouse;
uniform float u_mouse_str;
uniform float u_neutral;   // 0 = raw CPPN colors, 1 = emotion palette
uniform float u_valence, u_arousal;

// ---- Emotion palettes (circumplex corners). t = structural scalar in [0,1] ----
// vec3 Panger(float t){            // neg valence, high arousal -> fire
//   vec3 c = mix(vec3(0.12,0.0,0.0),  vec3(0.85,0.12,0.0), smoothstep(0.0,0.50,t));
//   c = mix(c, vec3(1.0,0.50,0.0),   smoothstep(0.45,0.78,t));
//   c = mix(c, vec3(1.0,0.92,0.25),  smoothstep(0.78,1.0,t));
//   return c;
// }
// vec3 Pjoy(float t){              // pos valence, high arousal -> vivid warm
//   vec3 c = mix(vec3(0.75,0.0,0.45), vec3(1.0,0.35,0.0), smoothstep(0.0,0.45,t));
//   c = mix(c, vec3(1.0,0.82,0.10),  smoothstep(0.45,0.80,t));
//   c = mix(c, vec3(1.0,1.0,0.85),   smoothstep(0.80,1.0,t));
//   return c;
// }
// vec3 Psad(float t){              // neg valence, low arousal -> cold blue
//   vec3 c = mix(vec3(0.02,0.03,0.13), vec3(0.10,0.20,0.45), smoothstep(0.0,0.50,t));
//   c = mix(c, vec3(0.26,0.36,0.62),  smoothstep(0.50,0.85,t));
//   c = mix(c, vec3(0.50,0.55,0.66),  smoothstep(0.85,1.0,t));
//   return c;
// }
// vec3 Pcalm(float t){             // pos valence, low arousal -> teal/green
//   vec3 c = mix(vec3(0.0,0.14,0.14), vec3(0.0,0.42,0.36), smoothstep(0.0,0.50,t));
//   c = mix(c, vec3(0.32,0.72,0.55), smoothstep(0.50,0.85,t));
//   c = mix(c, vec3(0.78,0.94,0.82), smoothstep(0.85,1.0,t));
//   return c;
// }

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

  // Network inputs
  float in0 = wx,             in1 = wy,          in2 = wr,
        in3 = sin(ang),       in4 = cos(ang),
        in5 = sin(wr*3.14159),                    in6 = sin(ts);

  float cplx = 0.7 + u_complexity * 1.3;

  // Layer 0  (7 inputs -> 6 neurons)
  float n0 = th((W(0.,0.,0.)*in0 + W(0.,1.,0.)*in1 + W(0.,2.,0.)*in2 + W(0.,3.,0.)*in3 + W(0.,4.,0.)*in4 + W(0.,5.,0.)*in5 + W(0.,6.,0.)*in6 + Wb(0.,0.)) * cplx);
  float n1 = th((W(0.,0.,1.)*in0 + W(0.,1.,1.)*in1 + W(0.,2.,1.)*in2 + W(0.,3.,1.)*in3 + W(0.,4.,1.)*in4 + W(0.,5.,1.)*in5 + W(0.,6.,1.)*in6 + Wb(0.,1.)) * cplx);
  float n2 = th((W(0.,0.,2.)*in0 + W(0.,1.,2.)*in1 + W(0.,2.,2.)*in2 + W(0.,3.,2.)*in3 + W(0.,4.,2.)*in4 + W(0.,5.,2.)*in5 + W(0.,6.,2.)*in6 + Wb(0.,2.)) * cplx);
  float n3 = th((W(0.,0.,3.)*in0 + W(0.,1.,3.)*in1 + W(0.,2.,3.)*in2 + W(0.,3.,3.)*in3 + W(0.,4.,3.)*in4 + W(0.,5.,3.)*in5 + W(0.,6.,3.)*in6 + Wb(0.,3.)) * cplx);
  float n4 = th((W(0.,0.,4.)*in0 + W(0.,1.,4.)*in1 + W(0.,2.,4.)*in2 + W(0.,3.,4.)*in3 + W(0.,4.,4.)*in4 + W(0.,5.,4.)*in5 + W(0.,6.,4.)*in6 + Wb(0.,4.)) * cplx);
  float n5 = th((W(0.,0.,5.)*in0 + W(0.,1.,5.)*in1 + W(0.,2.,5.)*in2 + W(0.,3.,5.)*in3 + W(0.,4.,5.)*in4 + W(0.,5.,5.)*in5 + W(0.,6.,5.)*in6 + Wb(0.,5.)) * cplx);

  // Layer 1  (6 -> 6)
  float m0 = th((W(1.,0.,0.)*n0 + W(1.,1.,0.)*n1 + W(1.,2.,0.)*n2 + W(1.,3.,0.)*n3 + W(1.,4.,0.)*n4 + W(1.,5.,0.)*n5 + Wb(1.,0.)) * cplx);
  float m1 = th((W(1.,0.,1.)*n0 + W(1.,1.,1.)*n1 + W(1.,2.,1.)*n2 + W(1.,3.,1.)*n3 + W(1.,4.,1.)*n4 + W(1.,5.,1.)*n5 + Wb(1.,1.)) * cplx);
  float m2 = th((W(1.,0.,2.)*n0 + W(1.,1.,2.)*n1 + W(1.,2.,2.)*n2 + W(1.,3.,2.)*n3 + W(1.,4.,2.)*n4 + W(1.,5.,2.)*n5 + Wb(1.,2.)) * cplx);
  float m3 = th((W(1.,0.,3.)*n0 + W(1.,1.,3.)*n1 + W(1.,2.,3.)*n2 + W(1.,3.,3.)*n3 + W(1.,4.,3.)*n4 + W(1.,5.,3.)*n5 + Wb(1.,3.)) * cplx);
  float m4 = th((W(1.,0.,4.)*n0 + W(1.,1.,4.)*n1 + W(1.,2.,4.)*n2 + W(1.,3.,4.)*n3 + W(1.,4.,4.)*n4 + W(1.,5.,4.)*n5 + Wb(1.,4.)) * cplx);
  float m5 = th((W(1.,0.,5.)*n0 + W(1.,1.,5.)*n1 + W(1.,2.,5.)*n2 + W(1.,3.,5.)*n3 + W(1.,4.,5.)*n4 + W(1.,5.,5.)*n5 + Wb(1.,5.)) * cplx);

  // Layer 2  (6 -> 3, with warp-frequency modulation injected)
  float warp = u_warp;
  float p0 = th((W(2.,0.,0.)*m0 + W(2.,1.,0.)*m1 + W(2.,2.,0.)*m2 + W(2.,3.,0.)*m3 + W(2.,4.,0.)*m4 + W(2.,5.,0.)*m5 + sin(wr*6.28318*warp)*0.4 + Wb(2.,0.)) * cplx);
  float p1 = th((W(2.,0.,1.)*m0 + W(2.,1.,1.)*m1 + W(2.,2.,1.)*m2 + W(2.,3.,1.)*m3 + W(2.,4.,1.)*m4 + W(2.,5.,1.)*m5 + cos(ang*2.*warp)*0.4     + Wb(2.,1.)) * cplx);
  float p2 = th((W(2.,0.,2.)*m0 + W(2.,1.,2.)*m1 + W(2.,2.,2.)*m2 + W(2.,3.,2.)*m3 + W(2.,4.,2.)*m4 + W(2.,5.,2.)*m5 + sin(ang*3.*warp+wr)*0.4  + Wb(2.,2.)) * cplx);

  // Output layer  (-> RGB)
  float cr = th(W(3.,0.,0.)*p0 + W(3.,1.,0.)*p1 + W(3.,2.,0.)*p2 + W(3.,3.,0.)*m0 + W(3.,4.,0.)*m3 + Wb(3.,0.));
  float cg = th(W(3.,0.,1.)*p0 + W(3.,1.,1.)*p1 + W(3.,2.,1.)*p2 + W(3.,3.,1.)*m1 + W(3.,4.,1.)*m4 + Wb(3.,1.));
  float cb = th(W(3.,0.,2.)*p0 + W(3.,1.,2.)*p1 + W(3.,2.,2.)*p2 + W(3.,3.,2.)*m2 + W(3.,4.,2.)*m5 + Wb(3.,2.));

  // Full CPPN colour = the variety. Keep it.
  vec3 base = vec3(cr, cg, cb) * .5 + .5;
  float lum = dot(base, vec3(0.2126, 0.7152, 0.0722));

  // Emotion grade: lean the whole image toward an emotion temperature,
  // multiply-tint preserves relative hue differences (greens vs blues survive).
  vec3  tint      = emotionTint();
  float intensity = clamp(length(vec2(u_valence, u_arousal)) / 1.41421, 0.0, 1.0);
  vec3  graded    = base * mix(vec3(1.0), tint * 1.7, intensity * 0.8); // Controls how much emotion takes over at corners, tint * 1.7 is the off-temp colors

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