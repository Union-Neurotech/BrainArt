// src/components/BrainArtCanvas.tsx
import { useEffect, useRef } from "react";
import { EegState } from "../App";

interface Props {
  eegStateRef: React.MutableRefObject<EegState>;
}

export default function BrainArtCanvas({ eegStateRef }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl", { preserveDrawingBuffer: true, antialias: false }) ||
               canvas.getContext("experimental-webgl", { preserveDrawingBuffer: true }) as WebGLRenderingContext;
    if (!gl) {
      console.error("WebGL required");
      return;
    }

    // --- 1. SHADERS ---
    const VS = `attribute vec2 a;void main(){gl_Position=vec4(a,0.,1.);}`;
    const FS = `
      precision highp float;
      uniform vec2 res; uniform float t,seed; uniform float u_zoom,u_warp,u_speed,u_sat;
      uniform float u_hue_shift,u_warm,u_complexity,u_chaos,u_radial;
      uniform float u_offset_x,u_offset_y; uniform vec2 u_mouse;uniform float u_mouse_str;
      float th(float x){float c=clamp(x,-8.,8.);float e=exp(-2.*c);return(1.-e)/(1.+e);}
      float h1(float n){return fract(sin(n)*43758.5453123);}
      float h2(float a,float b){return h1(a+b*127.1);}
      float W(float l,float i,float o){return th(h2(l*200.+i*13.+seed,o*7.+seed*3.)*4.-2.);}
      float Wb(float l,float o){return th(h2(l*300.+o*17.,seed*5.)*3.-1.5);}
      vec3 hueRot(vec3 col,float angle){
        float c=cos(angle),s=sin(angle);
        mat3 m=mat3(.299+.701*c+.168*s,.299-.299*c+.328*s,.299-.300*c-.328*s,
                    .587-.587*c-.330*s,.587+.413*c+.035*s,.587-.588*c+.292*s,
                    .114-.114*c+.497*s,.114-.114*c-.353*s,.114+.886*c+.023*s);
        return clamp(m*col,0.,1.);
      }
      void main(){
        vec2 uv=(gl_FragCoord.xy/res-.5)*2.;
        uv.x*=res.x/res.y; uv*=u_zoom;
        uv+=vec2(u_offset_x,u_offset_y)*0.6;
        float ts=t*u_speed;
        vec2 md=uv-u_mouse; float mr=length(md);
        float mw=u_mouse_str*exp(-mr*mr*1.8);
        float mang=atan(md.y,md.x);
        uv+=vec2(cos(mang+1.5708),sin(mang+1.5708))*mw*.4;
        uv+=md*(-mw*.25/(mr+.001));
        float r=length(uv),ang=atan(uv.y,uv.x);
        float sr=r+u_chaos*0.4*sin(ang*3.+ts*0.7+r*2.);
        float sa=ang+u_chaos*0.5*cos(r*4.+ts*0.5)*exp(-r*0.4);
        vec2 wv=vec2(sr*cos(sa),sr*sin(sa));
        float wx=wv.x,wy=wv.y,wr=length(wv);
        wx=mix(wx,wr*cos(ang),u_radial); wy=mix(wy,wr*sin(ang),u_radial);
        float in0=wx,in1=wy,in2=wr,in3=sin(ang),in4=cos(ang),in5=sin(wr*3.14159),in6=sin(ts);
        float cplx=0.7+u_complexity*1.3;
        float n0=th((W(0.,0.,0.)*in0+W(0.,1.,0.)*in1+W(0.,2.,0.)*in2+W(0.,3.,0.)*in3+W(0.,4.,0.)*in4+W(0.,5.,0.)*in5+W(0.,6.,0.)*in6+Wb(0.,0.))*cplx);
        float n1=th((W(0.,0.,1.)*in0+W(0.,1.,1.)*in1+W(0.,2.,1.)*in2+W(0.,3.,1.)*in3+W(0.,4.,1.)*in4+W(0.,5.,1.)*in5+W(0.,6.,1.)*in6+Wb(0.,1.))*cplx);
        float n2=th((W(0.,0.,2.)*in0+W(0.,1.,2.)*in1+W(0.,2.,2.)*in2+W(0.,3.,2.)*in3+W(0.,4.,2.)*in4+W(0.,5.,2.)*in5+W(0.,6.,2.)*in6+Wb(0.,2.))*cplx);
        float n3=th((W(0.,0.,3.)*in0+W(0.,1.,3.)*in1+W(0.,2.,3.)*in2+W(0.,3.,3.)*in3+W(0.,4.,3.)*in4+W(0.,5.,3.)*in5+W(0.,6.,3.)*in6+Wb(0.,3.))*cplx);
        float n4=th((W(0.,0.,4.)*in0+W(0.,1.,4.)*in1+W(0.,2.,4.)*in2+W(0.,3.,4.)*in3+W(0.,4.,4.)*in4+W(0.,5.,4.)*in5+W(0.,6.,4.)*in6+Wb(0.,4.))*cplx);
        float n5=th((W(0.,0.,5.)*in0+W(0.,1.,5.)*in1+W(0.,2.,5.)*in2+W(0.,3.,5.)*in3+W(0.,4.,5.)*in4+W(0.,5.,5.)*in5+W(0.,6.,5.)*in6+Wb(0.,5.))*cplx);
        float m0=th((W(1.,0.,0.)*n0+W(1.,1.,0.)*n1+W(1.,2.,0.)*n2+W(1.,3.,0.)*n3+W(1.,4.,0.)*n4+W(1.,5.,0.)*n5+Wb(1.,0.))*cplx);
        float m1=th((W(1.,0.,1.)*n0+W(1.,1.,1.)*n1+W(1.,2.,1.)*n2+W(1.,3.,1.)*n3+W(1.,4.,1.)*n4+W(1.,5.,1.)*n5+Wb(1.,1.))*cplx);
        float m2=th((W(1.,0.,2.)*n0+W(1.,1.,2.)*n1+W(1.,2.,2.)*n2+W(1.,3.,2.)*n3+W(1.,4.,2.)*n4+W(1.,5.,2.)*n5+Wb(1.,2.))*cplx);
        float m3=th((W(1.,0.,3.)*n0+W(1.,1.,3.)*n1+W(1.,2.,3.)*n2+W(1.,3.,3.)*n3+W(1.,4.,3.)*n4+W(1.,5.,3.)*n5+Wb(1.,3.))*cplx);
        float m4=th((W(1.,0.,4.)*n0+W(1.,1.,4.)*n1+W(1.,2.,4.)*n2+W(1.,3.,4.)*n3+W(1.,4.,4.)*n4+W(1.,5.,4.)*n5+Wb(1.,4.))*cplx);
        float m5=th((W(1.,0.,5.)*n0+W(1.,1.,5.)*n1+W(1.,2.,5.)*n2+W(1.,3.,5.)*n3+W(1.,4.,5.)*n4+W(1.,5.,5.)*n5+Wb(1.,5.))*cplx);
        float warp=u_warp;
        float p0=th((W(2.,0.,0.)*m0+W(2.,1.,0.)*m1+W(2.,2.,0.)*m2+W(2.,3.,0.)*m3+W(2.,4.,0.)*m4+W(2.,5.,0.)*m5+sin(wr*6.28318*warp)*0.4+Wb(2.,0.))*cplx);
        float p1=th((W(2.,0.,1.)*m0+W(2.,1.,1.)*m1+W(2.,2.,1.)*m2+W(2.,3.,1.)*m3+W(2.,4.,1.)*m4+W(2.,5.,1.)*m5+cos(ang*2.*warp)*0.4+Wb(2.,1.))*cplx);
        float p2=th((W(2.,0.,2.)*m0+W(2.,1.,2.)*m1+W(2.,2.,2.)*m2+W(2.,3.,2.)*m3+W(2.,4.,2.)*m4+W(2.,5.,2.)*m5+sin(ang*3.*warp+wr)*0.4+Wb(2.,2.))*cplx);
        float cr=th(W(3.,0.,0.)*p0+W(3.,1.,0.)*p1+W(3.,2.,0.)*p2+W(3.,3.,0.)*m0+W(3.,4.,0.)*m3+Wb(3.,0.));
        float cg=th(W(3.,0.,1.)*p0+W(3.,1.,1.)*p1+W(3.,2.,1.)*p2+W(3.,3.,1.)*m1+W(3.,4.,1.)*m4+Wb(3.,1.));
        float cb=th(W(3.,0.,2.)*p0+W(3.,1.,2.)*p1+W(3.,2.,2.)*p2+W(3.,3.,2.)*m2+W(3.,4.,2.)*m5+Wb(3.,2.));
        vec3 col=vec3(cr,cg,cb)*.5+.5;
        float lum=dot(col,vec3(0.2126,0.7152,0.0722));
        col=mix(vec3(lum),col,u_sat);
        col=hueRot(col,u_hue_shift*1.0472);
        col.r=mix(col.r,min(col.r*1.18,1.),max(u_warm,0.)*.35);
        col.b=mix(col.b,min(col.b*1.18,1.),max(-u_warm,0.)*.35);
        col=clamp(col,0.,1.);
        gl_FragColor=vec4(col,1.);
      }
    `;

    // --- 2. PROGRAM SETUP ---
    function mkS(t: number, s: string) {
      const sh = gl.createShader(t)!;
      gl.shaderSource(sh, s);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(sh));
      return sh;
    }
    const prog = gl.createProgram()!;
    gl.attachShader(prog, mkS(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, mkS(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    gl.useProgram(prog);

    const qb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, qb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const al = gl.getAttribLocation(prog, "a");
    gl.enableVertexAttribArray(al);
    gl.vertexAttribPointer(al, 2, gl.FLOAT, false, 0, 0);

    const UL: Record<string, WebGLUniformLocation | null> = {};
    ['res','t','seed','u_zoom','u_warp','u_speed','u_sat','u_hue_shift','u_warm','u_complexity','u_chaos','u_radial','u_offset_x','u_offset_y','u_mouse','u_mouse_str'].forEach((n) => {
      UL[n] = gl.getUniformLocation(prog, n);
    });

    // --- 3. STATE & RESIZE ---
    let W_ = 0, H_ = 0;
    const resize = () => {
      W_ = canvas.parentElement?.clientWidth || window.innerWidth;
      H_ = canvas.parentElement?.clientHeight || window.innerHeight;
      canvas.width = W_;
      canvas.height = H_;
      gl.viewport(0, 0, W_, H_);
      gl.uniform2f(UL.res, W_, H_);
    };
    window.addEventListener("resize", resize);
    resize();

    // --- 4. MOUSE INTERACTIONS ---
    // const mouse = { x: 0, y: 0, str: 0, down: false };
    // const toNDC = (cx: number, cy: number) => [((cx / W_) * 2 - 1) * (W_ / H_), -((cy / H_) * 2 - 1)];
    
    // const onMouseMove = (e: MouseEvent) => { [mouse.x, mouse.y] = toNDC(e.clientX, e.clientY); if (mouse.down) mouse.str = Math.min(mouse.str + 0.07, 1.2); };
    // const onMouseDown = () => { mouse.down = true; mouse.str = 0.35; };
    // const onMouseUp = () => { mouse.down = false; };
    
    // canvas.addEventListener("mousemove", onMouseMove);
    // canvas.addEventListener("mousedown", onMouseDown);
    // canvas.addEventListener("mouseup", onMouseUp);
    // canvas.addEventListener("mouseleave", onMouseUp);

    // --- 5. RENDER LOOP ---
    let t0 = performance.now();
    let seed = 5; // We can make this dynamic later
    let animationId: number;

    const render = (ts: number) => {
      const elapsed = (ts - t0) / 1000;
      // if (!mouse.down) mouse.str = Math.max(0, mouse.str - 0.025);

      // Read from the React Ref (this replaces your 'E' object)
      const E = eegStateRef.current;

      // Your exact math from etov()
      const v = E.valence, a = E.arousal, absV = Math.abs(v);
      const md = 1 - E.mindfulness * 0.6, cb = E.concentration * 0.5, rs = E.relaxation * 0.4;
      
      const V = {
        hue_shift: v, warm: v,
        sat: 1.0 + absV * 1.2 + E.gamma * 1.5,
        zoom: 0.7 + (1 - a) * 0.5 + (1 - E.beta) * 0.5 + rs,
        complexity: 0.3 + a * 0.5 + E.beta * 0.4 + cb,
        chaos: E.theta * 1.8 * md,
        radial: E.alpha * 0.85,
        speed: (0.02 + E.delta * 0.12 + Math.max(a, 0) * 0.06) * md,
        warp: 0.3 + E.gamma * 0.6 + Math.max(a, 0) * 0.4,
        offset_x: 0, offset_y: 0
      };

      // Push to Shaders
      gl.uniform1f(UL.t, elapsed); gl.uniform1f(UL.seed, seed);
      gl.uniform1f(UL.u_zoom, V.zoom); gl.uniform1f(UL.u_warp, V.warp);
      gl.uniform1f(UL.u_speed, V.speed); gl.uniform1f(UL.u_sat, V.sat);
      gl.uniform1f(UL.u_hue_shift, V.hue_shift); gl.uniform1f(UL.u_warm, V.warm);
      gl.uniform1f(UL.u_complexity, V.complexity); gl.uniform1f(UL.u_chaos, V.chaos);
      gl.uniform1f(UL.u_radial, V.radial);
      gl.uniform1f(UL.u_offset_x, V.offset_x); gl.uniform1f(UL.u_offset_y, V.offset_y);
      // gl.uniform2f(UL.u_mouse, mouse.x, mouse.y); gl.uniform1f(UL.u_mouse_str, mouse.str);
      
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      animationId = requestAnimationFrame(render);
    };
    animationId = requestAnimationFrame(render);

    // --- CLEANUP ---
    return () => {
      window.removeEventListener("resize", resize);
      // canvas.removeEventListener("mousemove", onMouseMove);
      // canvas.removeEventListener("mousedown", onMouseDown);
      // canvas.removeEventListener("mouseup", onMouseUp);
      // canvas.removeEventListener("mouseleave", onMouseUp);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return <canvas ref={canvasRef} className="block w-full h-full" />;
}