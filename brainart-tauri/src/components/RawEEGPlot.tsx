import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { EegState, CHANNEL_LABELS } from "../App";

interface Props {
  eegStateRef: React.MutableRefObject<EegState>;
}

const COLORS = ["#50a0ff", "#27c4a6", "#ffae42", "#ff5d73"]; // AF7, AF8, TP9, TP10
const SPACING = 1;          // vertical gap between stacked channels (montage units)
const SAMPLES = 256 * 5;    // 5 s window at 256 Hz

/**
 * Stacked "montage" view of the 4 Muse channels driven by live data in
 * eegStateRef.current.channels (populated from the WebSocket in App.tsx).
 * Each channel is mean-removed and scaled to its own recent amplitude so all
 * four read clearly regardless of absolute µV, then offset onto its own lane.
 */
export default function RawEEGPlot({ eegStateRef }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const nCh = CHANNEL_LABELS.length;

    const xs = Array.from({ length: SAMPLES }, (_, i) => i - SAMPLES);
    const initData: uPlot.AlignedData = [
      xs,
      ...Array.from({ length: nCh }, () => new Array(SAMPLES).fill(0) as number[]),
    ];

    const opts: uPlot.Options = {
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight || 160,
      title: "Raw EEG · 4 ch · 5 s",
      scales: {
        x: { time: false },
        // extra margin below the lowest lane (AF7) and above the highest (TP10)
        // so the traces don't clip at the panel edges
        y: { auto: false, range: [-1.8 * SPACING, (nCh - 0.3) * SPACING] },
      },
      axes: [
        { show: false, grid: { show: false } },
        {
          stroke: "#777",
          grid: { show: false },
          // label each lane with its electrode name
          values: (_u, splits) =>
            splits.map((v) => {
              const idx = Math.round(v / SPACING);
              return idx >= 0 && idx < nCh ? CHANNEL_LABELS[idx] : "";
            }),
          splits: () => Array.from({ length: nCh }, (_, i) => i * SPACING),
        },
      ],
      series: [
        {},
        ...CHANNEL_LABELS.map((lab, i) => ({
          stroke: COLORS[i], width: 1, label: lab, points: { show: false },
        })),
      ],
      legend: { show: false },
      cursor: { show: false },
    };

    const u = new uPlot(opts, initData, containerRef.current);

    let raf = 0;
    const render = () => {
      const chans = eegStateRef.current.channels;
      const data: uPlot.AlignedData = [xs];
      for (let i = 0; i < nCh; i++) {
        const src = chans[i] || [];
        const lane = i * SPACING;
        const out = new Array(SAMPLES).fill(lane) as number[];
        const n = Math.min(src.length, SAMPLES);
        if (n > 0) {
          // mean + robust amplitude (max abs deviation) over the visible window
          const start = src.length - n;
          let mean = 0;
          for (let k = 0; k < n; k++) mean += src[start + k];
          mean /= n;
          let maxAbs = 1e-6;
          for (let k = 0; k < n; k++) {
            const dvt = Math.abs(src[start + k] - mean);
            if (dvt > maxAbs) maxAbs = dvt;
          }
          const scale = (SPACING * 0.42) / maxAbs;
          const off = SAMPLES - n; // right-align newest sample
          for (let k = 0; k < n; k++) out[off + k] = lane + (src[start + k] - mean) * scale;
        }
        data.push(out);
      }
      u.setData(data);
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);

    const onResize = () => {
      if (!containerRef.current) return;
      u.setSize({
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight || 160,
      });
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      u.destroy();
    };
  }, [eegStateRef]);

  return (
    <div className="w-full h-full flex flex-col">
      <div ref={containerRef} className="w-full flex-1" />
    </div>
  );
}
