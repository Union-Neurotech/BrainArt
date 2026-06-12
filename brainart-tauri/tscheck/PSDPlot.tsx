import { useEffect, useRef } from "react";
import uPlot from "uplot";
import { EegState, CHANNEL_LABELS } from "./types";

interface Props {
  eegStateRef: React.MutableRefObject<EegState>;
}

const COLORS = ["#50a0ff", "#27c4a6", "#ffae42", "#ff5d73"];

export default function PSDPlot({ eegStateRef }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const nCh = CHANNEL_LABELS.length;

    const opts: uPlot.Options = {
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight || 160,
      title: "PSD · dB · 0–45 Hz",
      scales: { x: { time: false }, y: { auto: true } },
      axes: [
        { stroke: "#777", grid: { stroke: "#222" }, values: (_u, s) => s.map((v) => v + "") },
        { stroke: "#777", grid: { stroke: "#222" } },
      ],
      series: [
        { label: "Hz" },
        ...CHANNEL_LABELS.map((lab, i) => ({
          stroke: COLORS[i], width: 1.3, label: lab, points: { show: false },
        })),
      ],
      legend: { show: true },
      cursor: { show: false },
    };

    const u = new uPlot(opts, [[0], ...Array.from({ length: nCh }, () => [0])] as uPlot.AlignedData,
      containerRef.current);

    let raf = 0;
    let lastFreqLen = -1;
    const render = () => {
      const psd = eegStateRef.current.psd;
      if (psd && psd.freqs.length > 1) {
        const data: uPlot.AlignedData = [psd.freqs, ...psd.chans];
        u.setData(data);
        lastFreqLen = psd.freqs.length;
      } else if (lastFreqLen !== 0) {
        lastFreqLen = 0;
      }
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
