import { memo, useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { EegState } from "../App";

interface Props {
  eegStateRef: React.MutableRefObject<EegState>;
}

function RawEEGPlot({ eegStateRef }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // --- 1. SETUP ROLLING BUFFER ---
    // We want to show the last ~5 seconds of data. At 30 FPS, that's 150 points.
    const bufferSize = 150; 
    
    // Number of channels (Matches the 3 mock channels from our Rust backend)
    // Once you hook up the physical board, change this to 4, 8, or 16.
    const channels = 3; 

    // Initialize an empty data array formatted for uPlot: [ [X_Time], [Y_Ch1], [Y_Ch2], [Y_Ch3] ]
    const data: uPlot.AlignedData = [
      Array(bufferSize).fill(0).map((_, i) => i - bufferSize), // X-axis: -150 to 0
      ...Array(channels).fill(0).map(() => Array(bufferSize).fill(0))
    ];

    // --- 2. CONFIGURE uPLOT ---
    const opts: uPlot.Options = {
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight || 250,
      title: "Real-time EEG (Mock)",
      scales: {
        x: { time: false }, // Turn off timestamp parsing for a clean rolling integer axis
        y: { auto: false, range: [-2, 2] } // Fixed Y-axis range (prevents graph from bouncing)
      },
      axes: [
        { grid: { stroke: "#333333" }, stroke: "#aaaaaa" }, // Dark mode axes
        { grid: { stroke: "#333333" }, stroke: "#aaaaaa" }
      ],
      series: [
        {}, // X axis (Time)
        { stroke: "#ff7846", width: 1.5, label: "Ch 1" }, // Channel 1 (Orange)
        { stroke: "#50a0ff", width: 1.5, label: "Ch 2" }, // Channel 2 (Blue)
        { stroke: "#64e696", width: 1.5, label: "Ch 3" }, // Channel 3 (Green)
      ],
      cursor: {
        show: false // Turn off hover cursor for better performance
      }
    };

    const u = new uPlot(opts, data, containerRef.current);
    plotRef.current = u;

    // --- 3. RENDER LOOP ---
    let animationId: number;
    let tickCounter = 0;

    const updateData = () => {
      const state = eegStateRef.current;

      // 1. Shift the X axis data
      (data[0] as number[]).shift();
      (data[0] as number[]).push(tickCounter++);

      // 2. Shift the Y axis data and push the newest hardware values
      for (let i = 0; i < channels; i++) {
        (data[i + 1] as number[]).shift();
        // Fallback to 0 if the Rust backend hasn't populated raw_waves yet
        (data[i + 1] as number[]).push(state.raw_waves[i] || 0); 
      }

      // 3. Blast the new data to the chart
      u.setData(data);

      // Loop again on the next monitor refresh
      animationId = requestAnimationFrame(updateData);
    };

    animationId = requestAnimationFrame(updateData);

    // --- 4. HANDLE WINDOW RESIZE ---
    const handleResize = () => {
      if (!containerRef.current) return;
      u.setSize({
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight || 250,
      });
    };
    window.addEventListener("resize", handleResize);

    // --- 5. CLEANUP ---
    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", handleResize);
      u.destroy();
    };
  }, []);

  return (
    <div className="w-full h-full flex flex-col">
      {/* Container where uPlot will mount its canvas */}
      <div ref={containerRef} className="w-full flex-1" />
    </div>
  );
}

export default memo(RawEEGPlot);