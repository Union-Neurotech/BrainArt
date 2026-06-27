import { useEffect, useRef } from 'react'

interface Props {
  wavesRef: React.MutableRefObject<number[][]>
}

const COLORS = [
  '#ff7846', '#50a0ff', '#64e696', '#e6c84b',
  '#c864e6', '#46d7ff', '#ff6464', '#a0a0a0'
]

/**
 * Tiny oscilloscope: draws each channel as a polyline with a vertical offset,
 * auto-ranged per channel. Reads the latest snapshot from wavesRef each frame
 * (no React state churn at preview frequency).
 */
export default function RawWavePreview({ wavesRef }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    let raf = 0

    const draw = (): void => {
      const rect = canvas.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      const w = Math.max(1, Math.round(rect.width * dpr))
      const h = Math.max(1, Math.round(rect.height * dpr))
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }

      ctx.clearRect(0, 0, w, h)
      const chans = wavesRef.current
      if (chans && chans.length) {
        const rowH = h / chans.length
        chans.forEach((samples, ci) => {
          if (!samples || samples.length < 2) return
          let min = Infinity
          let max = -Infinity
          for (const v of samples) {
            if (v < min) min = v
            if (v > max) max = v
          }
          const range = max - min || 1
          const mid = rowH * ci + rowH / 2
          ctx.beginPath()
          ctx.strokeStyle = COLORS[ci % COLORS.length]
          ctx.lineWidth = 1 * dpr
          for (let i = 0; i < samples.length; i++) {
            const x = (i / (samples.length - 1)) * w
            const norm = (samples[i] - min) / range - 0.5 // -0.5..0.5
            const y = mid - norm * rowH * 0.8
            if (i === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          }
          ctx.stroke()
        })
      }
      raf = requestAnimationFrame(draw)
    }

    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [wavesRef])

  return <canvas ref={canvasRef} className="wave-canvas" />
}
