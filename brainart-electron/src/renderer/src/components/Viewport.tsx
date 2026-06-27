import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import { createRenderer } from '../gl/renderer.js'
import type { BrainState } from '../state'

interface Props {
  stateRef: React.MutableRefObject<BrainState>
}

export interface ViewportHandle {
  screenshot: () => string
  newSeed: () => void
}

/** 16:9 WebGL CPPN viewport. Runs its own RAF loop reading the shared stateRef. */
const Viewport = forwardRef<ViewportHandle, Props>(function Viewport({ stateRef }, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rendererRef = useRef<ReturnType<typeof createRenderer> | null>(null)

  useEffect(() => {
    if (!canvasRef.current) return
    const r = createRenderer(canvasRef.current, stateRef)
    rendererRef.current = r
    return () => r.destroy()
  }, [stateRef])

  useImperativeHandle(ref, () => ({
    screenshot: () => rendererRef.current?.screenshot() ?? '',
    newSeed: () => rendererRef.current?.newSeed()
  }))

  return (
    <div className="viewport-frame">
      <canvas ref={canvasRef} className="viewport-canvas" />
    </div>
  )
})

export default Viewport
