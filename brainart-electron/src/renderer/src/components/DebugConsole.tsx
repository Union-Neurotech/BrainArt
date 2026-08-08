import { useEffect, useRef } from 'react'
import RawWavePreview from './RawWavePreview'

export interface LogLine {
  level: string
  message: string
  time: string
}

interface Props {
  logs: LogLine[]
  wavesRef: React.MutableRefObject<number[][]>
}

/** Bottom bar: scrolling log console (left) + raw time-series preview (right). */
export default function DebugConsole({ logs, wavesRef }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [logs])

  return (
    <div className="console-grid">
      <div className="console-log" ref={scrollRef}>
        {logs.length === 0 && <div className="log-line log-info">- debug console -</div>}
        {logs.map((l, i) => (
          <div key={i} className={`log-line log-${l.level}`}>
            <span className="log-time">{l.time}</span> {l.message}
          </div>
        ))}
      </div>
      <div className="console-wave">
        <RawWavePreview wavesRef={wavesRef} />
      </div>
    </div>
  )
}
