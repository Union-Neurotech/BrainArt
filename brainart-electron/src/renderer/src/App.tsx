import { useCallback, useRef, useState } from 'react'
import { IonApp } from '@ionic/react'
import Viewport, { ViewportHandle } from './components/Viewport'
import StatusBar from './components/StatusBar'
import DebugConsole, { LogLine } from './components/DebugConsole'
import { BrainState, DEFAULT_STATE } from './state'
import { Board, StatusMsg, useBackend } from './ws'

const MAX_LOGS = 300

export default function App() {
  // Shared visual state: a mutable ref the GL loop reads + a UI copy for React.
  const stateRef = useRef<BrainState>({ ...DEFAULT_STATE })
  const [ui, setUi] = useState<BrainState>(stateRef.current)

  // Live raw waves: ref only (updated ~15 Hz, drawn on the preview's own RAF).
  const wavesRef = useRef<number[][]>([])

  const viewportRef = useRef<ViewportHandle>(null)

  const [boards, setBoards] = useState<Board[]>([])
  const [selectedBoard, setSelectedBoard] = useState('')
  const [port, setPort] = useState('')
  const [connected, setConnected] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [device, setDevice] = useState<string | null>(null)
  const [logs, setLogs] = useState<LogLine[]>([])

  const addLog = useCallback((level: string, message: string) => {
    const time = new Date().toLocaleTimeString()
    setLogs((prev) => {
      const next = [...prev, { level, message, time }]
      return next.length > MAX_LOGS ? next.slice(next.length - MAX_LOGS) : next
    })
  }, [])

  const applyPatch = useCallback((patch: Record<string, number>) => {
    setUi((prev) => {
      const next = { ...prev }
      for (const k of Object.keys(patch)) {
        if (k in next) (next as any)[k] = patch[k]
      }
      stateRef.current = next
      return next
    })
  }, [])

  const { send, online } = useBackend({
    onLog: addLog,
    onBoards: (b) => {
      setBoards(b)
      setSelectedBoard((cur) => cur || (b[0]?.name ?? ''))
    },
    onStatus: (s: StatusMsg) => {
      setConnected(s.connected)
      setStreaming(s.streaming)
      setDevice(s.device)
    },
    onWaves: (channels) => {
      wavesRef.current = channels
    },
    onState: applyPatch
  })

  const setSlider = useCallback((key: keyof BrainState, value: number) => {
    setUi((prev) => {
      const next = { ...prev, [key]: value }
      stateRef.current = next
      return next
    })
  }, [])

  const usingPort = boards.find((b) => b.name === selectedBoard)?.using_port ?? false

  const onConnect = () =>
    send({ type: 'connect', board: selectedBoard, port: usingPort ? port || null : null })
  const onDisconnect = () => send({ type: 'disconnect' })
  const onStart = () => send({ type: 'start' })
  const onStop = () => send({ type: 'stop' })

  const onSave = () => {
    const png = viewportRef.current?.screenshot()
    if (png) send({ type: 'save_image', png })
  }
  const onPrint = () => {
    const png = viewportRef.current?.screenshot()
    if (png) send({ type: 'print_image', png })
  }

  return (
    <IonApp>
      <div className="app-grid">
        <div className="cell-viewport">
          <Viewport ref={viewportRef} stateRef={stateRef} />
        </div>
        <div className="cell-status">
          <StatusBar
            boards={boards}
            selectedBoard={selectedBoard}
            onSelectBoard={setSelectedBoard}
            port={port}
            onPort={setPort}
            online={online}
            connected={connected}
            streaming={streaming}
            device={device}
            onConnect={onConnect}
            onDisconnect={onDisconnect}
            onStart={onStart}
            onStop={onStop}
            ui={ui}
            onSlider={setSlider}
            onSave={onSave}
            onPrint={onPrint}
          />
        </div>
        <div className="cell-console">
          <DebugConsole logs={logs} wavesRef={wavesRef} />
        </div>
      </div>
    </IonApp>
  )
}
