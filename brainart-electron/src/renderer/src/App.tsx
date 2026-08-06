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

  // Capture the canvas and hand it to the backend, which writes the file and
  // logs the path back. Both failure modes report themselves -- silence here is
  // what made a failed save indistinguishable from a dead button.
  const sendImage = (type: 'save_image' | 'print_image') => {
    const png = viewportRef.current?.screenshot()
    if (!png) {
      addLog('error', 'Could not capture the canvas.')
      return
    }
    if (!send({ type, png })) addLog('error', 'Backend offline — image not sent.')
  }

  const onSave = () => sendImage('save_image')
  const onPrint = () => sendImage('print_image')

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
