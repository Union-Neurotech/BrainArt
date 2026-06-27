import { useCallback, useEffect, useRef, useState } from 'react'

export interface Board {
  name: string
  id: number
  using_port: boolean
}

export interface StatusMsg {
  connected: boolean
  streaming: boolean
  device: string | null
}

export interface BackendHandlers {
  onLog?: (level: string, message: string) => void
  onBoards?: (boards: Board[]) => void
  onStatus?: (status: StatusMsg) => void
  onWaves?: (channels: number[][]) => void
  onState?: (patch: Record<string, number>) => void
}

const WS_PORT = (window as any).brainart?.wsPort ?? 17321
const WS_URL = `ws://127.0.0.1:${WS_PORT}`

/**
 * Connects to the Python backend WebSocket, auto-reconnecting if it drops.
 * Returns `send` for outbound commands and `online` for the socket state.
 */
export function useBackend(handlers: BackendHandlers): {
  send: (obj: unknown) => void
  online: boolean
} {
  const wsRef = useRef<WebSocket | null>(null)
  const handlersRef = useRef(handlers)
  handlersRef.current = handlers
  const [online, setOnline] = useState(false)

  useEffect(() => {
    let closed = false
    let retry: ReturnType<typeof setTimeout> | undefined

    function connect(): void {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => setOnline(true)
      ws.onclose = () => {
        setOnline(false)
        if (!closed) retry = setTimeout(connect, 1000)
      }
      ws.onerror = () => ws.close()
      ws.onmessage = (ev) => {
        let msg: any
        try {
          msg = JSON.parse(ev.data)
        } catch {
          return
        }
        const h = handlersRef.current
        switch (msg.type) {
          case 'log':
            h.onLog?.(msg.level, msg.message)
            break
          case 'boards':
            h.onBoards?.(msg.boards)
            break
          case 'status':
            h.onStatus?.(msg)
            break
          case 'waves':
            h.onWaves?.(msg.channels)
            break
          case 'state':
            h.onState?.(msg.patch)
            break
        }
      }
    }

    connect()
    return () => {
      closed = true
      if (retry) clearTimeout(retry)
      wsRef.current?.close()
    }
  }, [])

  const send = useCallback((obj: unknown) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj))
  }, [])

  return { send, online }
}
