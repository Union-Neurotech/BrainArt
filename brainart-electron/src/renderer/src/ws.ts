import { useCallback, useEffect, useRef, useState } from 'react'

// websockets interface

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
  /** Socket dropped: any device state the UI is holding is now stale. */
  onOffline?: () => void
}

const WS_PORT = (window as any).brainart?.wsPort ?? 17321
const WS_URL = `ws://127.0.0.1:${WS_PORT}`

/**
 * Connects to the Python backend WebSocket, auto-reconnecting if it drops.
 * Returns `send` for outbound commands and `online` for the socket state.
 */
export function useBackend(handlers: BackendHandlers): {
  send: (obj: unknown) => boolean
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
      ws.onclose = (ev) => {
        setOnline(false)
        // `connected`/`streaming`/`device` only ever arrive in a `status`
        // message, so without this the UI keeps claiming a live device after the
        // backend is gone -- leaving a red Disconnect button that can never
        // revert, because only an inbound status could flip it back.
        handlersRef.current.onOffline?.()
        // Surface abnormal closes. The reconnect below otherwise hides them
        // completely: a frame the backend refuses (close 1009, "message too
        // big") looks identical to nothing having happened at all.
        if (!closed && ev.code !== 1000) {
          handlersRef.current.onLog?.(
            'error',
            `Backend connection lost (code ${ev.code}${ev.reason ? `: ${ev.reason}` : ''}); reconnecting…`
          )
        }
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

  /** Returns false if the socket wasn't open, so callers can report the drop. */
  const send = useCallback((obj: unknown): boolean => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return false
    ws.send(JSON.stringify(obj))
    return true
  }, [])

  return { send, online }
}
