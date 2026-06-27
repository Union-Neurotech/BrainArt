import { contextBridge } from 'electron'

// The renderer dials the same WebSocket port the main process gave the backend.
const wsPort = Number(process.env.BRAINART_WS_PORT ?? 17321)

contextBridge.exposeInMainWorld('brainart', { wsPort })
