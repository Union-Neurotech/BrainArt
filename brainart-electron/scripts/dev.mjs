// Launch electron-vite dev with a clean environment.
//
// Some host terminals (notably anything spawned by another Electron app, e.g.
// an IDE) set ELECTRON_RUN_AS_NODE=1. If that leaks into the child Electron
// process it runs as plain Node, so `require('electron')` returns a path string
// and `electron.app` is undefined. Stripping it here makes `npm run dev` robust.
import { spawn } from 'node:child_process'

delete process.env.ELECTRON_RUN_AS_NODE

const child = spawn('npx', ['electron-vite', 'dev'], {
  stdio: 'inherit',
  shell: true,
  env: process.env
})

child.on('exit', (code) => process.exit(code ?? 0))
