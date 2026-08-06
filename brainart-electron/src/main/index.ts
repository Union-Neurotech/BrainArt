import { app, BrowserWindow } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import { join } from 'path'
import { existsSync } from 'fs'

// out/main/index.js -> out -> brainart-electron -> <project root>
const PROJECT_ROOT = join(__dirname, '..', '..', '..')
const SERVER_SCRIPT = join(PROJECT_ROOT, 'src', 'server.py')
const WS_PORT = Number(process.env.BRAINART_WS_PORT ?? 17321)

let py: ChildProcess | null = null

/** Prefer the project's .esp venv, then an explicit override, then system python. */
function resolvePython(): string {
  if (process.env.BRAINART_PYTHON) return process.env.BRAINART_PYTHON
  const venvWin = join(PROJECT_ROOT, '.esp', 'Scripts', 'python.exe')
  const venvNix = join(PROJECT_ROOT, '.esp', 'bin', 'python')
  if (existsSync(venvWin)) return venvWin
  if (existsSync(venvNix)) return venvNix
  return process.platform === 'win32' ? 'python' : 'python3'
}

function startBackend(): void {
  const python = resolvePython()
  const args = [SERVER_SCRIPT, '--port', String(WS_PORT)]

  // Images go to the user's Downloads folder by default. Resolve it here rather
  // than letting Python guess: app.getPath('downloads') asks the OS for the real
  // known-folder location, which is correct even when Downloads has been moved
  // off ~/. Opt out with BRAINART_SAVE_TO_PROJECT=1, or point somewhere specific
  // with BRAINART_IMAGE_DIR -- in that case pass no flag at all, since a CLI
  // flag would take precedence over the env var the backend is meant to read.
  if (process.env.BRAINART_IMAGE_DIR) {
    // inherited through env below
  } else if (process.env.BRAINART_SAVE_TO_PROJECT) {
    args.push('--project-images')
  } else {
    args.push('--image-dir', app.getPath('downloads'))
  }

  console.log(`[main] spawning backend: ${python} ${args.join(' ')}`)
  py = spawn(python, args, {
    cwd: join(PROJECT_ROOT, 'src'),
    env: { ...process.env, BRAINART_WS_PORT: String(WS_PORT) }
  })
  py.stdout?.on('data', (d) => process.stdout.write(`[py] ${d}`))
  py.stderr?.on('data', (d) => process.stderr.write(`[py] ${d}`))
  py.on('exit', (code) => console.log(`[main] backend exited (code ${code})`))
  py.on('error', (err) => console.error(`[main] failed to start backend: ${err}`))
}

function killBackend(): void {
  if (py) {
    py.kill()
    py = null
  }
}

function createWindow(): void {
  const win = new BrowserWindow({
    width: 1366,
    height: 820,
    minWidth: 1024,
    minHeight: 640,
    backgroundColor: '#000000',
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(__dirname, '..', 'preload', 'index.js'),
      sandbox: false
    }
  })

  if (process.env.ELECTRON_RENDERER_URL) {
    win.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    win.loadFile(join(__dirname, '..', 'renderer', 'index.html'))
  }
}

app.whenReady().then(() => {
  startBackend()
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  killBackend()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', killBackend)
process.on('exit', killBackend)
