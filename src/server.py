# Leonardo Ferrisi 2026
# ------------------------------
# BrainArt WebSocket backend
# ------------------------------
# Description:
#   Bridges the Electron renderer to BrainFlow. Speaks a small JSON protocol over
#   a local WebSocket. Reuses the existing device + preprocessing code:
#     - communications.Comms              (connect / stream / disconnect)
#     - preprocessing.get_simple_feature_vector  (averaged feature vector)
#     - assets.board_id_pairs             (board list + port requirement)
#
#   Renderer -> backend : list_boards | connect | disconnect | start | stop |
#                         save_image | print_image
#   backend  -> renderer: boards | log | status | waves | state
#
# Run standalone:  python server.py --port 17321

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import traceback

# Make local-module imports work regardless of the spawning cwd.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import websockets

from brainflow.board_shim import BoardShim

from assets import board_id_pairs
from communications import Comms
from preprocessing import get_simple_feature_vector

PROJECT_ROOT = os.path.dirname(THIS_DIR)
PROJECT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "generated", "images")


def downloads_dir():
    """Best-effort path to the current user's Downloads folder.

    Not just ~/Downloads: the folder is relocatable on Windows (and localized
    on Linux), so consult the OS's own record of it before falling back.
    """
    home = os.path.expanduser("~")

    if sys.platform.startswith("win"):
        # The "Downloads" known folder. Users can move it to another drive, in
        # which case ~/Downloads doesn't exist at all.
        guid = "{374DE290-123F-4565-9164-39C4925E467B}"
        try:
            import winreg

            key = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:
                path = os.path.expandvars(winreg.QueryValueEx(k, guid)[0])
                if path:
                    return path
        except Exception:
            pass

    elif not sys.platform == "darwin":
        # Linux/BSD: XDG user dirs, which may be localized (e.g. "Téléchargements").
        xdg = os.environ.get("XDG_DOWNLOAD_DIR")
        if xdg:
            return os.path.expandvars(xdg)
        try:
            cfg = os.path.join(
                os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config"),
                "user-dirs.dirs",
            )
            with open(cfg, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("XDG_DOWNLOAD_DIR"):
                        raw = line.split("=", 1)[1].strip().strip('"')
                        return os.path.expandvars(raw.replace("$HOME", home))
        except Exception:
            pass

    return os.path.join(home, "Downloads")

PREVIEW_HZ = 15            # how often live wave snapshots are pushed
PREVIEW_SAMPLES = 256      # samples per channel in each snapshot
PREVIEW_MAX_CHANNELS = 8   # cap traces in the debug preview

# Live rolling metrics pushed during streaming (non-destructive peek + EMA).
METRICS_HZ = 10            # target cadence; the loop is sequential, so it never
                           # overlaps computations and self-throttles to the
                           # actual ML compute time (slower hardware just emits
                           # less often, down toward ~1 Hz).
METRICS_WINDOW_SEC = 4.0   # recent-data window the rolling metrics summarize;
                           # also acts as the warm-up gate (no emit until full).
METRICS_EMA_ALPHA = 0.3    # EMA smoothing factor (higher = snappier, noisier).

# Keys returned by get_simple_feature_vector, in order. These mirror the
# renderer's visual-state fields exactly so a `state` patch drives the shader.
METRIC_KEYS = [
    "alpha", "beta", "delta", "theta", "gamma",
    "concentration", "mindfulness", "relaxation",
]


def _board_id(info):
    """board_id_pairs stores either an IntEnum or a raw int."""
    bid = info["id"]
    return bid.value if hasattr(bid, "value") else int(bid)


class Backend:
    def __init__(self, image_dir=None):
        # Default destination is the user's Downloads folder -- saved artwork is
        # something people go looking for, and it shouldn't be buried in the repo.
        self.image_dir = image_dir or downloads_dir()
        self.clients = set()
        self.comms = None
        self.board_id = None
        self.board_name = None
        self.streaming = False
        self.preview_task = None
        self.metrics_task = None
        self.eeg_channels = []
        self._metrics_ema = None        # smoothed live-metric state
        self._metrics_err_logged = False  # rate-limit live-metric error logs

    # ---- outbound messaging -------------------------------------------------
    async def broadcast(self, msg):
        if not self.clients:
            return
        data = json.dumps(msg)
        await asyncio.gather(
            *(c.send(data) for c in list(self.clients)),
            return_exceptions=True,
        )

    async def log(self, message, level="info"):
        print(f"[{level}] {message}", flush=True)
        await self.broadcast({"type": "log", "level": level, "message": message})

    async def send_status(self):
        await self.broadcast({
            "type": "status",
            "connected": self.comms is not None and self.comms.is_connected,
            "streaming": self.streaming,
            "device": self.board_name,
        })

    def boards_payload(self):
        return [
            {"name": name, "id": _board_id(info), "using_port": info["using_port"]}
            for name, info in board_id_pairs.items()
        ]

    # ---- command dispatch ---------------------------------------------------
    async def handle(self, ws, msg):
        t = msg.get("type")
        if t == "list_boards":
            await ws.send(json.dumps({"type": "boards", "boards": self.boards_payload()}))
        elif t == "connect":
            await self.connect(msg.get("board"), msg.get("port"))
        elif t == "disconnect":
            await self.disconnect()
        elif t == "start":
            await self.start()
        elif t == "stop":
            await self.stop()
        elif t == "save_image":
            await self.save_image(msg.get("png"))
        elif t == "print_image":
            await self.print_image(msg.get("png"))
        else:
            await self.log(f"Unknown message type: {t}", "error")

    # ---- device lifecycle ---------------------------------------------------
    async def connect(self, board_name, port):
        if self.comms is not None and self.comms.is_connected:
            await self.log("Already connected.", "error")
            return
        if board_name not in board_id_pairs:
            await self.log(f"Unknown board: {board_name}", "error")
            return
        bid = _board_id(board_id_pairs[board_name])
        await self.log(f"Connecting to {board_name} ...")
        try:
            # Comms(board_id, port, connect_on_init, debug). connect() swallows
            # errors and leaves is_connected False on failure.
            comms = await asyncio.to_thread(Comms, bid, port, True, False)
            if not comms.is_connected:
                await self.log(f"Failed to connect to {board_name}. Check device/port.", "error")
                return
            self.comms = comms
            self.board_id = bid
            self.board_name = board_name
            self.eeg_channels = BoardShim.get_eeg_channels(bid)
            await self.log(f"Connected to {board_name} ({len(self.eeg_channels)} EEG channels).")
            await self.send_status()
        except Exception as e:
            await self.log(f"Connection error: {e}", "error")

    async def disconnect(self):
        if self.streaming:
            await self.stop()
        if self.comms is not None:
            try:
                await asyncio.to_thread(self.comms.disconnect)
            except Exception as e:
                await self.log(f"Disconnect error: {e}", "error")
        name = self.board_name
        self.comms = None
        self.board_id = None
        self.board_name = None
        self.eeg_channels = []
        await self.log(f"Disconnected from {name}." if name else "Disconnected.")
        await self.send_status()

    # ---- streaming ----------------------------------------------------------
    async def start(self):
        if self.comms is None or not self.comms.is_connected:
            await self.log("Not connected.", "error")
            return
        if self.streaming:
            await self.log("Already streaming.", "error")
            return
        try:
            await asyncio.to_thread(self.comms.start_stream)
            # Reset the ring buffer so the start->stop averaging window is clean.
            await asyncio.to_thread(self.comms.board.get_board_data)
            self.streaming = True
            await self.log("Streaming started.")
            await self.send_status()
            # Fresh smoothing state for each streaming session.
            self._metrics_ema = None
            self._metrics_err_logged = False
            self.preview_task = asyncio.create_task(self.preview_loop())
            self.metrics_task = asyncio.create_task(self.metrics_loop())
        except Exception as e:
            await self.log(f"Start error: {e}", "error")

    async def preview_loop(self):
        period = 1.0 / PREVIEW_HZ
        try:
            while self.streaming:
                try:
                    data = await asyncio.to_thread(
                        self.comms.board.get_current_board_data, PREVIEW_SAMPLES
                    )
                    if data is not None and data.shape[1] > 0 and self.eeg_channels:
                        chans = [
                            data[ch].tolist()
                            for ch in self.eeg_channels[:PREVIEW_MAX_CHANNELS]
                        ]
                        await self.broadcast({"type": "waves", "channels": chans})
                except Exception as e:
                    await self.log(f"Preview error: {e}", "error")
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            pass

    async def metrics_loop(self):
        """Push smoothed rolling metrics while streaming.

        Peeks the most recent METRICS_WINDOW_SEC of data *non-destructively*
        (get_current_board_data), so the full start->stop buffer that stop()
        averages stays untouched. Computes the feature vector in a worker
        thread, applies an EMA, and broadcasts a `state` patch -- the same
        message the renderer already applies to drive the shader + indicators.

        The loop awaits each compute, so it never overlaps or piles up: it
        targets METRICS_HZ but degrades gracefully to whatever the ML compute
        can sustain on the current hardware.
        """
        period = 1.0 / METRICS_HZ
        sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        window = int(METRICS_WINDOW_SEC * sampling_rate)
        try:
            while self.streaming:
                t0 = time.monotonic()
                try:
                    data = await asyncio.to_thread(
                        self.comms.board.get_current_board_data, window
                    )
                    # Warm-up gate: emit only once a full window has buffered.
                    if data is not None and data.shape[1] >= window:
                        patch = await asyncio.to_thread(self.compute_metrics, data)
                        patch = self._smooth_metrics(patch)
                        await self.broadcast({"type": "state", "patch": patch})
                        self._metrics_err_logged = False
                except Exception as e:
                    # Avoid spamming the console at the loop rate: log once,
                    # then stay quiet until the next successful computation.
                    if not self._metrics_err_logged:
                        await self.log(f"Live metrics error: {e}", "error")
                        self._metrics_err_logged = True
                # Self-throttle: sleep only the remainder of the target period.
                elapsed = time.monotonic() - t0
                if self.streaming and elapsed < period:
                    await asyncio.sleep(period - elapsed)
        except asyncio.CancelledError:
            pass

    def _smooth_metrics(self, patch):
        """Exponential moving average across successive live patches."""
        ema = self._metrics_ema
        if ema is None:
            ema = dict(patch)
        else:
            a = METRICS_EMA_ALPHA
            for k, v in patch.items():
                ema[k] = a * v + (1 - a) * ema.get(k, v)
        self._metrics_ema = ema
        return {k: float(v) for k, v in ema.items()}

    async def stop(self):
        if not self.streaming:
            await self.log("Not streaming.", "error")
            return
        self.streaming = False
        if self.preview_task:
            self.preview_task.cancel()
            self.preview_task = None
        if self.metrics_task:
            self.metrics_task.cancel()
            self.metrics_task = None
        try:
            data = await asyncio.to_thread(self.comms.board.get_board_data)
            await asyncio.to_thread(self.comms.stop_stream)
            await self.send_status()
            n = data.shape[1] if data is not None else 0
            await self.log(f"Streaming stopped. Collected {n} samples. Computing metrics ...")
            patch = await asyncio.to_thread(self.compute_metrics, data)
            await self.broadcast({"type": "state", "patch": patch})
            await self.log("Metrics: " + ", ".join(f"{k}={v:.2f}" for k, v in patch.items()))
        except Exception as e:
            await self.log(f"Stop/metrics error: {e}", "error")
            traceback.print_exc()

    def compute_metrics(self, data):
        """Runs in a worker thread. Averaged feature vector -> state patch."""
        if data is None or data.shape[1] < 10:
            raise ValueError("Not enough data collected to compute metrics.")
        fv = get_simple_feature_vector(data=data, boardID=self.board_id)
        return {k: float(v) for k, v in zip(METRIC_KEYS, fv)}

    # ---- image save / print -------------------------------------------------
    def _write_png(self, b64):
        if not b64:
            return None
        if "," in b64:                       # strip a data: URL header if present
            b64 = b64.split(",", 1)[1]
        os.makedirs(self.image_dir, exist_ok=True)
        # Timestamps are second-resolution, so two saves in the same second would
        # otherwise overwrite each other. Suffix until the name is free.
        stem = os.path.join(self.image_dir, f"brainart_{time.strftime('%Y%m%d_%H%M%S')}")
        path = f"{stem}.png"
        n = 1
        while os.path.exists(path):
            path = f"{stem}_{n}.png"
            n += 1
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        return path

    async def save_image(self, b64):
        try:
            path = await asyncio.to_thread(self._write_png, b64)
            if path:
                await self.log(f"Image saved: {path}")
            else:
                await self.log("No image data to save.", "error")
        except Exception as e:
            await self.log(f"Save failed: {e}", "error")

    def _os_print(self, path):
        if sys.platform.startswith("win"):
            os.startfile(path, "print")      # default printer
        else:                                # macOS / Linux
            os.system(f'lpr "{path}"')

    async def print_image(self, b64):
        try:
            path = await asyncio.to_thread(self._write_png, b64)
        except Exception as e:
            await self.log(f"Print failed (save step): {e}", "error")
            return
        if not path:
            await self.log("No image data to print.", "error")
            return
        await self.log(f"Image saved for printing: {path}")
        try:
            await asyncio.to_thread(self._os_print, path)
            await self.log("Sent image to printer.")
        except Exception as e:
            await self.log(f"Print failed: {e}", "error")

    # ---- connection handler -------------------------------------------------
    async def ws_handler(self, ws):
        self.clients.add(ws)
        try:
            await ws.send(json.dumps({"type": "boards", "boards": self.boards_payload()}))
            await self.send_status()
            await self.log("Renderer connected.")
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                await self.handle(ws, msg)
        except websockets.ConnectionClosed as e:
            # Don't swallow this. A frame the server *refuses* (e.g. one that
            # exceeds max_size) shows up here as a close we sent, with no trace
            # anywhere else -- that is what made oversized screenshot saves fail
            # invisibly. Check both sides: the initiator carries the real code.
            # Print rather than self.log(), since the socket is already gone;
            # this lands in the Electron main process as "[py] ...".
            for side, close in (("sent", e.sent), ("recv", e.rcvd)):
                if close is not None and close.code != 1000:
                    print(
                        f"[error] Renderer socket closed ({side}): "
                        f"{close.code} {close.reason}",
                        flush=True,
                    )
        finally:
            self.clients.discard(ws)


async def main():
    parser = argparse.ArgumentParser(description="BrainArt WebSocket backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("BRAINART_WS_PORT", 17321)),
    )
    dest = parser.add_mutually_exclusive_group()
    dest.add_argument(
        "--downloads", action="store_true",
        help="save images to the user's Downloads folder (this is the default)",
    )
    dest.add_argument(
        "--project-images", action="store_true",
        help="save images to <project>/generated/images instead of Downloads",
    )
    dest.add_argument(
        "--image-dir", metavar="DIR",
        help="save images to DIR (overrides $BRAINART_IMAGE_DIR)",
    )
    args = parser.parse_args()

    # Precedence: explicit flag > $BRAINART_IMAGE_DIR > Downloads (the default).
    if args.downloads:
        image_dir = downloads_dir()
    elif args.project_images:
        image_dir = PROJECT_IMAGE_DIR
    else:
        image_dir = args.image_dir or os.environ.get("BRAINART_IMAGE_DIR") or None

    backend = Backend(image_dir)
    print(f"BrainArt backend listening on ws://{args.host}:{args.port}", flush=True)
    print(f"Saving images to: {backend.image_dir}", flush=True)
    # max_size=None: screenshots arrive on this socket as base64 PNG data URLs,
    # which run to several MB for a detailed frame. The websockets default is a
    # 1 MiB inbound cap, and exceeding it kills the connection (close 1009)
    # *without* delivering the message -- which is exactly how "Save Image"
    # silently did nothing for any PNG over ~768 KiB, i.e. the busy frames that
    # live EEG produces. Unbounded is fine here: the listener is loopback-only
    # and carries a single local renderer.
    async with websockets.serve(
        backend.ws_handler, args.host, args.port, max_size=None
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
