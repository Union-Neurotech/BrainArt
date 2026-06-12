"""
brainart_muse_server.py
Live Muse -> [valence, arousal] WebSocket server for the BrainArt web app.

Pipeline (matches train_muse_va_live.py exactly):
  BrainFlow Muse stream (4 ch: TP9, AF7, AF8, TP10 @ 256 Hz)
    -> reorder to model order [AF7, AF8, TP9, TP10]
    -> 60-90 s rest BASELINE: per-feature mean/std (the user's neutral point)
    -> sliding WINDOW (default 30 s), HOP (default 1 s):
         per-1s DE band powers [4ch, Wsec, 5band]
         -> feat_full (mean+std+slope + asymmetry/ratios) = 65 dims
         -> (x - baseline_mean) / baseline_std
         -> Ridge model -> /pred_std -> tanh -> EMA smooth
    -> broadcast JSON {valence, arousal, alpha, beta, theta, delta, gamma, status}

Run with a real Muse:
    pip install brainflow websockets numpy
    python brainart_muse_server.py --board muse_2          # or muse_s / muse_2016
No hardware? Test the web wiring with a synthetic stream:
    python brainart_muse_server.py --sim

The web app connects to ws://localhost:8765 .
"""
import argparse, asyncio, json, threading, time, math
import numpy as np

# ---- band defs (must match training feature semantics) ----
BANDS = [('delta',1,4),('theta',4,8),('alpha',8,13),('beta',13,30),('gamma',30,45)]
BAND_NAMES = [b[0] for b in BANDS]
ALPHA, BETA = 2, 3
FS = 256
WINDOW_SEC = 30
HOP_SEC = 1.0
BASELINE_SEC = 90
EMA = 0.6                      # output smoothing (higher = smoother/slower)

# ---- feature extractor (identical math to train_muse_va_live.feat_full) ----
def feat_full(x):
    """x: [4ch, Wsec, 5band] -> [65]."""
    x = x[None]                                   # [1,4,W,5]
    m = x.mean(2); s = x.std(2)
    t = np.linspace(-1, 1, x.shape[2])
    slope = ((x - x.mean(2,keepdims=True)) * t[None,None,:,None]).sum(2) / (t**2).sum()
    base = np.concatenate([m, s, slope], 1).reshape(1, -1)
    FP1,FP2,A1,A2 = m[:,0],m[:,1],m[:,2],m[:,3]
    eng = np.stack([
        FP2[:,ALPHA]-FP1[:,ALPHA],
        A2[:,ALPHA]-A1[:,ALPHA],
        (FP1[:,BETA]+FP2[:,BETA])-(FP1[:,ALPHA]+FP2[:,ALPHA]),
        m[:,:,BETA].mean(1)-m[:,:,ALPHA].mean(1),
        m[:,:,ALPHA].mean(1),
    ], 1)
    return np.concatenate([base, eng], 1)[0]      # [65]

def de_bandpowers(seg, fs):
    """seg: [4ch, fs] one second. -> [4ch, 5band] differential-entropy-like (log power)."""
    out = np.empty((seg.shape[0], len(BANDS)))
    freqs = np.fft.rfftfreq(seg.shape[1], 1/fs)
    win = np.hanning(seg.shape[1])
    for c in range(seg.shape[0]):
        psd = np.abs(np.fft.rfft((seg[c]-seg[c].mean())*win))**2
        for bi,(_,lo,hi) in enumerate(BANDS):
            p = psd[(freqs>=lo)&(freqs<hi)].mean() if np.any((freqs>=lo)&(freqs<hi)) else 1e-12
            out[c,bi] = math.log(p + 1e-12)        # DE ~ log power (offset absorbed by baseline z-score)
    return out

def window_features(buf, fs):
    """buf: [4ch, WINDOW_SEC*fs] -> feat_full + per-band mean (for art bands)."""
    nsec = buf.shape[1] // fs
    de = np.stack([de_bandpowers(buf[:, i*fs:(i+1)*fs], fs) for i in range(nsec)], 1)  # [4,nsec,5]
    f = feat_full(de)
    bands = de.mean((0,1))                          # [5] avg DE per band over window
    return f, bands

# ---- model ----
class Model:
    def __init__(self, path):
        z = np.load(path)
        self.coef = z['coef']; self.intercept = z['intercept']; self.pred_std = z['pred_std']
    def predict(self, x):                           # x: [65] already baseline-normalized
        p = self.coef @ x + self.intercept          # [2] subject-z units
        return np.tanh(p / self.pred_std)           # [-1,1]

# ---- shared latest result ----
latest = {"valence":0.0,"arousal":0.0,
          **{b:0.0 for b in BAND_NAMES},
          "status":"starting"}
lock = threading.Lock()

def set_latest(**kw):
    with lock: latest.update(kw)

# ---- acquisition + inference thread ----
def acquire_loop(args):
    win_n = int(WINDOW_SEC*FS)
    if args.sim:
        set_latest(status="sim")
        ph = 0.0
        # warm "baseline" instantly in sim
        while True:
            ph += 0.02
            v = 0.6*math.sin(ph*0.7) + 0.15*np.random.randn()
            a = 0.6*math.sin(ph*0.4+1.0) + 0.15*np.random.randn()
            bands = {b: float(np.clip(0.5+0.4*math.sin(ph*(0.3+i*0.2)),0,1)) for i,b in enumerate(BAND_NAMES)}
            set_latest(valence=float(np.clip(v,-1,1)), arousal=float(np.clip(a,-1,1)),
                       status="live", **bands)
            time.sleep(HOP_SEC)
        return

    # ---- real Muse ----
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    board_map = {"muse_2":BoardIds.MUSE_2_BOARD,"muse_s":BoardIds.MUSE_S_BOARD,
                 "muse_2016":BoardIds.MUSE_2016_BOARD}
    bid = board_map[args.board]
    params = BrainFlowInputParams()
    if args.serial: params.serial_number = args.serial
    board = BoardShim(bid, params)
    fs = BoardShim.get_sampling_rate(bid)
    eeg = BoardShim.get_eeg_channels(bid)            # BrainFlow Muse order: TP9,AF7,AF8,TP10
    # reorder to model order AF7,AF8,TP9,TP10:
    reorder = [eeg[1], eeg[2], eeg[0], eeg[3]]
    globals()['FS'] = fs; win_n = int(WINDOW_SEC*fs)
    board.prepare_session(); board.start_stream()
    model = Model(args.model)

    def grab(nsamp):
        d = board.get_current_board_data(nsamp)
        return d[reorder, :]                          # [4, <=nsamp]

    # ---- baseline ----
    set_latest(status="baseline")
    print(f"[baseline] sit still & relax for {BASELINE_SEC}s ...")
    time.sleep(WINDOW_SEC + 2)                        # let buffer fill one window
    feats=[]
    t_end = time.time() + (BASELINE_SEC - WINDOW_SEC)
    while time.time() < t_end:
        buf = grab(win_n)
        if buf.shape[1] >= win_n:
            f,_ = window_features(buf[:, -win_n:], fs); feats.append(f)
        time.sleep(HOP_SEC)
    feats = np.asarray(feats)
    base_mu, base_sd = feats.mean(0), feats.std(0).clip(1e-6)
    print(f"[baseline] done ({len(feats)} samples). going live.")
    set_latest(status="live")

    # ---- live ----
    # baseline band level/spread per band: first 20 features are per-ch mean DE
    # ([4ch x 5band]); average over channels -> per-band baseline mean/std.
    base_band_mu = base_mu[:20].reshape(4,5).mean(0)
    base_band_sd = base_sd[:20].reshape(4,5).mean(0).clip(1e-6)
    sv = sa = 0.0
    while True:
        buf = grab(win_n)
        if buf.shape[1] >= win_n:
            f, bands = window_features(buf[:, -win_n:], fs)
            v,a = model.predict((f-base_mu)/base_sd)
            sv = EMA*sv + (1-EMA)*float(v); sa = EMA*sa + (1-EMA)*float(a)
            # bands -> 0..1 vs the user's own baseline (sigmoid of z-score) for the art
            bvals = {BAND_NAMES[i]: float(1/(1+math.exp(-(bands[i]-base_band_mu[i])/base_band_sd[i])))
                     for i in range(5)}
            set_latest(valence=sv, arousal=sa, status="live", **bvals)
        time.sleep(HOP_SEC)

# ---- websocket broadcast ----
async def ws_main(args):
    import websockets
    clients = set()
    async def handler(ws):
        clients.add(ws)
        try:
            async for _ in ws: pass
        finally:
            clients.discard(ws)
    async def broadcaster():
        while True:
            with lock: msg = json.dumps(latest)
            if clients:
                await asyncio.gather(*[c.send(msg) for c in list(clients)], return_exceptions=True)
            await asyncio.sleep(0.25)               # 4 Hz; JS lerps between
    async with websockets.serve(handler, args.host, args.port):
        print(f"WebSocket up on ws://{args.host}:{args.port}")
        await broadcaster()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="muse_2", choices=["muse_2","muse_s","muse_2016"])
    ap.add_argument("--serial", default="")
    ap.add_argument("--model", default="muse_va_live.npz")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--sim", action="store_true", help="synthetic stream, no Muse/model")
    args = ap.parse_args()
    threading.Thread(target=acquire_loop, args=(args,), daemon=True).start()
    asyncio.run(ws_main(args))

if __name__ == "__main__":
    main()