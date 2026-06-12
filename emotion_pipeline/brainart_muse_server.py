"""
brainart_muse_server.py
Live Muse -> [valence, arousal] WebSocket server for the BrainArt web app.
Pipeline matches train_muse_va_live.py exactly. Supports per-user calibration
(affine rescale fit from a few known-VA clips) saved to muse_va_calib.json.
Control messages (client -> server):
    {"cmd":"calib_clip","label":"joy","v":0.8,"a":0.8,"secs":90}
    {"cmd":"calib_finish"}
    {"cmd":"calib_reset"}
Run: python brainart_muse_server.py --board muse_2   |   --sim
Web app connects to ws://localhost:8765 .

WebSocket payload (server -> client), JSON, ~4 Hz:
    valence, arousal            : float in [-1,1]
    delta..gamma                : float in [0,1]  (aggregate, normalized to baseline)
    eeg     : {"AF7":[...µV], "AF8":[...], "TP9":[...], "TP10":[...]}  last ~1s raw
    psd     : {"freqs":[Hz...], "AF7":[dB...], ...}  per-channel power spectrum (<=45 Hz)
    bands_ch: {"AF7":{delta,theta,alpha,beta,gamma}, ...}  per-channel log band power
    eeg_seq : int  monotonic id so the client only appends each raw chunk once
    status, calibrated
"""
import argparse, asyncio, itertools, json, os, queue, threading, time, math
import numpy as np

BANDS = [('delta',1,4),('theta',4,8),('alpha',8,13),('beta',13,30),('gamma',30,45)]
BAND_NAMES = [b[0] for b in BANDS]
ALPHA, BETA = 2, 3
FS = 256
# Channel labels in the SAME order the model sees them, i.e. reorder = [eeg[1],eeg[2],eeg[0],eeg[3]]
CH_LABELS = ["AF7", "AF8", "TP9", "TP10"]
SEQ = itertools.count()       # monotonically increasing id so the client can dedupe raw chunks
VIZ_MAXHZ = 45                # cap PSD output frequency (Hz)
WINDOW_SEC = 30
HOP_SEC = 1.0
BASELINE_SEC = 90
EMA = 0.6
CALIB_PATH = "muse_va_calib.json"

def feat_full(x):
    x = x[None]
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
    return np.concatenate([base, eng], 1)[0]

def de_bandpowers(seg, fs):
    out = np.empty((seg.shape[0], len(BANDS)))
    freqs = np.fft.rfftfreq(seg.shape[1], 1/fs)
    win = np.hanning(seg.shape[1])
    for c in range(seg.shape[0]):
        psd = np.abs(np.fft.rfft((seg[c]-seg[c].mean())*win))**2
        for bi,(_,lo,hi) in enumerate(BANDS):
            sel = (freqs>=lo)&(freqs<hi)
            p = psd[sel].mean() if np.any(sel) else 1e-12
            out[c,bi] = math.log(p + 1e-12)
    return out

def window_features(buf, fs):
    nsec = buf.shape[1] // fs
    de = np.stack([de_bandpowers(buf[:, i*fs:(i+1)*fs], fs) for i in range(nsec)], 1)
    return feat_full(de), de.mean((0,1))

def viz_payload(buf, fs, seq):
    """Build raw-EEG + PSD + per-channel band powers for the web visualizer.
    Uses only the most recent ~1s so the client can append a fresh chunk each tick."""
    if buf is None or buf.shape[1] < 16:
        return {"eeg_seq": seq}
    seg = buf[:, -fs:] if buf.shape[1] >= fs else buf
    n = seg.shape[1]
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1/fs)
    fsel = freqs <= VIZ_MAXHZ
    nch = min(len(CH_LABELS), seg.shape[0])
    eeg, bands_ch = {}, {}
    psd = {"freqs": np.round(freqs[fsel], 2).tolist()}
    for c in range(nch):
        eeg[CH_LABELS[c]] = np.round(seg[c], 2).astype(float).tolist()
        p = np.abs(np.fft.rfft((seg[c] - seg[c].mean()) * win))**2
        psd[CH_LABELS[c]] = np.round(10*np.log10(p[fsel] + 1e-12), 2).tolist()
        bp = {}
        for (name, lo, hi) in BANDS:
            sel = (freqs >= lo) & (freqs < hi)
            bp[name] = round(float(np.log((p[sel].mean() if np.any(sel) else 1e-12) + 1e-12)), 3)
        bands_ch[CH_LABELS[c]] = bp
    return {"eeg": eeg, "psd": psd, "bands_ch": bands_ch, "eeg_seq": seq}

class Model:
    def __init__(self, path):
        z = np.load(path)
        self.coef = z['coef']; self.intercept = z['intercept']; self.pred_std = z['pred_std']
        self.calib = None
        if os.path.exists(CALIB_PATH):
            self.load_calib()
    def raw(self, x):
        return self.coef @ x + self.intercept
    def output(self, raw):
        if self.calib is not None:
            return np.clip(self.calib['a']*raw + self.calib['b'], -1, 1)
        return np.tanh(raw / self.pred_std)
    def fit_calib(self, anchors):
        R = np.array([a[0] for a in anchors]); T = np.array([a[1] for a in anchors])
        a = np.zeros(2); b = np.zeros(2)
        for d in (0,1):
            if len(R) >= 2 and R[:,d].std() > 1e-6:
                co = np.polyfit(R[:,d], T[:,d], 1); a[d], b[d] = co[0], co[1]
            else:
                a[d], b[d] = 1.0/self.pred_std[d], 0.0
        self.calib = {'a':a, 'b':b}
        json.dump({'a':a.tolist(),'b':b.tolist(),'n_anchors':len(anchors)},
                  open(CALIB_PATH,'w'), indent=2)
        return a, b
    def load_calib(self):
        d = json.load(open(CALIB_PATH))
        self.calib = {'a':np.array(d['a']), 'b':np.array(d['b'])}
    def reset_calib(self):
        self.calib = None
        if os.path.exists(CALIB_PATH): os.remove(CALIB_PATH)

latest = {"valence":0.0,"arousal":0.0, **{b:0.0 for b in BAND_NAMES},
          "status":"starting","calibrated":False}
lock = threading.Lock()
CMDQ = queue.Queue()

def set_latest(**kw):
    with lock: latest.update(kw)

def process_cmds(model, anchors, recording, raw):
    if recording is not None:
        recording['raws'].append(np.asarray(raw, dtype=float))
        if time.time() >= recording['until']:
            mean_raw = np.mean(recording['raws'], 0)
            anchors.append((mean_raw, np.array(recording['target'])))
            print(f"[calib] '{recording['label']}' raw={mean_raw.round(3)} "
                  f"target={recording['target']} ({len(anchors)} anchors)")
            recording = None
    while not CMDQ.empty():
        try: c = CMDQ.get_nowait()
        except queue.Empty: break
        cmd = c.get('cmd')
        if cmd == 'calib_clip' and recording is None:
            recording = {'label':c.get('label','?'),
                         'target':[float(c.get('v',0)), float(c.get('a',0))],
                         'until':time.time()+float(c.get('secs',90)),
                         'raws':[], 'status':f"calibrating:{c.get('label','?')}"}
            print(f"[calib] recording '{recording['label']}' {c.get('secs',90)}s")
        elif cmd == 'calib_finish':
            if len(anchors) >= 2:
                a,b = model.fit_calib(anchors)
                print(f"[calib] fit a={a.round(3)} b={b.round(3)} -> {CALIB_PATH}")
            else:
                print(f"[calib] need >=2 anchors, have {len(anchors)}")
        elif cmd == 'calib_reset':
            anchors.clear(); model.reset_calib(); print("[calib] reset")
    return recording

def acquire_loop(args):
    if args.sim:
        run_sim(); return
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    bid = {"muse_2":BoardIds.MUSE_2_BOARD,"muse_s":BoardIds.MUSE_S_BOARD,
           "muse_2016":BoardIds.MUSE_2016_BOARD}[args.board]
    params = BrainFlowInputParams()
    if args.serial: params.serial_number = args.serial
    board = BoardShim(bid, params)
    fs = BoardShim.get_sampling_rate(bid)
    eeg = BoardShim.get_eeg_channels(bid)
    reorder = [eeg[1], eeg[2], eeg[0], eeg[3]]
    win_n = int(WINDOW_SEC*fs)
    board.prepare_session(); board.start_stream()
    model = Model(args.model)
    grab = lambda n: board.get_current_board_data(n)[reorder, :]
    set_latest(status="baseline")
    print(f"[baseline] relax & sit still for {BASELINE_SEC}s ...")
    time.sleep(WINDOW_SEC + 2)
    feats=[]; t_end = time.time() + (BASELINE_SEC - WINDOW_SEC); warned=False
    while time.time() < t_end:
        buf = grab(win_n)
        if buf.shape[1] >= win_n:
            f,_ = window_features(buf[:, -win_n:], fs); feats.append(f)
        elif not warned:
            print(f"[baseline] waiting for data... {buf.shape[1]}/{win_n} samples"); warned=True
        set_latest(**viz_payload(buf, fs, next(SEQ)))   # show live channels while baselining
        time.sleep(HOP_SEC)
    feats = np.asarray(feats)
    if feats.ndim < 2 or feats.shape[0] == 0:
        print("[baseline] WARNING: collected 0 full windows -- the Muse stream produced "
              "insufficient data (check the connection / battery). Using a neutral baseline.")
        nfeat = feat_full(np.zeros((4, 1, len(BANDS)))).shape[0]
        base_mu, base_sd = np.zeros(nfeat), np.ones(nfeat)
    else:
        base_mu, base_sd = feats.mean(0), feats.std(0).clip(1e-6)
    base_band_mu = base_mu[:20].reshape(4,5).mean(0)
    base_band_sd = base_sd[:20].reshape(4,5).mean(0).clip(1e-6)
    print(f"[baseline] done ({feats.shape[0] if feats.ndim>1 else 0} samples). live.")
    set_latest(status="live", calibrated=model.calib is not None)
    sv = sa = 0.0; anchors = []; recording = None
    while True:
        buf = grab(win_n)
        viz = viz_payload(buf, fs, next(SEQ))
        if buf.shape[1] >= win_n:
            f, bands = window_features(buf[:, -win_n:], fs)
            raw = model.raw((f-base_mu)/base_sd)
            out = model.output(raw)
            sv = EMA*sv + (1-EMA)*float(out[0]); sa = EMA*sa + (1-EMA)*float(out[1])
            bvals = {BAND_NAMES[i]: float(1/(1+math.exp(-(bands[i]-base_band_mu[i])/base_band_sd[i])))
                     for i in range(5)}
            recording = process_cmds(model, anchors, recording, raw)
            st = recording['status'] if recording else "live"
            set_latest(valence=sv, arousal=sa, status=st,
                       calibrated=model.calib is not None, **bvals, **viz)
        else:
            set_latest(**viz)
        time.sleep(HOP_SEC)

def run_sim():
    set_latest(status="sim")
    model = type('M', (), {'calib':None,'pred_std':np.array([0.24,0.22])})()
    anchors=[]; recording=None; ph=0.0; sv=sa=0.0
    tt = np.arange(FS)/FS                       # 1s time base for synthetic raw
    while True:
        ph += 0.02
        # synthetic 4-channel EEG: per-band sinusoids + noise, in microvolt-like scale
        sim_buf = np.empty((4, FS))
        for c in range(4):
            sig = (20*np.sin(2*np.pi*(2+0.3*c)*tt + ph) +      # delta/theta
                   14*np.sin(2*np.pi*10*tt + ph*1.3 + c) +      # alpha
                    6*np.sin(2*np.pi*20*tt + ph*0.7) +          # beta
                    3*np.sin(2*np.pi*38*tt))                     # gamma
            sim_buf[c] = sig + 8*np.random.randn(FS)
        raw = np.array([0.6*math.sin(ph*0.7), 0.6*math.sin(ph*0.4+1.0)]) + 0.1*np.random.randn(2)
        if getattr(model,'calib',None) is not None:
            out = np.clip(model.calib['a']*raw+model.calib['b'],-1,1)
        else:
            out = np.tanh(raw/model.pred_std)
        sv = EMA*sv+(1-EMA)*float(out[0]); sa = EMA*sa+(1-EMA)*float(out[1])
        bands = {b: float(np.clip(0.5+0.4*math.sin(ph*(0.3+i*0.2)),0,1)) for i,b in enumerate(BAND_NAMES)}
        recording = sim_process(model, anchors, recording, raw)
        st = recording['status'] if recording else ("sim" if model.calib is None else "live")
        set_latest(valence=float(np.clip(sv,-1,1)), arousal=float(np.clip(sa,-1,1)),
                   status=st, calibrated=model.calib is not None, **bands,
                   **viz_payload(sim_buf, FS, next(SEQ)))
        time.sleep(HOP_SEC)

def sim_process(model, anchors, recording, raw):
    if recording is not None:
        recording['raws'].append(np.asarray(raw,dtype=float))
        if time.time() >= recording['until']:
            anchors.append((np.mean(recording['raws'],0), np.array(recording['target'])))
            print(f"[sim-calib] '{recording['label']}' captured ({len(anchors)} anchors)")
            recording = None
    while not CMDQ.empty():
        try: c = CMDQ.get_nowait()
        except queue.Empty: break
        cmd=c.get('cmd')
        if cmd=='calib_clip' and recording is None:
            recording={'label':c.get('label','?'),'target':[float(c.get('v',0)),float(c.get('a',0))],
                       'until':time.time()+float(c.get('secs',8)),'raws':[],
                       'status':f"calibrating:{c.get('label','?')}"}
        elif cmd=='calib_finish' and len(anchors)>=2:
            R=np.array([a[0] for a in anchors]); T=np.array([a[1] for a in anchors])
            a=np.zeros(2); b=np.zeros(2)
            for d in (0,1):
                if R[:,d].std()>1e-6:
                    co=np.polyfit(R[:,d],T[:,d],1); a[d],b[d]=co[0],co[1]
                else: a[d],b[d]=1.0/model.pred_std[d],0.0
            model.calib={'a':a,'b':b}; print(f"[sim-calib] fit a={a.round(3)} b={b.round(3)}")
        elif cmd=='calib_reset':
            anchors.clear(); model.calib=None
    return recording

async def ws_main(args):
    import websockets
    clients = set()
    async def handler(ws):
        clients.add(ws)
        try:
            async for raw in ws:
                try: CMDQ.put(json.loads(raw))
                except Exception: pass
        finally:
            clients.discard(ws)
    async def broadcaster():
        while True:
            with lock: msg = json.dumps(latest)
            if clients:
                await asyncio.gather(*[c.send(msg) for c in list(clients)], return_exceptions=True)
            await asyncio.sleep(0.25)
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
    ap.add_argument("--sim", action="store_true")
    args = ap.parse_args()
    threading.Thread(target=acquire_loop, args=(args,), daemon=True).start()
    asyncio.run(ws_main(args))

if __name__ == "__main__":
    main()