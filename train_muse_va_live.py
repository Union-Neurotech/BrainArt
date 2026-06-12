"""
train_muse_va_live.py
Train the production 4-channel Muse VA regressor for live inference.

Key differences from the per-1s MLP:
  * sample unit = full 30 s trial (engineered features), not 1 s window
  * features    = per channel-band mean+std+slope (DE) + alpha asymmetry +
                  beta-alpha ratios  (65 dims for 4 channels)
  * normalization = PER-SUBJECT z-score of features (the training-time analogue
                    of a live per-user rest baseline) and PER-SUBJECT z-score of
                    labels -> the model learns *relative-to-your-own-baseline*
                    valence/arousal, which is what we can compute live.
  * model       = Ridge (linear). Boosting/LSTM did not beat it; signal is linear.

Saves muse_va_live.npz (coef/intercept + feature spec + output scaling) so the
live server can run with only numpy (no sklearn / torch).
"""
import pickle, glob, os, json, numpy as np, pandas as pd
from sklearn.linear_model import Ridge

DE_DIR   = "datasets/FACED/EEG_Features/DE"
VA_TABLE = "datasets/FACED/faced_va_table.csv"
MUSE_IDX = [0, 1, 30, 31]           # FP1,FP2,A1,A2  (AF7,AF8,TP9,TP10 proxies)
BANDS    = ['delta','theta','alpha','beta','gamma']
ALPHA, BETA = BANDS.index('alpha'), BANDS.index('beta')
ALPHA_RIDGE = 1.0
SEED = 0

# ---- labels: real per-trial VA, subject_z ----
df = pd.read_csv(VA_TABLE)
for c in ('valence','arousal'):
    g = df.groupby('subject_id')[c]
    df[c] = (df[c]-g.transform('mean'))/g.transform('std').clip(lower=1e-6)
LAB = {(int(r.subject_id),int(r.trial_id)):(r.valence,r.arousal) for r in df.itertuples()}

# ---- feature extractor (MUST match the live server exactly) ----
def feat_full(x):
    """x: [N, 4ch, W windows, 5 bands] (DE values). -> [N, 65]."""
    N = len(x); m = x.mean(2); s = x.std(2)
    t = np.linspace(-1, 1, x.shape[2])
    slope = ((x - x.mean(2,keepdims=True)) * t[None,None,:,None]).sum(2) / (t**2).sum()
    base = np.concatenate([m, s, slope], 1).reshape(N, -1)            # 60
    FP1,FP2,A1,A2 = m[:,0],m[:,1],m[:,2],m[:,3]
    eng = np.stack([
        FP2[:,ALPHA]-FP1[:,ALPHA],
        A2[:,ALPHA]-A1[:,ALPHA],
        (FP1[:,BETA]+FP2[:,BETA])-(FP1[:,ALPHA]+FP2[:,ALPHA]),
        m[:,:,BETA].mean(1)-m[:,:,ALPHA].mean(1),
        m[:,:,ALPHA].mean(1),
    ], 1)
    return np.concatenate([base, eng], 1)

# ---- load, per-subject z-score features ----
files = sorted(glob.glob(DE_DIR+"/*.pkl"))
Xtr, Y, SID = [], [], []
for f in files:
    sid = int(os.path.basename(f).replace("sub","").split(".")[0])
    d = np.asarray(pickle.load(open(f,'rb')))[:, MUSE_IDX, :, :]      # [28,4,30,5]
    feats = feat_full(d)                                             # [28,65]
    mu, sd = feats.mean(0), feats.std(0).clip(1e-6)                  # per-subject
    feats = (feats - mu) / sd
    for t in range(d.shape[0]):
        if (sid,t) not in LAB: continue
        Xtr.append(feats[t]); Y.append(LAB[(sid,t)]); SID.append(sid)
Xtr = np.asarray(Xtr); Y = np.asarray(Y); SID = np.asarray(SID)
print(f"{Xtr.shape[0]} trials, {Xtr.shape[1]} features, {len(set(SID))} subjects")

# ---- subject-independent CV sanity check ----
rng = np.random.default_rng(SEED); subs = np.array(sorted(set(SID))); rng.shuffle(subs)
folds = np.array_split(subs, 5); P=[[],[]]; T=[[],[]]
for k in range(5):
    val = set(folds[k].tolist()); m = np.array([s in val for s in SID])
    for d in (0,1):
        r = Ridge(alpha=ALPHA_RIDGE).fit(Xtr[~m], Y[~m,d]); pr = r.predict(Xtr[m])
        P[d].append(pr); T[d].append(Y[m,d])
rv = np.corrcoef(np.concatenate(P[0]),np.concatenate(T[0]))[0,1]
ra = np.corrcoef(np.concatenate(P[1]),np.concatenate(T[1]))[0,1]
print(f"subject-independent CV:  r_val={rv:+.3f}  r_aro={ra:+.3f}")

# ---- fit final model on all subjects ----
rv_model = Ridge(alpha=ALPHA_RIDGE).fit(Xtr, Y[:,0])
ra_model = Ridge(alpha=ALPHA_RIDGE).fit(Xtr, Y[:,1])
coef = np.stack([rv_model.coef_, ra_model.coef_])          # [2,65]
intr = np.array([rv_model.intercept_, ra_model.intercept_])# [2]

# output scaling: predictions are in subject-z units; record train-pred std so
# the live server can divide by it before tanh -> stable [-1,1] range.
pred = Xtr @ coef.T + intr
pred_std = pred.std(0).clip(1e-6)
print(f"train pred std (val,aro) = {pred_std.round(3)}")

np.savez("muse_va_live.npz",
         coef=coef, intercept=intr, pred_std=pred_std,
         muse_idx=np.array(MUSE_IDX), n_bands=5)
meta = {"channels":["AF7->FP1","AF8->FP2","TP9->A1","TP10->A2"],
        "muse_idx":MUSE_IDX, "bands":BANDS, "n_features":int(Xtr.shape[1]),
        "feature_order":"[mean(4x5), std(4x5), slope(4x5), 5 engineered]",
        "norm":"per-subject z-score of features (live: z-score vs user baseline)",
        "labels":"subject_z per-trial valence/arousal",
        "cv_r":{"valence":round(float(rv),3),"arousal":round(float(ra),3)},
        "output":"pred / pred_std -> tanh -> [-1,1]"}
json.dump(meta, open("muse_va_live.json","w"), indent=2)
print("saved muse_va_live.npz + muse_va_live.json")