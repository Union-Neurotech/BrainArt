"""
faced_window_experiment.py
Empirical test of: (1) longer aggregation windows, (2) engineered features,
(3) PCA, (4) K-means -> VA, for 4-channel Muse VA regression on FACED DE feats.

Subject-independent 5-fold CV. Pooled Pearson r per dimension reported.
Labels: real per-trial valence/arousal, subject_z normalized (matches the
production script's default).
"""
import pickle, glob, os, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

DE_DIR = "datasets/FACED/EEG_Features/DE"
VA_TABLE = "datasets/FACED/faced_va_table.csv"
MUSE_IDX = [0, 1, 30, 31]          # FP1, FP2, A1, A2  (AF7,AF8,TP9,TP10 proxies)
BANDS = ['delta','theta','alpha','beta','gamma']
ALPHA = BANDS.index('alpha'); BETA = BANDS.index('beta')
N_TRIAL, N_WIN = 28, 30
SEED = 0

# ---------- labels: subject_z per-trial valence/arousal ----------
df = pd.read_csv(VA_TABLE)
for col in ('valence','arousal'):
    g = df.groupby('subject_id')[col]
    df[col] = (df[col]-g.transform('mean'))/g.transform('std').clip(lower=1e-6)
LAB = {(int(r.subject_id),int(r.trial_id)):(r.valence,r.arousal) for r in df.itertuples()}

# ---------- load all subjects, select 4 channels ----------
files = sorted(glob.glob(DE_DIR+"/*.pkl"))
subjects=[]; X_raw=[]; Y=[]; SID=[]; TID=[]
for f in files:
    sid = int(os.path.basename(f).replace("sub","").split(".")[0])
    d = np.asarray(pickle.load(open(f,'rb')))[:, MUSE_IDX, :, :]   # [28,4,30,5]
    for t in range(d.shape[0]):
        if (sid,t) not in LAB: continue
        X_raw.append(d[t])            # [4,30,5]
        Y.append(LAB[(sid,t)]); SID.append(sid); TID.append(t)
X_raw=np.asarray(X_raw); Y=np.asarray(Y,dtype=float)
SID=np.asarray(SID); TID=np.asarray(TID)
print(f"loaded {X_raw.shape[0]} trials, X_raw {X_raw.shape}, {len(set(SID))} subjects")

# ---------- feature builders ----------
def feat_mean(x):                      # [N,4,30,5] -> mean over windows -> 20
    return x.mean(2).reshape(len(x),-1)
def feat_meanstd(x):                   # 40
    return np.concatenate([x.mean(2),x.std(2)],1).reshape(len(x),-1)
def feat_full(x):                      # mean+std+slope + engineered
    N=len(x); m=x.mean(2); s=x.std(2)
    # slope across the 30 windows (linear trend) per ch,band
    t=np.linspace(-1,1,x.shape[2]);
    slope=((x - x.mean(2,keepdims=True))*t[None,None,:,None]).sum(2)/ (t**2).sum()
    base=np.concatenate([m,s,slope],1).reshape(N,-1)   # 4*5*3=60
    # engineered, trial-mean band powers per channel
    FP1,FP2,A1,A2 = m[:,0],m[:,1],m[:,2],m[:,3]        # each [N,5]
    eng=np.stack([
        FP2[:,ALPHA]-FP1[:,ALPHA],                     # frontal alpha asymmetry
        A2[:,ALPHA]-A1[:,ALPHA],                       # temporal alpha asym
        (FP1[:,BETA]+FP2[:,BETA])-(FP1[:,ALPHA]+FP2[:,ALPHA]),  # frontal beta-alpha
        m[:,:,BETA].mean(1)-m[:,:,ALPHA].mean(1),      # global beta-alpha (arousal proxy)
        m[:,:,ALPHA].mean(1),                          # global alpha
    ],1)
    return np.concatenate([base,eng],1)

# windowed aggregation: split 30 windows into chunks of w, each chunk = one sample
def windowed(x, sid, tid, y, w):
    Xs=[]; Ss=[]; Ys=[]
    nchunk = N_WIN//w
    for i in range(len(x)):
        for c in range(nchunk):
            seg=x[i:i+1,:,c*w:(c+1)*w,:]
            Xs.append(feat_full(seg)[0]); Ss.append(sid[i]); Ys.append(y[i])
    return np.asarray(Xs), np.asarray(Ss), np.asarray(Ys)

def pooled_r(model, Xf, sid, y, folds=5):
    rng=np.random.default_rng(SEED); subs=np.array(sorted(set(sid))); rng.shuffle(subs)
    chunks=np.array_split(subs,folds)
    pv=[]; pa=[]; tv=[]; ta=[]
    for k in range(folds):
        val=set(chunks[k].tolist()); m=np.array([s in val for s in sid])
        mdl=make_pipeline(StandardScaler(), model.__class__(**model.get_params()))
        mdl.fit(Xf[~m], y[~m]); pr=mdl.predict(Xf[m])
        pv.append(pr[:,0]); pa.append(pr[:,1]); tv.append(y[m,0]); ta.append(y[m,1])
    pv,pa,tv,ta=map(np.concatenate,[pv,pa,tv,ta])
    rv=np.corrcoef(pv,tv)[0,1]; ra=np.corrcoef(pa,ta)[0,1]
    return rv,ra

print("\n=== A) window length sweep (features=full, Ridge) ===")
for w in [1,5,10,15,30]:
    Xw,Sw,Yw = windowed(X_raw,SID,TID,Y,w)
    rv,ra = pooled_r(Ridge(alpha=10.0), Xw, Sw, Yw)
    print(f" w={w:2d}s  n={len(Xw):6d}  r_val={rv:+.3f}  r_aro={ra:+.3f}")

print("\n=== B) feature set @ trial level (30s, Ridge) ===")
for name,fn in [("mean20",feat_mean),("meanstd40",feat_meanstd),("full65",feat_full)]:
    Xf=fn(X_raw); rv,ra=pooled_r(Ridge(alpha=10.0),Xf,SID,Y)
    print(f" {name:10s} d={Xf.shape[1]:3d}  r_val={rv:+.3f}  r_aro={ra:+.3f}")

print("\n=== C) PCA on full trial features (Ridge on top-k PCs) ===")
Xf=feat_full(X_raw)
for k in [5,10,20,40]:
    rng=np.random.default_rng(SEED); subs=np.array(sorted(set(SID))); rng.shuffle(subs)
    chunks=np.array_split(subs,5); pv=[];pa=[];tv=[];ta=[]
    for kk in range(5):
        val=set(chunks[kk].tolist()); m=np.array([s in val for s in SID])
        mdl=make_pipeline(StandardScaler(),PCA(n_components=k,random_state=SEED),Ridge(alpha=10.0))
        mdl.fit(Xf[~m],Y[~m]); pr=mdl.predict(Xf[m])
        pv.append(pr[:,0]);pa.append(pr[:,1]);tv.append(Y[m,0]);ta.append(Y[m,1])
    pv,pa,tv,ta=map(np.concatenate,[pv,pa,tv,ta])
    print(f" PCA k={k:2d}  r_val={np.corrcoef(pv,tv)[0,1]:+.3f}  r_aro={np.corrcoef(pa,ta)[0,1]:+.3f}")
ev=make_pipeline(StandardScaler(),PCA().fit(Xf)).steps[-1][1].explained_variance_ratio_
print(" cumPCA var@5/10/20:", [round(ev[:i].sum(),3) for i in (5,10,20)])

print("\n=== D) K-means clusters vs VA quadrant alignment ===")
Xs=StandardScaler().fit_transform(Xf)
for nc in [2,4,8]:
    km=KMeans(n_clusters=nc,random_state=SEED,n_init=10).fit(Xs)
    # how much VA variance is explained by cluster identity (eta^2)
    def eta2(y,lab):
        gm=y.mean(); ss_tot=((y-gm)**2).sum()
        ss_b=sum(((y[lab==c].mean()-gm)**2)*(lab==c).sum() for c in set(lab))
        return ss_b/ss_tot
    print(f" k={nc}  eta2_val={eta2(Y[:,0],km.labels_):.3f}  eta2_aro={eta2(Y[:,1],km.labels_):.3f}")
print("(eta2 = fraction of VA variance explained by cluster identity; ~0 means clusters are unrelated to VA)")
