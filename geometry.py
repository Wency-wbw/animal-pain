
import numpy as np

MOUTH_IDX = list(range(33,49))
EYES_IDX  = list(range(17,33))
EARS_IDX  = list(range(1,17))

def _stack_points(row):
    pts = []
    for i in range(1,49):
        pts.append([row[f"x{i}"], row[f"y{i}"]])
    return np.array(pts, dtype=float)

def basic_geometry(pts):
    def pairwise(idxs):
        sub = pts[np.array(idxs)-1]
        d = np.sqrt(((sub[:,None,:]-sub[None,:,:])**2).sum(-1))
        return d[np.triu_indices_from(d,1)]
    def boxrat(idxs):
        sub = pts[np.array(idxs)-1]
        x0,y0 = sub.min(0); x1,y1 = sub.max(0)
        w = max(x1-x0,1e-6); h=max(y1-y0,1e-6)
        return np.array([w,h,w/h])
    feats = [pairwise(MOUTH_IDX), pairwise(EYES_IDX), pairwise(EARS_IDX),
             boxrat(MOUTH_IDX), boxrat(EYES_IDX), boxrat(EARS_IDX)]
    return np.concatenate(feats)

def features_from_landmarks(df):
    X=[]; 
    for _,row in df.iterrows():
        X.append(basic_geometry(_stack_points(row)))
    X=np.stack(X,0)
    names=[f"f{i}" for i in range(X.shape[1])]
    return X,names

def feature_groups(names, Xshape):
    n=len(names); c=120; base=3*c
    masks={k:np.zeros(n,dtype=bool) for k in["mouth","eyes","ears"]}
    masks["mouth"][:c]=True; masks["eyes"][c:2*c]=True; masks["ears"][2*c:3*c]=True
    masks["mouth"][base:base+3]=True; masks["eyes"][base+3:base+6]=True; masks["ears"][base+6:base+9]=True
    return masks
