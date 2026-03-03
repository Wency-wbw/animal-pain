
import argparse, os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from utils import ensure_dir, macro_pr, save_json
from geometry import features_from_landmarks, feature_groups

def main(a):
    os.makedirs(ensure_dir(a.out), exist_ok=True)
    dfL=pd.read_csv(a.landmarks); dfM=pd.read_csv(a.metadata)
    df=dfL.merge(dfM[["id","label"]], on="id")
    X,names=features_from_landmarks(dfL.set_index("id").loc[df["id"]].reset_index())
    y=df["label"].astype(int).values

    cv=StratifiedKFold(n_splits=a.cv, shuffle=True, random_state=42)
    yt,yp=[],[]; imps=[]
    for fold,(tr,va) in enumerate(cv.split(X,y),1):
        pipe=Pipeline([("sc",StandardScaler()),
                       ("rf",RandomForestClassifier(n_estimators=400, random_state=fold, class_weight="balanced"))])
        pipe.fit(X[tr],y[tr]); pred=pipe.predict(X[va])
        yt+=list(y[va]); yp+=list(pred)
        pi=permutation_importance(pipe, X[va], y[va], n_repeats=10, random_state=0)
        imps.append(pi.importances_mean)
    save_json(macro_pr(yt,yp), os.path.join(a.out,"metrics.json"))
    np.save(os.path.join(a.out,"perm_importance.npy"), np.mean(np.stack(imps),0))

    masks=feature_groups(names, X.shape); res={}
    from sklearn.metrics import accuracy_score
    for k,m in masks.items():
        Xh=X.copy(); Xh[:,m]=0.0
        yt2,yp2=[],[]
        for tr,va in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(Xh,y):
            pipe=Pipeline([("sc",StandardScaler()),("rf",RandomForestClassifier(n_estimators=400, class_weight="balanced"))])
            pipe.fit(Xh[tr],y[tr]); yp2+=list(pipe.predict(Xh[va])); yt2+=list(y[va])
        res["hide_"+k]=macro_pr(yt2,yp2)
    only=np.ones_like(list(masks.values())[0]); only[masks["mouth"]]=False
    Xm=X.copy(); Xm[:,only]=0.0
    yt2,yp2=[],[]
    for tr,va in StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(Xm,y):
        pipe=Pipeline([("sc",StandardScaler()),("rf",RandomForestClassifier(n_estimators=400, class_weight="balanced"))])
        pipe.fit(Xm[tr],y[tr]); yp2+=list(pipe.predict(Xm[va])); yt2+=list(y[va])
    res["mouth_only"]=macro_pr(yt2,yp2)
    save_json(res, os.path.join(a.out,"occlusion_results.json"))
    print("RF done.")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--landmarks",required=True); ap.add_argument("--metadata",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--cv",type=int,default=10)
    a=ap.parse_args(); main(a)
