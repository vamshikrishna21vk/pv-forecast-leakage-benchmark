import numpy as np, pandas as pd
def holm(p):
    o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,j in enumerate(o): run=max(run,(m-i)*p[j]); adj[j]=min(run,1)
    return adj
L=["Linear Regression","XGBoost","Random Forest (full)"]
for pl in (1,2):
    d=pd.read_csv(f"analysis/out/plant{pl}/dm_bandwidth.csv")
    for ref in ("Persistence","CLIPER"):
        m=d[(d.ref==ref)&d.model.isin(L)].copy()
        for c in ("p_NW5","p_andrews","p_NWday"): m[c+"_H6"]=holm(m[c].values)
        print(f"P{pl}",ref); print(m[["model","loss","p_NW5_H6","p_andrews_H6","p_NWday_H6"]].round(4).to_string(index=False))
