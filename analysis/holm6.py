import sys, os, numpy as np, pandas as pd
sys.path.insert(0, ".")
def holm(p):
    o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,j in enumerate(o): run=max(run,(m-i)*p[j]); adj[j]=min(run,1)
    return adj
L=["Linear Regression","XGBoost","Random Forest (full)"]
for pl,path in (("P1","results"),("P2","results/plant2")):
    d=pd.read_csv(os.path.join(path,"dm_tests.csv"))
    for ref in ("Persistence","CLIPER"):
        m=d[(d.model_1==ref)&d.model_2.isin(L)].copy()
        m["holm6"]=holm(m.p_NW5_HLN.values)
        print(pl,ref); print(m[["model_2","loss","p_NW5_HLN","holm6"]].round(5).to_string(index=False))
    r=d[(d.model_1.isin(["Persistence","CLIPER"]))&(d.model_2=="Random Forest (reduced)")]
    print(pl,"RF-6 unadjusted"); print(r[["model_1","loss","p_NW5_HLN"]].round(5).to_string(index=False))
    z=pd.read_csv(os.path.join(path,"leakage_decomposition.csv")).set_index("quantity")["value"]
    A=z["MATCHED R2 leaked all-hours (A)"];B=z["MATCHED R2 leaked daylight (B)"];C=z["R2 causal all-hours (C)"];D=z["R2 causal daylight (D)"]
    gap=A-D; lk=((B-D)+(A-C))/2; ze=((A-B)+(C-D))/2
    print(pl,"gap %.4f leak %.4f (%.0f%%) zero %.4f (%.0f%%) leak|allhours %.4f leak|daylight %.4f interaction %.4f"%(gap,lk,100*lk/gap,ze,100*ze/gap,A-C,B-D,(B-D)-(A-C)))
