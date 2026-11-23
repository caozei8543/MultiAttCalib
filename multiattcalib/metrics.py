import csv
import json
from pathlib import Path
import numpy as np
import torch
from .geometry import rotation_euler_error


def pose_errors(T,Tref):
    r=rotation_euler_error(T[...,:3,:3],Tref[...,:3,:3])
    t=(T[...,:3,3]-Tref[...,:3,3]).abs()*100
    return r,t


def sample_rows(ids,T,Tref,T0=None,diagnostics=None):
    r,t=pose_errors(T,Tref)
    ri,ti=pose_errors(T0,Tref) if T0 is not None else (None,None)
    result=[]
    for i,name in enumerate(ids):
        row={"id":name,"rotation_deg":float(r[i].mean()),"translation_cm":float(t[i].mean())}
        for j,axis in enumerate("xyz"):
            row[f"rotation_{axis}_deg"]=float(r[i,j])
            row[f"translation_{axis}_cm"]=float(t[i,j])
        if ri is not None:
            row.update(initial_rotation_deg=float(ri[i].mean()),initial_translation_cm=float(ti[i].mean()))
        if diagnostics:
            for k,d in enumerate(diagnostics,1):
                for key in ("alpha","kappa","n_valid"):
                    row[f"{key}_{k}"]=float(d[key][i])
        result.append(row)
    return result


def summarize(rows,rotation_threshold=.5,translation_threshold=3.):
    if not rows:
        raise ValueError("Cannot summarize empty results")
    r=np.array([row["rotation_deg"] for row in rows],dtype=float)
    t=np.array([row["translation_cm"] for row in rows],dtype=float)
    if not (np.isfinite(r).all() and np.isfinite(t).all()) or (r<0).any() or (t<0).any():
        raise ValueError("Invalid error data")
    rs,ts=r<rotation_threshold,t<translation_threshold
    result={"n":len(rows),"rotation_mae_deg":float(r.mean()),"translation_mae_cm":float(t.mean()),
            "rotation_success_percent":float(rs.mean()*100),"translation_success_percent":float(ts.mean()*100),
            "joint_success_percent":float((rs&ts).mean()*100),
            "rotation_threshold_deg":rotation_threshold,"translation_threshold_cm":translation_threshold}
    for name in ("initial_rotation_deg","initial_translation_cm"):
        if name in rows[0]:
            result[name]=float(np.mean([row[name] for row in rows]))
    for key in rows[0]:
        if key.startswith(("alpha_","kappa_","n_valid_")):
            x=np.array([row[key] for row in rows])
            result[key]={"mean":float(x.mean()),"std":float(x.std()),"min":float(x.min()),"max":float(x.max())}
    return result


def write_metrics(rows,directory,cfg):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/"samples.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    summary=summarize(rows,cfg.success_rotation_deg,cfg.success_translation_cm)
    (directory/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary
