import argparse
import json
from pathlib import Path
import torch
from .config import load_config, VARIANTS
from .data import scan_kitti,make_splits,PairDataset,read_manifest,validate_transform
from .engine import train,test,load_model
from .metrics import sample_rows,write_metrics


def main(argv=None):
    p=argparse.ArgumentParser(description="MultiAttCalib: validity-guided reciprocal LiDAR-camera calibration")
    sub=p.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare-kitti")
    prep.add_argument("--root",required=True)
    prep.add_argument("--out",required=True)
    prep.add_argument("--date",default="2011_09_26")
    prep.add_argument("--drive-lists",help="JSON with train/val/test drive arrays")
    prep.add_argument("--counts",type=int,nargs=3,default=[7000,1500,1500])
    prep.add_argument("--config")
    for name in ("train","evaluate"):
        q=sub.add_parser(name)
        q.add_argument("--root",required=True)
        q.add_argument("--out",required=True)
        q.add_argument("--device",default="auto")
        if name=="train":
            q.add_argument("--train",required=True)
            q.add_argument("--val",required=True)
            q.add_argument("--config")
            q.add_argument("--variant",choices=VARIANTS)
            q.add_argument("--epochs",type=int)
        else:
            q.add_argument("--manifest",required=True)
            q.add_argument("--checkpoint",required=True)
            q.add_argument("--trials",type=int,default=1)
    q=sub.add_parser("plot-ablation")
    q.add_argument("--result",action="append",required=True,help="LABEL=path/to/samples.csv")
    q.add_argument("--out",required=True)
    q=sub.add_parser("visualize")
    q.add_argument("--root",required=True)
    q.add_argument("--manifest",required=True)
    q.add_argument("--checkpoint",required=True)
    q.add_argument("--index",type=int,default=0)
    q.add_argument("--out",required=True)
    q.add_argument("--device",default="auto")
    q=sub.add_parser("score-external")
    q.add_argument("--root",required=True)
    q.add_argument("--manifest",required=True)
    q.add_argument("--predictions",required=True)
    q.add_argument("--out",required=True)
    q.add_argument("--config")
    q.add_argument("--trials",type=int,default=1)
    a=p.parse_args(argv)
    device=getattr(a,"device","cpu")
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device=="auto" else device)
    if a.command=="prepare-kitti":
        cfg=load_config(a.config)
        drives=json.loads(Path(a.drive_lists).read_text()) if a.drive_lists else None
        make_splits(scan_kitti(a.root,a.date),a.out,a.counts,cfg.seed,drives,
                    cfg.rotation_perturb_deg,cfg.translation_perturb_m)
    elif a.command=="train":
        cfg=load_config(a.config,variant=a.variant,epochs=a.epochs)
        train(cfg,a.root,a.train,a.val,a.out,device)
    elif a.command=="evaluate":
        test(a.checkpoint,a.root,a.manifest,a.out,device,a.trials)
    elif a.command=="plot-ablation":
        from .visualize import ablation_plot
        ablation_plot(dict(x.split("=",1) for x in a.result),a.out)
    elif a.command=="visualize":
        from .visualize import qualitative
        import numpy as np
        model,cfg=load_model(a.checkpoint,device)
        row=PairDataset(a.root,a.manifest,cfg)[a.index]
        with torch.no_grad():
            poses,_=model(row["rgb"][None].to(device),[row["points"].to(device)],
                          row["K"][None].to(device),row["T0"][None].to(device))
        rgb=(row["rgb"].permute(1,2,0).numpy()*np.array(cfg.rgb_std)+np.array(cfg.rgb_mean)).clip(0,1)
        qualitative(rgb,row["points"],row["K"],row["T0"],poses[-1][0].cpu(),row["Tref"],a.out)
    elif a.command=="score-external":
        cfg=load_config(a.config)
        ds=PairDataset(a.root,a.manifest,cfg,trials=a.trials)
        predictions={r["id"]:r for r in read_manifest(a.predictions)}
        rows=[]
        for sample in ds:
            pred=predictions.pop(sample["id"],None)
            if pred is None:
                raise ValueError(f"Missing prediction for {sample['id']}")
            if sample["Tref"] is None:
                raise ValueError("Scoring requires labeled reference transforms")
            # Reject results from different perturbations, rather than silently compare them.
            T0=torch.from_numpy(validate_transform(pred["T0"],"external T0"))
            if not torch.allclose(T0,sample["T0"],atol=1e-5,rtol=1e-5):
                raise ValueError(f"Initial pose mismatch for {sample['id']}")
            T=torch.from_numpy(validate_transform(pred["T_pred"],"external T_pred"))
            rows.extend(sample_rows([sample["id"]],T[None],sample["Tref"][None],sample["T0"][None]))
        if predictions:
            raise ValueError("External predictions contain unexpected sample IDs")
        print(json.dumps(write_metrics(rows,a.out,cfg),indent=2))


if __name__=="__main__":
    main()
