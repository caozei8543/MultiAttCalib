from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import PairDataset,collate
from .model import MultiAttCalib
from .losses import calibration_loss
from .metrics import sample_rows,write_metrics,summarize


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def loader(dataset,cfg,training=False):
    return DataLoader(dataset,batch_size=cfg.batch_size,shuffle=training,num_workers=cfg.workers,
                      collate_fn=collate,generator=torch.Generator().manual_seed(cfg.seed))


def batch_to(batch,device):
    return {k:([x.to(device) for x in v] if k=="points" else v.to(device) if torch.is_tensor(v) else v)
            for k,v in batch.items()}


@torch.no_grad()
def evaluate(model,dl,device,steps,return_predictions=False):
    model.eval()
    rows,predictions=[],[]
    for raw in dl:
        b=batch_to(raw,device)
        poses,d=model(b["rgb"],b["points"],b["K"],b["T0"],steps)
        T=poses[-1]
        if not torch.isfinite(T).all():
            raise FloatingPointError("Nonfinite predicted transform")
        if b["Tref"] is not None:
            rows.extend(sample_rows(b["id"],T,b["Tref"],b["T0"],d))
        if return_predictions:
            for i,name in enumerate(b["id"]):
                predictions.append({"id":name,"T_pred":T[i].cpu().tolist(),"T0":b["T0"][i].cpu().tolist(),
                                    "alpha":[float(x["alpha"][i]) for x in d],
                                    "kappa":[float(x["kappa"][i]) for x in d]})
    return rows,predictions


def train(cfg,root,train_manifest,val_manifest,out,device):
    seed_all(cfg.seed)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"best.pt").exists():
        raise FileExistsError("Output contains a checkpoint; choose a new run directory")
    tr=PairDataset(root,train_manifest,cfg,training=True)
    va=PairDataset(root,val_manifest,cfg)
    drives_tr={r.get("drive",r["id"]) for r in tr.rows}
    drives_va={r.get("drive",r["id"]) for r in va.rows}
    if drives_tr&drives_va:
        raise ValueError("Training and validation share drives")
    train_loader,val_loader=loader(tr,cfg,True),loader(va,cfg)
    model=MultiAttCalib(cfg).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    sched=torch.optim.lr_scheduler.StepLR(opt,cfg.lr_step,gamma=cfg.lr_gamma)
    meta={"config":asdict(cfg),"torch":str(torch.__version__),"device":str(device),
          "train_manifest_sha256":hashlib.sha256(Path(train_manifest).read_bytes()).hexdigest(),
          "val_manifest_sha256":hashlib.sha256(Path(val_manifest).read_bytes()).hexdigest()}
    (out/"run.json").write_text(json.dumps(meta,indent=2))
    best=float("inf")
    for epoch in range(cfg.epochs):
        tr.set_epoch(epoch)
        model.train()
        total,count=0.,0
        for raw in train_loader:
            b=batch_to(raw,device)
            if b["Tref"] is None:
                raise ValueError("Training requires reference poses")
            opt.zero_grad(set_to_none=True)
            poses,_=model(b["rgb"],b["points"],b["K"],b["T0"],cfg.train_steps)
            loss=calibration_loss(poses,b["points"],b["K"],b["Tref"],cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            opt.step()
            total+=float(loss.detach())*len(b["id"])
            count+=len(b["id"])
        rows,_=evaluate(model,val_loader,device,cfg.eval_steps)
        s=summarize(rows,cfg.success_rotation_deg,cfg.success_translation_cm)
        # Validation-only selection; dimensionless score scaled by success thresholds.
        score=s["rotation_mae_deg"]/cfg.success_rotation_deg+s["translation_mae_cm"]/cfg.success_translation_cm
        record={"epoch":epoch+1,"loss":total/count,"lr":opt.param_groups[0]["lr"],"validation":s}
        with (out/"epochs.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
        print(json.dumps({"epoch":epoch+1,"train_loss":total/count,"val_score":score}),flush=True)
        sched.step()
        state={"model":model.state_dict(),"config":asdict(cfg),"epoch":epoch+1,
               "optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"validation":s}
        torch.save(state,out/"last.pt")
        if score<best:
            best=score
            torch.save(state,out/"best.pt")
    return model


def load_model(checkpoint,device):
    from .config import Config
    state=torch.load(checkpoint,map_location=device,weights_only=True)
    cfg=Config(**state["config"]).validate()
    model=MultiAttCalib(cfg).to(device)
    model.load_state_dict(state["model"],strict=True)
    model.eval()
    return model,cfg


def test(checkpoint,root,manifest,out,device,trials=1):
    model,cfg=load_model(checkpoint,device)
    ds=PairDataset(root,manifest,cfg,trials=trials)
    rows,pred=evaluate(model,loader(ds,cfg),device,cfg.eval_steps,True)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    from .data import save_manifest
    save_manifest(pred,out/"predictions.jsonl")
    meta={"checkpoint":Path(checkpoint).name,"manifest_sha256":hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
          "trials_per_pair":trials,"variant":cfg.variant,"config":asdict(cfg)}
    (out/"evaluation.json").write_text(json.dumps(meta,indent=2))
    if rows:
        print(json.dumps(write_metrics(rows,out,cfg),indent=2))
    else:
        print(f"Exported {len(pred)} poses (no reference labels).")
