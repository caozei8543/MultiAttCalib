"""Manifest-driven data; no dataset files or private paths are shipped."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .geometry import perturbation


def read_manifest(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    ids=[r["id"] for r in rows]
    if len(set(ids))!=len(ids):
        raise ValueError("Manifest IDs must be unique")
    return rows


def save_manifest(rows,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows))


def stable_rng(seed,*items):
    payload=json.dumps([seed,*items],separators=(",",":")).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8],"little"))


def validate_transform(T,name):
    T=np.asarray(T,dtype=np.float32)
    if T.shape!=(4,4) or not np.isfinite(T).all():
        raise ValueError(f"{name} must be finite 4x4")
    if not np.allclose(T[3],[0,0,0,1],atol=1e-5):
        raise ValueError(f"Invalid homogeneous row in {name}")
    R=T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(R),1,atol=1e-3):
        raise ValueError(f"{name} must have a proper rotation")
    return T


def read_calibration(path):
    result={}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key,value=line.split(":",1)
        try:
            result[key]=np.array([float(x) for x in value.split()],dtype=np.float64)
        except ValueError:
            pass
    return result


def kitti_calibration(date_dir):
    """P_rect_02 = K[I|baseline]; include rectification and camera-02 baseline in Tref."""
    c=read_calibration(date_dir/"calib_cam_to_cam.txt")
    v=read_calibration(date_dir/"calib_velo_to_cam.txt")
    P=c["P_rect_02"].reshape(3,4)
    K=P[:,:3]
    T0=np.eye(4)
    T0[:3,:3],T0[:3,3]=v["R"].reshape(3,3),v["T"]
    rect=np.eye(4)
    rect[:3,:3]=c["R_rect_00"].reshape(3,3)
    baseline=np.eye(4)
    baseline[:3,3]=np.linalg.solve(K,P[:,3])
    return K,baseline@rect@T0


def scan_kitti(root,date="2011_09_26"):
    root=Path(root).resolve()
    date_dir=root/date
    K,T=kitti_calibration(date_dir)
    by_drive={}
    for drive in sorted(date_dir.glob("*_sync")):
        rows=[]
        for image in sorted((drive/"image_02/data").glob("*.png")):
            cloud=drive/"velodyne_points/data"/(image.stem+".bin")
            if not cloud.is_file():
                continue
            rows.append({"id":f"{drive.name}/{image.stem}","drive":drive.name,
                         "image":image.relative_to(root).as_posix(),"points":cloud.relative_to(root).as_posix(),
                         "K":K.tolist(),"T_ref":T.tolist(),"sensor":"KITTI"})
        if rows:
            by_drive[drive.name]=rows
    if not by_drive:
        raise ValueError("No synchronized KITTI Raw drives found")
    return by_drive


def make_splits(by_drive, output, counts=(7000,1500,1500), seed=42, drive_lists=None,
                rotation_deg=10.,translation_m=.25):
    names=("train","val","test")
    if len(counts)!=3 or any(n<=0 for n in counts):
        raise ValueError("Three positive split counts required")
    if drive_lists is None:
        # Greedy deterministic split for independent experiments; use the original manifests for exact reproduction.
        rng=stable_rng(seed,"drives")
        pool=list(sorted(by_drive))
        rng.shuffle(pool)
        drive_lists={name:[] for name in names}
        for name,count in zip(names,counts):
            while sum(len(by_drive[d]) for d in drive_lists[name])<count and pool:
                drive_lists[name].append(pool.pop(0))
    assigned=[d for name in names for d in drive_lists[name]]
    if len(assigned)!=len(set(assigned)):
        raise ValueError("Drive leakage across splits")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for name,count in zip(names,counts):
        rows=[dict(r) for d in drive_lists[name] for r in by_drive[d]]
        if len(rows)<count:
            raise ValueError(f"{name}: only {len(rows)} frames, need {count}; specify drive lists")
        rng=stable_rng(seed,name,"subset")
        selected=[rows[i] for i in sorted(rng.permutation(len(rows))[:count])]
        for row in selected:
            row["split"]=name
            if name!="train":
                row["E"]=perturbation(stable_rng(seed,name,row["id"]),rotation_deg,translation_m).tolist()
        save_manifest(selected,output/f"{name}.jsonl")
    (output/"drives.json").write_text(json.dumps(drive_lists,indent=2)+"\n")


class PairDataset(Dataset):
    def __init__(self,root,manifest,cfg,training=False,trials=1):
        self.root=Path(root).resolve()
        self.rows=read_manifest(manifest)
        self.cfg,self.training,self.trials=cfg,training,trials
        self.epoch=0
        if trials<1:
            raise ValueError("trials must be positive")

    def set_epoch(self,epoch):
        self.epoch=epoch

    def __len__(self):
        return len(self.rows)*self.trials

    def _path(self,name):
        p=(self.root/name).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("Manifest path must be inside data root")
        return p

    def __getitem__(self,index):
        row=self.rows[index//self.trials]
        trial=index%self.trials
        cfg=self.cfg
        with Image.open(self._path(row["image"])) as im:
            im=im.convert("RGB")
            w,h=im.size
            image=np.asarray(im.resize((cfg.image_width,cfg.image_height),Image.Resampling.BILINEAR),dtype=np.float32)/255
        image=(image-np.array(cfg.rgb_mean,dtype=np.float32))/np.array(cfg.rgb_std,dtype=np.float32)
        K=np.array(row["K"],dtype=np.float32)
        if K.shape!=(3,3) or not np.isfinite(K).all() or not np.allclose(K[2],[0,0,1]):
            raise ValueError("K must be a finite rectified 3x3 pinhole intrinsic matrix")
        K[0]*=cfg.image_width/w
        K[1]*=cfg.image_height/h
        cloud=self._path(row["points"])
        if cloud.suffix==".bin":
            points=np.fromfile(cloud,dtype="<f4").reshape(-1,4)[:,:3]
        elif cloud.suffix==".npy":
            points=np.load(cloud,allow_pickle=False)[:,:3]
        else:
            raise ValueError("Point clouds must be .bin (KITTI XYZI) or .npy (Nx3/Nx4)")
        points=np.asarray(points,dtype=np.float32)
        points=points[np.isfinite(points).all(-1)&(np.linalg.norm(points,axis=-1)<=cfg.range_m)]
        Tref=validate_transform(row["T_ref"],"T_ref") if "T_ref" in row else None
        if "T0" in row:
            if self.trials!=1 or self.training:
                raise ValueError("Explicit T0 requires single-trial inference")
            T0=validate_transform(row["T0"],"T0")
        else:
            if Tref is None:
                raise ValueError("Supply T0 or T_ref")
            if not self.training and "E" in row and self.trials==1:
                E=validate_transform(row["E"],"E")
            else:
                rng=stable_rng(cfg.seed,"train" if self.training else "evaluation",
                               row["id"],self.epoch if self.training else 0,trial)
                E=perturbation(rng,cfg.rotation_perturb_deg,cfg.translation_perturb_m)
            T0=E@Tref
        return {"id":row["id"] if self.trials==1 else f'{row["id"]}/trial{trial}',
                "rgb":torch.from_numpy(image.transpose(2,0,1).copy()),"points":torch.from_numpy(points),
                "K":torch.from_numpy(K),"T0":torch.from_numpy(T0),
                "Tref":torch.from_numpy(Tref) if Tref is not None else None}


def collate(rows):
    refs=[r["Tref"] for r in rows]
    if any(r is None for r in refs) and not all(r is None for r in refs):
        raise ValueError("Do not mix reference-labeled and unlabeled pairs")
    return {"id":[r["id"] for r in rows],"points":[r["points"] for r in rows],
            **{k:torch.stack([r[k] for r in rows]) for k in ("rgb","K","T0")},
            "Tref":None if refs[0] is None else torch.stack(refs)}
