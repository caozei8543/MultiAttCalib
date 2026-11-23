"""Generate a small artificial dataset for plumbing tests only, not scientific evaluation."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from multiattcalib.data import save_manifest
from multiattcalib.geometry import perturbation


def generate(root,seed=7):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    h,w=64,96
    K=np.array([[70,0,w/2],[0,70,h/2],[0,0,1]],dtype=np.float32)
    for split,n in (("train",4),("val",2),("test",2)):
        d=root/split
        d.mkdir(exist_ok=True)
        rows=[]
        for i in range(n):
            z=rng.uniform(3,15,1000)
            u=rng.uniform(0,w,1000)
            v=rng.uniform(0,h,1000)
            xyz=np.stack(((u-w/2)*z/70,(v-h/2)*z/70,z),-1).astype(np.float32)
            rgb=np.zeros((h,w,3),dtype=np.uint8)
            rgb[:,:,1]=40
            rgb[v.astype(int),u.astype(int)]=[210,160,90]
            image=d/f"{i}.png"
            cloud=d/f"{i}.npy"
            Image.fromarray(rgb).save(image)
            np.save(cloud,xyz)
            row={"id":f"synthetic/{split}/{i}","drive":f"synthetic-{split}","image":f"{split}/{i}.png",
                 "points":f"{split}/{i}.npy","K":K.tolist(),"T_ref":np.eye(4).tolist()}
            if split!="train":
                row["E"]=perturbation(rng).tolist()
            rows.append(row)
        save_manifest(rows,root/f"{split}.jsonl")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True)
    parser.add_argument("--seed",type=int,default=7)
    a=parser.parse_args()
    generate(a.out,a.seed)
