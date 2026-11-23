"""Export inputs for independently implemented baselines without reference poses."""
import argparse
from multiattcalib.config import load_config
from multiattcalib.data import PairDataset,save_manifest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--config")
    p.add_argument("--trials",type=int,default=1)
    a=p.parse_args()
    cfg=load_config(a.config)
    ds=PairDataset(a.root,a.manifest,cfg,trials=a.trials)
    rows=[]
    for idx in range(len(ds)):
        item=ds[idx]
        raw=ds.rows[idx//ds.trials]
        rows.append({"id":item["id"],"image":raw["image"],"points":raw["points"],
                     "K_original":raw["K"],"K_resized":item["K"].tolist(),
                     "image_height":cfg.image_height,"image_width":cfg.image_width,
                     "range_m":cfg.range_m,"T0":item["T0"].tolist()})
    save_manifest(rows,a.out)
