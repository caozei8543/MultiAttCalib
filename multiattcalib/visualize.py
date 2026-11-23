import csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from .geometry import project,visible,rasterize
from .metrics import summarize


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def ablation_plot(results,out):
    """results: mapping label -> samples.csv; requires identical IDs."""
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),constrained_layout=True)
    samples=[read_csv(p) for p in results.values()]
    id_sets=[{r["id"] for r in rows} for rows in samples]
    if any(len(ids)!=len(rows) for ids,rows in zip(id_sets,samples)) or any(ids!=id_sets[0] for ids in id_sets[1:]):
        raise ValueError("Ablation samples must match exactly across methods")
    labels=list(results)
    for ax,key,ylabel,threshold in zip(axes,("rotation_deg","translation_cm"),
                                      ("Rotation MAE (deg)","Translation MAE (cm)"),(.5,3.)):
        values=[np.array([float(row[key]) for row in rows]) for rows in samples]
        boxes=ax.boxplot(values,whis=(5,95),patch_artist=True,showmeans=True,showfliers=False,
                         meanprops={"marker":"D","markerfacecolor":"black","markeredgecolor":"black","markersize":4})
        for i,b in enumerate(boxes["boxes"]):
            b.set_facecolor("#f3c4cb" if labels[i].lower()=="full" else "#bcd6ef")
        ax.set_xticks(range(1,len(labels)+1),labels,rotation=15,ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y",alpha=.2)
        twin=ax.twinx()
        twin.plot(range(1,len(labels)+1),[100*(v<threshold).mean() for v in values],"r--v",ms=4)
        twin.set_ylim(0,105)
        twin.set_ylabel("Success (%)",color="firebrick")
    fig.savefig(out,dpi=300)
    plt.close(fig)


def overlay(ax,rgb,points,K,T,label):
    h,w=rgb.shape[:2]
    uv,z=project(points,T,K)
    valid=visible(uv,z,h,w)
    ax.imshow(rgb)
    ax.scatter(uv[valid,0].cpu(),uv[valid,1].cpu(),c=z[valid].cpu(),cmap="turbo",
               vmin=0,vmax=80,s=1,linewidths=0,alpha=.85)
    ax.set_title(label)
    ax.axis("off")
    ax.set_xlim(0,w)
    ax.set_ylim(h,0)


def qualitative(rgb,points,K,T0,Tpred,Tref,out):
    h,w=rgb.shape[:2]
    depth,mask=rasterize(points,T0,K,h,w)
    d=depth.squeeze().cpu().numpy()
    fig,ax=plt.subplots(2,3,figsize=(12,5),constrained_layout=True)
    ax[0,0].imshow(rgb);ax[0,0].set_title("RGB")
    ax[0,1].imshow(np.ma.masked_equal(d,0),cmap="turbo",vmin=0,vmax=80);ax[0,1].set_title("Initial sparse depth")
    ax[0,2].imshow(mask.squeeze().cpu(),cmap="gray",vmin=0,vmax=1);ax[0,2].set_title("Validity")
    overlay(ax[1,0],rgb,points,K,T0,"Initial")
    overlay(ax[1,1],rgb,points,K,Tpred,"Refined")
    if Tref is not None:
        overlay(ax[1,2],rgb,points,K,Tref,"Reference")
    else:
        ax[1,2].text(.5,.5,"Reference unavailable",ha="center")
    for a in ax.flat:
        a.axis("off")
    fig.savefig(out,dpi=200)
    plt.close(fig)
