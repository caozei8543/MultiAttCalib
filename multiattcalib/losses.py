"""Equation (6): fixed reference-visible point set, including invalid-depth penalty."""
import torch
import torch.nn.functional as F
from .geometry import project,visible,rotation_geodesic


def reference_points(points,K,T,cfg):
    with torch.no_grad():
        uv,z=project(points,T,K)
        valid=visible(uv,z,cfg.image_height,cfg.image_width)
        return points[valid],uv[valid]


def projection_loss(points,K,T,ref_uv,cfg):
    if points.shape[0]==0:
        return T.sum()*0
    # Compute only valid positive-depth reprojections to avoid exploding masked gradients.
    cam=points@T[:3,:3].T+T[:3,3]
    z=cam[:,2]
    good=z>cfg.depth_epsilon_m
    hom=cam[good]@K.T
    uv=hom[:,:2]/hom[:,2,None]
    error=torch.linalg.vector_norm(uv-ref_uv[good],dim=-1)
    valid=F.smooth_l1_loss(error,torch.zeros_like(error),beta=cfg.smooth_l1_beta_px,reduction="sum")
    invalid=(cfg.invalid_penalty+cfg.invalid_slope*(cfg.depth_epsilon_m-z[~good])).sum()
    return (valid+invalid)/len(points)


def calibration_loss(poses,points,K,Tref,cfg):
    refs=[reference_points(p,k,t,cfg) for p,k,t in zip(points,K,Tref)]
    losses=[]
    for T in poses:
        r=rotation_geodesic(T[:,:3,:3],Tref[:,:3,:3]).mean()
        t=(T[:,:3,3]-Tref[:,:3,3]).abs().sum(-1).mean()
        p=torch.stack([projection_loss(p,k,t,uv,cfg) for (p,uv),k,t in zip(refs,K,T)]).mean()
        losses.append(cfg.lambda_rotation*r+cfg.lambda_translation*t+cfg.lambda_projection*p)
    return torch.stack(losses).mean()
