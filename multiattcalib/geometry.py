"""Column-vector transforms: T maps LiDAR coordinates into camera coordinates."""
import numpy as np
import torch


def euler_zyx_matrix(angles):
    """angles=(roll,pitch,yaw), radians; R=Rz(yaw) Ry(pitch) Rx(roll)."""
    x, y, z = angles
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                     [-sy, cy*sx, cy*cx]], dtype=np.float32)


def perturbation(rng, rotation_deg=10., translation_m=.25):
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = euler_zyx_matrix(np.deg2rad(rng.uniform(-rotation_deg, rotation_deg, 3)))
    E[:3, 3] = rng.uniform(-translation_m, translation_m, 3)
    return E


def skew(w):
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o,-z,y,z,o,-x,-y,x,o), -1).reshape(*w.shape[:-1],3,3)


def se3_exp(xi):
    """Eq. (5): twist order (omega_x,y,z, v_x,y,z); radians and metres."""
    w, v = xi[..., :3], xi[..., 3:]
    W = skew(w)
    W2 = W @ W
    q = (w*w).sum(-1)
    # Clamp the exact branch before evaluating it, to avoid singular gradients at zero.
    theta = q.clamp_min(1e-8).sqrt()
    small = q < 1e-4
    A = torch.where(small, 1-q/6+q*q/120, torch.sin(theta)/theta)
    B = torch.where(small, .5-q/24+q*q/720, (1-torch.cos(theta))/theta.square())
    C = torch.where(small, 1/6-q/120+q*q/5040, (theta-torch.sin(theta))/theta.pow(3))
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand_as(W)
    R = I + A[...,None,None]*W + B[...,None,None]*W2
    J = I + B[...,None,None]*W + C[...,None,None]*W2
    t = (J @ v.unsqueeze(-1)).squeeze(-1)
    upper = torch.cat((R, t.unsqueeze(-1)), -1)
    last = xi.new_tensor([0,0,0,1]).expand(*xi.shape[:-1],1,4)
    return torch.cat((upper, last), -2)


def transform(points, T):
    return points @ T[:3,:3].T + T[:3,3]


def project(points, T, K):
    cam = transform(points, T)
    hom = cam @ K.T
    z = cam[:,2]
    den = hom[:,2]
    den = torch.where(den.abs() > 1e-8, den, torch.full_like(den, 1e-8))
    return hom[:,:2]/den[:,None], z


def visible(uv, z, height, width, eps=0.):
    return ((z > eps) & torch.isfinite(uv).all(-1) & (uv[:,0]>=0)
            & (uv[:,0]<width) & (uv[:,1]>=0) & (uv[:,1]<height))


@torch.no_grad()
def rasterize(points, T, K, height, width):
    """Eq. (2): floor pixel coordinates; z-buffer min positive camera depth."""
    uv,z = project(points,T,K)
    ok = visible(uv,z,height,width)
    flat = z.new_full((height*width,), float("inf"))
    idx = uv[ok].floor().long()
    flat.scatter_reduce_(0, idx[:,1]*width+idx[:,0], z[ok], reduce="amin", include_self=True)
    mask = torch.isfinite(flat)
    depth = torch.where(mask, flat, 0.).reshape(1,height,width)
    return depth, mask.reshape(1,height,width).to(depth.dtype)


def rotation_geodesic(R, target):
    S = R @ target.transpose(-1,-2)
    c = ((S.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    v = torch.stack((S[...,2,1]-S[...,1,2], S[...,0,2]-S[...,2,0],
                     S[...,1,0]-S[...,0,1]), -1)/2
    return torch.atan2(torch.linalg.vector_norm(v,dim=-1), c)


def rotation_euler_error(R, target):
    """Absolute ZYX Euler-angle components of R R*^T, degrees."""
    S = R @ target.transpose(-1,-2)
    sy = torch.hypot(S[...,0,0], S[...,1,0])
    pitch = torch.atan2(-S[...,2,0], sy)
    roll = torch.where(sy>1e-7, torch.atan2(S[...,2,1], S[...,2,2]),
                       torch.atan2(-S[...,1,2], S[...,1,1]))
    yaw = torch.where(sy>1e-7, torch.atan2(S[...,1,0], S[...,0,0]), torch.zeros_like(sy))
    return torch.rad2deg(torch.stack((roll,pitch,yaw),-1)).abs()
