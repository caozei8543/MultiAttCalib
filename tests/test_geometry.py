import numpy as np
import torch
from multiattcalib.geometry import se3_exp,skew,rasterize,rotation_euler_error,perturbation
from multiattcalib.losses import projection_loss,reference_points
from multiattcalib.config import Config


def test_exp_matches_matrix_exp_and_has_zero_gradient():
    xi=torch.tensor([[.1,-.2,.3,.04,.05,.06],[0.,0.,0.,0.,0.,0.]],dtype=torch.float64,requires_grad=True)
    hat=torch.zeros(2,4,4,dtype=xi.dtype)
    hat[:,:3,:3]=skew(xi[:,:3])
    hat[:,:3,3]=xi[:,3:]
    T=se3_exp(xi)
    torch.testing.assert_close(T,torch.matrix_exp(hat),atol=1e-9,rtol=1e-9)
    T.sum().backward()
    assert torch.isfinite(xi.grad).all()


def test_zbuffer_and_invalid_points():
    pts=torch.tensor([[0.,0.,2.],[0.,0.,1.],[0.,0.,-1.],[99.,0.,1.]])
    K=torch.tensor([[1.,0.,2.],[0.,1.,2.],[0.,0.,1.]])
    d,m=rasterize(pts,torch.eye(4),K,5,5)
    assert d[0,2,2]==1 and m.sum()==1


def test_left_perturbation_and_euler_metric():
    E=torch.from_numpy(perturbation(np.random.default_rng(4)))
    Tref=torch.eye(4)
    Tref[:3,3]=torch.tensor([.5,.2,.1])
    T0=E@Tref
    torch.testing.assert_close(Tref@torch.linalg.inv(T0),torch.linalg.inv(E),atol=1e-6,rtol=1e-6)
    a=se3_exp(torch.tensor([0.,0.,torch.pi/6,0.,0.,0.]))
    torch.testing.assert_close(rotation_euler_error(a[:3,:3],torch.eye(3)),torch.tensor([0.,0.,30.]))


def test_loss_invalid_branch_and_empty():
    cfg=Config(image_height=10,image_width=10)
    pts=torch.tensor([[0.,0.,2.]])
    K=torch.eye(3)
    p,uv=reference_points(pts,K,torch.eye(4),cfg)
    assert projection_loss(p,K,torch.eye(4),uv,cfg)==0
    T=torch.eye(4)
    T[2,3]=-3
    loss=projection_loss(p,K,T,uv,cfg)
    torch.testing.assert_close(loss,torch.tensor(770.1))
    assert projection_loss(p[:0],K,T,uv[:0],cfg)==0
