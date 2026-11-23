import torch
import pytest
from multiattcalib.config import Config,VARIANTS
from multiattcalib.model import MultiAttCalib
from multiattcalib.losses import calibration_loss
from multiattcalib.metrics import summarize


@pytest.mark.parametrize("variant",VARIANTS)
def test_iterative_forward(variant):
    torch.set_num_threads(1)
    cfg=Config(image_height=64,image_width=96,token_height=4,token_width=6,channels=32,variant=variant)
    model=MultiAttCalib(cfg)
    rgb=torch.randn(2,3,64,96)
    points=[torch.randn(130,3) for _ in range(2)]
    for p in points:
        p[:,2]+=5
    K=torch.tensor([[60.,0.,48.],[0.,60.,32.],[0.,0.,1.]])[None].repeat(2,1,1)
    T=torch.eye(4)[None].repeat(2,1,1)
    poses,d=model(rgb,points,K,T,steps=2)
    assert len(poses)==len(d)==2 and poses[-1].shape==(2,4,4)
    loss=calibration_loss(poses,points,K,T,cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_joint_success_is_intersection_and_strict():
    s=summarize([{"rotation_deg":.1,"translation_cm":4.},
                 {"rotation_deg":.6,"translation_cm":2.},
                 {"rotation_deg":.1,"translation_cm":2.},
                 {"rotation_deg":.5,"translation_cm":3.}])
    assert s["rotation_success_percent"]==50 and s["translation_success_percent"]==50
    assert s["joint_success_percent"]==25
