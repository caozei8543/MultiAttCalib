import torch
from multiattcalib.matching import directional_probabilities,reciprocal,ConfidenceGate,ReciprocalMatcher


def test_normalization_and_empty():
    s=torch.randn(2,3,4)
    t=torch.randn(2,4,3)
    m=torch.tensor([[1.,0.,.2,0.],[0.,0.,0.,0.]])
    A,B=directional_probabilities(s,t,m)
    torch.testing.assert_close(A[0].sum(-1),torch.ones(3))
    torch.testing.assert_close(B[0].sum(-1),torch.tensor([1.,0.,1.,0.]))
    assert A[1].eq(0).all() and B[1].eq(0).all()
    W,k,n=reciprocal(A,B,m>0)
    assert n.tolist()==[2,0] and k[1]==0
    assert torch.isfinite(W).all()


def test_chance_baseline_and_permutation():
    A=torch.tensor([[[.6,.4],[.6,.4],[.6,.4]]])
    B=torch.softmax(torch.randn(1,2,3),-1)
    W,k,n=reciprocal(A,B,torch.ones(1,2,dtype=torch.bool))
    torch.testing.assert_close(k,torch.zeros(1),atol=1e-6,rtol=0)
    I=torch.eye(4)[None]
    _,k,_=reciprocal(I,I,torch.ones(1,4,dtype=torch.bool))
    torch.testing.assert_close(k,torch.ones(1))


def test_gate_monotonic_constant_and_no_gate():
    k=torch.tensor([-.5,0.,.5,1.])
    n=torch.tensor([3,3,3,0])
    full=ConfidenceGate()(k,n)
    assert full[0]<full[1]<full[2] and full[3]==0
    gate=ConfidenceGate("constant_gate")
    val=gate(k,n)
    torch.testing.assert_close(val,torch.tensor([.5,.5,.5,0.]))
    val.sum().backward()
    assert gate.b.grad>0 and gate.a.grad is None
    assert ConfidenceGate("no_gate")(k,n).eq(1).all()


def test_unobserved_depth_features_zero():
    matcher=ReciprocalMatcher(8)
    f,g=torch.randn(1,3,8),torch.randn(1,4,8)
    fp,gp,k,n=matcher(f,g,torch.zeros(1,4))
    torch.testing.assert_close(fp,f)
    assert gp.eq(0).all() and k.eq(0).all() and n.eq(0).all()
