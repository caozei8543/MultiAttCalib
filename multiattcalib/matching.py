"""Equations (3)-(5), including controlled ablations."""
import torch
from torch import nn
import torch.nn.functional as F


def directional_probabilities(s, t, occupancy, gamma=1.):
    """s:[B,NI,ND], t:[B,ND,NI], occupancy:[B,ND].

    Invalid keys are excluded from the first softmax. Invalid depth query
    rows are zero in the second. An empty valid set produces exact zeros.
    """
    valid = occupancy > 0
    logm = occupancy.clamp_min(torch.finfo(s.dtype).tiny).log()
    logits = (s + gamma*logm[:,None,:]).masked_fill(~valid[:,None,:], -torch.inf)
    # Softmax of an all-minus-inf row would produce NaNs.
    logits = torch.where(valid.any(-1)[:,None,None], logits, torch.zeros_like(logits))
    A = logits.softmax(-1) * valid[:,None,:]
    B = t.softmax(-1) * valid[:,:,None]
    return A, B


def reciprocal(A, B, valid):
    W = A * B.transpose(-1,-2)
    n = valid.sum(-1).clamp_max(A.shape[1])
    kappa = (W.sum((-2,-1))-1)/(n-1).clamp_min(1).to(W.dtype)
    kappa = torch.where(n>=2, kappa, torch.zeros_like(kappa))
    return W, kappa, n


class ReciprocalMatcher(nn.Module):
    def __init__(self, channels, gamma=1., variant="full"):
        super().__init__()
        self.gamma, self.variant = gamma, variant
        self.qi, self.ki, self.qd, self.kd = [nn.Linear(channels,channels) for _ in range(4)]
        self.vi, self.vd = nn.Linear(channels,channels), nn.Linear(channels,channels)
        self.scale = channels ** -.5

    def forward(self, f, g, occupancy):
        if self.variant == "no_validity":
            occupancy = torch.ones_like(occupancy)
        valid = occupancy > 0
        s = self.qi(f) @ self.kd(g).transpose(-1,-2) * self.scale
        t = self.qd(g) @ self.ki(f).transpose(-1,-2) * self.scale
        A,B = directional_probabilities(s,t,occupancy,self.gamma)
        W,kappa,n = reciprocal(A,B,valid)
        P,Q = (A,B) if self.variant == "no_reciprocal_weighting" else (W,W.transpose(-1,-2))
        f_out = f + P @ self.vd(g)
        g_out = (g + Q @ self.vi(f)) * valid.unsqueeze(-1)
        return f_out,g_out,kappa,n


class ConfidenceGate(nn.Module):
    def __init__(self, variant="full"):
        super().__init__()
        self.variant = variant
        self.a = nn.Parameter(torch.zeros(()), requires_grad=variant not in ("no_gate","constant_gate"))
        self.b = nn.Parameter(torch.zeros(()), requires_grad=variant != "no_gate")

    def forward(self,kappa,n):
        if self.variant == "no_gate":
            return torch.ones_like(kappa)
        logits = self.b + torch.zeros_like(kappa)
        if self.variant != "constant_gate":
            logits = logits + F.softplus(self.a)*kappa
        return (n>=2).to(kappa.dtype)*torch.sigmoid(logits)
