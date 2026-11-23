"""Dual ResNet-18, RGB multi-scale residual fusion, reciprocal matcher and SE(3)."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .matching import ReciprocalMatcher, ConfidenceGate
from .geometry import rasterize, se3_exp


def conv_bn(in_ch,out_ch,k=3,stride=1,groups=1):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,k,stride,k//2,groups=groups,bias=False),
                         nn.BatchNorm2d(out_ch))


class BasicBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1):
        super().__init__()
        self.c1=conv_bn(in_ch,out_ch,stride=stride)
        self.c2=conv_bn(out_ch,out_ch)
        self.skip=conv_bn(in_ch,out_ch,k=1,stride=stride) if stride!=1 or in_ch!=out_ch else nn.Identity()

    def forward(self,x):
        return F.relu(self.c2(F.relu(self.c1(x))) + self.skip(x))


class ResNet18(nn.Module):
    def __init__(self,in_ch):
        super().__init__()
        self.stem=nn.Sequential(conv_bn(in_ch,64,7,2),nn.ReLU(),nn.MaxPool2d(3,2,1))
        layers=[]
        last=64
        for i,ch in enumerate((64,128,256,512)):
            layers.append(nn.Sequential(BasicBlock(last,ch,1 if i==0 else 2),BasicBlock(ch,ch)))
            last=ch
        self.layers=nn.ModuleList(layers)

    def forward(self,x):
        x=self.stem(x)
        features=[]
        for layer in self.layers:
            x=layer(x)
            features.append(x)
        return features


class RCB(nn.Module):
    """CCAM-inspired depthwise 3x3 + pointwise 1x1 residual block."""
    def __init__(self,c):
        super().__init__()
        self.depthwise=conv_bn(c,c,3,groups=c)
        self.pointwise=conv_bn(c,c,1)

    def forward(self,x):
        return x+self.pointwise(F.relu(self.depthwise(x)))


class MultiScaleRGB(nn.Module):
    def __init__(self,c,shape):
        super().__init__()
        self.shape=shape
        self.proj=nn.ModuleList(conv_bn(ch,c,1) for ch in (128,256,512))
        self.blocks=nn.ModuleList(RCB(c) for _ in range(3))
        self.merge=conv_bn(3*c,c,1)

    def forward(self,features):
        result=[]
        for p,b,x in zip(self.proj,self.blocks,features[-3:]):
            x=F.interpolate(p(x),size=self.shape,mode="bilinear",align_corners=False)
            result.append(b(x))
        return self.merge(torch.cat(result,dim=1))


class PoseHead(nn.Module):
    """Convolutional pose regression head with spatial average pooling."""
    def __init__(self,c,shape):
        super().__init__()
        self.shape=shape
        self.layers=nn.Sequential(conv_bn(2*c,c,3),nn.ReLU(),
                                  conv_bn(c,c//2,1),nn.ReLU(),
                                  nn.Conv2d(c//2,6,1),nn.AdaptiveAvgPool2d(1))

    def forward(self,f,g):
        x=torch.cat((f,g),-1).transpose(1,2).reshape(f.shape[0],-1,*self.shape)
        return self.layers(x).flatten(1)


class MultiAttCalib(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        c=cfg.channels
        self.shape=(cfg.token_height,cfg.token_width)
        n=cfg.token_height*cfg.token_width
        self.rgb_backbone=ResNet18(3)
        self.depth_backbone=ResNet18(1)
        self.rgb_fusion=MultiScaleRGB(c,self.shape)
        self.depth_projection=conv_bn(256,c,1)
        self.pos_rgb=nn.Parameter(torch.zeros(1,n,c))
        self.pos_depth=nn.Parameter(torch.zeros(1,n,c))
        nn.init.normal_(self.pos_rgb,std=.02)
        nn.init.normal_(self.pos_depth,std=.02)
        self.rgb_attention=nn.MultiheadAttention(c,cfg.attention_heads,batch_first=True)
        self.depth_attention=nn.MultiheadAttention(c,cfg.attention_heads,batch_first=True)
        self.rgb_norm=nn.LayerNorm(c)
        self.depth_norm=nn.LayerNorm(c)
        self.matcher=ReciprocalMatcher(c,cfg.gamma,cfg.variant)
        self.gate=ConfidenceGate(cfg.variant)
        self.pose_head=PoseHead(c,self.shape)
        self.register_buffer("twist_bound",torch.tensor([math.radians(cfg.rotation_twist_bound_deg)]*3+
                                                       [cfg.translation_twist_bound_m]*3))

    def rgb_tokens(self,rgb):
        f=self.rgb_fusion(self.rgb_backbone(rgb)).flatten(2).transpose(1,2)+self.pos_rgb
        return self.rgb_norm(f+self.rgb_attention(f,f,f,need_weights=False)[0])

    def step(self,f,points,K,T):
        cfg=self.cfg
        dm=[rasterize(p,t,k,cfg.image_height,cfg.image_width) for p,k,t in zip(points,K,T)]
        depth=torch.stack([d for d,m in dm])
        mask=torch.stack([m for d,m in dm])
        occupancy=F.adaptive_avg_pool2d(mask,self.shape).flatten(1)
        g=self.depth_projection(self.depth_backbone(depth/cfg.range_m)[2])
        g=F.interpolate(g,size=self.shape,mode="bilinear",align_corners=False)
        g=g.flatten(2).transpose(1,2)+self.pos_depth
        g=self.depth_norm(g+self.depth_attention(g,g,g,need_weights=False)[0])
        fp,gp,kappa,n=self.matcher(f,g,occupancy)
        alpha=self.gate(kappa,n)
        xi=self.twist_bound*torch.tanh(self.pose_head(fp,gp))
        updated=se3_exp(alpha[:,None]*xi)@T
        return updated,{"kappa":kappa,"alpha":alpha,"n_valid":n,"xi":xi}

    def forward(self,rgb,points,K,T0,steps=None):
        steps=steps if steps is not None else (self.cfg.train_steps if self.training else self.cfg.eval_steps)
        f=self.rgb_tokens(rgb)
        poses,diagnostics=[],[]
        T=T0
        for _ in range(steps):
            T,d=self.step(f,points,K,T)
            poses.append(T)
            diagnostics.append(d)
        return poses,diagnostics
