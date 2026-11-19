"""DRC evidence, dual-reference correction head, boundary band and structure loss."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def evidence(rgb, margin):
    def pool(x):
        return F.avg_pool2d(F.pad(x, (4, 4, 4, 4), mode="reflect"), 9, stride=1)

    p = margin.sigmoid()
    masses, distances, inputs = [], [], []
    rgb_squared = rgb.square()
    for probability, weights in ((p, (2 * p - 1).relu()), (1 - p, (1 - 2 * p).relu())):
        mass = pool(weights)
        local = pool(weights * rgb) / mass.clamp_min(1e-6)
        whole = (weights * rgb).sum((-2, -1), keepdim=True) / weights.sum((-2, -1), keepdim=True).clamp_min(1e-6)
        variance = (pool(weights * rgb_squared) / mass.clamp_min(1e-6) - local.square()).clamp_min(0).mean(1, keepdim=True)
        shift = (local - whole).square().mean(1, keepdim=True)
        distances.extend((mean.sub(rgb).square().mean(1, keepdim=True) for mean in (local, whole)))
        inputs.extend(((81 * mass).log1p(), variance.log1p(), shift.log1p(), probability))
        masses.append(mass)
    gate = (9 * torch.minimum(*masses)).clamp(0, 1)
    return torch.cat((margin, *masses, gate, *distances, *inputs), dim=1)[0].permute(1, 2, 0).contiguous()


class DualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorers = nn.ModuleList(nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
                                     for _ in range(2))
        self.gain = nn.Linear(4, 1)
        for layer in (self.scorers[0][-1], self.scorers[1][-1], self.gain):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def mix(self, x):
        logits = torch.cat([self.scorers[i](x[..., 8+4*i:12+4*i]) for i in range(2)], dim=-1)
        return logits.sigmoid()

    def forward(self, x):
        alpha = self.mix(x)
        foreground = torch.lerp(x[..., 5], x[..., 4], alpha[..., 0])
        background = torch.lerp(x[..., 7], x[..., 6], alpha[..., 1])
        contrast = (background - foreground) / (background + foreground + .01)
        margin, fg_mass, bg_mass, gate = x[..., :4].unbind(-1)
        inputs = torch.stack((fg_mass, bg_mass, contrast.abs(), (margin.abs() / 4).clamp(0, 1)), dim=-1)
        score = margin + 4 * self.gain(inputs).squeeze(-1).sigmoid() * gate * contrast
        return score


def band(m):
 out=[]
 for z in (m.detach()>0).cpu().numpy()[:,0]:
  if not z.any() or z.all():out.append(np.zeros_like(z));continue
  a=cv2.distanceTransform(z.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
  b=cv2.distanceTransform((~z).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
  out.append((z&(a<=3))|(~z&(b<=3)))
 return torch.from_numpy(np.stack(out)).to(m.device)[:,None]

class Head(nn.Module):
    """The 392-parameter DRC head with the original J0-S state keys."""

    def __init__(self):
        super().__init__()
        self.original = DualHead()
        self.amplitude = nn.Parameter(torch.zeros(()))

    def forward(self, rgb, margin):
        mask = band(margin)
        delta = torch.stack([
            self.original(evidence(image[None], score[None])) - score[0]
            for image, score in zip(rgb, margin)
        ])[:, None]
        return margin + mask * self.amplitude.clamp(0, 1) * delta


def softcl(z,y):
 def erode(x):return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))
 def sk(x):
  opened=F.max_pool2d(erode(x),3,1,1);s=F.relu(x-opened)
  for _ in range(10):
   x=erode(x);d=F.relu(x-F.max_pool2d(erode(x),3,1,1));s=s+F.relu(d-s*d)
  return s
 p=z.sigmoid();y=y.float();sp,sy=sk(p),sk(y);eps=1e-6
 precision=((sp*y).sum((1,2,3))+eps)/(sp.sum((1,2,3))+eps)
 recall=((sy*p).sum((1,2,3))+eps)/(sy.sum((1,2,3))+eps)
 loss=1-2*precision*recall/(precision+recall+eps)
 return torch.where(y.sum((1,2,3))>0,loss,p.mean((1,2,3))).mean()
