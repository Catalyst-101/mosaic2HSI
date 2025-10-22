import torch
import torch.nn.functional as F
import numpy as np


def l1_loss(pred, target):
    return F.l1_loss(pred, target)


def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def spectral_angle_mapper(pred, target, eps=1e-8):
    # pred, target: (B, C, H, W)
    # compute SAM per pixel
    p = pred.view(pred.shape[0], pred.shape[1], -1)
    t = target.view(target.shape[0], target.shape[1], -1)
    dot = (p * t).sum(dim=1)
    pnorm = torch.norm(p, dim=1)
    tnorm = torch.norm(t, dim=1)
    cos = dot / (pnorm * tnorm + eps)
    cos = torch.clamp(cos, -1+1e-7, 1-1e-7)
    ang = torch.acos(cos)  # radians
    return ang.mean()

