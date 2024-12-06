# code from https://github.com/hou-yz/MVDet/tree/master
# modified by Erik Brorsson
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GaussianMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, target, kernel):

        target = self._traget_transform(x, target, kernel)
        return F.mse_loss(x, target)

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        if not kernel is None:
            with torch.no_grad():
                target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target
    
class WeightedGaussianMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, target, kernel, weight=None):

        target = self._traget_transform(x, target, kernel)

        if weight is not None:
            return F.mse_loss(x*weight, target*weight) # loss will be zero wherever weight is zero, and mse(x,target) where weight is 1
        else:
            return F.mse_loss(x, target)

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        if not kernel is None:
            with torch.no_grad():
                target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target
