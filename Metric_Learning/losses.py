import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, a, p, n):
        ha = 1
        loss = torch.norm(a - p, p=2, dim=1) - torch.norm(a - n, p=2, dim=1) + self.margin
        loss = F.relu(loss).mean()
        return loss


class TripletAngularLoss(nn.Module):
    def __init__(self, alpha=45, in_degree=True, margin=5.0):
        super(TripletAngularLoss, self).__init__()
        self.margin = margin
        if in_degree:
            alpha = np.deg2rad(alpha)
        self.tan_alpha = np.tan(alpha) ** 2

    def forward(self, a, p, n):
        c = (a + p) / 2
        loss = torch.norm(a - p, p=2, dim=1) - 4*self.tan_alpha*torch.norm(n - c, p=2, dim=1) + self.margin
        return F.relu(loss).mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0),inputs.size(1),-1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1,2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1,inputs.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(inputs)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()