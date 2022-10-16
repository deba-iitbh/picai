import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary Segmentation"""
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs = inputs.reshape(inputs.shape[0], -1)
        # targets = targets.reshape(targets.shape[0], -1)
        num = torch.sum(torch.mul(inputs, targets), dim=1) + self.smooth
        den = torch.sum(inputs.pow(self.p) + targets.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
