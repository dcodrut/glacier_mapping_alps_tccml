import torch
import torch.nn.functional as F


class MaskedDist(torch.nn.Module):
    def __init__(self, distance_type='L2'):
        super().__init__()
        self.distance_type = distance_type

    def forward(self, preds, targets, mask, samplewise=False):
        assert preds.shape == targets.shape and preds.shape == mask.shape

        # apply the mask
        preds_masked = preds * mask
        targets_masked = targets * mask

        # compute the loss
        reduction = 'none' if samplewise else 'sum'
        if self.distance_type == 'L2':
            loss = F.mse_loss(preds_masked, targets_masked, reduction=reduction)
        elif self.distance_type == 'L1':
            loss = F.l1_loss(preds_masked, targets_masked, reduction=reduction)
        elif self.distance_type == 'BCE':
            loss = F.binary_cross_entropy(preds_masked, targets_masked, reduction=reduction)
        else:
            raise NotImplementedError

        if samplewise:
            # reduce all but the first (batch) dimension
            axes = tuple(list(range(1, len(preds.shape))))
            loss = loss.sum(axes)
            den = (mask > 0).sum(axes)
        else:
            den = (mask > 0).sum()

        loss /= den

        return loss
