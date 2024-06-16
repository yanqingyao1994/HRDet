import torch
import torch.nn as nn
from mmrotate.registry import MODELS
from mmcv.ops import diff_iou_rotated_2d
from mmdet.models.losses.utils import weighted_loss

pi = 3.1415926
from mmrotate.structures.bbox import rbox2hbox 

@weighted_loss
def oiou_loss(pred, target, eps=1e-6):
    """OIoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """

    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    ious = ious.squeeze(0).clamp(min=eps)

    # dtheta
    pred_x, pred_y, pred_w, pred_h, pred_theta = pred.unbind(-1)
    target_x, target_y, target_w, target_h, target_theta = target.unbind(-1)
    pred, target = rbox2hbox(pred), rbox2hbox(target)
    dtheta = ((pred_theta - target_theta) / pi)

    # enclose area
    b1_x1, b1_y1, b1_x2, b1_y2 = pred.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target.unbind(-1)
    dx = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    dy = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    c2 = enclose_wh[:, 0]**2 + enclose_wh[:, 1]**2 + eps

    distance = (dx+dy)/c2 * torch.exp(abs(dtheta)-1)
    return 1 - ious + distance

@MODELS.register_module()
class OIoULoss(nn.Module):
    """OIoULoss.

    Args:
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 5) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.loss_weight * oiou_loss(
                pred,
                target,
                weight,
                eps=self.eps,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
        return loss
