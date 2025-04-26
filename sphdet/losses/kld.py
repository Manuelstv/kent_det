import torch
import pdb
import torch.nn as nn
from mmdet.models.builder import LOSSES
from mmdet.models.losses import weighted_loss
from sphdet.losses.kent_kld import get_kld, jiter_spherical_bboxes
import numpy as np
import math

def bfov_to_kent(annotations, epsilon=1e-6):
    if annotations.ndim == 1:
        annotations = annotations.unsqueeze(0)

    eta = 2*np.pi*annotations[:, 0] / 360.0
    alpha = np.pi * annotations[:, 1] / 180.0

    fov_theta = annotations[:, 2]
    fov_phi = annotations[:, 3]

    w = torch.sin(alpha)*torch.deg2rad(fov_theta)
    h = torch.deg2rad(fov_phi)

    varphi = (h**2) / 12 + epsilon
    vartheta = (w**2) / 12 + epsilon

    kappa = 0.5 * (1 / varphi + 1 / vartheta)
    beta = torch.abs(0.25 * (1 / vartheta - 1 / varphi))

    kent_dist = torch.stack([eta, alpha, kappa, beta, fov_theta, fov_phi], dim=1)

    return kent_dist

class SphBox2KentTransform:
    def __init__(self):
        self.transform = _sph_box2kent_transform
    def __call__(self, boxes):
        return self.transform(boxes)

def _sph_box2kent_transform(boxes):
    return bfov_to_kent(boxes)

@LOSSES.register_module()
class KentLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super().__init__()
        #assert mode in ['iou', 'giou', 'diou', 'ciou']
        #self.mode = mode
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.transform = SphBox2KentTransform()

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        pred, target = jiter_spherical_bboxes(pred, target)

        kent_pred = self.transform(pred)
        kent_target = self.transform(target)

        loss = self.loss_weight * kent_loss(
            kent_pred,
            kent_target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss

@weighted_loss
def kent_loss(y_pred, y_true, eps = 1e-6):
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(0)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)

    #kld part. really needs to improve variable names
    pred = y_pred[:, :4].double()
    true = y_true[:, :4].double()

    kld_pt = get_kld(pred, true)
    kld_tp = get_kld(true, pred)

    kld_pt = torch.clamp(kld_pt, min =0)
    kld_tp = torch.clamp(kld_tp, min =0)

    jsd = (kld_pt+kld_tp)/2
    const = 1.
    jsd_iou = 1 / (const + jsd)

    w2, h2 = y_pred[:,4], y_pred[:,5]
    w1, h1 = y_true[:,4], y_true[:,5]

    factor = 4 / torch.pi ** 2

    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    #Should we use masking like sph2pob?
    with torch.no_grad():
        alpha = v / (1 - jsd_iou + v + eps)

    '''''w_true = torch.deg2rad(y_true[:, 4])
    h_true = torch.deg2rad(y_true[:, 5])
    w_pred = torch.deg2rad(y_pred[:, 4])
    h_pred = torch.deg2rad(y_pred[:, 5])
    cx_pred, cy_pred = y_pred[:, 0], y_pred[:, 1]
    cx_true, cy_true = y_true[:, 0], y_true[:, 1]

    rho_squared = (cx_pred - cx_true)**2 + (cy_pred - cy_true)**2

    pred_left = cx_pred - w_pred / 2
    pred_right = cx_pred + w_pred / 2
    pred_bottom = cy_pred - h_pred / 2
    pred_top = cy_pred + h_pred / 2

    true_left = cx_true - w_true / 2
    true_right = cx_true + w_true / 2
    true_bottom = cy_true - h_true / 2
    true_top = cy_true + h_true / 2

    enclose_left = torch.minimum(pred_left, true_left)
    enclose_right = torch.maximum(pred_right, true_right)
    enclose_bottom = torch.minimum(pred_bottom, true_bottom)
    enclose_top = torch.maximum(pred_top, true_top)
    c_squared = (enclose_right - enclose_left)**2 + (enclose_top - enclose_bottom)**2

    epsilon = 1e-7
    distance_penalty = rho_squared / (c_squared + epsilon)'''

    kld_loss = 1 - jsd_iou + alpha*v# + distance_penalty
    return kld_loss


if __name__ == "__main__":
    pred = torch.tensor([[  2.2335,   1.7491, 331.7626, 146.6469]], dtype=torch.float32, requires_grad=True)#.half()
    target = torch.tensor([[5.4978, 2.3562, 1.2747, 0.4459]], dtype=torch.float32, requires_grad=True)#.half()
    loss = get_kld(target, pred)
    #loss.backward(retain_graph=True)
    print(loss)