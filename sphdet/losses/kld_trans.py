import torch
import pdb
import torch.nn as nn
from mmdet.models.builder import LOSSES
from mmdet.models.losses import weighted_loss
import numpy as np
import math

class SphBox2KentTransform:
    def __init__(self):
        self.transform = _sph_box2kent_transform
    def __call__(self, boxes):
        return self.transform(boxes)
    
def _sph_box2kent_transform(boxes):
    return bfov_to_kent_wh_stable(boxes)

def bfov_to_kent_wh_stable(annotations, epsilon=1e-6):
    if annotations.ndim == 1:
        annotations = annotations.unsqueeze(0)
   
    eta = 2*np.pi*annotations[:, 0] / 360.0 
    alpha = np.pi * annotations[:, 1] / 180.0

    h = torch.deg2rad(annotations[:, 3])
    w = torch.sin(alpha) * torch.deg2rad(annotations[:, 2])
    
    varphi_inv = 12 / (h**2 + epsilon)
    vartheta_inv = 12 / (w**2 + epsilon)
    
    log_kappa = (0.5 * (varphi_inv + vartheta_inv))  # log(κ)
    log_beta = (0.25 * (varphi_inv - vartheta_inv).abs() + epsilon)  # log(β)
    
    return torch.stack([eta, alpha, log_kappa, log_beta, w, h], dim=1)

def bfov_to_kent_wh(annotations, epsilon=1e-6):
    if annotations.ndim == 1:
        annotations = annotations.unsqueeze(0)

    #α ∈ [0, π] and η ∈ [0, 2π] be the co-latitude and longitude
    data_fov_w = annotations[:, 2]
    data_fov_h = annotations[:, 3]
   
    eta = 2*np.pi*annotations[:, 0] / 360.0 
    alpha = np.pi * annotations[:, 1] / 180.0

    h = torch.deg2rad(data_fov_h)
    varphi = (h**2) / 12 + epsilon
   
    w = torch.sin(alpha)*torch.deg2rad(data_fov_w)
    vartheta = (w**2) / 12 + epsilon

    kappa = 0.5 * (1 / varphi + 1 / vartheta)    
    beta = torch.abs(0.25 * (1 / vartheta - 1 / varphi))
        
    kent_dist = torch.stack([eta, alpha, kappa, beta, w ,h], dim=1)
        
    return kent_dist


def bfov_to_kent(annotations, epsilon=1e-6):
    if annotations.ndim == 1:
        annotations = annotations.unsqueeze(0)

    #α ∈ [0, π] and η ∈ [0, 2π] be the co-latitude and longitude
    data_fov_w = annotations[:, 2]
    data_fov_h = annotations[:, 3]
   
    eta = 2*np.pi*annotations[:, 0] / 360.0 
    alpha = np.pi * annotations[:, 1] / 180.0

    h = torch.deg2rad(data_fov_h)
    varphi = (h**2) / 12 + epsilon
   
    w = torch.sin(alpha)*torch.deg2rad(data_fov_w)
    vartheta = (w**2) / 12 + epsilon

    kappa = 0.5 * (1 / varphi + 1 / vartheta)    
    beta = torch.abs(0.25 * (1 / vartheta - 1 / varphi))
        
    kent_dist = torch.stack([eta, alpha, kappa, beta], dim=1)
        
    return kent_dist

def jiter_spherical_bboxes(bboxes1, bboxes2):
    eps = 1e-4 * 1.2345678
    similar_mask = (torch.abs(bboxes1 - bboxes2) < eps).any(dim=1)

    bboxes1[similar_mask] = bboxes1[similar_mask] - 2* eps
    bboxes2[similar_mask] = bboxes2[similar_mask] + eps

    pi = 180
    torch.clamp_(bboxes1[:, 0], 2*eps, 2*pi-eps)
    torch.clamp_(bboxes1[:, 1:4], 2*eps, pi-eps)
    torch.clamp_(bboxes2[:, 0], eps, 2*pi-2*eps)
    torch.clamp_(bboxes2[:, 1:4], eps, pi-2*eps)
    if bboxes1.size(1) == 5:
        torch.clamp_(bboxes2[:, 4], -2*pi+eps, max=2*pi-2*eps)
        torch.clamp_(bboxes2[:, 4], -2*pi+2*eps, max=2*pi-eps)

    return bboxes1, bboxes2

def check_nan_inf(tensor: torch.Tensor, name: str):
    """
    Check for NaN and Inf values in a tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to check.
        name (str): The name of the tensor for error reporting.
    """
    if torch.isnan(tensor).any():
        pdb.set_trace()
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        pdb.set_trace()
        raise ValueError(f"Inf detected in {name}")

def radians_to_Q(eta: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Convert angles in radians to a Q matrix.
    """

    N = alpha.size(0)
    alpha = alpha.view(N, 1)
    eta = eta.view(N, 1)
    
    gamma_1 = torch.cat([
        torch.cos(alpha),
        torch.sin(alpha) * torch.cos(eta),
        torch.sin(alpha) * torch.sin(eta)
    ], dim=1).unsqueeze(2)

    gamma_2 = torch.cat([
        - torch.sin(alpha),
        torch.cos(alpha) * torch.cos(eta),
        torch.cos(alpha) * torch.sin(eta)
    ], dim=1).unsqueeze(2)

    gamma_3 = torch.cat([
        torch.zeros(N, 1, device=eta.device),
        - torch.sin(eta),
        torch.cos(eta)
    ], dim=1).unsqueeze(2)

    gamma = torch.cat((gamma_1, gamma_2, gamma_3), dim=2)
    check_nan_inf(gamma, "gamma")
    return gamma

def log_approximate_c(log_kappa: torch.Tensor, log_beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-6  # Small value to avoid division by zero
    
    kappa = torch.exp(log_kappa)
    beta = torch.exp(log_beta)
    
    term1 = kappa - 2 * beta
    term2 = kappa + 2 * beta
    product = term1 * term2 + epsilon
    
    log_result = torch.log(torch.tensor(2 * torch.pi, device=log_kappa.device)) + kappa - 0.5 * torch.log(product)
    
    check_nan_inf(log_result, "approximate_c")
    return log_result

def expected_x(gamma_a1: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    const = (torch.exp(c_k - c)).view(-1, 1)
    result = const * gamma_a1
    check_nan_inf(result, "expected_x")
    return result

def expected_xxT(log_kappa: torch.Tensor, log_beta: torch.Tensor, Q_matrix: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    kappa = torch.exp(log_kappa)
    beta = torch.exp(log_beta)
    
    c_kk = log_del_2_kappa(log_kappa, log_beta)
    c_beta = log_del_beta(log_kappa, log_beta)
    epsilon = 1e-6  # Small value to avoid division by zero

    lambda_1 = torch.exp(c_k - c)
    lambda_2 = 0.5*(1 - torch.exp(c_kk - c) + torch.exp(c_beta - c))
    lambda_3 = 0.5*(1 - torch.exp(c_kk - c) - torch.exp(c_beta - c))

    lambdas = torch.stack([lambda_1, lambda_2, lambda_3], dim=-1)  # Shape: [N, 3]
    lambda_matrix = torch.diag_embed(lambdas)  # Shape: [N, 3, 3]

    Q_matrix_T = Q_matrix.transpose(-1, -2)  # Transpose the last two dimensions: [N, 3, 3]
    result = torch.matmul(Q_matrix, torch.matmul(lambda_matrix, Q_matrix_T))  # Shape: [N, 3, 3]
    check_nan_inf(result, "expected_xxT")
    return result

def beta_gamma_exxt_gamma(log_beta: torch.Tensor, gamma: torch.Tensor, ExxT: torch.Tensor) -> torch.Tensor:
    beta = torch.exp(log_beta)
    gamma_unsqueezed = gamma.unsqueeze(1)  # Shape: (N, 1, 3)
    intermediate_result = torch.bmm(gamma_unsqueezed, ExxT)  # Shape: (N, 1, 3)
    gamma_unsqueezed_2 = gamma.unsqueeze(2)  # Shape: (N, 3, 1)
    result = torch.bmm(intermediate_result, gamma_unsqueezed_2).squeeze()  # Shape: (N,)
    result = beta * result  # Shape: (N,)
    check_nan_inf(result, "beta_gamma_exxt_gamma")
    return result

def calculate_kappa_term(log_kappa_a, gamma_a1, log_kappa_b, gamma_b1, Ex_a): 
    kappa_a = torch.exp(log_kappa_a)
    kappa_b = torch.exp(log_kappa_b)
    
    kappa_a_gamma_a1 = kappa_a.view(-1, 1) * gamma_a1
    kappa_b_gamma_b1 = kappa_b.view(-1, 1) * gamma_b1
    diff_kappa_term_diag = kappa_a_gamma_a1 - kappa_b_gamma_b1
    result_diag = torch.sum(diff_kappa_term_diag*(Ex_a), dim=1)   
    
    return result_diag

def log_del_kappa(log_kappa: torch.Tensor, log_beta: torch.Tensor) -> torch.Tensor:
    kappa = torch.exp(log_kappa)
    beta = torch.exp(log_beta)
    
    epsilon = 1e-6    
    argument_numerator = torch.clamp(2 * torch.pi * (kappa**2 - kappa - 4 * beta**2), min=0)
    numerator = torch.log(argument_numerator + epsilon) + log_kappa
    
    argument_denominator = torch.clamp((kappa - 2 * beta) * (kappa + 2 * beta), min=0)
    denominator = 1.5 * torch.log(argument_denominator + epsilon)
    
    result = numerator - denominator
    
    check_nan_inf(result, "del_kappa")
    return result

def log_del_2_kappa(log_kappa: torch.Tensor, log_beta: torch.Tensor) -> torch.Tensor:
    kappa = torch.exp(log_kappa)
    beta = torch.exp(log_beta)
    
    epsilon = 1e-6
    
    polynomial_term = kappa**4 - 2 * kappa**3 + \
                     (2 - 8 * beta**2) * kappa**2 + \
                     8 * beta**2 * kappa + \
                     16 * beta**4 + 4 * beta**2
    
    log_numerator_inner = 2 * torch.pi * polynomial_term
    log_numerator_inner = torch.clamp(log_numerator_inner, min=epsilon)
    
    kappa_minus_2beta = kappa - 2 * beta
    kappa_plus_2beta = kappa + 2 * beta
    
    kappa_minus_2beta_safe = torch.clamp(kappa_minus_2beta, min=epsilon)
    kappa_plus_2beta_safe = torch.clamp(kappa_plus_2beta, min=epsilon)
    
    numerator = torch.log(log_numerator_inner) + log_kappa
    denominator = 2.5 * (torch.log(kappa_minus_2beta_safe) + 
                        torch.log(kappa_plus_2beta_safe)) + epsilon
    
    result = numerator - denominator
    
    check_nan_inf(result, "del_2_kappa")
    return result

def log_del_beta(log_kappa: torch.Tensor, log_beta: torch.Tensor) -> torch.Tensor:
    kappa = torch.exp(log_kappa)
    beta = torch.exp(log_beta)
    
    if torch.any(kappa <= 2 * beta):
        raise ValueError("kappa must be greater than 2 * beta to ensure log validity.")

    epsilon = 1e-6
    
    numerator = torch.log(torch.tensor(8 * torch.pi, device=log_kappa.device)) + log_kappa + log_beta
    denominator = 1.5*(torch.log(kappa - 2 * beta) + torch.log(kappa + 2 * beta)) + epsilon
    result = numerator - denominator

    return result

def calculate_beta_term(log_beta_a: torch.Tensor, gamma_a2: torch.Tensor, log_beta_b: torch.Tensor, gamma_b2: torch.Tensor, ExxT_a: torch.Tensor) -> torch.Tensor:
    beta_a = torch.exp(log_beta_a)
    beta_b = torch.exp(log_beta_b)

    beta_a_gamma_a2 = beta_a.view(-1, 1) * gamma_a2
    beta_a_gamma_a2_expanded = beta_a_gamma_a2.unsqueeze(1)
    intermediate_result_a2 = torch.bmm(beta_a_gamma_a2_expanded, ExxT_a)
    beta_a_term_1 = torch.bmm(intermediate_result_a2, gamma_a2.unsqueeze(2)).squeeze(-1)
    beta_b_gamma_b2 = beta_b.view(-1, 1) * gamma_b2
    product_diag = ExxT_a * beta_b_gamma_b2.unsqueeze(1)
    result_diag = product_diag.sum(dim=-1)
    
    gamma_b2_expanded = gamma_b2.unsqueeze(0)
    beta_b_term_1 = torch.sum(result_diag * gamma_b2_expanded, dim=-1)

    return beta_a_term_1, beta_b_term_1

def kld_diagonal(log_kappa_a: torch.Tensor, log_beta_a: torch.Tensor, gamma_a1: torch.Tensor, gamma_a2: torch.Tensor, gamma_a3: torch.Tensor,
               log_kappa_b: torch.Tensor, log_beta_b: torch.Tensor, gamma_b1: torch.Tensor, gamma_b2: torch.Tensor, gamma_b3: torch.Tensor,
               Ex_a: torch.Tensor, ExxT_a: torch.Tensor, c_a: torch.Tensor, c_b: torch.Tensor, c_ka: torch.Tensor) -> torch.Tensor:

    log_term = c_b - c_a
    ex_a_term = calculate_kappa_term(log_kappa_a, gamma_a1, log_kappa_b, gamma_b1, Ex_a)
    beta_a_term_1_expanded, beta_b_term_1 = calculate_beta_term(log_beta_a, gamma_a2, log_beta_b, gamma_b2, ExxT_a)
    beta_a_term_2_expanded, beta_b_term_2 = calculate_beta_term(log_beta_a, gamma_a3, log_beta_b, gamma_b3, ExxT_a)

    kld = log_term + ex_a_term + beta_a_term_1_expanded.T - beta_b_term_1 - beta_a_term_2_expanded.T + beta_b_term_2
    check_nan_inf(kld, "kld")
    return kld

def get_kld(kent_pred: torch.Tensor, kent_target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the KLD between predicted and target Kent distributions.
    Inputs are expected to have:
    - eta (orientation)
    - alpha (rotation)
    - log_kappa (concentration)
    - log_beta (ovalness)
    """
    eta_a, alpha_a, log_kappa_a, log_beta_a = kent_target[:, 0], kent_target[:, 1], kent_target[:, 2], kent_target[:, 3]
    eta_b, alpha_b, log_kappa_b, log_beta_b = kent_pred[:, 0], kent_pred[:, 1], kent_pred[:, 2], kent_pred[:, 3]

    Q_matrix_a = radians_to_Q(eta_a, alpha_a)
    Q_matrix_b = radians_to_Q(eta_b, alpha_b)

    gamma_a1, gamma_a2, gamma_a3 = Q_matrix_a[:, :, 0], Q_matrix_a[:, :, 1], Q_matrix_a[:, :, 2]
    gamma_b1, gamma_b2, gamma_b3 = Q_matrix_b[:, :, 0], Q_matrix_b[:, :, 1], Q_matrix_b[:, :, 2]

    c_a = log_approximate_c(log_kappa_a, log_beta_a)
    c_b = log_approximate_c(log_kappa_b, log_beta_b)
    c_ka = log_del_kappa(log_kappa_a, log_beta_a)

    ExxT_a = expected_xxT(log_kappa_a, log_beta_a, Q_matrix_a, c_a, c_ka)
    Ex_a = expected_x(gamma_a1, c_a, c_ka)

    kld = kld_diagonal(log_kappa_a, log_beta_a, gamma_a1, gamma_a2, gamma_a3,
                     log_kappa_b, log_beta_b, gamma_b1, gamma_b2, gamma_b3,
                     Ex_a, ExxT_a, c_a, c_b, c_ka)

    return kld

@LOSSES.register_module()
class JsdRawLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super().__init__()
        #assert mode in ['iou', 'giou', 'diou', 'ciou']
        #self.mode = mode
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

        loss = self.loss_weight * jsd_raw_loss(
            target,
            pred,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        
        return loss

@weighted_loss
def jsd_raw_loss(pred, true, eps = 1e-6):
    # Ensure inputs are 2D[]
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if true.dim() == 1:
        true = true.unsqueeze(0)

    y_pred = bfov_to_kent(pred).double()
    y_true = bfov_to_kent(true).double()

    m = 0.5*(y_pred+y_true)

    kld_pt = get_kld(y_pred, m)
    kld_tp = get_kld(y_true, m)

    kld_pt = torch.clamp(kld_pt, min =0)
    kld_tp = torch.clamp(kld_tp, min =0)
    
    jsd = (kld_pt+kld_tp)/2
    
    const = 1.
    jsd_iou = 1 / (const + jsd)  

    #w_h_pred = torch.sqrt((y_pred[:,2]-2*y_pred[:,3])/(y_pred[:,2]+2*y_pred[:,3]))
    #w_h_true = torch.sqrt((y_true[:,2]-2*y_true[:,3])/(y_true[:,2]+2*y_true[:,3])) 

    #w_h_pred = y_pred[:,4]/y_pred[:,5]
    #w_h_true = y_true[:,4]/y_true[:,5]

    #arctan_pred = torch.atan(w_h_pred)
    #arctan_true = torch.atan(w_h_true)

    #v = (4 / (torch.pi ** 2)) * ((arctan_true - arctan_pred) ** 2) 
    #alpha = v / (1 - jsd_iou + v)

    #center_dist = torch.pow(y_pred[:, 0] - y_true[:, 0], 2) + \
    #             torch.pow(y_pred[:, 1] - y_true[:, 1], 2)
    
    #enclose_x1 = torch.min(y_pred[:, 0] - y_pred[:, 4]/2, y_true[:, 0] - y_true[:, 4]/2)
    #enclose_y1 = torch.min(y_pred[:, 1] - y_pred[:, 5]/2, y_true[:, 1] - y_true[:, 5]/2)
    #enclose_x2 = torch.max(y_pred[:, 0] + y_pred[:, 4]/2, y_true[:, 0] + y_true[:, 4]/2)
    #enclose_y2 = torch.max(y_pred[:, 1] + y_pred[:, 5]/2, y_true[:, 1] + y_true[:, 5]/2)
    
    #enclose_diag = torch.pow(enclose_x2 - enclose_x1, 2) + torch.pow(enclose_y2 - enclose_y1, 2) + eps

    kld_loss = 1 - jsd_iou # + alpha*v # + (center_dist / enclose_diag)**2
    return kld_loss






@LOSSES.register_module()
class JsdRawCiouLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super().__init__()
        #assert mode in ['iou', 'giou', 'diou', 'ciou']
        #self.mode = mode
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

        loss = self.loss_weight * jsd_raw_ciou_loss(
            target,
            pred,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        
        return loss

@weighted_loss
def jsd_raw_ciou_loss(pred, true, eps = 1e-6):
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if true.dim() == 1:
        true = true.unsqueeze(0)

    y_pred = bfov_to_kent(pred).double()
    y_true = bfov_to_kent(true).double()

    m = 0.5*(y_pred+y_true)

    kld_pt = get_kld(y_pred, m)
    kld_tp = get_kld(y_true, m)

    kld_pt = torch.clamp(kld_pt, min =0)
    kld_tp = torch.clamp(kld_tp, min =0)
    
    jsd = (kld_pt+kld_tp)/2
    
    const = 1.
    jsd_iou = 1 / (const + jsd)  

    w_h_pred = torch.sqrt((y_pred[:,2]-2*y_pred[:,3])/(y_pred[:,2]+2*y_pred[:,3]))
    w_h_true = torch.sqrt((y_true[:,2]-2*y_true[:,3])/(y_true[:,2]+2*y_true[:,3])) 

    arctan_pred = torch.atan(w_h_pred)
    arctan_true = torch.atan(w_h_true)

    v = (4 / (torch.pi ** 2)) * ((arctan_true - arctan_pred) ** 2) 
    alpha = v / (1 - jsd_iou + v)

    kld_loss = 1 - jsd_iou + alpha*v
    return kld_loss


@LOSSES.register_module()
class JsdRawCiouLossWH(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super().__init__()
        #assert mode in ['iou', 'giou', 'diou', 'ciou']
        #self.mode = mode
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

        loss = self.loss_weight * jsd_raw_ciou_loss_wh(
            target,
            pred,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        
        return loss

@weighted_loss
def jsd_raw_ciou_loss_wh(pred, true, eps = 1e-6):
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if true.dim() == 1:
        true = true.unsqueeze(0)

    y_pred = bfov_to_kent_wh(pred).double()
    y_true = bfov_to_kent_wh(true).double()

    m = 0.5*(y_pred+y_true)

    kld_pt = get_kld(y_pred, m)
    kld_tp = get_kld(y_true, m)

    kld_pt = torch.clamp(kld_pt, min =0)
    kld_tp = torch.clamp(kld_tp, min =0)
    
    jsd = (kld_pt+kld_tp)/2
    
    const = 1.
    jsd_iou = 1 / (const + jsd)  

    w_h_pred = ((y_pred[:,4])/(y_pred[:,5]))
    w_h_true = ((y_true[:,4])/(y_true[:,5])) 

    arctan_pred = torch.atan(w_h_pred)
    arctan_true = torch.atan(w_h_true)

    v = (4 / (torch.pi ** 2)) * ((arctan_true - arctan_pred) ** 2) 
    alpha = v / (1 - jsd_iou + v)

    kld_loss = 1 - jsd_iou + alpha*v
    return kld_loss



@LOSSES.register_module()
class StableJefRawCiouLossWH(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super().__init__()
        #assert mode in ['iou', 'giou', 'diou', 'ciou']
        #self.mode = mode
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
        pred, target = bfov_to_kent_wh_stable(pred), bfov_to_kent_wh_stable(target)

        loss = self.loss_weight * jef_raw_ciou_loss_wh(
            target,
            pred,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        
        return loss

@weighted_loss
def jef_raw_ciou_loss_wh(pred, true, eps = 1e-6):
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if true.dim() == 1:
        true = true.unsqueeze(0)

    y_pred = (pred).double()
    y_true = (true).double()

    #m = 0.5*(y_pred+y_true)

    kld_pt = get_kld(y_pred, y_true)
    kld_tp = get_kld(y_true, y_pred)

    kld_pt = torch.clamp(kld_pt, min =0)
    kld_tp = torch.clamp(kld_tp, min =0)
    
    jsd = (kld_pt+kld_tp)/2
    
    const = 1.
    jsd_iou = 1 / (const + jsd)  

    w_h_pred = ((y_pred[:,4])/(y_pred[:,5]))
    w_h_true = ((y_true[:,4])/(y_true[:,5])) 

    arctan_pred = torch.atan(w_h_pred)
    arctan_true = torch.atan(w_h_true)

    v = (4 / (torch.pi ** 2)) * ((arctan_true - arctan_pred) ** 2) 
    alpha = v / (1 - jsd_iou + v)

    kld_loss = 1 - jsd_iou + alpha*v
    return kld_loss
    
if __name__ == "__main__":
    pred = torch.tensor([[  2.2335,   1.7491, 331.7626, 146.6469]], dtype=torch.float32, requires_grad=True)#.half()
    target = torch.tensor([[5.4978, 2.3562, 1.2747, 0.4459]], dtype=torch.float32, requires_grad=True)#.half()
    loss = get_kld(target, pred)
    #loss.backward(retain_graph=True)
    print(loss)