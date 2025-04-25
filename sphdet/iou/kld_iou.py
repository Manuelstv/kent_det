import torch
import pdb
import torch.nn as nn
from mmdet.models.builder import LOSSES
from mmdet.models.losses import weighted_loss
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

    kent_dist = torch.stack([eta, alpha, kappa, beta], dim=1)

    return kent_dist

class SphBox2KentTransform:
    def __init__(self):
        self.transform = _sph_box2kent_transform
    def __call__(self, boxes):
        return self.transform(boxes)
    
def _sph_box2kent_transform(boxes):
    return bfov_to_kent(boxes)

'''def debug_negative_kld(kent_pred: torch.Tensor, kent_target: torch.Tensor, kld: torch.Tensor):
    """
    Debug function to identify which Kent distribution parameters lead to negative KLD values.
    
    Args:
        kent_pred (torch.Tensor): Predicted Kent distribution parameters (eta, alpha, psi, kappa, beta)
        kent_target (torch.Tensor): Target Kent distribution parameters (eta, alpha, psi, kappa, beta)
        kld (torch.Tensor): The calculated KLD matrix
    """
    # Find indices where KLD is negative
    negative_indices = torch.where(kld < 0)
    
    if len(negative_indices[0]) == 0:
        print("No negative KLD values found.")
        return
    
    print(f"\nFound {len(negative_indices[0])} negative KLD values")
    print("\nDetailed analysis of negative KLD cases:")
    
    for i, j in zip(negative_indices[0], negative_indices[1]):
        kld_value = kld[i, j]
        
        # Get corresponding parameters
        pred_params = kent_pred[i]
        target_params = kent_target[j]
        
        print(f"\nKLD Value: {kld_value:.6f}")
        print("\nPredicted parameters:")
        print(f"  eta: {pred_params[0]:.6f}")
        print(f"  alpha: {pred_params[1]:.6f}")
        print(f"  psi: {pred_params[2]:.6f}")
        print(f"  kappa: {pred_params[3]:.6f}")
        print(f"  beta: {pred_params[4]:.6f}")
        
        print("\nTarget parameters:")
        print(f"  eta: {target_params[0]:.6f}")
        print(f"  alpha: {target_params[1]:.6f}")
        print(f"  psi: {target_params[2]:.6f}")
        print(f"  kappa: {target_params[3]:.6f}")
        print(f"  beta: {target_params[4]:.6f}")
        
        # Calculate parameter differences
        diff = pred_params - target_params
        print("\nParameter differences (pred - target):")
        print(f"  eta diff: {diff[0]:.6f}")
        print(f"  alpha diff: {diff[1]:.6f}")
        print(f"  psi diff: {diff[2]:.6f}")
        print(f"  kappa diff: {diff[3]:.6f}")
        print(f"  beta diff: {diff[4]:.6f}")
        print("-" * 50)
'''
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

def normalized_l1_loss(y_true, y_pred, scale_factor=30.0):
    """
    Compute normalized L1 loss for 5-tuple where:
    - First 3 values range from -pi to pi
    - Last 2 values range from 0 to inf
    
    Args:
        y_pred: torch.Tensor of shape (n, 5) or (5,) - predicted values
        y_true: torch.Tensor of shape (n, 5) or (5,) - true values
        scale_factor: float - scaling factor for sigmoid normalization
    
    Returns:
        torch.Tensor: normalized L1 loss
    """
    # Ensure inputs are 2D
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(0)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)
        
    #print(y_pred, y_true)
    
    #y_pred, y_true = jiter_spherical_bboxes(y_pred, y_true)
    
    mask = (y_true != 0).any(dim=1)
    if not mask.any():
        return torch.tensor(0.0, device=y_pred.device)
    
    #pdb.set_trace()
    pred = torch.stack([y_pred[mask, 0]/360, y_pred[mask, 1]/180, y_pred[mask, 2]/180, y_pred[mask, 3]/180], dim=1)
    true = torch.stack([y_true[mask, 0]/360, y_true[mask, 1]/180, y_true[mask, 2]/180, y_true[mask, 3]/180], dim=1)

    # Compute L1 loss for the stacked tensors
    l1_loss = torch.mean(torch.abs(pred - true))  # Calculate the mean L1 loss
    return l1_loss  # Return the computed L1 loss

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

def log_approximate_c(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:

    epsilon = 1e-6  # Small value to avoid division by zero
    
    term1 = kappa - 2 * beta
    term2 = kappa + 2 * beta
    product = term1 * term2 + epsilon
    
    log_result = torch.log(torch.tensor(2 * torch.pi, device=kappa.device)) + kappa - 0.5 * torch.log(product)
    
    check_nan_inf(log_result, "approximate_c")
    return log_result

def expected_x(gamma_a1: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:

    const = (torch.exp(c_k - c)).view(-1, 1)
    result = const * gamma_a1
    check_nan_inf(result, "expected_x")
    return result

def expected_xxT(kappa: torch.Tensor, beta: torch.Tensor, Q_matrix: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:

    c_kk = log_del_2_kappa(kappa, beta)
    c_beta = log_del_beta(kappa, beta)
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

def beta_gamma_exxt_gamma(beta: torch.Tensor, gamma: torch.Tensor, ExxT: torch.Tensor) -> torch.Tensor:
    gamma_unsqueezed = gamma.unsqueeze(1)  # Shape: (N, 1, 3)
    intermediate_result = torch.bmm(gamma_unsqueezed, ExxT)  # Shape: (N, 1, 3)
    gamma_unsqueezed_2 = gamma.unsqueeze(2)  # Shape: (N, 3, 1)
    result = torch.bmm(intermediate_result, gamma_unsqueezed_2).squeeze()  # Shape: (N,)
    result = beta * result  # Shape: (N,)
    check_nan_inf(result, "beta_gamma_exxt_gamma")
    return result

def calculate_kappa_term(kappa_a, gamma_a1, kappa_b, gamma_b1, Ex_a): 
    kappa_a_gamma_a1 = kappa_a.view(-1, 1) * gamma_a1
    kappa_b_gamma_b1 = kappa_b.view(-1, 1) * gamma_b1
    diff_kappa_term_diag = kappa_a_gamma_a1 - kappa_b_gamma_b1
    result_diag = torch.sum(diff_kappa_term_diag*(Ex_a), dim=1)   
    
    return result_diag

def log_del_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-6    
    argument_numerator = torch.clamp(2 * torch.pi * (kappa**2 - kappa - 4 * beta**2), min =0)
    numerator = torch.log(argument_numerator + epsilon) + kappa
    
    argument_denominator = torch.clamp((kappa - 2 * beta) * (kappa + 2 * beta), min =0)
    denominator = 1.5 * torch.log(argument_denominator + epsilon)
    
    result = numerator - denominator
    
    check_nan_inf(result, "del_kappa")
    return result

def log_del_2_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
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
    
    numerator = torch.log(log_numerator_inner) + kappa
    denominator = 2.5 * (torch.log(kappa_minus_2beta_safe) + 
                        torch.log(kappa_plus_2beta_safe)) + epsilon
    
    result = numerator - denominator
    
    check_nan_inf(result, "del_2_kappa")
    return result

def log_del_beta(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:

    if torch.any(kappa <= 2 * beta):
        raise ValueError("kappa must be greater than 2 * beta to ensure log validity.")


    epsilon = 1e-6
    
    numerator = torch.log(torch.tensor(8 * torch.pi, device=kappa.device))+kappa + torch.log(beta+epsilon)
    denominator = 1.5*(torch.log(kappa - 2 * beta) + torch.log(kappa + 2 * beta)) + epsilon
    result = numerator - denominator

    return result

def calculate_beta_term(beta_a: torch.Tensor, gamma_a2: torch.Tensor, beta_b: torch.Tensor, gamma_b2: torch.Tensor, ExxT_a: torch.Tensor) -> torch.Tensor:

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

def kld_diagonal(kappa_a: torch.Tensor, beta_a: torch.Tensor, gamma_a1: torch.Tensor, gamma_a2: torch.Tensor, gamma_a3: torch.Tensor,
               kappa_b: torch.Tensor, beta_b: torch.Tensor, gamma_b1: torch.Tensor, gamma_b2: torch.Tensor, gamma_b3: torch.Tensor,
               Ex_a: torch.Tensor, ExxT_a: torch.Tensor, c_a: torch.Tensor, c_b: torch.Tensor, c_ka: torch.Tensor) -> torch.Tensor:

    log_term = c_b - c_a
    ex_a_term = calculate_kappa_term(kappa_a, gamma_a1, kappa_b, gamma_b1, Ex_a)
    beta_a_term_1_expanded, beta_b_term_1 = calculate_beta_term(beta_a, gamma_a2, beta_b, gamma_b2, ExxT_a)
    beta_a_term_2_expanded, beta_b_term_2 = calculate_beta_term(beta_a, gamma_a3, beta_b, gamma_b3, ExxT_a)

    kld = log_term + ex_a_term + beta_a_term_1_expanded.T - beta_b_term_1 - beta_a_term_2_expanded.T + beta_b_term_2
    check_nan_inf(kld, "kld")
    return kld

def clip_kent_parameters(kappa: torch.Tensor, beta: torch.Tensor, max_kappa: float = 1e13) -> torch.Tensor:
    """
    Clip kappa and beta to reasonable ranges to avoid numerical instability.
    """
    #pdb.set_trace()
    #kappa = torch.clamp(kappa, max=max_kappa)
    beta = torch.clamp(beta, max=(kappa / 2.2) - 1e-4)
    return kappa, beta

def validate_kent_parameters(kent_params: torch.Tensor) -> torch.Tensor:
    """
    Validate Kent parameters and enforce constraints.
    """
    eta, alpha, kappa, beta = kent_params[:, 0], kent_params[:, 1], kent_params[:, 2], kent_params[:, 3]
    kappa, beta = clip_kent_parameters(kappa, beta)
    #beta = enforce_kent_constraints(kappa, beta)
    return torch.stack([eta, alpha, kappa, beta], dim=1)

def get_kld(kent_pred: torch.Tensor, kent_target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the KLD between predicted and target Kent distributions.
    """
    # Validate and enforce constraints on parameters
    #kent_target = validate_kent_parameters(kent_target)
    #kent_pred = validate_kent_parameters(kent_pred)

    eta_a, alpha_a, kappa_a, beta_a = kent_target[:, 0], kent_target[:, 1], kent_target[:, 2], kent_target[:, 3]
    eta_b, alpha_b, kappa_b, beta_b = kent_pred[:, 0], kent_pred[:, 1], kent_pred[:, 2], kent_pred[:, 3]

    Q_matrix_a = radians_to_Q(eta_a, alpha_a)
    Q_matrix_b = radians_to_Q(eta_b, alpha_b)

    gamma_a1, gamma_a2, gamma_a3 = Q_matrix_a[:, :, 0], Q_matrix_a[:, :, 1], Q_matrix_a[:, :, 2]
    gamma_b1, gamma_b2, gamma_b3 = Q_matrix_b[:, :, 0], Q_matrix_b[:, :, 1], Q_matrix_b[:, :, 2]

    c_a = log_approximate_c(kappa_a, beta_a)
    c_b = log_approximate_c(kappa_b, beta_b)
    c_ka = log_del_kappa(kappa_a, beta_a)

    ExxT_a = expected_xxT(kappa_a, beta_a, Q_matrix_a, c_a, c_ka)
    Ex_a = expected_x(gamma_a1, c_a, c_ka)

    kld = kld_diagonal(kappa_a, beta_a, gamma_a1, gamma_a2, gamma_a3,
                     kappa_b, beta_b, gamma_b1, gamma_b2, gamma_b3,
                     Ex_a, ExxT_a, c_a, c_b, c_ka)

    return kld

def kld_kent_iou(y_pred, y_true, eps = 1e-6):
    # Ensure inputs are 2D[]
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(0)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)

    y_pred = y_pred.double()
    y_true = y_true.double()

    kld_pt = get_kld(y_pred, y_true)
    kld_tp = get_kld(y_true, y_pred)

    kld_pt = torch.clamp(kld_pt, min =0)
    kld_tp = torch.clamp(kld_tp, min =0)

    jsd = (kld_pt+kld_tp)/2

    const = 1.
    kld_loss = 1 / (const + jsd)

    return kld_loss

    
if __name__ == "__main__":
    pred = torch.tensor([[  2.2335,   1.7491, 331.7626, 146.6469]], dtype=torch.float32, requires_grad=True)#.half()
    target = torch.tensor([[5.4978, 2.3562, 1.2747, 0.4459]], dtype=torch.float32, requires_grad=True)#.half()
    loss = get_kld(target, pred)
    #loss.backward(retain_graph=True)
    print(loss)