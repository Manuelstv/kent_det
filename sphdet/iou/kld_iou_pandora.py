import torch
import pdb
import torch.nn as nn

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
        raise ValueError(f"Inf detected in {name}")
    
    
def check_nan_inf_2(tensor: torch.Tensor, kappa: torch.Tensor, beta: torch.Tensor,  name: str):
    """
    Check for NaN and Inf values in a tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to check.
        name (str): The name of the tensor for error reporting.
    """
    if torch.isnan(tensor).any():
        #pdb.set_trace()
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

def radians_to_Q(psi: torch.Tensor, alpha: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    Convert angles in radians to a Q matrix.
    
    Args:
        psi (torch.Tensor): The psi angles.
        alpha (torch.Tensor): The alpha angles.
        eta (torch.Tensor): The eta angles.
    
    Returns:
        torch.Tensor: The resulting Q matrix.
    """
    N = alpha.size(0)
    psi = psi.view(N, 1)
    alpha = alpha.view(N, 1)
    eta = eta.view(N, 1)
    
    gamma_1 = torch.cat([
        torch.cos(alpha),
        torch.sin(alpha) * torch.cos(eta),
        torch.sin(alpha) * torch.sin(eta)
    ], dim=1).unsqueeze(2)

    gamma_2 = torch.cat([
        -torch.cos(psi) * torch.sin(alpha),
        torch.cos(psi) * torch.cos(alpha) * torch.cos(eta) - torch.sin(psi) * torch.sin(eta),
        torch.cos(psi) * torch.cos(alpha) * torch.sin(eta) + torch.sin(psi) * torch.cos(eta)
    ], dim=1).unsqueeze(2)

    gamma_3 = torch.cat([
        torch.sin(psi) * torch.sin(alpha),
        -torch.sin(psi) * torch.cos(alpha) * torch.cos(eta) - torch.cos(psi) * torch.sin(eta),
        -torch.sin(psi) * torch.cos(alpha) * torch.sin(eta) + torch.cos(psi) * torch.cos(eta)
    ], dim=1).unsqueeze(2)

    gamma = torch.cat((gamma_1, gamma_2, gamma_3), dim=2)
    if torch.isnan(gamma).any():
        pdb.set_trace()
    
    return gamma

def log_approximate_c(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Approximate the c value based on kappa and beta.
    
    Args:
        kappa (torch.Tensor): The kappa values.
        beta (torch.Tensor): The beta values.
    
    Returns:
        torch.Tensor: The approximated c value.
    """
    epsilon = 1e-6
    term1 = kappa - 2 * beta
    term2 = kappa + 2 * beta
    product = term1 * term2 + epsilon
    log_result = torch.log(torch.tensor(2 * torch.pi, device=kappa.device)) + kappa - 0.5 * torch.log(product)
    
    check_nan_inf(log_result, "approximate_c")
    return log_result

def expected_x(gamma_a1: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    """
    Calculate the expected value of x based on gamma and c values.
    
    Args:
        gamma_a1 (torch.Tensor): The first gamma values.
        c (torch.Tensor): The c values.
        c_k (torch.Tensor): The kappa values.
    
    Returns:
        torch.Tensor: The expected value of x.
    """
    const = (torch.exp(c_k - c)).view(-1, 1)
    result = const * gamma_a1
    check_nan_inf(result, "expected_x")
    return result

def expected_xxT(kappa: torch.Tensor, beta: torch.Tensor, Q_matrix: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    """
    Calculate the expected value of xx^T based on kappa, beta, Q_matrix, c, and c_k values.
    
    Args:
        kappa (torch.Tensor): The kappa values.
        beta (torch.Tensor): The beta values.
        Q_matrix (torch.Tensor): The Q matrix.
        c (torch.Tensor): The c values.
        c_k (torch.Tensor): The kappa values.
    
    Returns:
        torch.Tensor: The expected value of xx^T.
    """
    c_kk = log_del_2_kappa(kappa, beta)
    c_beta = log_del_beta(kappa, beta)

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
    """
    Calculate the product of beta, gamma, ExxT, and gamma.
    
    Args:
        beta (torch.Tensor): The beta values.
        gamma (torch.Tensor): The gamma values.
        ExxT (torch.Tensor): The expected value of xx^T.
    
    Returns:
        torch.Tensor: The result of the calculation.
    """
    gamma_unsqueezed = gamma.unsqueeze(1)  # Shape: (N, 1, 3)
    intermediate_result = torch.bmm(gamma_unsqueezed, ExxT)  # Shape: (N, 1, 3)
    gamma_unsqueezed_2 = gamma.unsqueeze(2)  # Shape: (N, 3, 1)
    result = torch.bmm(intermediate_result, gamma_unsqueezed_2).squeeze()  # Shape: (N,)
    result = beta * result  # Shape: (N,)
    check_nan_inf(result, "beta_gamma_exxt_gamma")
    return result

def calculate_kappa_term(kappa_a, gamma_a1, kappa_b, gamma_b1, Ex_a):
    """
    Calculate the kappa term of the KLD matrix.
    
    Args:
        kappa_a (torch.Tensor): The kappa_a values.
        gamma_a1 (torch.Tensor): The gamma_a1 values.
        kappa_b (torch.Tensor): The kappa_b values.
        gamma_b1 (torch.Tensor): The gamma_b1 values.
        Ex_a (torch.Tensor): The expected value of x.
    
    Returns:
        torch.Tensor: The kappa term of the KLD matrix.
    """
    
    kappa_a_gamma_a1 = kappa_a.view(-1, 1) * gamma_a1
    kappa_b_gamma_b1 = kappa_b.view(-1, 1) * gamma_b1
    kappa_a_gamma_a1_expanded = kappa_a_gamma_a1.unsqueeze(1)
    kappa_b_gamma_b1_expanded = kappa_b_gamma_b1.unsqueeze(0)
    diff_kappa_term = kappa_a_gamma_a1_expanded - kappa_b_gamma_b1_expanded
    diff_kappa_term_diag = kappa_a_gamma_a1 - kappa_b_gamma_b1
    
    Ex_a_expanded = Ex_a.unsqueeze(1).expand(-1, diff_kappa_term.size(1), -1)
    Ex_a_expanded_diag = Ex_a
    
    result = torch.sum(diff_kappa_term * Ex_a_expanded, dim=-1).T
    result_diag = torch.sum(diff_kappa_term_diag*(Ex_a_expanded_diag), dim=1) # Shape: [5, ] @ [3, 5] -> [5, 1]        
    
    check_nan_inf(result, "calculate_kappa_term")
    return result_diag

def log_del_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-6  # Small value for numerical stability
    
    # Numerator: log(2 * pi * (k^2 - k - 4 * b^2)) + k
    argument_numerator = 2 * torch.pi * (kappa**2 - kappa - 4 * beta**2)
    numerator = torch.log(argument_numerator + epsilon) + kappa
    
    # Denominator: 1.5 * log((k - 2b) * (k + 2b))
    argument_denominator = (kappa - 2 * beta) * (kappa + 2 * beta)
    denominator = 1.5 * torch.log(argument_denominator + epsilon)
    
    # Final result
    result = numerator - denominator
    
    # Check for NaNs or Infs
    check_nan_inf(result, "del_kappa")
    return result

def log_del_2_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-6  # Small value for numerical stability
    
    # Numerator: log(2 * pi * (k^4 - 2 * k^3 + (2 - 8 * b^2) * k^2 + 8 * b^2 * k + 16 * b^4 + 4 * b^2)) + k
    argument_numerator = 2 * torch.pi * (kappa**4 - 2 * kappa**3 + (2 - 8 * beta**2) * kappa**2 
                                        + 8 * beta**2 * kappa + 16 * beta**4 + 4 * beta**2)
    numerator = torch.log(argument_numerator + epsilon) + kappa
    
    # Denominator: 2.5 * log((k - 2b) * (k + 2b))
    argument_denominator = (kappa - 2 * beta) * (kappa + 2 * beta)
    denominator = 2.5 * torch.log(argument_denominator + epsilon)
    
    # Final result
    result = numerator - denominator
    
    # Check for NaNs or Infs
    check_nan_inf(result, "del_2_kappa")
    return result


def log_del_beta(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Compute the logarithm of the expression:
        (8 * pi * exp(kappa) * beta) / ((kappa - 2 * beta) * (kappa + 2 * beta))^(3 / 2)
    
    Parameters:
    -----------
    kappa : torch.Tensor
        Concentration parameter κ.
    beta : torch.Tensor
        Ovalness parameter β, where 0 ≤ β < κ/2.

    Returns:
    --------
    torch.Tensor
        Logarithm of the given expression.
    """
    # Stability check: ensure kappa > 2*beta to avoid invalid logs
    if torch.any(kappa <= 2 * beta):
        raise ValueError("kappa must be greater than 2 * beta to ensure log validity.")

    # Logarithmic components
    log_const = torch.log(torch.tensor(8.0 * torch.pi))  # log(8π)
    log_beta = torch.log(beta)  # log(β)
    log_kappa_terms = 1.5 * (torch.log(kappa - 2 * beta) + torch.log(kappa + 2 * beta))  # 3/2 log(terms)
    
    # Final result: log(8π) + κ + log(β) - 3/2 [log(κ - 2β) + log(κ + 2β)]
    result = log_const + kappa + log_beta - log_kappa_terms

    return result

def calculate_beta_term(beta_a: torch.Tensor, gamma_a2: torch.Tensor, beta_b: torch.Tensor, gamma_b2: torch.Tensor, ExxT_a: torch.Tensor) -> torch.Tensor:
    """
    Calculate the beta term of the KLD matrix.
    
    Args:
        beta_a (torch.Tensor): The beta_a values.
        gamma_a2 (torch.Tensor): The gamma_a2 values.
        beta_b (torch.Tensor): The beta_b values.
        gamma_b2 (torch.Tensor): The gamma_b2 values.
        ExxT_a (torch.Tensor): The expected value of xx^T.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The beta term of the KLD matrix for beta_a and beta_b.
    """
    
    beta_a_gamma_a2 = beta_a.view(-1, 1) * gamma_a2
    beta_a_gamma_a2_expanded = beta_a_gamma_a2.unsqueeze(1)
    
    intermediate_result_a2 = torch.bmm(beta_a_gamma_a2_expanded, ExxT_a)
    
    beta_a_term_1 = torch.bmm(intermediate_result_a2, gamma_a2.unsqueeze(2)).squeeze(-1)
    beta_a_term_1_expanded = beta_a_term_1.expand(-1, beta_b.size(0))

    beta_b_gamma_b2 = beta_b.view(-1, 1) * gamma_b2
    beta_b_gamma_b2_expanded = beta_b_gamma_b2.unsqueeze(0)
    
    ExxT_a_expanded = ExxT_a.unsqueeze(1)
    product = beta_b_gamma_b2_expanded.unsqueeze(2) * ExxT_a_expanded
    product_diag = ExxT_a * beta_b_gamma_b2.unsqueeze(1)  # Unsqueeze to shape [5, 1, 3] for broadcasting

    
    result = product.sum(dim=-1)
    result_diag = product_diag.sum(dim=-1)
    
    gamma_b2_expanded = gamma_b2.unsqueeze(0)
    beta_b_term_1 = torch.sum(result_diag * gamma_b2_expanded, dim=-1)

    return beta_a_term_1, beta_b_term_1

def kld_matrix(kappa_a: torch.Tensor, beta_a: torch.Tensor, gamma_a1: torch.Tensor, gamma_a2: torch.Tensor, gamma_a3: torch.Tensor,
               kappa_b: torch.Tensor, beta_b: torch.Tensor, gamma_b1: torch.Tensor, gamma_b2: torch.Tensor, gamma_b3: torch.Tensor,
               Ex_a: torch.Tensor, ExxT_a: torch.Tensor, c_a: torch.Tensor, c_b: torch.Tensor, c_ka: torch.Tensor) -> torch.Tensor:
    """
    Calculate the KLD matrix.
    
    Args:
        kappa_a (torch.Tensor): The kappa_a values.
        beta_a (torch.Tensor): The beta_a values.
        gamma_a1 (torch.Tensor): The gamma_a1 values.
        gamma_a2 (torch.Tensor): The gamma_a2 values.
        gamma_a3 (torch.Tensor): The gamma_a3 values.
        kappa_b (torch.Tensor): The kappa_b values.
        beta_b (torch.Tensor): The beta_b values.
        gamma_b1 (torch.Tensor): The gamma_b1 values.
        gamma_b2 (torch.Tensor): The gamma_b2 values.
        gamma_b3 (torch.Tensor): The gamma_b3 values.
        Ex_a (torch.Tensor): The expected value of x.
        ExxT_a (torch.Tensor): The expected value of xx^T.
        c_a (torch.Tensor): The c_a values.
        c_b (torch.Tensor): The c_b values.
        c_ka (torch.Tensor): The kappa values.
    
    Returns:
        torch.Tensor: The KLD matrix.
    """
    log_term = c_b.unsqueeze(1) - c_a.unsqueeze(0)
    ex_a_term = calculate_kappa_term(kappa_a, gamma_a1, kappa_b, gamma_b1, Ex_a)
    beta_a_term_1_expanded, beta_b_term_1 = calculate_beta_term(beta_a, gamma_a2, beta_b, gamma_b2, ExxT_a)
    beta_a_term_2_expanded, beta_b_term_2 = calculate_beta_term(beta_a, gamma_a3, beta_b, gamma_b3, ExxT_a)
    
    pdb.set_trace()
    
    kld = log_term + ex_a_term + beta_a_term_1_expanded.T - beta_b_term_1.T - beta_a_term_2_expanded.T + beta_b_term_2.T
    check_nan_inf(kld, "kld_matrix")
    return kld

def kld_diagonal(kappa_a: torch.Tensor, beta_a: torch.Tensor, gamma_a1: torch.Tensor, gamma_a2: torch.Tensor, gamma_a3: torch.Tensor,
               kappa_b: torch.Tensor, beta_b: torch.Tensor, gamma_b1: torch.Tensor, gamma_b2: torch.Tensor, gamma_b3: torch.Tensor,
               Ex_a: torch.Tensor, ExxT_a: torch.Tensor, c_a: torch.Tensor, c_b: torch.Tensor, c_ka: torch.Tensor) -> torch.Tensor:
    """
    Calculate only the diagonal elements of the KLD matrix.

    Args:
        kappa_a, beta_a, gamma_a1, gamma_a2: Parameters for distribution A.
        kappa_b, beta_b, gamma_b1, gamma_b2: Parameters for distribution B.
        Ex_a: Expected value of x for distribution A.
        ExxT_a: Expected value of xx^T for distribution A.
        c_a, c_b, c_ka: Normalizing constants for the distributions.

    Returns:
        torch.Tensor: Diagonal KLD terms as a 1D tensor.
    """
    # Kappa term: Only diagonal elements
    log_term = c_b - c_a
    ex_a_term = calculate_kappa_term(kappa_a, gamma_a1, kappa_b, gamma_b1, Ex_a)
    beta_a_term_1_expanded, beta_b_term_1 = calculate_beta_term(beta_a, gamma_a2, beta_b, gamma_b2, ExxT_a)
    beta_a_term_2_expanded, beta_b_term_2 = calculate_beta_term(beta_a, gamma_a3, beta_b, gamma_b3, ExxT_a)
        
    kld = log_term + ex_a_term + beta_a_term_1_expanded.T - beta_b_term_1 - beta_a_term_2_expanded.T + beta_b_term_2
    check_nan_inf(kld, "kld_matrix")
    return kld
    


def get_kld(kent_pred: torch.Tensor, kent_target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the KLD between predicted and target Kent distributions.
    
    Args:
        kent_pred (torch.Tensor): The predicted Kent distribution parameters.
        kent_target (torch.Tensor): The target Kent distribution parameters.
    
    Returns:
        torch.Tensor: The KLD matrix.
    """

    eta_a, alpha_a, psi_a, kappa_a, beta_a = kent_target[:, 0], kent_target[:, 1], kent_target[:, 2], kent_target[:, 3], kent_target[:, 4]
    Q_matrix_a = radians_to_Q(psi_a, alpha_a, eta_a)
        
    eta_b, alpha_b, psi_b, kappa_b, beta_b = kent_pred[:, 0], kent_pred[:, 1], kent_pred[:, 2], kent_pred[:, 3], kent_pred[:, 4]
    Q_matrix_b = radians_to_Q(psi_b, alpha_b, eta_b)

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

if __name__ == "__main__":

    kent_target = torch.tensor([[  1.6673,   1.6248,   0.0000, 138.8834,  66.3045], [  3.8534,   1.1306,   0.0000, 310.9839, 152.0818], [1.7453e-09, 1.7453e-09, 0.0000e+00, 1.2279e+01, 2.2395e-02], [ 3.0483,  1.7917,  0.0000, 27.2595, 11.4925], [ 2.4396,  1.0456,  0.0000, 63.4280,  7.4796]])
    kent_pred = torch.tensor([[  3.8534,   1.1306,   0.0000, 30.9839, 10.0818], [  1.6673,   1.6248,   0.0000, 138.8834,  65.5045], [ 2.4396,  1.0456,  0.0000, 23.4280,  7.4796], [ 2.4396,  1.0456,  0.0000, 43.4280,  7.4796], [ 2.4396,  1.0456,  0.0000, 63.4280,  2.4796]])
    
    #kent_target = torch.tensor([[1.7453e-09, 1.7453e-09, 0.0000e+00, 1.2279e+01, 2.2395e-02]])
    #kent_pred = torch.tensor([[  1.6673,   1.6248,   0.0000, 138.8834,  66.3045]])
    
    kld = get_kld(kent_target, kent_pred)

    result = 1 - 1 / (2 + torch.sqrt(kld+1e-6))
    result_1 = 1 - 1 / (2 + kld)
    #pdb.set_trace()
    print(kld)
    print(result)
    print(result_1)