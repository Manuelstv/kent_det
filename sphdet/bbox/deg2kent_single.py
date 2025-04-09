import torch
import sys
import numpy as np
import torch.nn as nn
import pdb
from sphdet.bbox.kent_formator import kent_me_matrix_torch, get_me_matrix_torch

def bfov_to_kent(annotations, epsilon=1e-6):
    if annotations.ndim == 1:
        annotations = annotations.unsqueeze(0)
    
    data_x = annotations[:, 0] / 360.0 
    data_y = annotations[:, 1] / 180.0

    #α ∈ [0, π] and η ∈ [0, 2π] be the co-latitude and longitude
    data_fov_h = annotations[:, 2]
    data_fov_v = annotations[:, 3]
        
    eta = 2*np.pi*data_x
    alpha = np.pi * data_y

    varphi = (torch.deg2rad(data_fov_h) ** 2) / 12 + epsilon  # Horizontal variance
    vartheta = (torch.deg2rad(data_fov_v) ** 2) / 12 + epsilon  # Vertical variance
    
    kappa = 0.5 * (1 / varphi + 1 / vartheta)    
    beta = torch.abs(0.25 * (1 / vartheta - 1 / varphi))

    print('hello')
    max_kappa = 50000
    kappa = torch.clamp(kappa, max=max_kappa)
    beta = torch.clamp(beta, max=(kappa / 2.0001) - 1e-4)
    
    kent_dist = torch.stack([eta, alpha,  kappa, beta], dim=1)
    
    return kent_dist

if __name__ == '__main__':
    annotations = torch.tensor([35.0, 0.0, 23.0, 20.0], dtype=torch.float32, requires_grad=True)
    
    annotations_2 = torch.tensor([[180.28125, 133.03125,  5.     ,  5.    ], 
                            [35.0, 0.0, 23.0, 50.0], 
                            [35.0, 10.0, 23.0, 20.0]], dtype=torch.float32, requires_grad=True)
    print(annotations_2)

    kent = deg2kent_single(annotations_2, 480, 960)
    print("Kent:", kent)
    
    if not kent.requires_grad:
        kent = kent.detach().requires_grad_(True)
    
    loss = kent.sum()
    print("Loss:", loss)
    
    if loss.requires_grad:
        loss.retain_grad()
        loss.backward()
    else:
        print("Loss does not require gradients")
    
    print("Loss Grad:", loss.grad)
    print("Annotations Grad:", annotations_2.grad)