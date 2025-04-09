import torch
import numpy as np
from mmdet.core.anchor import AnchorGenerator
from mmdet.core.anchor.builder import PRIOR_GENERATORS
import random
import math
import pdb

def staggered_centers(featmap_size, stride):
    """
    Generate centers in a staggered grid pattern.

    Args:
        featmap_size (tuple[int]): Size of the feature map (height, width).
        stride (tuple[int]): Stride of the feature map (stride_w, stride_h).

    Returns:
        torch.Tensor: Center coordinates, shaped as [N, 2].
    """
    h, w = featmap_size
    stride_x, stride_y = stride
    centers_x, centers_y = [], []
    for i in range(w):
        for j in range(h):
            offset = (stride_x // 2) if (j % 2 == 1) else 0  # Offset every other row
            cx = i * stride_x + offset
            cy = j * stride_y
            centers_x.append(cx)
            centers_y.append(cy)
    return torch.tensor([centers_x, centers_y]).T  # Shape: [N, 2]

from sphdet.bbox.box_formator import Planar2SphBoxTransform

@PRIOR_GENERATORS.register_module()
class CustomAnchorGenerator(AnchorGenerator):
    """Custom anchor generator with user-defined anchor center distributions."""

    def __init__(self, box_formator = 'sph2pix', box_version = 4, custom_center_sampler=True, **kwargs):
        """
        Args:
            custom_center_sampler (callable): A function that generates custom anchor centers.
                                              It should take `featmap_size` and `stride` as inputs
                                              and return a tensor of shape [N, 2] (center coordinates).
            **kwargs: Arguments for the base `AnchorGenerator` class.
        """
        super().__init__(**kwargs)
        self.custom_center_sampler = custom_center_sampler
        box_version = 4
        self.box_formator = Planar2SphBoxTransform(box_formator, box_version)
        self.base_anchors = self.gen_base_anchors()

    def single_level_grid_priors(self, featmap_size, level_idx, dtype=torch.float32, device='cuda'):
        feat_h, feat_w = featmap_size
        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        stride_w, stride_h = self.strides[level_idx]
        image_width = feat_w * stride_w
        image_height = feat_h * stride_h
        dtype=torch.float32

        N = feat_h*feat_w
        #i = torch.arange(0, N, dtype=torch.float32, device=device)
        #golden_ratio = (1 + math.sqrt(5)) / 2
        
        #theta = 2 * torch.pi * i / golden_ratio
        #phi = torch.acos(1 - 2 * (i + 0.5) / N)
        
        #lon = (theta * 180 / torch.pi) % 360
        #lat = (phi * 180 / torch.pi)

        lon = torch.linspace(0, 360, feat_w)

        theta = torch.linspace(0, math.pi, feat_h)
        lat = torch.asin(torch.sin(theta)) * 180 / math.pi                   

        shift_xx, shift_yy = self._meshgrid(lon, lat)
        sph_centers = torch.stack([shift_xx, shift_yy], dim=-1).to(device).to(dtype)

        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)

        fovs = torch.cat([(base_anchors[:,2]*2/(1024)*360).unsqueeze(1), 
                        (base_anchors[:,3]*2/(1024)*360).unsqueeze(1)], dim=1)

        centers_expanded = sph_centers.repeat_interleave(9, dim=0)
        fovs_tiled = fovs.repeat(sph_centers.shape[0], 1)
        result = torch.cat((centers_expanded, fovs_tiled), dim=1)

        return result

    def staggered_centers(self, featmap_size, stride):
        """
        Generate centers in a staggered grid pattern.

        Args:
            featmap_size (tuple[int]): Size of the feature map (height, width).
            stride (tuple[int]): Stride of the feature map (stride_w, stride_h).

        Returns:
            torch.Tensor: Center coordinates, shaped as [N, 2].
        """
        h, w = featmap_size
        stride_x, stride_y = stride
        centers_x, centers_y = [], []
        for i in range(w):
            for j in range(h):
                offset = (stride_x // 2) if (j % 2 == 1) else 0  # Offset every other row
                cx = i * stride_x + offset
                cy = j * stride_y
                centers_x.append(cx)
                centers_y.append(cy)

        return torch.tensor([centers_x, centers_y]).T  # Shape: [N, 2]

    def random_centers(self, featmap_size, stride):
        """
        Generate centers in a random pattern.

        Args:
            featmap_size (tuple[int]): Size of the feature map (height, width).
            stride (tuple[int]): Stride of the feature map (stride_w, stride_h).

        Returns:
            torch.Tensor: Center coordinates, shaped as [N, 2].
        """
        h, w = featmap_size
        stride_x, stride_y = stride

        # Total number of centers to generate
        num_centers = h * w

        centers_x = []
        centers_y = []

        for _ in range(num_centers):
            # Generate random x and y coordinates within the bounds of the feature map
            cx = random.randint(0, w * stride_x - 1)
            cy = random.randint(0, h * stride_y - 1)
            centers_x.append(cx)
            centers_y.append(cy)

        # Combine x and y into a single tensor of shape [N, 2]
        centers = torch.stack([torch.tensor(centers_x), torch.tensor(centers_y)], dim=-1)
        return centers
    
    def fibonacci_erp_centers(self, featmap_size, stride, device='cuda'):
        """
        Generate anchor centers using a Fibonacci lattice on a sphere, projected to ERP.
        
        Args:
            featmap_size (tuple[int]): Feature map size (height, width).
            stride (tuple[int]): Stride of the feature map (stride_w, stride_h).
            device (str): Device for tensor operations.
        
        Returns:
            torch.Tensor: Center coordinates in ERP space, shaped [N, 2].
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = stride
        image_width = feat_w * stride_w
        image_height = feat_h * stride_h
        N = image_width * image_height
        dtype=torch.float32

        shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h

        # Generate Fibonacci lattice points on the sphere
        golden_ratio = (1 + math.sqrt(5)) / 2
        theta = 2 * torch.pi * shift_x / golden_ratio  # Longitude (0 to 2π)
        phi = torch.acos(1 - 2 * (shift_y + 0.5) / N)  # Polar angle
        
        lon = (theta * 180 / torch.pi) % 360  # Convert to degrees and wrap to [0,360]
        lat = phi * 180 / torch.pi  # Convert to degrees [0,180]


        #shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        #shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h

        shift_xx, shift_yy = self._meshgrid(lon, lat)
        shifts = torch.stack([shift_xx, shift_yy], dim=-1)

        pdb.set_trace()

        #centers = torch.stack([lon, lat], dim=-1)

        #sorted_indices = torch.argsort(centers[:, 0] * 1e6 + centers[:, 1])  # Combine keys for sorting
        #sorted_centers = centers[sorted_indices]

        return shifts


# Example: Custom Center Sampler (Radial Pattern)
def radial_centers(featmap_size, stride):
    """
    Generate centers in a radial pattern.

    Args:
        featmap_size (tuple[int]): Size of the feature map (height, width).
        stride (tuple[int]): Stride of the feature map (stride_w, stride_h).

    Returns:
        torch.Tensor: Center coordinates, shaped as [N, 2].
    """
    h, w = featmap_size
    centers = []
    for r in [0, 1, 2]:  # Radii
        for theta in np.linspace(0, 2 * np.pi, 8 * (r + 1), endpoint=False):
            cx = int(w // 2 + r * stride[0] * np.cos(theta))
            cy = int(h // 2 + r * stride[1] * np.sin(theta))
            centers.append((cx, cy))
    return torch.tensor(centers)


# Example Usage
if __name__ == "__main__":
    # Configuration
    anchor_generator = CustomAnchorGenerator(
        custom_center_sampler=staggered_centers,  # Use staggered centers
        strides=[16, 32, 64],
        ratios=[0.5, 1.0, 2.0],
        scales=[8, 16, 32],
        base_sizes=None,
        center_offset=0.0  # Set to 0 since centers are fully custom
    )

    # Generate anchors for a dummy feature map
    featmap_size = (4, 4)  # (height, width)
    custom_anchors = anchor_generator.single_level_grid_priors(featmap_size, level_idx=0)

    # Print and visualize anchor centers
    print("Generated Anchors:\n", custom_anchors)

    import matplotlib.pyplot as plt
    plt.scatter(custom_anchors[:, 0].cpu(), custom_anchors[:, 1].cpu(), s=10)
    plt.title('Custom Anchor Centers')
    plt.show()