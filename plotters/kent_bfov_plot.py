import os
import cv2
import numpy as np
import torch
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Mapping
from dataclasses import dataclass
import logging
from numpy.typing import NDArray
from scipy.special import jv as I_, gamma as G_

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KentParams:
    """Kent distribution parameters."""
    eta: float  # longitude
    alpha: float  # colatitude
    psi: float  # rotation
    kappa: float  # concentration
    beta: float  # ellipticity

@dataclass
class BBox:
    """Bounding box parameters."""
    u00: float  # center x
    v00: float  # center y
    a_long: float  # width
    a_lat: float  # height
    category_id: int

class Rotation:
    @staticmethod
    def Rx(alpha):
        return np.asarray([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])

    @staticmethod
    def Ry(beta):
        return np.asarray([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])

    @staticmethod
    def Rz(gamma):
        return np.asarray([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

def bfov_to_kent(annotations, epsilon=1e-6):
    if annotations.ndim == 1:
        annotations = annotations.unsqueeze(0)

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
        
    kent_dist = torch.stack([eta, alpha, eta*0, kappa, beta], dim=1)
        
    return kent_dist

def project_equirectangular_to_sphere(points: NDArray, width: int, height: int) -> NDArray:
    """Convert equirectangular coordinates to spherical coordinates."""
    theta = points[:, 0] * (2.0 * np.pi / width)  # Longitude [0, 2π]
    phi = points[:, 1] * (np.pi / height)         # Colatitude [0, π]
    
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])

def project_sphere_to_equirectangular(points: NDArray, width: int, height: int) -> NDArray:
    """Convert spherical coordinates to equirectangular coordinates."""
    phi = np.arccos(np.clip(points[:, 2], -1, 1))
    theta = np.arctan2(points[:, 1], points[:, 0])
    theta[theta < 0] += 2 * np.pi
    
    return np.vstack([
        theta * width / (2.0 * np.pi),
        phi * height / np.pi
    ])

def compute_kent_distribution(params: KentParams, points: NDArray) -> NDArray:
    """Compute Kent distribution values for given points."""
    def compute_log_normalization(kappa: float, beta: float, epsilon: float = 1e-6) -> float:
        term1 = kappa - 2 * beta
        term2 = kappa + 2 * beta
        return np.log(2 * np.pi) + kappa-0.5* np.log(term1 * term2 + epsilon)

    # Compute orthonormal basis
    gamma_1 = np.array([
        np.sin(params.alpha) * np.cos(params.eta),
        np.sin(params.alpha) * np.sin(params.eta),
        np.cos(params.alpha)
    ])
    
    temp = np.array([-np.sin(params.eta), np.cos(params.eta), 0])
    gamma_2 = np.cross(gamma_1, temp)
    gamma_2 /= np.linalg.norm(gamma_2)
    gamma_3 = np.cross(gamma_1, gamma_2)
    
    # Apply rotation
    cos_psi, sin_psi = np.cos(params.psi), np.sin(params.psi)
    gamma_2_new = cos_psi * gamma_2 + sin_psi * gamma_3
    gamma_3_new = -sin_psi * gamma_2 + cos_psi * gamma_3
    
    Q = np.array([gamma_1, gamma_2_new, gamma_3_new])
    
    # Compute distribution
    dot_products = points @ Q.T
    normalization = compute_log_normalization(params.kappa, params.beta)
    
    return np.exp(
        params.kappa * dot_products[:, 0] + 
        params.beta * (dot_products[:, 1] ** 2 - dot_products[:, 2] ** 2)
        - normalization)

def create_heatmap(
    distribution: NDArray,
    original_image: NDArray,
    gamma: float = 0.3,  # Reduced from 0.5 for more contrast
    alpha: float = 1.,  # Increased from 0.5 for stronger heatmap
    heatmap_intensity: float = 1.5  # New parameter to boost heatmap values
) -> NDArray:
    """Create a more visible heatmap visualization of the distribution."""
    # Clip and normalize with more aggressive percentile clipping
    p_low, p_high = np.percentile(distribution, [1, 99])  # Changed from [0, 97]
    dist_clip = np.clip(distribution, p_low, p_high)
    
    # Apply intensity boost before normalization
    dist_boosted = dist_clip * heatmap_intensity
    
    # Normalize after boosting
    dist_norm = (dist_boosted - dist_boosted.min()) / (
        dist_boosted.max() - dist_boosted.min() + 1e-6
    )
    
    # Apply gamma correction (lower gamma = more contrast)
    dist_gamma = np.power(dist_norm, gamma)
    
    # Use a more vibrant colormap (JET instead of HOT)
    heatmap_raw = cv2.applyColorMap(
        (dist_gamma * 255).astype(np.uint8),
        cv2.COLORMAP_JET  # Changed from COLORMAP_HOT
    )
    
    # More aggressive blending
    heatmap = np.clip(heatmap_raw.astype(np.float32) / 255.0 * 1.5, 0, 1)  # Increased from 1.2
    original_float = original_image.astype(np.float32) / 255.0
    
    # Use non-linear blending for better visibility
    blend_alpha = np.power(np.mean(heatmap, axis=2), 0.7) * alpha  # Power scaling
    blend_alpha = np.stack([blend_alpha] * 3, axis=2)
    
    blended = original_float * (1 - blend_alpha) + heatmap * blend_alpha
    return np.clip(blended * 255, 0, 255).astype(np.uint8)

def load_coco_annotations(image_name: str, annotation_path: Path) -> List[BBox]:
    """Load COCO format annotations for specified image."""
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to load annotations: {e}")
        raise

    # Find image ID
    image_id = next(
        (img['id'] for img in data['images'] 
        if img['file_name'] == image_name), None
    )
    
    if image_id is None:
        raise ValueError(f"Image {image_name} not found in annotations")
    
    # Extract relevant boxes
    return [
        BBox(
            u00=ann['bbox'][0],
            v00=ann['bbox'][1],
            a_long=ann['bbox'][2],
            a_lat=ann['bbox'][3],
            category_id=ann['category_id']
        )
        for ann in data['annotations']
        if ann['image_id'] == image_id
    ]

def load_category_mapping(annotation_path: Path) -> Dict[str, int]:
    """Create a mapping of category names to category IDs."""
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    return {cat['name'].lower(): cat['id'] for cat in data['categories']}

def project_point(point, R, w, h):
    """Helper function to calculate the projected coordinates of a point in 3D space."""
    point_rotated = np.dot(R, point / np.linalg.norm(point))
    phi = np.arctan2(point_rotated[0], point_rotated[2])
    theta = np.arcsin(point_rotated[1])
    u = (phi / (2 * np.pi) + 0.5) * w
    v = h - (-theta / np.pi + 0.5) * h
    return u, v

def plot_bfov(
    image: np.ndarray, 
    v00: float, 
    u00: float, 
    fov_lat: float, 
    fov_long: float,
    color: Tuple[int, int, int], 
    h: int, 
    w: int
) -> np.ndarray:
    """Plots BFOV by densely sampling points and computing their convex hull."""
    # Shift image to center the BFOV (handles wrapping)
    t = int(w // 2 - u00)
    u00 += t
    image = np.roll(image, t, axis=1)

    # Calculate angles (phi00 = longitude, theta00 = latitude)
    phi00 = (u00 - w / 2) * (2 * np.pi / w)
    theta00 = -(v00 - h / 2) * (np.pi / h)

    # Rotation matrix (first rotate around Y, then X)
    R = np.dot(Rotation.Ry(phi00), Rotation.Rx(theta00))

    # BFOV dimensions in tangent space
    tan_half_lat = np.tan(fov_lat / 2)
    tan_half_long = np.tan(fov_long / 2)

    # Generate a dense grid of points inside the BFOV rectangle
    num_samples = 500  # Increase for higher precision
    x = np.linspace(-tan_half_long, tan_half_long, num_samples)
    y = np.linspace(-tan_half_lat, tan_half_lat, num_samples)
    xx, yy = np.meshgrid(x, y)
    points_tangent = np.vstack([xx.ravel(), yy.ravel(), np.ones_like(xx.ravel())]).T

    # Project all points to equirectangular coordinates
    points_projected = np.array([project_point(pt, R, w, h) for pt in points_tangent])

    # Compute convex hull to get the outline
    hull = cv2.convexHull(points_projected.astype(np.int32))

    # Draw the outline
    color_bgr = (color[2], color[1], color[0]) if len(color) == 3 else color
    cv2.polylines(image, [hull], isClosed=True, color=color_bgr, thickness=2)

    # Shift image back to original position
    image = np.roll(image, w - t, axis=1)

    return image

def process_image(
    image_path: Path,
    annotation_path: Path,
    target_category: Union[int, str] = 13,
    output_dir: Path = Path("output"),
    plot_both: bool = True
) -> None:
    """Process a 360-degree image and generate combined visualizations."""
    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load category mapping and resolve category ID
    '''if isinstance(target_category, str):
        category_mapping = load_category_mapping(annotation_path)
        target_category = category_mapping.get(target_category.lower())
        if target_category is None:
            available_categories = ', '.join(sorted(category_mapping.keys()))
            raise ValueError(
                f"Category '{target_category}' not found. Available categories: {available_categories}"
            )
    '''
    # Load data
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width = image.shape[:2]
    logger.info(f"Processing image {image_path.name} ({width}x{height})")
    
    boxes = load_coco_annotations(image_path.name, annotation_path)
    logger.info(f"Found {len(boxes)} boxes")

    # Create coordinate grid
    sphere_points = create_sphere_points(height, width)
    
    # Define colors for visualization
    color_map = {
        4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 0, 0), 12: (255, 255, 0),
        17: (0, 255, 255), 25: (255, 0, 255), 26: (128, 128, 0),
        27: (0, 128, 128), 30: (128, 0, 128), 34: (128, 128, 128),
        35: (64, 0, 0), 36: (0, 64, 0)
    }
    
    # Create copies for different visualizations
    heatmap_image = image.copy()
    bfov_image = image.copy()
    combined_image = image.copy()
    
    # Process boxes
    combined_distribution = np.zeros((height, width), dtype=np.float32)
    
    for i, box in enumerate(boxes):
        #if box.category_id != target_category:
        #    continue
        
        # Get color for this category
        color = color_map.get(box.category_id, (255, 255, 255))
        color_bgr = (color[2], color[1], color[0])  # Convert to BGR for OpenCV
        
        # Convert box to Kent parameters
        bbox_tensor = torch.tensor(
            [box.u00, box.v00, box.a_long, box.a_lat],
            dtype=torch.float32
        )
        kent_params = bfov_to_kent(bbox_tensor).detach().numpy()[0]
        params = KentParams(*kent_params)
        logger.info(f"Box {i} Kent parameters: {params}")
        
        # Compute Kent distribution
        kent_values = compute_kent_distribution(params, sphere_points)
        kent_image = kent_values.reshape((height, width))
        combined_distribution += kent_image
        
        # Create heatmap for this box
        box_heatmap = create_heatmap(kent_image, image)
        cv2.imwrite(
            str(output_dir / f"kent_box_{i}_class_{box.category_id}.png"),
            box_heatmap
        )
        
        # Plot BFOV
        u00_px = box.u00 / 360 * width
        v00_px = box.v00 / 180 * height
        a_lat_rad = np.radians(box.a_lat)
        a_long_rad = np.radians(box.a_long)
        
        bfov_image = plot_bfov(
            bfov_image, 
            v00_px, 
            u00_px, 
            a_lat_rad, 
            a_long_rad, 
            color_bgr, 
            height, 
            width
        )
    
    # Create combined heatmap
    heatmap_image = create_heatmap(combined_distribution, heatmap_image)
    cv2.imwrite(str(output_dir / "kent_combined.png"), heatmap_image)
    
    # Save BFOV visualization
    cv2.imwrite(str(output_dir / "bfov_visualization.png"), bfov_image)
    
    # Create combined visualization (heatmap + BFOV)
    if plot_both:
        # Blend heatmap with original image (50% opacity)
        heatmap_blend = cv2.addWeighted(heatmap_image, 0.5, image, 0.5, 0)
        
        # Add BFOV outlines to the blended image
        combined_image = cv2.addWeighted(heatmap_blend, 0.7, bfov_image, 0.3, 0)
        
        # Save combined visualization
        cv2.imwrite(str(output_dir / "combined_visualization.png"), combined_image)
    
    logger.info("Processing complete!")

def create_sphere_points(height: int, width: int) -> NDArray:
    """Create spherical coordinate points for the entire image."""
    v, u = np.mgrid[0:height:1, 0:width:1]
    points = np.vstack((u.reshape(-1), v.reshape(-1))).T
    return project_equirectangular_to_sphere(points, width, height)

if __name__ == "__main__":
    # Configuration
    IMAGE_PATH = Path("datasets/360INDOOR/images/7l0yq.jpg")
    ANNOTATION_PATH = Path("datasets/360INDOOR/annotations/instances_train2017.json")
    OUTPUT_DIR = Path("output")

    # Available Categories:
    # ========================================
    # 1. toilet              2. board               3. mirror             
    # 4. bed                 5. potted plant        6. book              
    # 7. clock               8. phone               9. keyboard          
    # 10. tv                 11. fan                12. backpack          
    # 13. light              14. refrigerator       15. bathtub           
    # 16. wine glass         17. airconditioner     18. cabinet           
    # 19. sofa              20. bowl               21. sink              
    # 22. computer          23. cup                24. bottle            
    # 25. washer            26. chair              27. picture           
    # 28. window            29. door               30. heater            
    # 31. fireplace         32. mouse              33. oven              
    # 34. microwave         35. person             36. vase              
    # 37. table       
    
    try:
        process_image(
            IMAGE_PATH, 
            ANNOTATION_PATH, 
            target_category="person",  # Can use string name instead of ID
            output_dir=OUTPUT_DIR,
            plot_both=True  # Set to False to only generate heatmap or BFOV separately
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise