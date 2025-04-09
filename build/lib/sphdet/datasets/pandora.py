from math import pi

from mmdet.datasets.builder import DATASETS

from .base_sph_dataset import SphBaseDataset
from .evaluation import RotatedSphCOCOeval


@DATASETS.register_module()
class PANDORA(SphBaseDataset):
    
    CLASSES = ('airconditioner', 'backpack', 'bathtub', 'bed', 'board', 'book', 
               'bottle', 'bowl', 'bucket', 'cabinet', 'chair', 'clock', 'clothes', 
               'computer', 'cup', 'cushion', 'door', 'extinguisher', 'fan', 'faucet', 
               'fireplace', 'heater', 'keyboard', 'light', 'microwave', 'mirror', 
               'mouse', 'outlet', 'oven', 'paper extraction', 'person', 'phone', 
               'picture', 'potted plant', 'refrigerator', 'shoes', 'shower head', 
               'sink', 'sofa', 'table', 'toilet', 'towel', 'tv', 'vase', 'washer', 
               'window', 'wine glass')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),(197, 226, 255), 
               (171, 134, 1), (109, 63, 54), (207, 138, 255), (151, 0, 95), 
               (9, 80, 61), (84, 105, 51), (74, 65, 105)]

    EVALUATOR = RotatedSphCOCOeval

    BOX_VERSION = 5