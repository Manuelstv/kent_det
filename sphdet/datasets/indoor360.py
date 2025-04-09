from math import pi

from mmdet.datasets.builder import DATASETS

from .base_sph_dataset import SphBaseDataset, IgnoreClasses
from .evaluation import SphCOCOeval, VanillaCOCOeval


@DATASETS.register_module()
class INDOOR360(SphBaseDataset):
    
    CLASSES = ('toilet', 'board', 'mirror','bed', 'potted plant', 'book','clock',
               'phone', 'keyboard', 'tv', 'fan', 'backpack', 'light', 'refrigerator',
               'bathtub', 'wine glass', 'airconditioner', 'cabinet', 'sofa','bowl',
               'sink', 'computer', 'cup', 'bottle', 'washer', 'chair', 'picture',
               'window', 'door', 'heater', 'fireplace', 'mouse', 'oven', 'microwave',
               'person', 'vase', 'table')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103)]

    EVALUATOR = SphCOCOeval
    
    BOX_VERSION = 4

@DATASETS.register_module()
@IgnoreClasses(ignore=('mouse', 'bowl', 'wine glass'))
class INDOOR360v1(INDOOR360):
    pass

@DATASETS.register_module()
@IgnoreClasses(ignore=('mouse', 'bowl', 'wine glass', 'oven', 'refrigerator'))
class INDOOR360v2(INDOOR360):
    pass