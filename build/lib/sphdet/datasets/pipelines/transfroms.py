from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip, Resize


@PIPELINES.register_module()
class SphResize(Resize):
    """Resize images & rotated bbox Inherit Resize pipeline class to handle
    rotated bboxes.

    """

    def _resize_bboxes(self, results):
        """Spherical BBoxes is ratio-based definition, which unchange 
        when image_size changes. """
        pass


@PIPELINES.register_module()
class SphRandomFlip(RandomFlip):
    """Flip the image & bbox & mask.

    """

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        w, h = 360, 180
        if direction == 'horizontal':
            flipped[..., 0::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            flipped[..., 1::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            flipped[..., 0::4] = w - bboxes[..., 0::4]
            flipped[..., 1::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped


@PIPELINES.register_module()
class SphRotatedRandomFlip(RandomFlip):
    """Flip the image & bbox & mask.

    """

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 5 == 0
        flipped = bboxes.copy()
        w, h = 360, 180
        if direction == 'horizontal':
            flipped[..., 0::5] = w - bboxes[..., 0::5]
            flipped[..., 4::5] = -flipped[..., 4::5]
        elif direction == 'vertical':
            flipped[..., 1::5] = h - bboxes[..., 1::5]
            flipped[..., 4::5] = -flipped[..., 4::5]
        elif direction == 'diagonal':
            flipped[..., 0::5] = w - bboxes[..., 0::5]
            flipped[..., 1::5] = h - bboxes[..., 1::5]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped