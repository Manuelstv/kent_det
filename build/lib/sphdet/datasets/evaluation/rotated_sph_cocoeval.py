from pycocotools.cocoeval import COCOeval
from sphdet.iou import unbiased_iou, sph2pob_standard_iou
import numpy as np
import torch

class RotatedSphCOCOeval(COCOeval):
    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'][:5] for g in gt]
            d = [d['bbox'][:5] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        #iscrowd = [int(o['iscrowd']) for o in gt]
        #ious = maskUtils.iou(d,g,iscrowd)
        #ious = Sph().sphIoU(np.deg2rad(d), np.deg2rad(g))
        with torch.no_grad():
            d = torch.tensor(d).view((-1, 5))
            g = torch.tensor(g).view((-1, 5))
            ious = unbiased_iou(d, g)
        return ious.cpu().numpy()