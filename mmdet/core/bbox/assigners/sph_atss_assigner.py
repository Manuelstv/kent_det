# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import pdb

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

#mstv - LOSS BBOX absurda, provavelmente o problema esta aqui no assigner na goa d emudar a logica para esféricas
@BBOX_ASSIGNERS.register_module()
class ATSSAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    If ``alpha`` is not None, it means that the dynamic cost
    ATSSAssigner is adopted, which is currently only used in the DDOD.

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk,
                 alpha=None,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.alpha = alpha
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    """Assign a corresponding gt bbox or background to each bbox.

    Args:
        topk (int): number of bbox selected in each level.
        alpha (float): param of cost rate for each proposal only in DDOD.
            Default None.
        iou_calculator (dict): builder of IoU calculator.
            Default dict(type='BboxOverlaps2D').
        ignore_iof_thr (int): whether ignore max overlaps or not.
            Default -1 (1 or -1).
    """

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py
    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               cls_scores=None,
               bbox_preds=None):
        """Assign gt to bboxes for spherical bounding boxes.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level.
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
            cls_scores (list[Tensor]): Classification scores for all scale
                levels. Default None.
            bbox_preds (list[Tensor]): Box energies / deltas for all scale
                levels. Default None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]  # Ensure bboxes are in (center_x, center_y, fov_h, fov_v) format
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        message = 'Invalid alpha parameter because cls_scores or ' \
                  'bbox_preds are None. If you want to use the ' \
                  'cost-based ATSSAssigner, please set cls_scores, ' \
                  'bbox_preds and self.alpha at the same time.'

        if self.alpha is None:
            # ATSSAssigner
            overlaps = self.iou_calculator(bboxes, gt_bboxes)
            if cls_scores is not None or bbox_preds is not None:
                warnings.warn(message)
        else:
            # Dynamic cost ATSSAssigner in DDOD
            assert cls_scores is not None and bbox_preds is not None, message

            # Compute cls cost for bbox and GT
            cls_cost = torch.sigmoid(cls_scores[:, gt_labels])

            # Compute IoU between all bbox and gt
            overlaps = self.iou_calculator(bbox_preds, gt_bboxes)

            # Ensure element-wise multiplication
            assert cls_cost.shape == overlaps.shape

            # Overlaps is actually a cost matrix
            overlaps = cls_cost**(1 - self.alpha) * overlaps**self.alpha

        # Assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,), 0, dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # Compute angular distance between centers
        gt_centers = gt_bboxes[:, :2]  # (center_x, center_y)
        bbox_centers = bboxes[:, :2]   # (center_x, center_y)

        # Angular distance in spherical coordinates
        delta_x = torch.abs(bbox_centers[:, None, 0] - gt_centers[None, :, 0])
        delta_y = torch.abs(bbox_centers[:, None, 1] - gt_centers[None, :, 1])
        distances = torch.sqrt(delta_x**2 + delta_y**2)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the angular distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # On each pyramid level, for each gt, select k bbox whose center
            # are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)

            _, topk_idxs_per_level = distances_per_level.topk(selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # Get corresponding IoU for these candidates, and compute the
        # mean and std, set mean + std as the IoU threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # If an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)