# Copyright (c) OpenMMLab. All rights reserved.
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch

from mmtrack.models import TRACKERS
from .byte_tracker_reid import ByteTrackerReID


@TRACKERS.register_module()
class ByteTrackerReIDStable(ByteTrackerReID):
    """Stable ByteTrackerReID variant with controllable ReID fusion strength."""

    def __init__(self,
                 reid_alpha=0.6,
                 reid_dist_scale=2.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.reid_alpha = reid_alpha
        self.reid_dist_scale = reid_dist_scale

    def assign_ids(self,
                   ids,
                   det_bboxes,
                   det_labels,
                   det_embeds=None,
                   cv_gray_img=None,
                   weight_iou_with_det_scores=False,
                   match_iou_thr=0.5,
                   apply_thermal_iou_adjustment=False,
                   apply_weighted_matching=False):
        track_bboxes = np.zeros((0, 4))
        for id in ids:
            track_bboxes = np.concatenate((track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
        track_bboxes = self.kf_bbox_to_xyxy(track_bboxes)

        ious = self._bbox_overlaps(track_bboxes, det_bboxes[:, :4])
        if weight_iou_with_det_scores:
            ious *= det_bboxes[:, 4][None]
        d_iou = 1 - ious.cpu().numpy()

        dists = d_iou
        if self.with_reid and det_embeds is not None and len(ids) > 0:
            valid_ids, valid_inds = [], []
            for i, id in enumerate(ids):
                if len(self.tracks[id].get('embeds', [])) > 0:
                    valid_ids.append(id)
                    valid_inds.append(i)

            if len(valid_ids) > 0:
                track_embeds = self.get('embeds', valid_ids, self.reid.get('num_samples', None), behavior='mean')
                reid_d = torch.cdist(track_embeds, det_embeds).cpu().numpy() / max(self.reid_dist_scale, 1e-6)
                full_reid_d = np.ones((len(ids), len(det_bboxes)), dtype=np.float32)
                full_reid_d[valid_inds, :] = reid_d
                a = float(np.clip(self.reid_alpha, 0.0, 1.0))
                dists = (1.0 - a) * d_iou + a * full_reid_d

        if dists.size > 0:
            dists = np.where(np.isfinite(dists), dists, 1.0)
            row_ind, col_ind = linear_sum_assignment(dists)
            row = np.zeros(len(ids), dtype=np.int32) - 1
            col = np.zeros(len(det_bboxes), dtype=np.int32) - 1
            cost_limit = 1 - match_iou_thr
            for r, c in zip(row_ind, col_ind):
                if dists[r, c] < cost_limit:
                    row[r] = c
                    col[c] = r
        else:
            row = np.zeros(len(ids), dtype=np.int32) - 1
            col = np.zeros(len(det_bboxes), dtype=np.int32) - 1
        return row, col

    @staticmethod
    def kf_bbox_to_xyxy(track_bboxes):
        from mmtrack.core.bbox import bbox_cxcyah_to_xyxy
        return bbox_cxcyah_to_xyxy(track_bboxes)

    @staticmethod
    def _bbox_overlaps(a, b):
        from mmdet.core import bbox_overlaps
        return bbox_overlaps(a, b)
