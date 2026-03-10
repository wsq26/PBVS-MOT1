# Copyright (c) OpenMMLab. All rights reserved.
"""
OATM + TGF v2: Enhanced Thermal Gradient Flow for Infrared MOT.

Improvements over v1:
1. Robust gradient computation with Gaussian smoothing
2. Optimized feature weighting
3. Scale-invariant gradient descriptors
4. Adaptive TGF weighting based on gradient quality
"""
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import cv2

from mmtrack.models import TRACKERS
from .byte_tracker_reid_stable import ByteTrackerReIDStable


@TRACKERS.register_module()
class ByteTrackerReIDOATMTGFv2(ByteTrackerReIDStable):
    """Enhanced ByteTracker with OATM and improved Thermal Gradient Flow.

    Args:
        oatm_alpha (float): Weight for OATM thermal matching.
        tgf_alpha (float): Weight for thermal gradient flow matching.
        center_ratio (float): Ratio of center region for OATM.
        gradient_bins (int): Number of gradient orientation bins.
        gaussian_sigma (float): Sigma for Gaussian smoothing before gradient computation.
        adaptive_tgf (bool): Enable adaptive TGF weighting based on gradient quality.
        **kwargs: Other arguments for ByteTrackerReID.
    """

    def __init__(self,
                 oatm_alpha=0.22,
                 tgf_alpha=0.12,
                 center_ratio=0.6,
                 gradient_bins=8,
                 gaussian_sigma=1.0,
                 adaptive_tgf=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.oatm_alpha = oatm_alpha
        self.tgf_alpha = tgf_alpha
        self.center_ratio = center_ratio
        self.gradient_bins = gradient_bins
        self.gaussian_sigma = gaussian_sigma
        self.adaptive_tgf = adaptive_tgf

    def extract_center_region(self, bbox_xyxy):
        """Extract center region coordinates (from OATM)."""
        x1, y1, x2, y2 = bbox_xyxy
        w = x2 - x1
        h = y2 - y1
        margin_w = w * (1 - self.center_ratio) / 2
        margin_h = h * (1 - self.center_ratio) / 2
        cx1 = x1 + margin_w
        cy1 = y1 + margin_h
        cx2 = x2 - margin_w
        cy2 = y2 - margin_h
        return np.array([cx1, cy1, cx2, cy2])

    def compute_thermal_mean(self, gray_img, bbox_xyxy):
        """Compute mean thermal intensity."""
        if gray_img is None:
            return 128.0

        h, w = gray_img.shape
        x1, y1, x2, y2 = bbox_xyxy
        x1 = int(max(0, min(w-1, x1)))
        y1 = int(max(0, min(h-1, y1)))
        x2 = int(max(x1+1, min(w, x2)))
        y2 = int(max(y1+1, min(h, y2)))

        if x2 <= x1 or y2 <= y1:
            return 128.0

        roi = gray_img[y1:y2, x1:x2]
        return float(roi.mean())

    def compute_oatm_similarity(self, track_bbox, det_bbox, cv_gray_img):
        """Compute OATM thermal similarity (from original OATM)."""
        if cv_gray_img is None:
            return 0.5

        # Full bbox
        track_full = self.compute_thermal_mean(cv_gray_img, track_bbox)
        det_full = self.compute_thermal_mean(cv_gray_img, det_bbox)

        # Center region
        track_center_bbox = self.extract_center_region(track_bbox)
        det_center_bbox = self.extract_center_region(det_bbox)
        track_center = self.compute_thermal_mean(cv_gray_img, track_center_bbox)
        det_center = self.compute_thermal_mean(cv_gray_img, det_center_bbox)

        # Similarities
        full_sim = 1.0 - min(abs(track_full - det_full) / 128.0, 1.0)
        center_sim = 1.0 - min(abs(track_center - det_center) / 128.0, 1.0)

        # Weighted: prioritize center
        similarity = 0.3 * full_sim + 0.7 * center_sim
        return float(similarity)

    def compute_thermal_gradient_descriptor(self, gray_img, bbox_xyxy):
        """IMPROVED: Compute robust Thermal Gradient Flow descriptor.

        Improvements:
        1. Gaussian smoothing for noise reduction
        2. Normalized gradient magnitudes for scale invariance
        3. Enhanced radial pattern with more bins
        4. Gradient quality score for adaptive weighting
        """
        if gray_img is None:
            return np.ones(self.gradient_bins + 8) / (self.gradient_bins + 8), 0.0

        h, w = gray_img.shape
        x1, y1, x2, y2 = bbox_xyxy
        x1 = int(max(0, min(w-1, x1)))
        y1 = int(max(0, min(h-1, y1)))
        x2 = int(max(x1+1, min(w, x2)))
        y2 = int(max(y1+1, min(h, y2)))

        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 5 or (y2 - y1) < 5:
            return np.ones(self.gradient_bins + 8) / (self.gradient_bins + 8), 0.0

        roi = gray_img[y1:y2, x1:x2].astype(np.float32)

        # IMPROVEMENT 1: Gaussian smoothing for robust gradient computation
        roi_smooth = cv2.GaussianBlur(roi, (5, 5), self.gaussian_sigma)

        # Compute thermal gradients
        grad_x = cv2.Sobel(roi_smooth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi_smooth, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude and orientation
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)

        # IMPROVEMENT 2: Compute gradient quality score
        # High quality: strong, consistent gradients
        mean_magnitude = magnitude.mean()
        std_magnitude = magnitude.std()
        gradient_quality = np.clip(mean_magnitude / 50.0, 0, 1) * np.clip(1.0 - std_magnitude / 30.0, 0, 1)

        # IMPROVEMENT 3: Histogram of Oriented Thermal Gradients (HOTG)
        # Weight by magnitude for more discriminative features
        orientation_deg = np.degrees(orientation) % 360
        bins = np.linspace(0, 360, self.gradient_bins + 1)
        hotg = np.zeros(self.gradient_bins, dtype=np.float32)

        for i in range(self.gradient_bins):
            mask = (orientation_deg >= bins[i]) & (orientation_deg < bins[i+1])
            hotg[i] = magnitude[mask].sum()

        # Normalize with L2 norm for scale invariance
        hotg_norm = np.linalg.norm(hotg)
        if hotg_norm > 1e-6:
            hotg = hotg / hotg_norm
        else:
            hotg = np.ones(self.gradient_bins) / self.gradient_bins

        # IMPROVEMENT 4: Enhanced Radial Thermal Gradient Pattern (8 sectors)
        # More fine-grained spatial structure
        cy, cx = roi.shape[0] / 2, roi.shape[1] / 2
        h_roi, w_roi = roi.shape

        # Create 8-sector radial pattern
        y_coords, x_coords = np.ogrid[:h_roi, :w_roi]
        y_centered = y_coords - cy
        x_centered = x_coords - cx

        # Compute angle for each pixel
        angles = np.arctan2(y_centered, x_centered)
        angles_deg = np.degrees(angles) % 360

        # 8 sectors (45 degrees each)
        sector_bins = np.linspace(0, 360, 9)
        rtgp = np.zeros(8, dtype=np.float32)

        for i in range(8):
            mask = (angles_deg >= sector_bins[i]) & (angles_deg < sector_bins[i+1])
            rtgp[i] = magnitude[mask].mean() if mask.sum() > 0 else 0.0

        # Normalize
        rtgp_norm = np.linalg.norm(rtgp)
        if rtgp_norm > 1e-6:
            rtgp = rtgp / rtgp_norm
        else:
            rtgp = np.ones(8) / 8

        # Combine HOTG and RTGP
        descriptor = np.concatenate([hotg, rtgp])

        return descriptor, gradient_quality

    def compute_tgf_similarity(self, track_id, det_bbox, cv_gray_img):
        """IMPROVED: Compute Thermal Gradient Flow similarity with quality weighting."""
        if cv_gray_img is None or self.tgf_alpha <= 0:
            return 0.5, 0.5

        # Get detection's thermal gradient descriptor
        det_bbox_xyxy = det_bbox[:4]
        det_descriptor, det_quality = self.compute_thermal_gradient_descriptor(cv_gray_img, det_bbox_xyxy)

        # Get track's thermal gradient descriptor history
        if 'tgf_descriptors' not in self.tracks[track_id]:
            return 0.5, 0.5

        tgf_history = self.tracks[track_id]['tgf_descriptors']
        if len(tgf_history) == 0:
            return 0.5, 0.5

        # Use recent descriptors with quality weighting
        recent_data = tgf_history[-5:]

        # Compute weighted similarity
        similarities = []
        qualities = []
        for hist_descriptor, hist_quality in recent_data:
            # Cosine similarity (better for normalized vectors)
            dot_product = np.dot(det_descriptor, hist_descriptor)
            cosine_sim = np.clip(dot_product, 0, 1)

            # Weight by quality
            weight = (det_quality + hist_quality) / 2
            similarities.append(cosine_sim * weight)
            qualities.append(weight)

        # Weighted average similarity
        if sum(qualities) > 0:
            avg_similarity = sum(similarities) / sum(qualities)
        else:
            avg_similarity = 0.5

        # Average quality for adaptive weighting
        avg_quality = np.mean([det_quality] + [q for _, q in recent_data])

        return float(np.clip(avg_similarity, 0.0, 1.0)), float(avg_quality)

    def assign_ids(self,
                   ids,
                   det_bboxes,
                   det_labels,
                   det_embeds=None,
                   cv_gray_img=None,
                   weight_iou_with_det_scores=False,
                   match_iou_thr=0.5,
                   **kwargs):
        """Assign IDs with OATM + improved TGF."""
        track_bboxes = np.zeros((0, 4))
        for id in ids:
            track_bboxes = np.concatenate((track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)

        from mmtrack.core.bbox import bbox_cxcyah_to_xyxy
        track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)

        # Standard IoU
        from mmdet.core import bbox_overlaps
        ious = bbox_overlaps(track_bboxes, det_bboxes[:, :4]).cpu().numpy()

        if weight_iou_with_det_scores:
            ious *= det_bboxes[:, 4].cpu().numpy()[None]

        d_iou = 1 - ious

        # Compute OATM costs
        oatm_costs = np.zeros((len(ids), len(det_bboxes)), dtype=np.float32)
        # Compute TGF costs with adaptive weighting
        tgf_costs = np.zeros((len(ids), len(det_bboxes)), dtype=np.float32)
        tgf_weights = np.ones((len(ids), len(det_bboxes)), dtype=np.float32)

        if cv_gray_img is not None:
            track_bboxes_np = track_bboxes.cpu().numpy()
            det_bboxes_np = det_bboxes[:, :4].cpu().numpy()

            for i, track_id in enumerate(ids):
                for j in range(len(det_bboxes)):
                    # OATM: multi-scale thermal similarity
                    if self.oatm_alpha > 0:
                        oatm_sim = self.compute_oatm_similarity(
                            track_bboxes_np[i],
                            det_bboxes_np[j],
                            cv_gray_img
                        )
                        oatm_costs[i, j] = 1.0 - oatm_sim

                    # TGF: thermal gradient flow similarity with quality
                    if self.tgf_alpha > 0:
                        tgf_sim, tgf_quality = self.compute_tgf_similarity(
                            track_id,
                            det_bboxes_np[j],
                            cv_gray_img
                        )
                        tgf_costs[i, j] = 1.0 - tgf_sim

                        # Adaptive weighting: use TGF more when quality is high
                        if self.adaptive_tgf:
                            tgf_weights[i, j] = 0.5 + 0.5 * tgf_quality  # Range [0.5, 1.0]

        # Combine costs with adaptive TGF weighting
        if self.adaptive_tgf:
            adaptive_tgf_costs = tgf_costs * tgf_weights
            dists = d_iou + self.oatm_alpha * oatm_costs + self.tgf_alpha * adaptive_tgf_costs
        else:
            dists = d_iou + self.oatm_alpha * oatm_costs + self.tgf_alpha * tgf_costs

        # Add ReID distance
        if self.with_reid and det_embeds is not None and len(ids) > 0:
            valid_ids = []
            valid_inds = []
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
                dists = (1.0 - a) * dists + a * full_reid_d

        # Hungarian assignment
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

    def update(self, ids, bboxes, embeds, cv_gray_img=None, **kwargs):
        """Override update to maintain TGF descriptor history with quality."""
        super().update(ids, bboxes, embeds, **kwargs)

        if cv_gray_img is not None:
            for i, id in enumerate(ids):
                bbox_xyxy = bboxes[i, :4].cpu().numpy() if torch.is_tensor(bboxes) else bboxes[i, :4]

                # Compute and store thermal gradient descriptor with quality
                tgf_descriptor, gradient_quality = self.compute_thermal_gradient_descriptor(cv_gray_img, bbox_xyxy)

                if 'tgf_descriptors' not in self.tracks[id]:
                    self.tracks[id]['tgf_descriptors'] = []

                self.tracks[id]['tgf_descriptors'].append((tgf_descriptor, gradient_quality))

                # Keep only recent history (last 10 frames)
                if len(self.tracks[id]['tgf_descriptors']) > 10:
                    self.tracks[id]['tgf_descriptors'].pop(0)
