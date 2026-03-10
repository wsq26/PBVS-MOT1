_base_ = [
    './bytetrack_reid_swin_l.py'
]

model = dict(
    tracker=dict(
        type='ByteTrackerReIDOATMTGFv2',
        obj_score_thrs=dict(high=0.5, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        reid_alpha=0.45,
        reid_dist_scale=2.0,
        oatm_alpha=0.22,
        tgf_alpha=0.12,
        center_ratio=0.6,
        gradient_bins=8,
        gaussian_sigma=1.0,
        adaptive_tgf=True,
        reid=dict(
            num_samples=15,
            img_scale=(256, 128),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            match_score_thr=2.0),
        match_reid_thr=0.2,
        num_frames_retain=30))
