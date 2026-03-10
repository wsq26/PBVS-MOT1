
_base_ = [
    '../../_base_/models/tood.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]

img_scale = (800, 1440)
samples_per_gpu = 4

model = dict(
    type='DeepSORT',
    detector=dict(
        backbone=dict(
            _delete_=True,
            type='SwinTransformer',
            pretrain_img_size=384,
            embed_dims=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True,
            init_cfg=None),
        neck=dict(in_channels=[192, 384, 768, 1536]),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='mmdetection/work_dirs/tood_tmot_finetune_swin_l/best_coco_bbox_mAP_epoch_9.pth'
        )
    ),
    motion=dict(type='KalmanFilter', center_only=False),
    reid=dict(
        type='BaseReID',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=10778,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            loss_pairwise=dict(
                type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/mnt/disk1/wusiqing/MOT/0216tracking/reid_resnet50_vtmot/epoch_1.pth'
        )
    ),
    tracker=dict(
        type='ByteTrackerReID',
        obj_score_thrs=dict(high=0.5, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_tentatives=3,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            match_score_thr=2.0),
        match_reid_thr=0.2,
        num_frames_retain=30
    )
)
