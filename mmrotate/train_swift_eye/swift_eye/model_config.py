import numpy as np

tracking_head=dict(
        type='RotatedRetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator_tracking',
            scales=[6,8,10],
            ratios=[1.0, 0.5, 2.0],
            strides=[4]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range='le90',
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        sampler=dict(
                type='RandomSampler',
                num=64,
                pos_fraction=0.25,
                add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.5),
        max_per_img=2000)
        )

correlation_head=dict(
    type='CorrelationHead',
    in_channels=256,
    mid_channels=256,
    kernel_size=3,
)


backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
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
        )

neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5)

mask_head=dict(type='FCNMaskHead',
                         class_agnostic=True,
                         roi_feat_size=(72,88),
                         num_convs=4,
                         in_channels=256,
                         num_classes=1,
                         )
mask_extractor= dict(type ='RotatedSingleRoIExtractor',
                        roi_layer=dict(
                        type='RoIAlignRotated',
                        out_size=(144,176),
                        sample_num=2,
                        clockwise=True),
                        out_channels=1,
                        featmap_strides=[1])   

test=dict(
    pipeline=[
        
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(346, 346),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[81.49, 81.49, 81.49],
                        std=[38.74, 38.74, 38.74],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],

)



detection_head=dict(
        type='RotatedRetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            scales=[6,8,10],
            ratios=[1.0, 0.5, 2.0],
            strides=[4]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range='le90',
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        sampler=dict(
                type='RandomSampler',
                num=64,
                pos_fraction=0.25,
                add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.5),
        max_per_img=2000)
        )
