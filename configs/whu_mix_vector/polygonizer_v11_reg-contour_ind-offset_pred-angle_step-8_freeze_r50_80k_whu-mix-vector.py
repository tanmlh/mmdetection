_base_ = [
    '../_base_/datasets/whu_mix_vector.py', '../_base_/default_runtime.py',
]
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    # batch_augments=batch_augments
)

load_from='work_dirs/mask2former_r50_query-300_50e_whu-mix-vector/epoch_50.pth'
num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='PolyFormer',
    data_preprocessor=data_preprocessor,
    frozen_parameters=[
        'backbone',
        'panoptic_head.pixel_decoder',
        'panoptic_head.transformer_decoder',
        'panoptic_head.decoder_input_projs',
        'panoptic_head.query_embed',
        'panoptic_head.query_feat',
        'panoptic_head.level_embed',
        'panoptic_head.cls_embed',
        'panoptic_head.mask_embed',
    ],
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    panoptic_head=dict(
        type='PolygonizerHeadV11',
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=300,
        num_transformer_feat_level=3,
        poly_cfg=dict(
            num_inter_points=96,
            num_primitive_queries=96,
            apply_prim_pred=True,
            poly_align_type='align_by_roll',
            poly_inter_type='fixed_step',
            step_size=8,
            init_poly_type='polygonized',
            polygonized_scale=2.,
            max_offsets=10,
            use_coords_in_poly_feat=True,
            use_decoded_feat_in_poly_feat=True,
            use_point_feat_in_poly_feat=True,
            use_prim_offsets=False,
            ts_type='none',
            point_as_prim=True,
            pred_angle=True,
            prim_cls_thre=0.2,
            num_cls_channels=96,
            stride_size=64,
            use_ind_offset=True,
            poly_decode_type='dp',
            reg_targets_type='contour'
        ),
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        polyformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=3,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                # self_attn2_cfg=dict(  # MultiheadAttention
                #     embed_dims=256,
                #     num_heads=8,
                #     dropout=0.0,
                #     batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]
        ),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        loss_poly_cls=dict(
            type='BCELoss',
            use_sigmoid=False,
            loss_weight=1.0,
            reduction='mean',
            # class_weight=[1.0] * num_classes + [0.1]
        ),
        loss_poly_reg=dict(
            type='SmoothL1Loss',
            reduction='mean',
            loss_weight=0.1
        ),
        loss_poly_vec=dict(
            type='SmoothL1Loss',
            reduction='mean',
            loss_weight=10.
        ),
        loss_poly_ts=dict(
            type='MSELoss',
            reduction='mean',
            loss_weight=5.
        ),
        loss_poly_ang=dict(
            type='SmoothL1Loss',
            reduction='mean',
            loss_weight=1.
        )),
    panoptic_fusion_head=dict(
        type='PolyFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ]),
        # prim_assigner=dict(
        #     type='HungarianAssigner',
        #     match_costs=[
        #         dict(type='PointL1Cost', weight=0.1),
        #         dict(type='ClassificationCost', weight=5.)
        #     ]),
        sampler=dict(type='MaskPseudoSampler'),
        add_target_to_data_samples=True,
    ),
    test_cfg=dict(
        panoptic_on=False,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        # filter_low_score=True
        filter_low_score=False
    ),
    init_cfg=None)

val_evaluator = [
    # dict(
    #     type='CocoPanopticMetric',
    #     ann_file='../../Datasets/Dataset4EO/CrowdAI/0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/annotation.json',
    #     # ann_file=data_root + 'annotations/panoptic_val2017.json',
    #     seg_prefix=data_root + 'annotations/panoptic_val2017/',
    #     backend_args={{_base_.backend_args}}),
    dict(
        type='CocoMetric',
        # ann_file=data_root + 'annotations/instances_val2017.json',
        ann_file='../../Datasets/Dataset4EO/WHU-Mix/val/val.json',
        # ann_file='../../Datasets/Dataset4EO/CrowdAI/0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/annotation.json',
        metric=['segm'],
        mask_type='polygon',
        backend_args={{_base_.backend_args}})
]
test_evaluator = val_evaluator

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
max_iters = 80000
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[40000, 75000],
    gamma=0.1)

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
interval = 5000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    # val_interval=interval,
    val_interval=80000,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        max_keep_ckpts=2,
        interval=interval),
    # visualizer=dict(type='WandbVisualizer', wandb_cfg=wandb_cfg, name='wandb_vis')
    visualization=dict(type='TanmlhVisualizationHook', draw=True, interval=50)
)

vis_backends = [
    dict(
        type='WandbVisBackend', save_dir='./wandb/',
        init_kwargs=dict(
            project = 'mmdetection',
            entity = 'tum-tanmlh',
            name = 'test_polygonizer_v11_reg-contour_ind-offset_pred-angle_step-8_freeze_r50_80k_whu-mix-vector',
            resume = 'never',
            dir = './work_dirs/',
            allow_val_change=True
        ),
    )
]
# vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TanmlhVisualizer', vis_backends=vis_backends, name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

auto_scale_lr = dict(enable=True, base_batch_size=8)

train_dataloader = dict(
    dataset=dict(
        ann_file='val/val.json',
        data_prefix=dict(img='val/image'),
    )
)
