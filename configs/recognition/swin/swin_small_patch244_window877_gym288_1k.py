_base_ = [
    '../../_base_/models/swin/swin_small.py', '../../_base_/default_runtime.py'
]
model=dict(backbone=dict(patch_size=(2,4,4), drop_path_rate=0.1),
            cls_head=dict(
                        type='I3DHead',
                        in_channels=768,
                        num_classes=288,
                        spatial_type='avg',
                        dropout_ratio=0.5),
           test_cfg=dict(max_testing_views=4))

load_from = '/home/wangjiahao/models/swin_small_patch244_window877_kinetics400_1k.pth'

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/home/wangjiahao/datasets/FineGym/frames'
data_root_val = '/home/wangjiahao/datasets/FineGym/frames'
ann_file_train = '/home/wangjiahao/datasets/FineGym/parsed_labels/train_tpn.txt'
ann_file_val = '/home/wangjiahao/datasets/FineGym/parsed_labels/test_tpn.txt'
ann_file_test = '/home/wangjiahao/datasets/FineGym/parsed_labels/test_tpn.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
# train_pipeline = [
#     dict(type='DecordInit'),
#     dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='RandomResizedCrop'),
#     dict(type='Resize', scale=(224, 224), keep_ratio=False),
#     dict(type='Flip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs', 'label'])
# ]
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
# val_pipeline = [
#     dict(type='DecordInit'),
#     dict(
#         type='SampleFrames',
#         clip_len=32,
#         frame_interval=2,
#         num_clips=1,
#         test_mode=True),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='Flip', flip_ratio=0),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs'])
# ]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    # dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
# test_pipeline = [
#     dict(type='DecordInit'),
#     dict(
#         type='SampleFrames',
#         clip_len=32,
#         frame_interval=2,
#         num_clips=4,
#         test_mode=True),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 224)),
#     dict(type='ThreeCrop', crop_size=224),
#     dict(type='Flip', flip_ratio=0),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs'])
# ]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    # dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train_dataloader=dict(
        videos_per_gpu=8,
        workers_per_gpu=4
    ),
    val_dataloader=dict(
        videos_per_gpu=8,
        workers_per_gpu=4
    ),
    test_dataloader=dict(
        videos_per_gpu=8,
        workers_per_gpu=4
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        filename_tmpl='{:06}.jpg',),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        filename_tmpl='{:06}.jpg',),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        filename_tmpl='{:06}.jpg',))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 30

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = '/home/wangjiahao/experiments/finegym/swin_trans/exp1'
find_unused_parameters = False


# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=8,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
