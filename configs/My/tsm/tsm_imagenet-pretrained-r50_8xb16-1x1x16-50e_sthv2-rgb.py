_base_ = ['../../_base_/models/tsm_r50.py', '../../_base_/default_runtime.py']

model = dict(backbone=dict(num_segments=8), cls_head=dict(num_segments=8, num_classes=2))

file_client_args = dict(io_backend='disk')

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '../<Your path to frames>/frames'
data_root_val = '../<Your path to frames>/frames'
ann_file_train = '../<Your path to frames>/frames/QAQ.txt'
ann_file_val = '../<Your path to frames>/frames/QAQ.txt'
ann_file_test = '../<Your path to frames>/frames/QAQ.txt'

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

test_pipeline = val_pipeline
train_dataloader = dict(
    batch_size=8,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='{:01}.jpg',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:01}.jpg',
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:01}.jpg',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.00002),
    clip_grad=dict(max_norm=20, norm_type=2))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=60, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=60,
        by_epoch=True,
        milestones=[30, 40, 45],
        gamma=0.1)
]

default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (6 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=8)
work_dir = './work_dirs/tsmr50-HW'
load_from = './configs/My/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb_20230317-be0fc26e.pth'