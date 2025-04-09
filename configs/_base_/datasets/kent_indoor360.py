dataset_type = 'INDOOR360'
data_root = 'datasets/360INDOOR/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='SphResize', img_scale=(256, 128), keep_ratio=True),
    dict(type='SphRandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256,128),
        flip=False,
        transforms=[
            dict(type='SphResize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            #dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


#revisar isso
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_small/instances_train2017.json',
        img_prefix=data_root + 'images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_small/instances_train2017.json',
        img_prefix=data_root + 'images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_small/instances_train2017.json',
        img_prefix=data_root + 'images',
        pipeline=test_pipeline))
evaluation = dict(interval=100, metric='bbox', save_best='bbox_mAP_50')