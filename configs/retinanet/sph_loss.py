_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_120e.py',
    '../_base_/datasets/indoor360.py',
    '../_base_/models/sph_retinanet_r50_fpn.py',
]

# log
checkpoint_config = dict(interval=50)
evaluation = dict(interval=5)

optimizer_config = dict(_delete_ = True, grad_clip=dict(max_norm=15, norm_type=2))

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')])

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1)


model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        anchor_generator=dict(
            box_formator='sph2pix'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
            loss_bbox=dict(
            _delete_=True,
            type='Sph2PobIoULoss',
            mode='ciou',
            loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            iou_calculator=dict(
                backend='sph2pob_efficient_iou')),),
    test_cfg=dict(
        nms=dict(iou_threshold=0.5),
        iou_calculator='naive_iou',
        box_formator='sph2pix'))