# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)
#optimizer = dict(type='Adam')
#optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.0001)


optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[80, 110])
runner = dict(type='EpochBasedRunner', max_epochs=120)