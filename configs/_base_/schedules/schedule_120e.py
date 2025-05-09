# optimizer
optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.0001)


optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[110])
runner = dict(type='EpochBasedRunner', max_epochs=150)