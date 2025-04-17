#from mmcv.runner import Hook
from mmcv.runner.hooks import HOOKS, Hook

import torch

@HOOKS.register_module()
class GradientNormHook(Hook):
    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            total_norm = 0
            for p in runner.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            runner.log_buffer.update({'grad_norm': total_norm})