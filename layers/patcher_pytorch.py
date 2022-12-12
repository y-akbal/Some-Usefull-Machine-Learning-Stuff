import torch
import torch.nn as nn
class patcher(nn.Module):
    def __init__(self, shapes):
        super().__init__()
        self.n, self.m = shapes
        iscompiled = False
    def forward(self, tensor):
        b, d, k, l = tensor.shape
        assert k % self.n == 0 and l % self.m == 0, "Keep in mind divisibility criterions"
        return tensor.reshape(b, d * (k//self.n)*(l//self.m), self.n, self.m)
