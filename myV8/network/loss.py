import torch
from torchsummary import summary
import cv2
from PIL import Image
from torch import nn
import torch.nn.functional as F
from typing import Union
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import math
from pathlib import Path


__all__ = ("DFL", )


class DFL(nn.Module):
    """glenn, 经过DFL之后直接变成了四点的定位坐标

    Args:
        c1[]: c1 is reg_max
    """

    def __init__(self, c1=16):
        super().__init__()
        # 单层的感知机
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        # 这里定义了0-15的16个参数
        x = torch.arange(c1, dtype=torch.float)
        # 给卷积核赋值
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        # [1, 64, 8400]
        b, c, a = x.shape
        # 第一次的定位信息纬度变化， 这里是将坐标信息在前， reg_max信息在后
        x = x.view(b, 4, self.c1, a).transpose(2, 1)
        # 先在16个纬度上实现softmax，再使用单层感知机（16, 1）实现16个只
        distribute = self.conv(x.softmax(1)).view(b, 4, a)
        # [1, 4, 8400]
        return distribute


if __name__ == '__main__':
    dfl = DFL()
    a = torch.randn(1, 64, 8400)
    print(dfl(a).shape)

