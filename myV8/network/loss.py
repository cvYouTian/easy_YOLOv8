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
    """glenn
    Args:
        c1[]: c1 is reg_max
    """
    def __init__(self, c1=16):
        super().__init__()
        # 单层的感知机
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        # [1, 64, 8400]
        b, c, a = x.shape
        x = x.view(b, 4, self.c1, a)

        # 将其拆成4个坐标点和reg_max的数量
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

