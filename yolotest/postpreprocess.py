import torch
import torch.nn as nn
from typing import Union
from pathlib import Path, PurePath
import os
from Model import Conv, C2f, Bottleneck, SPPF


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    anchors: torch.Tensor = torch.empty(0)
    stride: torch.Tensor = torch.empty(0)

    def __init__(self,
                 nc: int,
                 ch: Union[list, tuple]):
        super(Detect, self).__init__()
        self.nc = nc
        self.nh = len(ch)
        self.cla_conv = nn.ModuleList(
            nn.Sequential(Conv(x,), Conv(), nn.Conv2d()) for x in ch
        )



class Fuse:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def is_fused(self):
        ...
