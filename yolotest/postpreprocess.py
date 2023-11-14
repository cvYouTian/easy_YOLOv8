import torch
import torch.nn as nn
from typing import Union
from pathlib import Path, PurePath
import os
from Model import Conv, C2f, Bottleneck, SPPF


class Fuse:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def is_fused(self):
        ...

