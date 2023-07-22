import torch
from torch import nn



class C2f(nn.Module):
    pass
class Conv(nn.Module):
    def __init__(self, i=None, o=None, k=None, s=None, p=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, stride=s, padding=p)
    @staticmethod
    def autopading(kernelsize,paded=None):
        if paded is None:
            pad = kernelsize // 2 if isinstance(kernelsize, int) else [x // 2 for x in kernelsize]
        return pad

class SPPF(nn.Module):
    pass

class Bottleneck(nn.Module):
    pass

class forward(nn.Module):
    pass


class YOLOv8l(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self,x):


