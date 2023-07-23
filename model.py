import torch
from torch import nn



class C2f(nn.Module):
    pass
class Conv(nn.Module):
    def __init__(self, i=None, o=None, k=None, s=None, p=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, stride=s, padding=self.autopad(k,p))
        self.batchnorm = nn.BatchNorm2d(o)
        self.silu = nn.SiLU()
    @staticmethod
    def autopad(kernelsize,pad=None):
        if pad is None:
            pad = kernelsize // 2 if isinstance(kernelsize, int) else [x // 2 for x in kernelsize]
        return pad
    def forward(self, x):
        return self.silu(self.batchnorm(self.conv(x)))



class SPPF(nn.Module):
    def __init__(self, i=None, o=None):
        super().__init__()
        self.conv110 = Conv(i=512, o=512, k=1, s=1, p=0)
        # self.conv110 = nn.Sequential(Conv(i=512, o=512, k=1, s=1, p=0))
        self.maxpooling = nn.MaxPool2d(kernel_size=5, stride=1)


class Bottleneck(nn.Module):
    pass

class forward(nn.Module):
    pass


class YOLOv8l(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self,x):


