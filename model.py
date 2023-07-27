import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, i=None, o=None, k=None, s=None, p=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, stride=s,
                              padding=self.autopad(k, p), bias=False)
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
    def __init__(self, in_channel=512, out_channel=512):
        super().__init__()
        self.conv = Conv(i=in_channel, o=out_channel, k=1, s=1, p=0)
        self.maxpooling = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpooling(x0)
        x2 = self.maxpooling(x1)
        x3 = self.maxpooling(x2)
        x4 = torch.cat((x0, x1, x2, x3), 1)
        out = self.conv(x4)
        return out


class Bottleneck(nn.Module):
    def __init__(self, shortcut=True, in_channel=None, out_channel=None):
        super().__init__()
        self.shortcut = shortcut
        self.conv1 = Conv(i=in_channel, o=0.5*out_channel, k=3, s=1, p=1)
        self.conv2 = Conv(i=0.5*out_channel, o=out_channel, k=3, s=1, p=1)

    def forward(self, x):
        if self.shortcut:
            x0 = self.conv1(x)
            x1 = self.conv2(x0)
            out = x0 + x1
        else:
            x0 = self.conv1(x)
            out = self.conv2(x0)
        return out


class C2f(nn.Module):
    def __init__(self, shortcut=True, Block=None, in_channel=None, out_channel=None):
        super().__init__()
        self.shortcut = shortcut
        self.Block = Block
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = Conv(i=in_channel, o=out_channel, k=1, s=1, p=0)
        self.bottleneck = nn.ModuleList(Bottleneck(shortcut=shortcut, in_channel=self.out_channel,
                                                   out_channel=self.out_channel) for _ in range(Block))

    def forward(self, x):
        x0 = list(self.conv(x).chunk(2, 1))
        x0.extend(i(x0[-1]) for i in self.bottleneck)
        return self.conv(torch.cat(x0, 1))



class YOLOv8l(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone
        self.conv0 = Conv(i=3, o=64, k=3, s=2, p=1)
        self.conv1 = Conv(i=64, o=128, k=3, s=2, p=1)
        self.c2f_2 = C2f(shortcut=True, Block=3, in_channel=128, out_channel=128)
        self.conv3 = Conv(i=128, o=256, k=3, s=2, p=1)
        self.c2f_4 = C2f(shortcut=True, Block=6, in_channel=256, out_channel=256)
        self.conv5 = Conv(i=256, o=512, k=3, s=2, p=1)
        self.c2f_6 = C2f(shortcut=True, Block=6, in_channel=512, out_channel=512)
        self.conv7 = Conv(i=512, o=512, k=3, s=2, p=1)
        self.c2f_8 = C2f(shortcut=True, Block=3, in_channel=512, out_channel=512)
        self.sppf_9 = SPPF(in_channel=512, out_channel=512)
        # head
        self.upsample = nn.Upsample(scale_factor=2)
        self.c2f_12 = C2f(shortcut=False, Block=3, in_channel=1024, out_channel=512)
        self.c2f_15 = C2f(shortcut=False, Block=3, in_channel=512, out_channel=512)
        self.conv16 = Conv(i=256, o=256, k=3, s=2, p=1)
        self.c2f_18 = C2f(shortcut=False, Block=3, in_channel=512, out_channel=512)
        self.conv19 = Conv(i=512, o=512, k=3, s=2, p=1)
        self.c2f_21 = C2f(shortcut=False, Block=3, in_channel=1024, out_channel=512)
    @staticmethod
    def Detect():





    def forward(self,x):
        pass


