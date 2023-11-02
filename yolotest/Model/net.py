import torch
from torch import nn
from typing import Union


def auto_padding(kernel_size, pad=None):
    if pad is None:
        pad = kernel_size // 2 if isinstance(kernel_size, int) else (x // 2 for x in kernel_size)
    return pad

class Conv(nn.Module):
    def __init__(self, input_chanel: int, output_chanel: int, kernel_size: Union[int, tuple, list] = (1, 1),
                 stride: int = 1, padding: Union[int, None] = None, ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_chanel,
                              out_channels=output_chanel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=self.autopadding(kernel_size, padding),
                              bias=False)
        self.BN = nn.BatchNorm2d(output_chanel)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.BN(self.conv(x)))


class SPPF(nn.Module):
    def __init__(self, in_channel=512, out_channel=512):
        super().__init__()
        self.conv = Conv(i=in_channel, o=out_channel, k=[1, 1], s=1, p=None)
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
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 shortcut: bool=True,
                 kernel_size: Union[int, tuple, list]=3):
        super(Bottleneck, self).__init__()
        self.shortcut = shortcut
        self.conv1 = Conv(input_chanel=in_channel, output_chanel=0.5*out_channel, kernel_size=kernel_size, stride=1,
                          padding=auto_padding(kernel_size=kernel_size))
        self.conv2 = Conv(input_chanel=0.5*out_channel, output_chanel=out_channel, kernel_size=3, stride=1, p=1)

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
        self.conv1 = Conv(i=in_channel, o=out_channel, k=1, s=1, p=0)
        self.conv2 = Conv(i=out_channel*0.5*3, o=out_channel, k=1, s=1, p=0)
        self.bottleneck = nn.ModuleList(Bottleneck(shortcut=shortcut, in_channel=self.out_channel*0.5,
                                                   out_channel=self.out_channel*0.5) for _ in range(Block))

    def forward(self, x):
        # x0 = list(self.conv(x).chunk(2, 1))
        # x0.extend(i(x0[-1]) for i in self.bottleneck)
        # return self.conv(torch.cat(x0, 1))
        x0, x1 = torch.split(x, [self.out_channel*0.5, self.out_channel*0.5], dim=1)
        x2 = (i(x) for i in self.bottleneck)
        x3 = torch.cat(())


class YOLOv8l(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone
        self.conv0 = Conv(input=3, output=64, kernel_size=3, stride=2, padding=None)
        self.conv1 = Conv(input=64, output=128, kernel_size=3, stride=2, padding=None)
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
        pass

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.c2f_2(x1)
        x3 = self.conv3(x2)
        x4 = self.c2f_4(x3)
        x5 = self.conv5(x4)
        x6 = self.c2f_6(x5)
        x7 = self.conv7(x6)
        x8 = self.c2f_8(x7)
        x9 = self.sppf_9(x8)
        x10 = self.upsample(x9)
        x11 = torch.cat((x6, x10), 1)
        x12 = self.c2f_12(x11)
        x13 = self.upsample(x12)
        x14 = torch.cat((x4, x13), 1)
        x15 = self.c2f_15(x14)

        x16 = self.conv16(x15)
        x17 = torch.cat((x12, x16), 1)
        x18 = self.c2f_18(x17)

        x19 = self.conv19(x18)
        x20 = torch.cat((x9, x19), 1)
        x21 = self.c2f_21(x20)
        return x15





net = YOLOv8l()
print(net)


