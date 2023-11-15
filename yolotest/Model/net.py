import torch
from torch import nn
import torch.nn.functional as F
from typing import Union


def auto_padding(kernel_size, pad=None):
    if pad is None:
        pad = kernel_size // 2 if isinstance(kernel_size, int) else (x // 2 for x in kernel_size)
    return pad


class Conv(nn.Module):
    def __init__(self, input_chanel,
                 output_chanel,
                 kernel_size: Union[int, tuple, list],
                 stride: int,
                 padding: Union[int, None]):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_chanel,
                              out_channels=output_chanel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=auto_padding(kernel_size, padding),
                              bias=False)
        self.BN = nn.BatchNorm2d(output_chanel)

    def forward(self, x):
        return F.silu(self.BN(self.conv(x)))


class SPPF(nn.Module):
    def __init__(self, in_channel=512, out_channel=512):
        super(SPPF, self).__init__()
        self.conv = Conv(in_channel, out_channel, (1, 1), 1, padding=auto_padding(1, None))
        self.maxpooling = nn.MaxPool2d(kernel_size=5, stride=1, padding=auto_padding(5, None))

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
                 in_channel,
                 out_channel,
                 shortcut: bool = True,
                 kernel_size: Union[int, tuple, list] = 3):
        super(Bottleneck, self).__init__()
        self.shortcut = shortcut
        self.conv1 = Conv(input_chanel=in_channel,
                          output_chanel=int(out_channel * 0.5),
                          kernel_size=kernel_size,
                          stride=1,
                          padding=auto_padding(kernel_size=kernel_size))
        self.conv2 = Conv(input_chanel=int(out_channel * 0.5),
                          output_chanel=out_channel,
                          kernel_size=kernel_size,
                          stride=1,
                          padding=auto_padding(kernel_size=kernel_size, pad=None))

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
    def __init__(self,
                 shortcut: bool,
                 block: int,
                 in_channel,
                 out_channel):
        super(C2f, self).__init__()
        self.conv1 = Conv(input_chanel=in_channel,
                          output_chanel=out_channel,
                          kernel_size=1,
                          stride=1,
                          padding=auto_padding(kernel_size=1, pad=None))
        self.conv2 = Conv(input_chanel=int(out_channel * 0.5 * 3),
                          output_chanel=out_channel,
                          kernel_size=1,
                          stride=1,
                          padding=auto_padding(kernel_size=1, pad=None))
        self.bottleneck = nn.ModuleList(
            [Bottleneck(int(out_channel * 0.5), int(out_channel * 0.5), shortcut) for _ in range(block)])

    def forward(self, x):
        module = list(self.conv1(x).chunk(2, 1))
        module.extend(m(module[-1]) for m in self.bottleneck)
        x = self.conv2(torch.cat(module, 1))
        return x

        # x = self.conv1(x)
        # x0, x1 = torch.split(x, [self.out_channel*0.5, self.out_channel*0.5], dim=1)
        # x2 = [i(x) for i in self.bottleneck]
        # x3 = torch.cat(x2, dim=1)
        #



class YOLOv8l(nn.Module):
    def __init__(self):
        super(YOLOv8l, self).__init__()
        # backbone
        self.conv_0 = Conv(3, 64, 3, 2,
                           padding=auto_padding(kernel_size=3, pad=None)),
        self.conv_1 = Conv(64, 128, 3, 2,
                           padding=auto_padding(kernel_size=3, pad=None)),
        self.c2f_2 = C2f(True, 3, 128, 128),
        self.conv_3 = Conv(128, 256, 3, 2,
                           padding=auto_padding(kernel_size=3, pad=None)),
        self.c2f_4 = C2f(True, 6, 256, 256),
        self.conv_5 = Conv(256, 512, 3, 2,
                           padding=auto_padding(kernel_size=3, pad=None)),
        self.c2f_6 = C2f(True, 6, 512, 512),
        self.conv_7 = Conv(512, 512, 3, 2,
                           padding=auto_padding(kernel_size=3, pad=None)),
        self.c2f_8 = C2f(True, 3, 512, 512),
        self.sppf_9 = SPPF(512, 512)

        # head
        self.upsample = nn.Upsample(2)
        self.c2f_12 = C2f(False, 3, 1024, 512)
        self.c2f_15 = C2f(False, 3, 512, 512)
        self.conv_16 = Conv(256, 256, 3, 2,
                            padding=auto_padding(3, None))
        self.c2f_18 = C2f(False, 3, 512, 512)
        self.conv_19 = Conv(512, 512, 3, 2,
                            padding=auto_padding(3, None))
        self.c2f_21 = C2f(False, 3, 1024, 512)

    @staticmethod
    def detector(feat):
        ch = feat[:]
        Bbox = nn.ModuleList(
            nn.Sequential()
        )
    def forward(self, x):
        # backbone
        p1 = self.conv_0(x)
        p2 = self.conv_1(p1)
        x2 = self.c2f_2(p2)
        p3 = self.conv_3(x2)
        x4 = self.c2f_4(p3)
        p4 = self.conv_5(x4)
        x6 = self.c2f_6(p4)
        p5 = self.conv_7(x6)
        x8 = self.c2f_8(p5)
        sppf = self.sppf_9(x8)
        # neck
        x10 = self.upsample(sppf)
        x11 = torch.cat((x6, x10), 1)
        x12 = self.c2f_12(x11)
        x13 = self.upsample(x12)
        x14 = torch.cat((x4, x13), 1)
        x15 = self.c2f_15(x14)

        x16 = self.conv_16(x15)
        x17 = torch.cat((x12, x16), 1)
        x18 = self.c2f_18(x17)

        x19 = self.conv_19(x18)
        x20 = torch.cat((sppf, x19), 1)
        x21 = self.c2f_21(x20)
        return x15, x18, x21


net = YOLOv8l()
print(net)
