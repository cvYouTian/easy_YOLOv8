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
    """这里的sppf结构作者使用了中间的隐藏层

    """

    def __init__(self, in_channel=512, out_channel=512):
        super(SPPF, self).__init__()
        c_ = in_channel // 2
        self.cv1 = Conv(in_channel, c_, 1, 1, padding=auto_padding(1, None))
        self.cv2 = Conv(c_ * 4, out_channel, 1, 1, padding=auto_padding(1, None))
        self.m = nn.MaxPool2d(5, 1, padding=auto_padding(5, None))

    def forward(self, x):
        x0 = self.cv1(x)
        x1 = self.m(x0)
        x2 = self.m(x1)
        return self.cv2(torch.cat((x0, x1, x2, self.m(x2)), 1))


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 shortcut: bool = True,
                 kernel_size: Union[int, tuple, list] = 3):
        super(Bottleneck, self).__init__()
        self.shortcut = shortcut
        self.conv1 = Conv(input_chanel=in_channel,
                          output_chanel=out_channel // 2,
                          kernel_size=kernel_size,
                          stride=1,
                          padding=auto_padding(kernel_size, None))

        self.conv2 = Conv(input_chanel=out_channel // 2,
                          output_chanel=out_channel,
                          kernel_size=kernel_size,
                          stride=1,
                          padding=auto_padding(kernel_size, None))

    def forward(self, x):
        if self.shortcut:
            x0 = self.conv1(x)
            x1 = self.conv2(x0)
            logits = x + x1
        else:
            x0 = self.conv1(x)
            logits = self.conv2(x0)
        return logits


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
                          padding=auto_padding(1, None))

        self.conv2 = Conv(input_chanel=out_channel // 2 * (block + 2),
                          output_chanel=out_channel,
                          kernel_size=1,
                          stride=1,
                          padding=auto_padding(1, None))

        self.bottleneck = nn.ModuleList(
            [Bottleneck(out_channel // 2, out_channel // 2, shortcut) for _ in range(block)])

    def forward(self, x):
        module = list(self.conv1(x).chunk(2, 1))
        module.extend(m(module[-1]) for m in self.bottleneck)
        x = self.conv2(torch.cat(module, 1))
        return x


class Detect(nn.Module):

    def __init__(self, in_chanel, nc, reg_max):
        super().__init__()
        self.in_chanel = in_chanel

        self.bbox = nn.Sequential(Conv(in_chanel, 64, 3, 1, 1),
                                  Conv(64, 64, 3, 1, 1),
                                  Conv(64, reg_max * 4, 1, 1, 0))

        self.cls = nn.Sequential(Conv(in_chanel, 256, 3, 1, 1),
                                 Conv(256, 256, 3, 1, 1),
                                 Conv(256, nc, 1, 1, 0))

    def forward(self, x):
        bbox = self.bbox(x)
        cls = self.cls(x)

        return torch.cat((bbox, cls), 1)


class YOLOv8l(nn.Module):
    def __init__(self, nc, reg_max):
        super(YOLOv8l, self).__init__()
        self.nc = nc
        self.reg_max = reg_max

        # backbone
        self.conv_0 = Conv(3, 64, 3, 2,
                           padding=auto_padding(3, None))

        self.conv_1 = Conv(64, 128, 3, 2,
                           padding=auto_padding(3, None))

        self.c2f_2 = C2f(True, 3, 128, 128)

        self.conv_3 = Conv(128, 256, 3, 2,
                           padding=auto_padding(3, None))

        self.c2f_4 = C2f(True, 6, 256, 256)

        self.conv_5 = Conv(256, 512, 3, 2,
                           padding=auto_padding(3, None))

        self.c2f_6 = C2f(True, 6, 512, 512)

        self.conv_7 = Conv(512, 512, 3, 2,
                           padding=auto_padding(3, None))

        self.c2f_8 = C2f(True, 3, 512, 512)

        self.sppf_9 = SPPF(512, 512)

        # head
        self.upsample = nn.Upsample(None, 2, "nearest")

        self.c2f_12 = C2f(False, 3, 1024, 512)

        self.c2f_15 = C2f(False, 3, 768, 256)

        self.conv_16 = Conv(256, 256, 3, 2,
                            padding=auto_padding(3, None))

        self.c2f_18 = C2f(False, 3, 768, 512)

        self.conv_19 = Conv(512, 512, 3, 2,
                            padding=auto_padding(3, None))

        self.c2f_21 = C2f(False, 3, 1024, 512)

        self.det_low = Detect(256, nc, reg_max)
        self.det_mid = Detect(512, nc, reg_max)
        self.det_high = Detect(512, nc, reg_max)

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

        det_low = self.det_low(x15)
        det_mid = self.det_low(x18)
        det_high = self.det_low(x21)

        return torch.cat([det_low.view(1, self.nc * 4 * self.reg_max, -1),
                          det_mid.view(1, self.nc * 4 * self.reg_max, -1),
                          det_high.view(1, self.nc * 4 * self.reg_max, -1)],
                         2)


if __name__ == '__main__':
    image_path = "/home/youtian/Pictures/Screenshots/bird.png"
    net = YOLOv8l(80, 16)
    net.to("cuda:0")
    net.eval()
    transform = transforms.Compose([transforms.Resize((640, 640)),
                                    transforms.ToTensor()])

    # image = Image.open(image_path)
    image = cv2.imread(image_path)
    image = Image.fromarray(image)
    image = transform(image)

    batch_input = image.unsqueeze(0).to("cuda:0")

    with torch.no_grad():
        out = net(batch_input)

    print(out)
    # feature_visualization(out, "nn.Conv2d, nn.Conv2d, nn.Conv2d", 5)
    # summary(net, input_size=(3, 640, 640))
