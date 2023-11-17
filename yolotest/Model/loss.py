import os
import torch
import torch.nn as nn
import torch.functional as F
from typing import Union
from pathlib import Path, PurePath
from matplotlib import pyplot as plt


class Test(object):
    def __init__(self, r, g, b):
        self.r = r
        self.b = b
        self.g = g

    def __repr__(self):
        return "shit"
    
# a = Test(1,2, 3)
# print(a)
# print(getattr(a, "r"))

# a= 80
# b= 20
# sx = torch.arange(0, a).type(torch.FloatTensor) + 0.5
# sy = torch.arange(0, b).type(torch.FloatTensor) + 0.5
# x, y = torch.meshgrid(sx, sy)
# a = torch.stack((x, y), -1).view(-1, 2).type(torch.FloatTensor)
# print(a)
t = torch.randn([1, 8400, 4, 16])
a = t.softmax(3)
print(a[:, :, 0, 0])