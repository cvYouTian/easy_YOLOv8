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



a = Test(1,2, 3)
# print(a)
print(getattr(a, "r"))
