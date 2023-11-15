import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import re
import torch.nn.functional as F
import torch
import yaml

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linar1 = nn.Linear(6, 512)
        self.linar2 = nn.Linear(512, 256)
        self.linar3 = nn.Linear(256, 128)
        self.linar4 = nn.Linear(128, 64)
        self.linar5 = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.linar1(x))
        x = F.relu(self.linar2(x))
        x = F.relu(self.linar3(x))
        x = F.relu(self.linar4(x))
        output = F.relu(self.linar5(x))
        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(800, 32)
        self.BN1 = nn.BatchNorm2d(32)
        self.Avpooling = nn.AvgPool1d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(800, 32)
        self.BN1 = nn.BatchNorm2d(32)
        self.Avpooling = nn.AvgPool1d(kernel_size=3, stride=3)
        self.conv1 = nn.Conv1d(800, 32)
        self.BN1 = nn.BatchNorm2d(32)
        self.Avpooling = nn.AvgPool1d(kernel_size=3, stride=3)
        self.conv1 = nn.Conv1d(800, 32)
        self.BN1 = nn.BatchNorm2d(32)
        self.Avpooling = nn.AvgPool1d(kernel_size=3, stride=3)
        self.conv1 = nn.Conv1d(800, 32)
        self.BN1 = nn.BatchNorm2d(32)



if __name__ == '__main__':
    ...
    # net = Net()
    # print(net)
    #
    # t = torch.randn(512, 6)
    # print(t)



