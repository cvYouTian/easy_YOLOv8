import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import re
import torch.nn.functional as F
import torch
import torch.nn as nn
from yolotest.Model.net import YOLOv8l
n = 100
l = [2, 3, 4, 5, 6]

l.insert(2, n)

print(l)



