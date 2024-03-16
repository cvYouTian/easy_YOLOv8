import torch
import torch.nn as nn
from YOLOv8lite.network.model import YOLOv8l


net = YOLOv8l(80, 16).to("cuda:0")
data = xxx

net(data)




print(type(net))