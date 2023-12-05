import torch
import torch.nn as nn
from torch import optim


class MyModule(nn.Module):
    def __init__(self, num):
        super(MyModule, self).__init__()
        params = torch.ones(num, requires_grad=True)
        # self.conv = nn.Conv2d(3, 64, 3, 1, 1)
        self.params = nn.Parameter(params)

    def forward(self, x):
        y = self.params * x
        return y


if __name__ == '__main__':
    net = MyModule(10)
    # my_module = MyModule(10)
    # inputs = torch.ones(10)
    # outputs = my_module(inputs)
    # optimizer = optim.SGD([{"params": net.parameters()},
    #                        {"params": my_module.parameters()}], lr=0.01)
    #
    # tensor = torch.randn([2, 10, 30])
    # # print(tensor[:, 1, :].shape)
    #
    # a = lambda x : print(x)
    # a(4)
