import torch
from torch import nn

class myNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 10, 3)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
net = myNet()
pic = torch.randn((3, 5, 5))
print(net(pic).shape)