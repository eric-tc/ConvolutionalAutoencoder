import torch

import torch.nn as nn
from torch.autograd import  Variable

import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv11 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(256, 1, kernel_size=3, padding=1)



    def forward(self, x):
        in_size = x.size(0)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))

        x = F.softmax(x, 1) #this line is changed

        return x

net = Net()
inputs = torch.rand(1,1,4,4)
out = net (Variable(inputs))
print (out)
out.sum(dim=1)