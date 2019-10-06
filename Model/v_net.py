import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VanillaSACValue(nn.Module):
    def __init__(self, state_dim, lr, device):
        super(VanillaSACValue, self).__init__()

        self.state_dim = state_dim
        self.actor_lr = lr
        self.device = device


        self.fc1 = nn.Linear(state_dim,256).to(device)
        self.fc2 = nn.Linear(256,256).to(device)
        self.fc3 = nn.Linear(256,1).to(device)

        nn.init.uniform_(tensor=self.fc3.weight, a = -3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.fc3.bias, a=-3e-3, b=3e-3)

        self.optimizer = optim.Adam(self.parameters(),lr)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ConvSACValue(nn.Module):
    def __init__(self, step_channelsize, height, width, lr, device):
        super(ConvSACValue, self).__init__()

        self.step_channelsize = step_channelsize
        self.height = height
        self.width = width
        self.out_channels = 16

        self.conv_flatten_size = int(height * width * self.out_channels / (8 ** 2))

        print("Value Net conv flatten size : ", self.conv_flatten_size)

        self.actor_lr = lr
        self.device = device

        self.conv1 = torch.nn.Conv2d(in_channels=self.step_channelsize, out_channels=64,
                                     kernel_size=3, stride=2, padding=1).to(device)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=self.out_channels,
                                     kernel_size=7, stride=4, padding=3).to(device)

        self.fc1 = nn.Linear(self.conv_flatten_size, 128).to(device)
        self.fc2 = nn.Linear(128, 128).to(device)
        self.fc3 = nn.Linear(128, 1).to(device)

        nn.init.uniform_(tensor=self.fc3.weight, a = -3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.fc3.bias, a=-3e-3, b=3e-3)

        self.optimizer = optim.Adam(self.parameters(),lr)


    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        assert x.shape[-1] * x.shape[-2] * x.shape[-3] == self.conv_flatten_size

        x = x.view(-1, self.conv_flatten_size)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
