import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VanillaSACQNet(nn.Module):
    def __init__(self, state_dim, action_dim, lr, device):
        super(VanillaSACQNet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = lr
        self.device = device

        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, 256).to(device)
        self.fc2 = nn.Linear(256,256).to(device)
        self.fc3 = nn.Linear(256,1).to(device)

        nn.init.uniform_(tensor=self.fc3.weight, a = -3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.fc3.bias, a=-3e-3, b=3e-3)

        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, 256).to(device)
        self.fc5 = nn.Linear(256, 256).to(device)
        self.fc6 = nn.Linear(256, 1).to(device)

        nn.init.uniform_(tensor=self.fc6.weight, a=-3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.fc6.bias, a=-3e-3, b=3e-3)

        self.optimizer = optim.Adam(self.parameters(),lr)


    def forward(self, s, a):
        x = torch.cat([s,a],dim=1) # [batch, s+a]

        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc4(x))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)

        return x1, x2

class ConvSACQNet(nn.Module):
    def __init__(self, step_channelsize, height, width, action_dim, lr, device):
        super(ConvSACQNet, self).__init__()

        self.step_channelsize = step_channelsize
        self.height = height
        self.width = width
        self.out_channels = 16

        self.conv_flatten_size = int(height * width * self.out_channels / (16 ** 2))

        self.action_dim = action_dim
        self.actor_lr = lr
        self.device = device

        # Q1 architecture
        self.conv1 = torch.nn.Conv2d(in_channels=self.step_channelsize, out_channels=64,
                                kernel_size=7, stride=4, padding=3).to(device)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=self.out_channels,
                                kernel_size=7, stride=4, padding=3).to(device)

        self.fc1 = nn.Linear(self.conv_flatten_size + self.action_dim, 128).to(device)
        self.fc2 = nn.Linear(128, 1).to(device)

        nn.init.uniform_(tensor=self.fc2.weight, a = -3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.fc2.bias, a=-3e-3, b=3e-3)

        # Q2 architecture
        self.conv3 = torch.nn.Conv2d(in_channels=self.step_channelsize, out_channels=64,
                                     kernel_size=7, stride=4, padding=3).to(device)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=self.out_channels,
                                     kernel_size=7, stride=4, padding=3).to(device)

        self.fc3 = nn.Linear(self.conv_flatten_size + self.action_dim, 128).to(device)
        self.fc4 = nn.Linear(128, 1).to(device)

        nn.init.uniform_(tensor=self.fc4.weight, a=-3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.fc4.bias, a=-3e-3, b=3e-3)

        self.optimizer = optim.Adam(self.parameters(),lr)

    def forward(self, s, a):

        x1 = F.relu(self.conv1(s))
        x1 = F.relu(self.conv2(x1))

        x1 = x1.view(-1, self.conv_flatten_size)
        x1 = torch.cat([x1, a], dim=1)  # [batch, s+a]

        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)

        x2 = F.relu(self.conv3(s))
        x2 = F.relu(self.conv4(x2))

        x2 = x2.view(-1, self.conv_flatten_size)
        x2 = torch.cat([x2, a], dim=1)  # [batch, s+a]

        x2 = F.relu(self.fc3(x2))
        x2 = self.fc4(x2)

        return x1, x2