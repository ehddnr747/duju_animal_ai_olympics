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

