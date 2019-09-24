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

