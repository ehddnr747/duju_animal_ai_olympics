import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class VanillaSACPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, lr, device):
        super(VanillaSACPolicy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = lr
        self.device = device


        self.fc1 = nn.Linear(state_dim, 256).to(device)
        self.fc2 = nn.Linear(256, 256).to(device)
        self.mu = nn.Linear(256, action_dim).to(device)
        self.log_std = nn.Linear(256, action_dim).to(device)

        nn.init.uniform_(tensor=self.mu.weight, a = -3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.mu.bias, a=-3e-3, b=3e-3)

        nn.init.uniform_(tensor=self.log_std.weight, a=-3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.log_std.bias, a=-3e-3, b=3e-3)

        self.optimizer = optim.Adam(self.parameters(),lr)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)

        # It should return mu, log_std

        return mu, log_std

    def sample_with_logp(self, x):
        mu, log_std = self.forward(x)
        std = torch.exp(log_std)

        normal = Normal(mu, std)
        x_t = normal.rsample()
        logp = normal.log_prob(x_t)

        y_t = torch.tanh(x_t)
        logp -= torch.log(1 - torch.pow(y_t, 2) + 1e-6)

        return y_t, logp

    def sample(self, x):

        with torch.no_grad():
            mu, log_std = self.forward(x)
            std = torch.exp(log_std)

            normal = Normal(mu, std)

            return torch.tanh(normal.sample())

    def mean_action(self, x):
        with torch.no_grad():
            mu, _log_std = self.forward(x)
            return torch.tanh(mu)






class ConvSACPolicy(nn.Module):
    def __init__(self, stepsize, height, width, action_dim, lr, device):
        super(VanillaSACPolicy, self).__init__()

        self.stepsize = stepsize
        self.height = height
        self.width = width

        self.conv_flatten_size = stepsize * height * width * 32 / (16 ** 2)

        self.action_dim = action_dim
        self.actor_lr = lr
        self.device = device

        self.conv1 = torch.nn.Conv2d(in_channels=self.stepsize, out_channels=256,
                                kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=128,
                                kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=2, padding=1)

        self.mu = nn.Linear(self.conv_flatten_size, action_dim).to(device)
        self.log_std = nn.Linear(self.conv_flatten_size, action_dim).to(device)

        nn.init.uniform_(tensor=self.mu.weight, a = -3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.mu.bias, a=-3e-3, b=3e-3)

        nn.init.uniform_(tensor=self.log_std.weight, a=-3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.log_std.bias, a=-3e-3, b=3e-3)

        self.optimizer = optim.Adam(self.parameters(),lr)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)

        # It should return mu, log_std

        return mu, log_std

    def sample_with_logp(self, x):
        mu, log_std = self.forward(x)
        std = torch.exp(log_std)

        normal = Normal(mu, std)
        x_t = normal.rsample()
        logp = normal.log_prob(x_t)

        y_t = torch.tanh(x_t)
        logp -= torch.log(1 - torch.pow(y_t, 2) + 1e-6)

        return y_t, logp

    def sample(self, x):

        with torch.no_grad():
            mu, log_std = self.forward(x)
            std = torch.exp(log_std)

            normal = Normal(mu, std)

            return torch.tanh(normal.sample())

    def mean_action(self, x):
        with torch.no_grad():
            mu, _log_std = self.forward(x)
            return torch.tanh(mu)


