import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class VisionModel(torch.nn.Module):
    def __init__(self,  channel_size, height, width, dim_z, lr, device):
        super(VisionModel, self).__init__()

        self.channel_size = channel_size
        self.height = height
        self.width = width
        self.dim_z = dim_z
        self.lr = lr
        self.device = device

        ## Input size : [batch, 3, 84, 84]
        ### Encoder

        self.conv1 = nn.Conv2d(in_channels=self.channel_size, out_channels=32,
                               kernel_size=4, stride=2, padding=0).to(device) # [32, 41, 41]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=5, stride=2, padding=2).to(device) # [64, 21, 21]
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=5, stride=2, padding=2).to(device)  # [128, 11, 11]
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=5, stride=2, padding=2).to(device)  # [256, 6, 6]
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=4, stride=2, padding=0).to(device)  # [256, 2, 2]

        self.mu = nn.Linear(256 * 4, self.dim_z).to(device)
        self.log_std = nn.Linear(256 * 4, self.dim_z).to(device)

        self.recons = nn.Linear(dim_z, 256).to(device)


        ### Decoder
        self.t_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                          kernel_size=6, stride=1, padding=0).to(device)  # [128, 6, 6]
        self.t_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=5, stride=2, padding=2).to(device)  # [64, 11, 11]
        self.t_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                          kernel_size=5, stride=2, padding=2).to(device)  # [32, 21, 21]
        self.t_conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                          kernel_size=5, stride=2, padding=2).to(device)  # [16, 41, 41]
        self.t_conv5 = nn.ConvTranspose2d(in_channels=16, out_channels= self.channel_size,
                                          kernel_size=4, stride=2, padding=0).to(device)  # [3, 84, 84]

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)

    def encode(self, x):
        assert len(x.shape) == 4

        x = F.celu(self.conv1(x))
        x = F.celu(self.conv2(x))
        x = F.celu(self.conv3(x))
        x = F.celu(self.conv4(x))
        x = F.celu(self.conv5(x))

        x = x.view(-1, 256 * 4)

        mu = self.mu(x)
        sigma = torch.exp(torch.clamp(self.log_std(x),-3,3))

        return mu, sigma

    def reparameterization(self, mu, sigma):
        temp_normal = Normal(mu, sigma)

        return temp_normal.rsample()  # [batch, 16]

    def decode(self, z):
        x = F.celu(self.recons(z))  # [batch, 256]

        x = x.view(-1, 256, 1, 1)

        x = F.celu(self.t_conv1(x))
        x = F.celu(self.t_conv2(x))
        x = F.celu(self.t_conv3(x))
        x = F.celu(self.t_conv4(x))
        x = torch.clamp(torch.sigmoid(self.t_conv5(x)), 0, 1)

        return x

    def reconstruct_image(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterization(mu, sigma)
        image = self.decode(z)

        return image

    def to_dim_z(self, x):
        mus, sigmas = self.encode(x)
        zs = self.reparameterization(mus, sigmas)

        return zs

    def train(self, batch):
        assert len(batch.shape) == 4

        mus, sigmas = self.encode(batch)
        zs = self.reparameterization(mus, sigmas)

        recons = self.decode(zs)

        #print(torch.max(recons), torch.min(recons))
        #print(torch.max(mus),torch.min(mus),torch.max(sigmas),torch.min(sigmas))
        reconstruction_error = F.binary_cross_entropy(recons, batch, reduction="sum") / batch.shape[0]

        var = sigmas ** 2
        mu_2 = mus ** 2

        regularization_error = torch.mean(torch.sum(-0.5 * (1 + torch.log(var) - mu_2 - var), dim=1, keepdim=False))

        loss = reconstruction_error  + regularization_error

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
        self.optimizer.step()

        return reconstruction_error.detach().cpu().numpy(), regularization_error.detach().cpu().numpy(), loss.detach().cpu().numpy()

    def trainer(self, replay_buffer, image_buffer, batch_size, device):
        image_idx_batch = replay_buffer.sample_image_uniform(batch_size)

        image_batch = []
        for s_idx in image_idx_batch:
            frame = image_buffer.get_current_image(int(s_idx))
            image_batch.append(frame)

        image_batch = torch.FloatTensor(np.array(image_batch)).to(device)

        return self.train(image_batch)
