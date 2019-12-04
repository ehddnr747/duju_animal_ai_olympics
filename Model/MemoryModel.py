import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class MemoryModel(nn.Module):
    def __init__(self, z_dim, hidden_size, lr, device):
        super(MemoryModel, self).__init__()

        self.action_dim = 3  # 0 : forward, 1 : left, 2 : right

        self.z_dim = z_dim # 64
        input_size = self.z_dim + self.action_dim

        self.input_size =  input_size# 67
        self.hidden_size = hidden_size # 512
        self.lr = lr
        self.device = device

        self.gaussian_const = ((np.pi * 2) ** (1 / 2))


        # [seq, batch, input_size]
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1).to(device)

        # [seq, batch, hidden_size], + (h_n, c_n)

        # [seq * batch, hidden_size]

        self.fc1 = nn.Linear(hidden_size, hidden_size).to(device)

        # [seq * batch, hidden_size]

        self.fc2 = nn.Linear(hidden_size, hidden_size).to(device)

        # [seq * batch, hidden_size]

        #MDN
        self.mu = nn.Linear(hidden_size, z_dim * 5).to(device)
        self.log_std = nn.Linear(hidden_size, z_dim * 5).to(device)
        self.prob = nn.Linear(hidden_size, z_dim * 5).to(device)

        # [seq * batch, z_dim * 5]

        # [seq, batch, z_dim, 5]

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)

    def gaussian_pdf(self, y, mu, sigma):
        return torch.clamp_min(torch.exp((y - mu) ** 2 / (-2) / (sigma ** 2)) / (sigma * self.gaussian_const), 1e-3)

    def gumbel_sample(self, prob, dim):
        z = np.random.gumbel(loc=0.0, scale=1.0, size=prob.shape)
        return torch.argmax(torch.log(prob) + torch.FloatTensor(z).to(self.device), dim=dim, keepdim=True)

    def full_forward(self, x, a, h, c):
        # x.shape == (seq, batch, z_dim)
        # a.shape == (seq, batch, a_dim)
        # h.shape == (1, batch, hidden_size)
        # c.shape == (1, batch, hidden_size)
        assert len(x.shape) == 3 and x.shape[2] == self.z_dim
        assert len(a.shape) == 3 and a.shape[2] == self.action_dim
        assert len(h.shape) == 3 and h.shape[2] == self.hidden_size
        assert len(c.shape) == 3 and c.shape[2] == self.hidden_size

        lstm_input = torch.cat([x,a], dim=2)

        lstm_output, (h_n, c_n) = self.lstm(lstm_input, (h, c))

        seq = lstm_output.shape[0]
        batch = lstm_output.shape[1]

        lstm_output = lstm_output.view([seq * batch, self.hidden_size])

        fc_output = torch.celu(self.fc1(lstm_output))
        fc_output = torch.celu(self.fc2(fc_output))

        mu = self.mu(fc_output)
        mu = mu.view([seq, batch, self.z_dim, 5])

        log_std = self.log_std(fc_output)
        sigma = torch.exp(torch.clamp(log_std, -3, 3)).view([seq, batch, self.z_dim, 5])

        prob = self.prob(fc_output).view([seq, batch, self.z_dim, 5])
        prob = F.softmax(prob, dim=3)

        gaussian_index = self.gumbel_sample(prob, dim=3)

        selected_mu = torch.gather(mu, dim=3, index= gaussian_index)
        selected_sigma = torch.gather(sigma, dim=3, index= gaussian_index)

        mdn_output = selected_mu + selected_sigma * torch.randn_like(selected_sigma)

        return mdn_output, mu, sigma, prob, h_n, c_n





    def hidden_forward(self, x, a, h, c):
        # x.shape == (1, 1, z_dim)
        # a.shape == (1, 1, a_dim)
        # h.shape == (1, 1, hidden_size)
        # c.shape == (1, 1, hidden_size)
        assert len(x.shape) == 3 and x.shape[2] == self.z_dim
        assert len(a.shape) == 3 and a.shape[2] == self.action_dim
        assert len(h.shape) == 3 and h.shape[2] == self.hidden_size
        assert len(c.shape) == 3 and c.shape[2] == self.hidden_size

        lstm_input = torch.cat([x, a], dim=2)

        lstm_output, (h_n, c_n) = self.lstm(lstm_input, (h, c))

        return h_n, c_n


    def train(self, z_in_batch, z_out_batch, a_batch, mask, batch_size, dim_z):
        assert z_in_batch.shape == (40, batch_size, dim_z)
        assert z_out_batch.shape == (40, batch_size, dim_z)
        assert a_batch.shape == (40, batch_size, 3)
        assert mask.shape == (40, batch_size)

        mask = mask.unsqueeze(2).float()
        z_out_batch = z_out_batch.unsqueeze(3)

        h_0 = torch.zeros([1, batch_size, self.hidden_size]).to(self.device)
        c_0 = torch.zeros([1, batch_size, self.hidden_size]).to(self.device)

        _, mu, sigma, prob, _, _ = self.full_forward(z_in_batch, a_batch, h_0, c_0)

        assert mu.shape == (40, batch_size, dim_z, 5)
        assert sigma.shape == (40, batch_size, dim_z, 5)
        assert prob.shape == (40, batch_size, dim_z, 5)

        p_y = self.gaussian_pdf(z_out_batch, mu, sigma) # [40, 16, 64, 5]


        loss = torch.mean(-torch.log(torch.sum(p_y * prob, dim=3)) * mask)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
        self.optimizer.step()

        return loss.detach().cpu().numpy()


    def train_batch(self, vision_model, replay_buffer, image_buffer, batch_size):
        s_return, a_return, r_return, t_return, s2_return, batch_size = replay_buffer.sample_sequence_uniform(batch_size, image_buffer)

        # s_return : [40, 16, 3, 84, 84]
        # a_return : [40, 16, 1]
        # r_return : [40, 16, 1]
        # t_return : [40, 16, 1]
        # s2_return : [40,16, 3, 84, 84]
        # batch_size : 4

        seq = s_return.shape[0]
        mask = np.cumprod(~t_return, axis=0)
        a_batch = self.one_hot(a_return.reshape(-1),3).reshape([seq, batch_size, 3])

        s_batch = torch.FloatTensor(s_return).to(self.device)
        a_batch = torch.FloatTensor(a_batch).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        s2_batch = torch.FloatTensor(s2_return).to(self.device)

        z_in_batch = vision_model.to_dim_z(s_batch.view(-1, 3, 84, 84)).view([seq, batch_size, self.z_dim])
        z_out_batch = vision_model.to_dim_z(s2_batch.view(-1, 3, 84, 84)).view([seq, batch_size, self.z_dim])
        mask = mask.squeeze(2)

        dim_z = vision_model.dim_z

        return self.train(z_in_batch, z_out_batch, a_batch, mask, batch_size, dim_z)

    def one_hot(self,indices,total_len):
        return np.eye(total_len)[indices]

