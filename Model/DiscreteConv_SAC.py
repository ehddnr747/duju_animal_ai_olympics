import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from Model.SAC_base import soft_target_update


class DiscreteConvSAC(nn.Module):
    def __init__(self, step_size, channel_size, height, width, action_dim, lr, device):
        super(DiscreteConvSAC, self).__init__()

        self.step_size = step_size
        self.channel_size = channel_size
        self.height = height
        self.width = width
        self.action_dim = action_dim
        self.lr = lr
        self.device = device

        self.input_channel_size = self.step_size * self.channel_size
        self.fc_input_size = int(16 * (height / 8) * (width / 8))

        print("input channel size : ", self.input_channel_size)
        print("fc input size : ", self.fc_input_size )

        # [input_channel_size, 64, 96]

        # Q1
        self.q1_conv1 = nn.Conv2d(in_channels=self.input_channel_size, out_channels=32,
                                  kernel_size=5, stride=2, padding=2).to(device)   #[32, 32, 48]
        self.q1_conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                                  kernel_size=5, stride=2, padding=2).to(device)   #[32, 16, 24]
        self.q1_conv3 = nn.Conv2d(in_channels=32, out_channels=16,
                                  kernel_size=3, stride=2, padding=1).to(device)   #[16, 8, 12]
        self.q1_fc1 = nn.Linear(self.fc_input_size, 256).to(device)
        self.q1_fc2 = nn.Linear(256, action_dim).to(device)


        # Q2
        self.q2_conv1 = nn.Conv2d(in_channels=self.input_channel_size, out_channels=32,
                                  kernel_size=5, stride=2, padding=2).to(device)  # [32, 32, 48]
        self.q2_conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                                  kernel_size=5, stride=2, padding=2).to(device)  # [32, 16, 24]
        self.q2_conv3 = nn.Conv2d(in_channels=32, out_channels=16,
                                  kernel_size=3, stride=2, padding=1).to(device)  # [32, 8, 12]
        self.q2_fc1 = nn.Linear(self.fc_input_size, 256).to(device)
        self.q2_fc2 = nn.Linear(256, action_dim).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, x):
        # dim x : [batch, input_channel_size, height, width]
        assert len(x.shape) == 4

        x1 = F.relu(self.q1_conv1(x))
        x1 = F.relu(self.q1_conv2(x1))
        x1 = F.relu(self.q1_conv3(x1))

        assert x1.shape[-1] * x1.shape[-2] * x1.shape[-3] == self.fc_input_size, (x1.shape[-1] * x1.shape[-2] * x1.shape[-3], self.fc_input_size)
        x1 = x1.view(-1, self.fc_input_size)

        x1 = F.relu(self.q1_fc1(x1))
        x1 = self.q1_fc2(x1)


        x2 = F.relu(self.q2_conv1(x))
        x2 = F.relu(self.q2_conv2(x2))
        x2 = F.relu(self.q2_conv3(x2))

        assert x2.shape[-1] * x2.shape[-2] * x2.shape[-3] == self.fc_input_size
        x2 = x2.view(-1, self.fc_input_size)

        x2 = F.relu(self.q2_fc1(x2))
        x2 = self.q2_fc2(x2)

        return x1, x2

    # def get_mean_distribution_from_Qs(self, q1, q2):
    #     return F.softmax(q1, 1)

    def get_mean_distribution_from_Qs(self, q1, q2):
        return (F.softmax(q1, 1) + F.softmax(q2, 1)) / 2.0

    def get_stochastic_action(self, x):
        with torch.no_grad():
            assert len(x.shape) == 4

            q1, q2 = self.forward(x)
            probs = self.get_mean_distribution_from_Qs(q1, q2)

            action = Categorical(probs).sample()
            assert action.shape == (1,)

            return action.detach().cpu().numpy()[0]

    def get_max_action(self, x):
        with torch.no_grad():
            assert len(x.shape) == 4

            q1, q2 = self.forward(x)
            probs = self.get_mean_distribution_from_Qs(q1, q2)

            action = torch.argmax(probs, dim=1)
            assert action.shape == (1,)

            return action.detach().cpu().numpy()[0]


# def train_discrete_Conv_SAC_stochastic(Q_main, Q_target, replay_buffer, batch_size, gamma, alpha):
#     device = Q_main.device
#     input_channel_size = Q_main.input_channel_size
#     height = Q_main.height
#     width = Q_main.width
#     action_dim = Q_main.action_dim
#
#     s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(batch_size)
#     s_batch = torch.FloatTensor(s_batch).to(device)  # (batch_size, input_channel_size, height, width)
#     a_batch = torch.LongTensor(a_batch).to(device)  # (batch_size, action_dim)
#     r_batch = torch.FloatTensor(r_batch).to(device)  # (batch_size, 1)
#     s2_batch = torch.FloatTensor(s2_batch).to(device)  # (batch_size, input_channel_size, height, width)
#
#     assert s_batch.shape == (batch_size, input_channel_size, height, width)
#     assert a_batch.shape == (batch_size, 1)
#     assert r_batch.shape == (batch_size, 1)
#     assert s2_batch.shape == (batch_size, input_channel_size, height, width)
#
#     q1, q2 = Q_main.forward(s_batch)
#
#     assert q1.shape == (batch_size, action_dim)
#     assert q2.shape == (batch_size, action_dim)
#
#     # Q[s,a] = r + gamma * ( H[s'] + E_(a')[Q(s',a')] )
#
#     # valid check!
#     q1_action = q1.gather(1, a_batch)  # Q[s,a]
#     q2_action = q2.gather(1, a_batch)
#
#     assert q1_action.shape == (batch_size, 1)
#     assert q2_action.shape == (batch_size, 1)
#
#     with torch.no_grad():
#         ### Q part ###
#         q1_t, q2_t = Q_target.forward(s2_batch)
#         assert q1_t.shape == (batch_size, action_dim)
#         assert q2_t.shape == (batch_size, action_dim)
#
#         target_probs = Q_target.get_mean_distribution_from_Qs(q1_t, q2_t)
#         assert target_probs.shape == (batch_size, action_dim)
#
#         E_q1_s2_t = torch.sum(target_probs * q1_t, dim=1, keepdim=True)
#         E_q2_s2_t = torch.sum(target_probs * q2_t, dim=1, keepdim=True)
#         assert E_q1_s2_t.shape == (batch_size, 1)
#         assert E_q2_s2_t.shape == (batch_size, 1)
#
#         q_target_min = torch.min(E_q1_s2_t, E_q2_s2_t)
#         assert q_target_min.shape == (batch_size, 1)
#
#         ### H part ###
#         q1_m_s2, q2_m_s2 = Q_main.forward(s2_batch)
#
#         main_probs = Q_main.get_mean_distribution_from_Qs(q1_m_s2, q2_m_s2)
#         assert main_probs.shape == (batch_size, action_dim)
#
#         main_policy_distribution_s2 = Categorical(main_probs)
#         main_entropy_s2 = main_policy_distribution_s2.entropy().view(-1, 1)  # H[s']
#         assert main_entropy_s2.shape == (batch_size, 1)
#
#         y_v = alpha * main_entropy_s2 + q_target_min
#         y_q = r_batch + gamma * y_v
#
#     q1_loss = F.mse_loss(q1_action, y_q)
#     q2_loss = F.mse_loss(q2_action, y_q)
#
#     Q_main.optimizer.zero_grad()
#     q1_loss.backward()
#     q2_loss.backward()
#     torch.nn.utils.clip_grad_value_(Q_main.parameters(), 1.0)
#     Q_main.optimizer.step()
#
#     soft_target_update(Q_main, Q_target, 0.005)
#
#     with torch.no_grad():
#         return torch.max(q1), torch.max(q2), torch.mean(main_entropy_s2), torch.mean(r_batch)


def train_discrete_Conv_SAC_max(Q_main, Q_target, replay_buffer, batch_size, gamma, alpha):
    device = Q_main.device
    input_channel_size = Q_main.input_channel_size
    height = Q_main.height
    width = Q_main.width
    action_dim = Q_main.action_dim

    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(batch_size)
    s_batch = torch.FloatTensor(s_batch).to(device)  # (batch_size, input_channel_size, height, width)
    a_batch = torch.LongTensor(a_batch).to(device)  # (batch_size, action_dim)
    r_batch = torch.FloatTensor(r_batch).to(device)  # (batch_size, 1)
    s2_batch = torch.FloatTensor(s2_batch).to(device)  # (batch_size, input_channel_size, height, width)

    assert s_batch.shape == (batch_size, input_channel_size, height, width)
    assert a_batch.shape == (batch_size, 1)
    assert r_batch.shape == (batch_size, 1)
    assert s2_batch.shape == (batch_size, input_channel_size, height, width)

    q1, q2 = Q_main.forward(s_batch)

    assert q1.shape == (batch_size, action_dim)
    assert q2.shape == (batch_size, action_dim)

    # Q[s,a] = r + gamma * ( H[s'] + max[Q(s',a')] )

    q1_action = q1.gather(1, a_batch)  # Q[s,a]
    q2_action = q2.gather(1, a_batch)

    assert q1_action.shape == (batch_size, 1)
    assert q2_action.shape == (batch_size, 1)

    with torch.no_grad():
        q1_t, q2_t = Q_target.forward(s2_batch)
        assert q1_t.shape == (batch_size, action_dim)
        assert q2_t.shape == (batch_size, action_dim)

        q1_m, q2_m = Q_main.forward(s2_batch)
        assert q1_m.shape == (batch_size, action_dim)
        assert q2_m.shape == (batch_size, action_dim)

        main_probs = Q_main.get_mean_distribution_from_Qs(q1_m, q2_m)
        assert main_probs.shape == (batch_size, action_dim)

        main_policy_distribution_s2 = Categorical(main_probs)
        main_entropy_s2 = main_policy_distribution_s2.entropy().view(-1, 1)  # H[s']
        assert main_entropy_s2.shape == (batch_size, 1)

        E_q1_s2_t = torch.max(q1_t, dim=1, keepdim=True)[0]
        E_q2_s2_t = torch.max(q2_t, dim=1, keepdim=True)[0]
        assert E_q1_s2_t.shape == (batch_size, 1)
        assert E_q2_s2_t.shape == (batch_size, 1)

        q_target_min = torch.min(E_q1_s2_t, E_q2_s2_t)
        assert q_target_min.shape == (batch_size, 1)

        y_v = alpha * main_entropy_s2 + q_target_min
        y_q = r_batch + gamma * y_v

    q1_loss = F.mse_loss(q1_action, y_q)
    q2_loss = F.mse_loss(q2_action, y_q)

    Q_main.optimizer.zero_grad()
    q1_loss.backward()
    q2_loss.backward()
    torch.nn.utils.clip_grad_value_(Q_main.parameters(), 1.0)
    Q_main.optimizer.step()

    soft_target_update(Q_main, Q_target, 0.005)

    with torch.no_grad():
        return torch.max(q1), torch.max(q2), torch.mean(main_entropy_s2), torch.mean(r_batch)