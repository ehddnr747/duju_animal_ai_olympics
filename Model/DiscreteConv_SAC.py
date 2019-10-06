import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from Model.SAC_base import soft_target_update


class DiscreteConvSAC(nn.Module):
    def __init__(self, state_dim, action_dim, lr, device):
        super(DiscreteConvSAC, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.device = device

        # Q1
        self.fc1 = nn.Linear(state_dim, 256).to(device)
        self.fc2 = nn.Linear(256, 256).to(device)
        self.fc3 = nn.Linear(256, action_dim).to(device)

        # Q2
        self.fc4 = nn.Linear(state_dim, 256).to(device)
        self.fc5 = nn.Linear(256, 256).to(device)
        self.fc6 = nn.Linear(256, action_dim).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, x):
        # dim x : [batch, state_dim]
        assert len(x.shape) == 2

        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc4(x))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)

        return x1, x2

    def get_mean_distribution_from_Qs(self, q1, q2):
        return (F.softmax(q1, 1) + F.softmax(q2, 1)) / 2.0

    def get_stochastic_action(self, x):
        with torch.no_grad():
            assert len(x.shape) == 2

            q1, q2 = self.forward(x)
            probs = self.get_mean_distribution_from_Qs(q1, q2)

            action = Categorical(probs).sample()
            assert action.shape == (1,)

            return action.detach().cpu().numpy()[0]

    def get_max_action(self, x):
        with torch.no_grad():
            assert len(x.shape) == 2

            q1, q2 = self.forward(x)
            probs = self.get_mean_distribution_from_Qs(q1, q2)

            action = torch.argmax(probs, dim=1)
            assert action.shape == (1,)

            return action.detach().cpu().numpy()[0]

class DiscreteSAC(nn.Module):
    def __init__(self, state_dim, action_dim, lr, device):
        super(DiscreteSAC, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.device = device

        # Q1
        self.fc1 = nn.Linear(state_dim, 256).to(device)
        self.fc2 = nn.Linear(256, 256).to(device)
        self.fc3 = nn.Linear(256, action_dim).to(device)

        # Q2
        self.fc4 = nn.Linear(state_dim, 256).to(device)
        self.fc5 = nn.Linear(256, 256).to(device)
        self.fc6 = nn.Linear(256, action_dim).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, x):
        # dim x : [batch, state_dim]
        assert len(x.shape) == 2

        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc4(x))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)

        return x1, x2

    def get_mean_distribution_from_Qs(self, q1, q2):
        return (F.softmax(q1, 1) + F.softmax(q2, 1)) / 2.0

    def get_stochastic_action(self, x):
        with torch.no_grad():
            assert len(x.shape) == 2

            q1, q2 = self.forward(x)
            probs = self.get_mean_distribution_from_Qs(q1, q2)

            action = Categorical(probs).sample()
            assert action.shape == (1,)

            return action.detach().cpu().numpy()[0]

    def get_max_action(self, x):
        with torch.no_grad():
            assert len(x.shape) == 2

            q1, q2 = self.forward(x)
            probs = self.get_mean_distribution_from_Qs(q1, q2)

            action = torch.argmax(probs, dim=1)
            assert action.shape == (1,)

            return action.detach().cpu().numpy()[0]


def train_discrete_Conv_SAC(Q_main, Q_target, replay_buffer, batch_size, gamma):
    device = Q_main.device
    state_dim = Q_main.state_dim
    action_dim = Q_main.action_dim

    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(batch_size)
    s_batch = torch.FloatTensor(s_batch).to(device)  # (batch_size, state_dim)
    a_batch = torch.LongTensor(a_batch).to(device)  # (batch_size, action_dim)
    r_batch = torch.FloatTensor(r_batch).to(device)  # (batch_size, 1)
    s2_batch = torch.FloatTensor(s2_batch).to(device)  # (batch_szie, state_dim)

    assert s_batch.shape == (batch_size, state_dim)
    assert a_batch.shape == (batch_size, 1)
    assert r_batch.shape == (batch_size, 1)
    assert s2_batch.shape == (batch_size, state_dim)

    q1, q2 = Q_main.forward(s_batch)

    assert q1.shape == (batch_size, action_dim)
    assert q2.shape == (batch_size, action_dim)

    # Q[s,a] = r + gamma * ( H[s'] + E_(a')[Q(s',a')] )

    # valid check!
    q1_action = q1.gather(1, a_batch)  # Q[s,a]
    q2_action = q2.gather(1, a_batch)

    assert q1_action.shape == (batch_size, 1)
    assert q2_action.shape == (batch_size, 1)

    with torch.no_grad():
        q1_t, q2_t = Q_target.forward(s2_batch)
        assert q1_t.shape == (batch_size, action_dim)
        assert q2_t.shape == (batch_size, action_dim)

        target_probs = Q_target.get_mean_distribution_from_Qs(q1_t, q2_t)
        assert target_probs.shape == (batch_size, action_dim)

        target_policy_distribution_s2 = Categorical(target_probs)
        target_entropy_s2 = target_policy_distribution_s2.entropy().view(-1, 1)  # H[s']
        assert target_entropy_s2.shape == (batch_size, 1)

        E_q1_s2_t = torch.sum(target_probs * q1_t, dim=1, keepdim=True)
        E_q2_s2_t = torch.sum(target_probs * q2_t, dim=1, keepdim=True)
        assert E_q1_s2_t.shape == (batch_size, 1)
        assert E_q2_s2_t.shape == (batch_size, 1)

        q_target_min = torch.min(E_q1_s2_t, E_q2_s2_t)
        assert q_target_min.shape == (batch_size, 1)

        y_v = target_entropy_s2 + q_target_min
        y_q = r_batch + gamma * y_v

    q1_loss = F.mse_loss(q1_action, y_q)
    q2_loss = F.mse_loss(q2_action, y_q)

    Q_main.optimizer.zero_grad()
    q1_loss.backward()
    q2_loss.backward()
    torch.nn.utils.clip_grad_value_(Q_main.parameters(), 1.0)
    Q_main.optimizer.step()

    soft_target_update(Q_main, Q_target, 0.001)

    with torch.no_grad():
        return torch.max(q1), torch.max(q2)