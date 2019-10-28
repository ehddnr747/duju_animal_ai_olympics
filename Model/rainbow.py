import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Model.SAC_base import soft_target_update
from numpy.random import binomial
from numpy.random import multinomial
import numpy as np

class Rainbow_DQN(nn.Module):
    def __init__(self, state_dim, action_dim, lr, device):
        super(Rainbow_DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.device = device

        self.uniform_probs = [1.0/self.action_dim] * self.action_dim

        # Q1
        self.q1_fc1 = nn.Linear(self.state_dim, 256).to(device)
        self.q1_fc2_V = nn.Linear(256, 256).to(device)
        self.q1_fc2_A = nn.Linear(256, 256).to(device)

        self.q1_V = nn.Linear(256, 1).to(device)
        self.q1_A = nn.Linear(256, action_dim).to(device)

        # Q2
        self.q2_fc1 = nn.Linear(self.state_dim, 256).to(device)
        self.q2_fc2_V = nn.Linear(256, 256).to(device)
        self.q2_fc2_A = nn.Linear(256, 256).to(device)

        self.q2_V = nn.Linear(256, 1).to(device)
        self.q2_A = nn.Linear(256, action_dim).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr, weight_decay=1e-6)

        self.milestones = [100000000]
        self.step_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones,gamma=0.5)

    def forward(self, x):
        assert len(x.shape) == 2

        x1 = F.celu(self.q1_fc1(x))
        x1_V = F.celu(self.q1_fc2_V(x1))
        x1_A = F.celu(self.q1_fc2_A(x1))

        x1_V = self.q1_V(x1_V)
        x1_A = self.q1_A(x1_A)

        x1_A_mean = torch.mean(x1_A, dim=1,keepdim=True)
        x1_A_mean = torch.cat([x1_A_mean]*self.action_dim, dim=1)

        x1_A = x1_A - x1_A_mean

        x1 = x1_V + x1_A

        x2 = F.celu(self.q2_fc1(x))
        x2_V = F.celu(self.q2_fc2_V(x2))
        x2_A = F.celu(self.q2_fc2_A(x2))

        x2_V = self.q2_V(x2_V)
        x2_A = self.q2_A(x2_A)

        x2_A_mean = torch.mean(x2_A, dim=1, keepdim=True)
        x2_A_mean = torch.cat([x2_A_mean] * self.action_dim, dim=1)

        x2_A = x2_A - x2_A_mean

        x2 = x2_V + x2_A

        return x1, x2

    def epsilon_sample(self, x, epsilon):
        exploration_flag = bool(binomial(1, epsilon))

        if exploration_flag:
            action = np.argmax(multinomial(1, self.uniform_probs))
            return action

        else:
            q1, q2 = self.forward(x)
            Q = q1 + q2

            action = torch.argmax(Q, dim=1)
            assert action.shape == (1,)

            return action.detach().cpu().numpy()[0]

#Image Buffer
def train_rainbow_dqn(Q_main, Q_target, replay_buffer, batch_size, gamma):
    device = Q_main.device

    state_dim = Q_main.state_dim
    action_dim = Q_main.action_dim

    s_batch, a_batch, r_batch, t_batch, s2_batch, indices, weights = replay_buffer.sample_batch(batch_size)


    s_batch = torch.FloatTensor(np.array(s_batch)).to(device)
    s2_batch = torch.FloatTensor(np.array(s2_batch)).to(device)
    a_batch = torch.LongTensor(a_batch).to(device)  # (batch_size, action_dim)
    r_batch = torch.FloatTensor(r_batch).to(device)  # (batch_size, 1)

    weights = torch.FloatTensor(weights).to(device).view(-1, 1) #(batch_size, 1)


    assert s_batch.shape == (batch_size, state_dim)
    assert s2_batch.shape == (batch_size, state_dim)
    assert a_batch.shape == (batch_size, 1)
    assert r_batch.shape == (batch_size, 1)
    assert weights.shape == (batch_size, 1)

    q1, q2 = Q_main.forward(s_batch)

    assert q1.shape == (batch_size, action_dim)
    assert q2.shape == (batch_size, action_dim)

    q1_action = q1.gather(1, a_batch)  # Q[s,a]
    q2_action = q2.gather(1, a_batch)

    assert q1_action.shape == (batch_size, 1)
    assert q2_action.shape == (batch_size, 1)

    # Q[s,a] = r + gamma * targetmin(mainargmax(Q[s',a']))

    with torch.no_grad():
        q1_s2_m, q2_s2_m = Q_main.forward(s2_batch)
        q1_a2_m = torch.argmax(q1_s2_m, dim=1, keepdim=True)
        q2_a2_m = torch.argmax(q2_s2_m, dim=1, keepdim=True)
        assert q1_a2_m.shape == (batch_size, 1)
        assert q2_a2_m.shape == (batch_size, 1)

        q1_s2_t, q2_s2_t = Q_target.forward(s2_batch)
        assert q1_s2_t.shape == (batch_size, action_dim)
        assert q2_s2_t.shape == (batch_size, action_dim)

        q1_max = q1_s2_t.gather(1, q1_a2_m)
        q2_max = q2_s2_t.gather(1, q2_a2_m)
        assert q1_max.shape == (batch_size, 1)
        assert q2_max.shape == (batch_size, 1)


        min_Q_max = torch.min(q1_max, q2_max)
        assert min_Q_max.shape == (batch_size, 1)

        y_q = r_batch + gamma * min_Q_max

    q1_l1 = torch.abs(q1_action - y_q)
    q2_l1 = torch.abs(q2_action - y_q)

    # PER Buffer Update
    with torch.no_grad():
        priority = (q1_l1 + q2_l1) / 2.0
        priority = priority.cpu().numpy()
        replay_buffer.update_priorities(indices, priority)
        replay_buffer.update_beta()

    q1_loss = torch.mean((q1_l1 ** 2) * weights)
    q2_loss = torch.mean((q2_l1 ** 2) * weights)

    q1_loss = torch.clamp(q1_loss, -1.0, 1.0)
    q2_loss = torch.clamp(q2_loss, -1.0, 1.0)

    Q_main.optimizer.zero_grad()
    q1_loss.backward()
    q2_loss.backward()
    torch.nn.utils.clip_grad_value_(Q_main.parameters(), 1.0)
    Q_main.optimizer.step()
    Q_main.step_lr_scheduler.step()


    soft_target_update(Q_main, Q_target, 0.005)

    with torch.no_grad():
        return torch.mean(q1), torch.min(q1), torch.max(q1), torch.mean(q2), torch.mean(r_batch)