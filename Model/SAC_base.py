import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def soft_target_update(main, target, tau):
    params_main = list(main.parameters())
    params_target = list(target.parameters())

    assert len(params_main) == len(params_target)

    for pi in range(len(params_main)):
        params_target[pi].data.copy_((1 - tau) * params_target[pi].data + tau * params_main[pi].data)

def target_initialize(main, target):
    params_main = list(main.parameters())
    params_target = list(target.parameters())

    assert len(params_main) == len(params_target)

    for pi in range(len(params_main)):
        params_target[pi].data.copy_(params_main[pi].data)



def train(policy, Qnet, Value_main, Value_target, replay_buffer, batch_size, alpha, gamma):

    device = policy.device

    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(batch_size)
    s_batch = torch.FloatTensor(s_batch).to(device)
    a_batch = torch.FloatTensor(a_batch).to(device)
    r_batch = torch.FloatTensor(r_batch).to(device)
    s2_batch = torch.FloatTensor(s2_batch).to(device)

    MSE = nn.MSELoss()


    q1, q2 = Qnet.forward(s_batch, a_batch)
    v_main = Value_main.forward(s_batch)

    pi, logp_pi = policy.sample_with_logp(s_batch)
    pi_no_grad = pi.detach()

    with torch.no_grad():
        y_q = r_batch + gamma * Value_target.forward(s2_batch)
        y_v = torch.min(torch.cat(list(Qnet.forward(s_batch, pi_no_grad)),dim=1),dim=1,keepdim=True)[0] + alpha * logp_pi  # The shape must be [Batch, 1]

    V_loss = MSE(v_main, y_v)
    Q1_loss = MSE(q1, y_q)
    Q2_loss = MSE(q2, y_q)

    PI_loss = torch.mean((-1.0) * torch.mean(torch.cat(list(Qnet.forward(s_batch, pi_no_grad)),dim=1), dim=1, keepdim=True) + alpha * logp_pi)



    Value_main.optimizer.zero_grad()
    V_loss.backward()
    torch.nn.utils.clip_grad_value_(Value_main.parameters(), 1.0)
    Value_main.optimizer.step()

    Qnet.optimizer.zero_grad()
    Q1_loss.backward()
    Q2_loss.backward()
    torch.nn.utils.clip_grad_value_(Qnet.parameters(), 1.0)
    Qnet.optimizer.step()

    policy.optimizer.zero_grad()
    PI_loss.backward()
    torch.nn.utils.clip_grad_value_(policy.parameters(), 1.0)
    policy.optimizer.step()

    soft_target_update(Value_main, Value_target, 0.001)

    return np.max(v_main.detach().cpu().numpy())