import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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



def train(policy, Qnet, Value_main, Value_target, replay_buffer, batch_size):

    device = policy.device

    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(batch_size)
    s_batch = torch.FloatTensor(s_batch).to(device)
    a_batch = torch.FloatTensor(a_batch).to(device)
    r_batch = torch.FloatTensor(r_batch).to(device)
    s2_batch = torch.FloatTensor(s2_batch).to(device)

    


def evaluate(policy, env, video_info = None):