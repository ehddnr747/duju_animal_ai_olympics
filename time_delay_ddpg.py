import os

import utils.utils as utils
import utils.graph_reward as graph_reward
import models.ReplayBuffer as ReplayBuffer
import models.Noise as Noise

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dm_control import suite
import numpy as np
import argparse

framework = "PyTorch"
exp_type = "Time_Delay"

actor_lr = 1e-4
critic_lr = 1e-4
tau = 1e-3
batch_size = 100
buffer_size = 1e6
sigma = 0.3
gamma = 0.99


class DDPGActor(nn.Module):
    def __init__(self, observation_dim, action_dim, actor_lr, device):
        super(DDPGActor, self).__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.device = device

        self.fc1 = nn.Linear(observation_dim, 400).to(device)
        self.fc2 = nn.Linear(400, 300).to(device)
        self.fc3 = nn.Linear(300, action_dim).to(device)
        nn.init.uniform_(tensor=self.fc3.weight, a=-3e-3, b=3e-3)
        nn.init.uniform_(tensor=self.fc3.bias, a=-3e-3, b=3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x

class DDPGCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, critic_lr, device):
        super(DDPGCritic, self).__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.critic_lr = critic_lr
        self.device = device

        self.fc1 = nn.Linear(observation_dim + action_dim, 400).to(device)
        self.fc2 = nn.Linear(400, 300).to(device)
        self.fc3 = nn.Linear(300, 1).to(device)
        nn.init.uniform_(tensor=self.fc3.weight, a=-3e-4, b=3e-4)
        nn.init.uniform_(tensor=self.fc3.bias, a=-3e-4, b=3e-4)

        self.optimizer = optim.Adam(self.parameters(), lr=self.critic_lr)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

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

def train(actor_main, critic_main, actor_target, critic_target, replay_buffer, criterion):

    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(batch_size)
    s_batch = torch.FloatTensor(s_batch).to(device)
    a_batch = torch.FloatTensor(a_batch).to(device)
    r_batch = torch.FloatTensor(r_batch).to(device)
    s2_batch = torch.FloatTensor(s2_batch).to(device)

    with torch.no_grad():
        next_target_q = critic_target.forward(s2_batch,
                                              actor_target.forward(s2_batch)
                                              )
        y_i = r_batch.view([-1, 1]) + gamma * next_target_q

    q = critic_main.forward(s_batch, a_batch.view([-1, actor_main.action_dim]))

    loss = criterion(q, y_i)

    critic_main.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(critic_main.parameters(), 1.0)
    critic_main.optimizer.step()

    # max_q print

    actor_main.optimizer.zero_grad()
    a_out = actor_main.forward(s_batch)
    loss = -critic_main.forward(s_batch, a_out).mean()

    loss.backward()
    torch.nn.utils.clip_grad_value_(actor_main.parameters(), 1.0)
    actor_main.optimizer.step()

    soft_target_update(actor_main, actor_target, tau)
    soft_target_update(critic_main, critic_target, tau)

    return np.max(q.detach().cpu().numpy())

def evaluate(actor_main, env, control_stepsize, observation_dim, action_dim):

    timestep = env.reset()
    _, _, _, s = timestep

    step_i = 0
    ep_reward = 0
    prev_action = np.zeros([action_dim])

    s = utils.state_1d_flat(s)
    s_a = np.append(s,prev_action)
    s_a = torch.FloatTensor(s_a).to(device)

    while step_i < 1000:
        with torch.no_grad():
            a = actor_main.forward(s_a.view(-1,observation_dim)).cpu().numpy()[0]

        for _ in range(control_stepsize):
            timestep = env.step(np.reshape(prev_action,(action_dim,)))
            step_i += 1

            if step_i > 1000:
                break
        if step_i > 1000:
            break

        t, r, _, s2 = timestep
        s2 = utils.state_1d_flat(s2)

        s2_a = np.append(s2, a)
        s2_a = torch.FloatTensor(s2_a).to(device)

        s_a = s2_a
        ep_reward += r*control_stepsize # normalize episode reward approximately 1000
        prev_action = a

    return ep_reward


def save_networks(actor_main, critic_main, actor_target, critic_target, record_dir, iter_i):
    actor_main_path = os.path.join(record_dir, "actor_main_" + str(iter_i) + ".torch")
    critic_main_path = os.path.join(record_dir, "critic_main_" + str(iter_i) + ".torch")
    actor_target_path = os.path.join(record_dir, "actor_target" + str(iter_i) + ".torch")
    critic_target_path = os.path.join(record_dir, "critic_target_" + str(iter_i) + ".torch")

    torch.save(actor_main, actor_main_path)
    torch.save(critic_main, critic_main_path)
    torch.save(actor_target, actor_target_path)
    torch.save(critic_target, critic_target_path)

    print("Saved networks")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments for Experiment")

    parser.add_argument('--domain', help='which domain?')
    parser.add_argument('--task', help='which task?')
    parser.add_argument('--control_stepsize', help='how large the control stepsize?')
    parser.add_argument('--max_iteration', help="how many iterations?", default=1000000)
    parser.add_argument('--device', help="which device?", default=0)

    # arguments
    domain_name = parser.parse_args().domain
    task_name = parser.parse_args().task
    control_stepsize = int(parser.parse_args().control_stepsize)
    max_iteration = int(parser.parse_args().max_iteration)
    model_save_period = max_iteration / 10
    device_index = int(parser.parse_args().device)

    device = torch.device("cuda",device_index)

    record_dir = utils.directory_setting("../results", exp_type + "_" + domain_name, task_name, control_stepsize)

    utils.append_file_writer(record_dir, "exp_detail.txt", "exp_type : " + str(exp_type) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "framework : " + str(framework) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "actor_lr : " + str(actor_lr) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "critic_lr : " + str(critic_lr) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "tau : " + str(tau) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "batch_size : " + str(batch_size) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "buffer_size : " + str(buffer_size) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "sigma : " + str(sigma) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "gamma : " + str(gamma) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "domain_name : " + str(domain_name) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "task_name : " + str(task_name) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "control_stepsize : " + str(control_stepsize) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "max_iteration : " + str(max_iteration) + "\n")

    env = suite.load(domain_name=domain_name, task_name=task_name)

    state_dim = utils.state_1d_dim_calc(env)[-1]
    action_dim = env.action_spec().shape[-1]
    observation_dim = state_dim + action_dim

    utils.append_file_writer(record_dir, "exp_detail.txt", "state_dim : " + str(state_dim) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "action_dim : " + str(action_dim) + "\n")
    utils.append_file_writer(record_dir, "exp_detail.txt", "observation_dim : " + str(observation_dim) + "\n")

    replay_buffer = ReplayBuffer.ReplayBuffer(buffer_size=buffer_size)

    MSEcriterion = nn.MSELoss()

    actor_main = DDPGActor(observation_dim, action_dim, actor_lr, device)
    actor_target = DDPGActor(observation_dim, action_dim, actor_lr, device)
    critic_main = DDPGCritic(observation_dim, action_dim, critic_lr, device)
    critic_target = DDPGCritic(observation_dim, action_dim, critic_lr, device)

    target_initialize(actor_main, actor_target)
    target_initialize(critic_main, critic_target)

    iter_i = 0
    epi_i = 0
    save_flag = False

    while iter_i < max_iteration:

        noise = Noise.OrnsteinUhlenbeckActionNoise(mu=np.zeros([action_dim]), sigma=sigma)

        noise.reset()
        timestep = env.reset()
        ep_reward = 0.0
        prev_action = np.zeros([action_dim])

        # timestep, reward, discount, observation
        _, _, _, s = timestep

        s = utils.state_1d_flat(s)

        s_a = np.append(s,prev_action)
        s_a = torch.FloatTensor(s_a).to(device)

        step_i = 0
        while step_i < 1000:

            with torch.no_grad():
                if iter_i < 10000:
                    a = noise().reshape((1,action_dim))
                else:
                    a = actor_main.forward(s_a.view(-1, observation_dim)).cpu().numpy()
                    a = a + noise()
                a = np.clip(a[0],-1.0, 1.0)

            for _ in range(control_stepsize):
                timestep = env.step(prev_action)
                step_i += 1

                if step_i > 1000:
                    break

            if step_i > 1000:
                break

            t, r, _, s2 = timestep
            s2 = utils.state_1d_flat(s2)

            s2_a = np.append(s2, a)
            s2_a = torch.FloatTensor(s2_a).to(device)

            replay_buffer.add(s_a.cpu().numpy(), a, r, t, s2_a.cpu().numpy())

            s_a = s2_a
            ep_reward += r*control_stepsize # normalize episode reward approximately 1000
            prev_action = a

            if iter_i%model_save_period == 0:
                save_flag = True
            iter_i += 1

        for _ in range(int(1000/control_stepsize)):
            max_q = train(actor_main, critic_main, actor_target, critic_target, replay_buffer, MSEcriterion)

        max_q_from_laststep = max_q

        eval_return = evaluate(actor_main, env, control_stepsize, observation_dim, action_dim)

        rewards_str = str(epi_i) + " *** " + str(ep_reward) + " *** " \
                      + str(max_q_from_laststep) + " *** " + str(eval_return)+"\n"
        utils.append_file_writer(record_dir, "rewards.txt", rewards_str)

        if epi_i % 50 == 0:
            graph_reward.save_graph(record_dir)

        if save_flag:

            save_networks(actor_main, critic_main, actor_target, critic_target, record_dir, iter_i)
            save_flag = False

        epi_i += 1

