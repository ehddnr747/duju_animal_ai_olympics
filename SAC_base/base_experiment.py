from Model.ReplayBuffer import ReplayBuffer
from Model.policy_net import VanillaSACPolicy
from Model.q_net import VanillaSACQNet
from Model.v_net import VanillaSACValue
from Model.SAC_base import target_initialize
from Model.SAC_base import train
import lib_duju.utils as duju_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

from dm_control import suite
import numpy as np

env = suite.load(domain_name="cartpole",task_name="swingup")

state_dim = duju_utils.state_1d_dim_calc(env)[-1]
action_dim = env.action_spec().shape[-1]

lr = 1e-3
device = torch.device("cuda")


replay_buffer = ReplayBuffer(buffer_size=1e6)

policy = VanillaSACPolicy(state_dim, action_dim, lr, device)
QNet = VanillaSACQNet(state_dim, action_dim, lr, device)
VNet_main = VanillaSACValue(state_dim, lr, device)
VNet_target = VanillaSACValue(state_dim, lr, device)

target_initialize(VNet_main, VNet_target)

max_episode = 1000

for epi_i in range(1, max_episode + 1):

    timestep = env.reset()
    ep_reward = 0.0

    # timestep, reward, discount, observation
    end, _, _, s = timestep
    end = end.last()
    s = duju_utils.state_1d_flat(s)

    while not end:
        a = policy.sample(torch.FloatTensor(s).to(device).view(1,-1)).cpu().numpy()[0]
        timestep = env.step(a)

        end, r, _, s2 = timestep
        end = end.last()
        s2 = duju_utils.state_1d_flat(s2)

        replay_buffer.add(s, a, np.array([r]),np.array([end]), s2)

        s = s2
        ep_reward += r

    for _idx in range(1000):
        #print(_idx)
        max_v = train(policy, QNet, VNet_main, VNet_target, replay_buffer, batch_size=128, alpha=0.05, gamma=0.99)

    print(ep_reward, "***", max_v)

    timestep = env.reset()
    end, _, _, s = timestep
    end = end.last()
    s = duju_utils.state_1d_flat(s)

    eval_ep_reward = 0.0

    if (epi_i % 1) == 0 :
        while not end:
            a = policy.mean_action(torch.FloatTensor(s).to(device).view(1, -1)).cpu().numpy()[0]
            timestep = env.step(a)

            end, r, _, s2 = timestep
            end = end.last()
            s2 = duju_utils.state_1d_flat(s2)

            s = s2
            eval_ep_reward += r

            frame = env.physics.render(camera_id=0, width=640, height=480) #[height, width, channel]
            cv2.imshow("test", frame)
            cv2.waitKey(1)


        print("Eval! *** ", eval_ep_reward)

cv2.destroyAllWindows()

