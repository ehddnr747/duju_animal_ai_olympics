import numpy as np
import torch
import cv2
from dm_control import suite

import lib_duju.utils as duju_utils

from Model.ReplayBuffer import ReplayBuffer
from Model.SAC_base import target_initialize

from Model.Discrete_SAC import DiscreteSAC
from Model.Discrete_SAC import train_discrete_SAC

env = suite.load(domain_name="cartpole",task_name="swingup")

state_dim = duju_utils.state_1d_dim_calc(env)[-1]
action_dim = 5

action_dict = { 0 : -1.0,
               1 : -0.5,
               2 : 0.0,
               3 : 0.5,
               4 : 1.0 }

reward_compensate = 10 # inverse alpha

lr = 1e-3
gamma = 0.99
device = torch.device("cuda")
max_episode = 10000
batch_size = 100

replay_buffer = ReplayBuffer(buffer_size=1e6)

q_main = DiscreteSAC(state_dim, action_dim, lr, device)
q_target = DiscreteSAC(state_dim, action_dim, lr, device)

target_initialize(q_main, q_target)

for epi_i in range(1, max_episode + 1):
    print(epi_i)

    timestep = env.reset()
    ep_reward = 0.0

    # timestep, reward, discount, observation
    end, _, _, s = timestep
    end = end.last()
    s = duju_utils.state_1d_flat(s)

    while not end:
        a_category = q_main.get_stochastic_action(torch.FloatTensor(s).to(device).view(1,-1))
        a_deploy = action_dict[a_category]
        timestep = env.step(a_deploy)

        end, r, _, s2 = timestep
        end = end.last()
        s2 = duju_utils.state_1d_flat(s2)

        replay_buffer.add(s, np.array([a_category]), np.array([r * reward_compensate]),np.array([end]), s2)

        frame = env.physics.render(camera_id=0, width=640, height=480)  # [height, width, channel]
        cv2.imshow("train", frame)
        cv2.waitKey(1)

        s = s2
        ep_reward += r

    for _idx in range(1000):
        #print(_idx)
        max_q1, max_q2 = train_discrete_SAC(q_main, q_target, replay_buffer, batch_size, gamma)

    print(ep_reward, "***", (max_q1, max_q2))

    timestep = env.reset()
    end, _, _, s = timestep
    end = end.last()
    s = duju_utils.state_1d_flat(s)

    eval_ep_reward = 0.0
    eval_action = []

    if (epi_i % 1) == 0 :
        while not end:
            a_category = q_main.get_max_action(torch.FloatTensor(s).to(device).view(1,-1))
            a_deploy = action_dict[a_category]
            eval_action.append(a_deploy)

            timestep = env.step(a_deploy)

            end, r, _, s2 = timestep
            end = end.last()
            s2 = duju_utils.state_1d_flat(s2)

            s = s2
            eval_ep_reward += r

            frame = env.physics.render(camera_id=0, width=640, height=480) #[height, width, channel]
            cv2.imshow("eval", frame)
            cv2.waitKey(1)


        print("Eval! *** ", eval_ep_reward)
        #print(eval_action)

cv2.destroyAllWindows()