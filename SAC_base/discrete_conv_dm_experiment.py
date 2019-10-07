import numpy as np
import torch
import cv2
from dm_control import suite

import lib_duju.utils as duju_utils

from Model.ReplayBuffer import ReplayBuffer
from Model.FrameBuffer import FrameBuffer
from Model.SAC_base import target_initialize

from Model.DiscreteConv_SAC import DiscreteConvSAC
from Model.DiscreteConv_SAC import train_discrete_Conv_SAC

exp_title = "Conv_Discrete_SAC_black_and_white_skipframe"
print(exp_title)

env = suite.load(domain_name="cartpole",task_name="swingup")

action_dim = 3

# state related variables
step_size = 4
channel_size = 1
height = 48
width = 64
skip_frame = 4

input_channel_size = step_size * channel_size

action_dict = { 0 : -1.0,
               1 : 0.0,
               2 : 1.0 }

reward_compensate = 10 # inverse alpha

lr = 3e-4
gamma = 0.99
device = torch.device("cuda")
max_episode = 10000
batch_size = 100
buffer_size = 5e4

replay_buffer = ReplayBuffer(buffer_size)
frame_buffer = FrameBuffer(step_size, channel_size, height, width)

q_main = DiscreteConvSAC(step_size, channel_size, height, width, action_dim, lr, device)
q_target = DiscreteConvSAC(step_size, channel_size, height, width, action_dim, lr, device)

target_initialize(q_main, q_target)

for epi_i in range(1, max_episode + 1):
    print(epi_i, end = "\t")

    timestep = env.reset()
    ep_reward = 0.0

    # timestep, reward, discount, observation
    end, _, _, _ = timestep
    end = end.last()

    frame = env.physics.render(camera_id=0, height = 48, width = 64)
    for _ in range(step_size):
        frame_buffer.dm_add(frame)
    s = frame_buffer.get_buffer()

    while not end:
        a_category = q_main.get_stochastic_action(
                        torch.FloatTensor(s).to(device).view(1, input_channel_size, height, width)
                                                  )
        for _ in range(skip_frame):
            a_deploy = action_dict[a_category]
        timestep = env.step(a_deploy)

        end, r, _, _ = timestep
        end = end.last()
        frame = env.physics.render(camera_id=0, height=48, width=64)
        frame_buffer.dm_add(frame)

        s2 = frame_buffer.get_buffer()

        replay_buffer.add(s, np.array([a_category]), np.array([r * reward_compensate]),np.array([end]), s2)

        frame = env.physics.render(camera_id=0, height=480, width=640)  # [height, width, channel]
        cv2.imshow("train", frame)
        cv2.waitKey(1)

        s = s2
        ep_reward += r

    for _idx in range(1000):
        #print(_idx)
        max_q1, max_q2 = train_discrete_Conv_SAC(q_main, q_target, replay_buffer, batch_size, gamma)

    print(ep_reward, "***", (float(max_q1), float(max_q2)))

    #### Eval ####

    timestep = env.reset()
    eval_ep_reward = 0.0
    eval_action = []

    end, _, _, _ = timestep
    end = end.last()

    frame = env.physics.render(camera_id=0, height=48, width=64)
    for _ in range(step_size):
        frame_buffer.dm_add(frame)
    s = frame_buffer.get_buffer()

    if (epi_i % 1) == 0 :
        while not end:
            a_category = q_main.get_max_action(
                        torch.FloatTensor(s).to(device).view(1, input_channel_size, height, width)
                                                  )
            a_deploy = action_dict[a_category]
            for _ in range(skip_frame):
                eval_action.append(a_deploy)

            timestep = env.step(a_deploy)

            end, r, _, _ = timestep
            end = end.last()
            frame = env.physics.render(camera_id=0, height=48, width=64)
            frame_buffer.dm_add(frame)

            s2 = frame_buffer.get_buffer()

            s = s2
            eval_ep_reward += r

            frame = env.physics.render(camera_id=0, height=480, width=640) #[height, width, channel]
            cv2.imshow("eval", frame)
            cv2.waitKey(1)


        print("Eval! *** ", eval_ep_reward)
        #print(eval_action)

    if (epi_i % 10) == 0:
        print("Networks Saved!")
        duju_utils.torch_network_save(q_main,"../trained/"+exp_title+"_q_main_"+str(epi_i)+".torch")
        duju_utils.torch_network_save(q_target, "../trained/"+exp_title+"_q_target_"+str(epi_i)+".torch")

cv2.destroyAllWindows()