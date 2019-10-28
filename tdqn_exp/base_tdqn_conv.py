import numpy as np
import torch
import cv2
from dm_control import suite

import lib_duju.utils as duju_utils

from Model.ReplayBuffer import ReplayBuffer
from Model.ImageBuffer import ImageBuffer
from Model.SAC_base import target_initialize

from Model.triple_dqn import Triple_DQN
from Model.triple_dqn import train_triple_dqn

exp_title = "Conv_Discrete_TDQN_celu_larger networks"
print(exp_title)

train_print_flag = True
eval_print_flag = True

load_flag = False

env = suite.load(domain_name="cartpole",task_name="swingup")

action_dim = 3

# state related variables
step_size = 3
channel_size = 1
height = 48
width = 64
skip_frame = 4

input_channel_size = step_size * channel_size

rb_state_dim = 1
rb_action_dim = 1

action_dict = { 0 : -1.0,
                1 : 1.0,
                2 : 0.0}

reward_compensate = 1

print("reward_compensate", reward_compensate)
print("skip_frame", skip_frame)

lr = 5e-6
gamma = 0.99
device = torch.device("cuda")
max_episode = 10000
batch_size = 32
buffer_size = int(5e5)

print("lr : ", lr)


replay_buffer = ReplayBuffer(rb_state_dim, rb_action_dim, buffer_size)
image_buffer = ImageBuffer(height, width, step_size, channel_size, int(buffer_size * 1.1))
eval_image_buffer = ImageBuffer(height, width, step_size, channel_size, 2000)

q_main = Triple_DQN(step_size, channel_size, height, width, action_dim, lr, device)
q_target = Triple_DQN(step_size, channel_size, height, width, action_dim, lr, device)

target_initialize(q_main, q_target)

print(q_main)

if load_flag:
    duju_utils.torch_network_load(q_main, "../trained/Conv_Discrete_TDQN_celu_larger networks_q_main_250.torch")
    duju_utils.torch_network_load(q_target, "../trained/Conv_Discrete_TDQN_celu_larger networks_q_target_250.torch")


for epi_i in range(1, max_episode + 1):
    print(epi_i, end = "\t")

    timestep = env.reset()
    ep_reward = 0.0

    # timestep, reward, discount, observation
    end, _, _, _ = timestep
    end = end.last()

    frame = env.physics.render(camera_id=0, height = height, width =width)
    for _ in range(step_size):
        image_buffer.dm_add_gray(frame)
    s_idx = image_buffer.get_current_index()
    s_frame = image_buffer.get_state(s_idx)

    epsilon = np.maximum((1- epi_i/100.0)/2.0, np.random.sample() * 0.1) # mean 0.05 min 0.0 max 0.1

    while not end:
        a_category = q_main.epsilon_sample(
                        torch.FloatTensor(s_frame).to(device).view(1, input_channel_size, height, width),
                        epsilon
                )
        a_deploy = action_dict[a_category]

        for _ in range(skip_frame):
            timestep = env.step(a_deploy)

        end, r, _, _ = timestep
        end = end.last()
        frame = env.physics.render(camera_id=0, height=height, width=width)
        image_buffer.dm_add_gray(frame)

        s2_idx = image_buffer.get_current_index()
        s2_frame = image_buffer.get_state(s2_idx)

        replay_buffer.add(  np.array([s_idx]),
                          np.array([a_category]),
                          np.array([r * reward_compensate]),
                          np.array([end]),
                          np.array([s2_idx])    )

        #frame = env.physics.render(camera_id=0, height=480, width=640)  # [height, width, channel]

        if train_print_flag:
            cv2.imshow("train", cv2.resize(np.moveaxis(s2_frame,[0,1,2],[2,0,1]),(width*4,height*4)))
            #cv2.imshow("train", frame)
            cv2.waitKey(1)

        s_idx = s2_idx
        s_frame = s2_frame

        ep_reward += r * skip_frame

    for _idx in range(int(1000)):
        #print(_idx)
            mean_q1, min_q1, max_q1, mean_q2, mean_reward = train_triple_dqn(q_main, q_target, replay_buffer, image_buffer, batch_size, gamma)

    print(int(ep_reward), "***", (float(mean_q1), float(min_q1), float(max_q1), float(mean_q2), float(mean_reward)))

    #### Eval ####

    timestep = env.reset()
    eval_ep_reward = 0.0
    eval_action = []

    end, _, _, _ = timestep
    end = end.last()

    frame = env.physics.render(camera_id=0, height=height, width=width)
    for _ in range(step_size):
        eval_image_buffer.dm_add_gray(frame)
    s_idx = eval_image_buffer.get_current_index()
    s_frame = eval_image_buffer.get_state(s_idx)

    if (epi_i % 25) == 0 :
        while not end:
            a_category = q_main.epsilon_sample(
                        torch.FloatTensor(s_frame).to(device).view(1, input_channel_size, height, width),
                        0.0
                                                  )
            a_deploy = action_dict[a_category]
            eval_action.append(a_deploy)

            for _ in range(skip_frame):
                timestep = env.step(a_deploy)

            end, r, _, _ = timestep
            end = end.last()
            frame = env.physics.render(camera_id=0, height=height, width=width)
            eval_image_buffer.dm_add_gray(frame)

            s2_idx = eval_image_buffer.get_current_index()
            s2_frame = eval_image_buffer.get_state(s2_idx)

            s_idx = s2_idx
            s_frame = s2_frame

            eval_ep_reward += r * skip_frame

            # frame = env.physics.render(camera_id=0, height=480, width=640) #[height, width, channel]
            if eval_print_flag:
                cv2.imshow("eval", cv2.resize(np.moveaxis(s2_frame,[0,1,2],[2,0,1]),(width*8,height*8)))
                cv2.waitKey(1)


        print("Eval! *** ", eval_ep_reward, " *** lr *** ", q_main.optimizer.state_dict()["param_groups"][0]["lr"])
        #print(eval_action)

    if (epi_i % 25) == 0:
        print("Networks Saved!")
        duju_utils.torch_network_save(q_main,"../trained/"+exp_title+"_q_main_"+str(epi_i)+".torch")
        duju_utils.torch_network_save(q_target, "../trained/"+exp_title+"_q_target_"+str(epi_i)+".torch")

cv2.destroyAllWindows()