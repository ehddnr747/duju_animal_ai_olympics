from Model.ReplayBuffer import ReplayBuffer
from Model.policy_net import ConvSACPolicy
from Model.q_net import ConvSACQNet
from Model.v_net import ConvSACValue
from Model.SAC_base import target_initialize
from Model.SAC_base import conv_train
import lib_duju.utils as duju_utils
from Model.ImageBuffer import ImageBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

from dm_control import suite
import numpy as np

env = suite.load(domain_name="cartpole",task_name="swingup")

state_dim = duju_utils.state_1d_dim_calc(env)[-1]

stepsize = 5
channelsize = 3

step_channelsize = stepsize * channelsize

height = 48
width = 64

action_dim = env.action_spec().shape[-1]

buffer_size = 2e5
batch_size = 64

lr = 1e-3
device = torch.device("cuda")

replay_buffer = ReplayBuffer(buffer_size)
image_buffer = ImageBuffer(height,width, stepsize, channelsize, int(buffer_size*1.01))
eval_image_buffer = ImageBuffer(height, width, stepsize, channelsize, 1001)

policy = ConvSACPolicy(step_channelsize, height, width, action_dim, lr, device)
QNet = ConvSACQNet(step_channelsize, height, width, action_dim, lr, device)
VNet_main = ConvSACValue(step_channelsize, height, width, lr, device)
VNet_target = ConvSACValue(step_channelsize, height, width, lr, device)

target_initialize(VNet_main, VNet_target)

max_episode = 10000
image_idx = 0
eval_image_idx = 0

### Image Idx starts from 1

for epi_i in range(1, max_episode + 1):

    print(epi_i)

    timestep = env.reset()
    ep_reward = 0.0

    # timestep, reward, discount, observation
    end, _, _, _ = timestep
    image_buffer.dm_add(env.physics.render(camera_id=0, width=64, height=48))
    image_idx += 1
    end = end.last()

    s = image_buffer.get_state(image_idx)

    while not end:
        a = policy.sample(torch.FloatTensor(s).to(device).view(1,step_channelsize,height,width)).cpu().numpy()[0]
        timestep = env.step(a)

        end, r, _, _ = timestep
        image_buffer.dm_add(env.physics.render(camera_id=0, width=64, height=48))
        image_idx += 1
        end = end.last()
        s2 = image_buffer.get_state(image_idx)

        replay_buffer.add(image_idx - 1, a, np.array([r]),np.array([end]), image_idx)

        s = s2
        ep_reward += r

    for _idx in range(500):
        #print(_idx)
        max_v = conv_train(policy, QNet, VNet_main, VNet_target, replay_buffer, image_buffer, batch_size, alpha=0.2, gamma=0.99)

    print(ep_reward, "***", max_v)

    if (epi_i % 10) == 0 :

        timestep = env.reset()
        end, _, _, _ = timestep
        eval_image_buffer.dm_add(env.physics.render(camera_id=0, width=64, height=48))
        eval_image_idx += 1
        end = end.last()
        s = eval_image_buffer.get_state(eval_image_idx)

        eval_ep_reward = 0.0

        while not end:
            a = policy.mean_action(torch.FloatTensor(s).to(device).view(1,step_channelsize,height,width)).cpu().numpy()[0]
            timestep = env.step(a)

            end, r, _, s2 = timestep
            eval_image_buffer.dm_add(env.physics.render(camera_id=0, width=64, height=48))
            eval_image_idx += 1
            end = end.last()

            s2 = eval_image_buffer.get_state(eval_image_idx)

            s = s2
            eval_ep_reward += r

            frame = env.physics.render(camera_id=0, width=640, height=480) #[height, width, channel]
            cv2.imshow("test", frame)
            cv2.waitKey(1)


        print("Eval! *** ", eval_ep_reward)

cv2.destroyAllWindows()

