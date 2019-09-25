from Model.ReplayBuffer import ReplayBuffer
from Model.vanilla_policy_net import VanillaSACPolicy
from Model.vanilla_q_net import VanillaSACQNet
from Model.vanilla_v_net import VanillaSACValue
from Model.SAC_base import target_initialize
from Model.SAC_base import soft_target_update
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

lr = 3e-4
device = torch.device("cuda")


replay_buffer = ReplayBuffer(buffer_size=1e6)

policy = VanillaSACPolicy(state_dim, action_dim, lr, device)
QNet = VanillaSACQNet(state_dim, action_dim, lr, device)
VNet = VanillaSACValue(state_dim, lr, device)



for _ in range(1000):
    env.step(1)
    frame = env.physics.render(camera_id=0,width=320,height=240)
    cv2.imshow("test",frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()

