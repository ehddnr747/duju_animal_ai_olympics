from Model.ReplayBuffer import ReplayBuffer as replaybuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

from dm_control import suite
import numpy as np

env = suite.load(domain_name="cartpole",task_name="swingup")

for _ in range(1000):
    env.step(1)
    frame = env.physics.render(camera_id=0,width=320,height=240)
    cv2.imshow("test",frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()

