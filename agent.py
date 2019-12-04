### This is for submission ###

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from numpy.random import binomial
from numpy.random import multinomial


def torch_network_load(net, path, device=torch.device("cpu")):
    load_dict = torch.load(path, device)
    parameters = load_dict["model_state_dict"]
    optimizer = load_dict["optimizer_state_dict"]

    net.load_state_dict(parameters)
    net.optimizer.load_state_dict(optimizer)


class Rainbow_DQN_Conv(nn.Module):
    def __init__(self, step_size, channel_size, height, width, action_dim, lr, device):
        super(Rainbow_DQN_Conv, self).__init__()

        self.step_size = step_size
        self.channel_size = channel_size
        self.height = height
        self.width = width
        self.action_dim = action_dim
        self.lr = lr
        self.device = device

        self.input_channel_size = self.step_size * self.channel_size
        self.fc_input_size = 32 * 9

        self.uniform_probs = [1.0 / self.action_dim] * self.action_dim

        # Q1
        self.q1_conv1 = nn.Conv2d(in_channels=self.input_channel_size, out_channels=32,
                                  kernel_size=4, stride=2, padding=0).to(device)  # [32, 41, 41]
        self.q1_conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                                  kernel_size=5, stride=2, padding=2).to(device)  # [32, 21, 21]
        self.q1_conv3 = nn.Conv2d(in_channels=32, out_channels=32,
                                  kernel_size=7, stride=7, padding=0).to(device)  # [32, 3, 3]
        # self.q1_conv4 = nn.Conv2d(in_channels=64, out_channels=128,
        #                           kernel_size=3, stride=1, padding=0).to(device)  # [128, 1, 1]

        self.q1_fc1 = nn.Linear(self.fc_input_size, 128).to(device)
        self.q1_fc2_V = nn.Linear(128, 128).to(device)
        self.q1_fc2_A = nn.Linear(128, 128).to(device)

        self.q1_V = nn.Linear(128, 1).to(device)
        self.q1_A = nn.Linear(128, action_dim).to(device)

        # Q2
        self.q2_conv1 = nn.Conv2d(in_channels=self.input_channel_size, out_channels=32,
                                  kernel_size=4, stride=2, padding=0).to(device)  # [32, 41, 41]
        self.q2_conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                                  kernel_size=5, stride=2, padding=2).to(device)  # [32, 21, 21]
        self.q2_conv3 = nn.Conv2d(in_channels=32, out_channels=32,
                                  kernel_size=7, stride=7, padding=0).to(device)  # [64, 3, 3]
        # self.q2_conv4 = nn.Conv2d(in_channels=64, out_channels=128,
        #                           kernel_size=3, stride=1, padding=0).to(device)  # [128, 1, 1]

        self.q2_fc1 = nn.Linear(self.fc_input_size, 128).to(device)
        self.q2_fc2_V = nn.Linear(128, 128).to(device)
        self.q2_fc2_A = nn.Linear(128, 128).to(device)

        self.q2_V = nn.Linear(128, 1).to(device)
        self.q2_A = nn.Linear(128, action_dim).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr, weight_decay=1e-6)

        self.milestones = [100000000]
        self.step_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.5)

    def forward(self, x):
        assert len(x.shape) == 4

        x1 = F.celu(self.q1_conv1(x))
        x1 = F.celu(self.q1_conv2(x1))
        x1 = F.celu(self.q1_conv3(x1))
        # x1 = F.celu(self.q1_conv4(x1))

        x1 = x1.view(-1, self.fc_input_size)

        x1 = F.celu(self.q1_fc1(x1))
        x1_V = F.celu(self.q1_fc2_V(x1))
        x1_A = F.celu(self.q1_fc2_A(x1))

        x1_V = self.q1_V(x1_V)
        x1_A = self.q1_A(x1_A)

        x1_A_mean = torch.mean(x1_A, dim=1, keepdim=True)
        x1_A_mean = torch.cat([x1_A_mean] * self.action_dim, dim=1)

        x1_A = x1_A - x1_A_mean

        x1 = x1_V + x1_A

        x2 = F.celu(self.q2_conv1(x))
        x2 = F.celu(self.q2_conv2(x2))
        x2 = F.celu(self.q2_conv3(x2))
        # x2 = F.celu(self.q2_conv4(x2))

        x2 = x2.view(-1, self.fc_input_size)

        x2 = F.celu(self.q2_fc1(x2))
        x2_V = F.celu(self.q2_fc2_V(x2))
        x2_A = F.celu(self.q2_fc2_A(x2))

        x2_V = self.q2_V(x2_V)
        x2_A = self.q2_A(x2_A)

        x2_A_mean = torch.mean(x2_A, dim=1, keepdim=True)
        x2_A_mean = torch.cat([x2_A_mean] * self.action_dim, dim=1)

        x2_A = x2_A - x2_A_mean

        x2 = x2_V + x2_A

        return x1, x2

    def epsilon_sample(self, x, epsilon):
        exploration_flag = bool(binomial(1, epsilon))

        if exploration_flag:
            action = np.argmax(multinomial(1, self.uniform_probs))
            return action

        else:
            q1, q2 = self.forward(x)
            Q = q1 + q2

            action = torch.argmax(Q, dim=1)
            assert action.shape == (1,)

            return action.detach().cpu().numpy()[0]

class ImageBuffer(object):

    # idx starts from 1. The first image will have idx 1.
    # When the buffer get full with batch size 100, then there will be 100 iamges and the current idx will be 100.

    def __init__(self, height, width, stepsize, channel_size, buffer_size):
        self.height = height
        self.width = width
        self.stepsize = stepsize
        self.channel_size = channel_size
        self.step_channelsize = stepsize * channel_size

        self.buffer = []
        self.buffer_size = buffer_size
        self.count = 0

        assert self.stepsize <= self.buffer_size

        self.full_count = 0

    # # [height, width, channel]
    # def dm_add(self, frame):
    #     frame = np.moveaxis(frame, [0, 1, 2], [1, 2, 0])
    #     frame = (frame / 128.0) - 1.0
    #
    #     if self.count < self.buffer_size:
    #         self.count += 1
    #         self.buffer.append(frame)
    #     else:
    #         self.buffer.pop(0)
    #         self.buffer.append(frame)
    #         self.full_count += 1

    # [height, width, channel]
    def dm_add_gray(self, frame):
        frame = frame / 256.0
        frame = frame[:, :, [0]] * 0.2989 + frame[:, :, [1]] * 0.5870 + frame[:, :, [2]] * 0.1140
        frame = np.moveaxis(frame, [0, 1, 2], [1, 2, 0])  # [1, height, width]

        assert frame.shape == (1, self.height, self.width)

        if self.count < self.buffer_size:
            self.count += 1
            self.buffer.append(frame)
        else:
            self.buffer.pop(0)
            self.buffer.append(frame)
            self.full_count += 1

    def animal_add(self, frame):

        # store as np.ubyte

        frame = np.moveaxis(frame, [0, 1, 2], [1, 2, 0]) * 256
        frame = np.array(frame, dtype=np.ubyte)

        assert frame.shape == (self.channel_size, self.height, self.width)

        if self.count < self.buffer_size:
            self.count += 1
            self.buffer.append(frame)
        else:
            self.buffer.pop(0)
            self.buffer.append(frame)
            self.full_count += 1

    def get_state(self, idx):
        assert idx > self.full_count
        assert idx <= self.count + self.full_count

        if self.count < self.buffer_size:
            temp_idx = idx
        else:
            temp_idx = idx - self.full_count

        return_array = np.concatenate(self.buffer[temp_idx - self.stepsize: temp_idx], axis=0)
        # because image idx starts from 1 and list idx starts from 0

        assert return_array.shape == (self.step_channelsize, self.height, self.width)

        return return_array / np.array(256, dtype=np.float32)
        # [stepsize, height, width]

    def get_state_and_next(self, idx):

        assert idx > 0 and idx < self.count + self.full_count

        return self.get_state(idx), self.get_state(idx + 1)

    def get_current_index(self):
        return self.count + self.full_count


class Agent(object):

    def __init__(self):

        self.action_dict = {0: np.array([1, 0]),  # forward
                            1: np.array([0, 2]),  # left
                            2: np.array([0, 1]),  # right
                            }
        self.net = Rainbow_DQN_Conv(4, 3, 84, 84, 3, 3e-5, torch.device("cpu"))

        torch_network_load(self.net, "/aaio/data/nets.torch")


        self.image_buffer = ImageBuffer(84, 84, 4, 3, 2000)
        self.start_flag = True

    def reset(self, t=250):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param t the number of timesteps in the episode
        """
        self.start_flag = True

    def step(self, obs, reward, done, info):
        """
        :param obs: agent's observation of the current environment
        :param reward: amount of reward returned after previous action
        :param done: whether the episode has ended.
        :param info: contains auxiliary diagnostic information, including BrainInfo.
        :return: the action to take, a list or size 2
        """

        info = info["brain_info"]

        if self.start_flag:
            self.start_flag = False

            initial_frame = info.visual_observations[0][0]  # [height, width, channel]

            for _ in range(4):
                self.image_buffer.animal_add(initial_frame)

            s_idx = self.image_buffer.get_current_index()
            input_state = self.image_buffer.get_state(s_idx)

        else:
            s_frame = info.visual_observations[0][0]  # [height, width, channel]
            self.image_buffer.animal_add(s_frame)
            s_idx = self.image_buffer.get_current_index()
            input_state = self.image_buffer.get_state(s_idx)

        a_category = self.net.epsilon_sample(
            torch.FloatTensor(input_state).view(1, 4 * 3, 84, 84),
            0.0  # epsilon
        )

        action = self.action_dict[a_category]

        return action