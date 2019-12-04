from Model.ReplayBuffer import PERBuffer
from Model.ImageBuffer import ImageBuffer
from Model.rainbow_conv import Rainbow_DQN_Conv
from Model.rainbow_conv import train_rainbow_dqn_conv
from Model.SAC_base import target_initialize
import lib_duju.utils as duju_utils
import matplotlib.pyplot as plt
import sys

import torch
import cv2

from animalai.envs import UnityEnvironment
from animalai.envs import ArenaConfig
import numpy as np

import argparse
import os

parser = argparse.ArgumentParser(description="Arguments for Baseline Experiment")

parser.add_argument("--env", help="which env?")

env = UnityEnvironment(worker_id=int(parser.parse_args().env)+40,
        base_port=5005 + int(parser.parse_args().env),
        file_name='/home/duju/animal_ai_olympics/AnimalAI-Olympics/env/AnimalAI.x86_64',   # Path to the environment
)


config_dict = {
    1 : "1-Food.yaml",
    2 : "2-Preferences.yaml",
    3 : "3-Obstacles.yaml",
    4 : "4-Avoidance.yaml",
    5 : "5-SpatialReasoning.yaml",
    6 : "6-Generalization.yaml",
    7 : "7-InternalMemory.yaml"
}

env_file = config_dict[int(parser.parse_args().env)]


arena_config = ArenaConfig("/home/duju/animal_ai_olympics/duju_animal_ai_olympics/configs/"+env_file)

exp_title = "Animal_AI_Baseline_"+ env_file.split(".")[0]
print(exp_title)

exp_dir = "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/baseline_results/"+exp_title

video_dir = "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/videos/"+exp_title
#image_dir = "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/images/"+exp_title

os.makedirs(video_dir)
#os.makedirs(image_dir)

action_dict = { 0 : np.array([1,0]), # forward
                1 : np.array([0,2]), # left
                2 : np.array([0,1]), # right
#                3 : np.array([2,0]) # backward
                }

action_semantic = { 0 : "forward", 1 : "left", 2 : "right" }
#action_semantic = { 0 : "forward", 1 : "left", 2 : "right", 3: "backward" }

state_dim = (84, 84, 3)

height = state_dim[0]
width = state_dim[1]
channel = state_dim[2]

step_size = 4

input_channel_size= step_size * channel

action_dim = len(action_dict)

rb_state_dim = 1
rb_action_dim = 1

lr = 3e-5
gamma = 0.99
device = torch.device("cuda")

max_episode = 25
max_step = max_episode * 1000

batch_size = 32
buffer_size = int(1e5)

alpha = 0.6
initial_beta = 0.4

reward_scale = 10

image_buffer = ImageBuffer(height, width, step_size, channel, int(buffer_size * 1.1))

q_main = Rainbow_DQN_Conv(step_size, channel, height, width, action_dim, lr, device)

q_path_dict = {
1 : "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/baseline_results/Animal_AI_Baseline2_1-Food/Animal_AI_Baseline2_1-Food_q_main_800.torch",
2 : "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/baseline_results/Animal_AI_Baseline4_2-Preferences/Animal_AI_Baseline4_2-Preferences_q_main_800.torch",
3 : "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/baseline_results/Animal_AI_Baseline3_3-Obstacles/Animal_AI_Baseline3_3-Obstacles_q_main_400.torch",
4 : "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/baseline_results/Animal_AI_Baseline4_4-Avoidance/Animal_AI_Baseline4_4-Avoidance_q_main_500.torch",
5 : "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/baseline_results/Animal_AI_Baseline2_5-SpatialReasoning/Animal_AI_Baseline2_5-SpatialReasoning_q_main_600.torch",
6 : "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/baseline_results/Animal_AI_Baseline4_6-Generalization/Animal_AI_Baseline4_6-Generalization_q_main_700.torch",
7 : "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/baseline_results/Animal_AI_Baseline3_7-InternalMemory/Animal_AI_Baseline3_7-InternalMemory_q_main_500.torch"
}
duju_utils.torch_network_load(q_main, q_path_dict[int(parser.parse_args().env)])

train_episodic_reward = []
eval_episodic_reward = []

info = env.reset(arenas_configurations = arena_config)["Learner"]

for epi_i in range(1, max_episode + 1):
    initial_frame = info.visual_observations[0][0] #[height, width, channel]
    end = info.local_done[0]

    s_fl = []

    for _ in range(step_size):
        image_buffer.animal_add(initial_frame)

    s_idx = image_buffer.get_current_index()
    input_state = image_buffer.get_state(s_idx)

    ep_reward = 0.0
    ep_count = 0

    epsilon = 0.05

    while True:
        a_category = q_main.epsilon_sample(
            torch.FloatTensor(input_state).to(device).view(1, input_channel_size, height, width),
            epsilon
        )
        a_deploy = action_dict[a_category]

        info = env.step(a_deploy)["Learner"]

        end = info.local_done[0]

        ep_count += 1
        r = info.rewards[0] * reward_scale
        s2_frame = info.visual_observations[0][0]

        image_buffer.animal_add(s2_frame)
        s2_idx = image_buffer.get_current_index()
        input_state = image_buffer.get_state(s2_idx)

        s_idx = s2_idx
        ep_reward += r

        if end:
            break

        img_str = "R : " + str(r) + " A : " + action_semantic[a_category]
        img = cv2.putText(cv2.resize(s2_frame[:, :, [2,1,0]], (420, 420)), img_str, (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
        cv2.imshow("train", img)
        s_fl.append(np.uint8(cv2.resize(s2_frame[:, :, [2,1,0]], (420, 420))*256))
        cv2.waitKey(1)

    if ep_reward > 0:
        tag = "_success"
    else:
        tag = "_fail"

    s_video_path = os.path.join(video_dir, "s_" + str(epi_i) + tag + ".avi")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(s_video_path, fourcc, 25, (420, 420))

    for i in s_fl:
        out.write(i)
    out.release()

    train_episodic_reward.append(ep_reward)


cv2.destroyAllWindows()
env.close()
