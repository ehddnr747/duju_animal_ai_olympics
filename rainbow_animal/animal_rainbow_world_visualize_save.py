from Model.WorldReplayBuffer import PERBuffer
from Model.WorldReplayBuffer import ReplayBuffer
from Model.ImageBuffer import ImageBuffer
from Model.rainbow import Rainbow_DQN
from Model.rainbow import train_rainbow_dqn
from Model.SAC_base import target_initialize
import lib_duju.utils as duju_utils
import matplotlib.pyplot as plt
import sys
import time

import torch
import cv2

from animalai.envs import UnityEnvironment
from animalai.envs import ArenaConfig
import numpy as np

import argparse
import os

from Model.VisionModel import  VisionModel
from Model.MemoryModel import MemoryModel

parser = argparse.ArgumentParser(description="Arguments for World Experiment")

parser.add_argument("--env", help="which env?")

env = UnityEnvironment(worker_id=int(parser.parse_args().env)+10,
        base_port=5005 + int(parser.parse_args().env)+10,
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

exp_title = "Animal_AI_World_Proposed_"+ env_file.split(".")[0]
print(exp_title)

exp_dir = "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/proposed_results/"+exp_title

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

dim_z = 256
hidden_size = 256

step_size = 4

input_channel_size= step_size * channel

action_dim = len(action_dict)

rb_state_dim = 1
rb_action_dim = 1
control_state_dim = dim_z + hidden_size * 2

lr = 1e-4
gamma = 0.99
device = torch.device("cuda")

max_episode = 25
max_step = max_episode * 1000

model_data_collection_episode = 250

batch_size = 16
buffer_size = int(5e5)

alpha = 0.6
initial_beta = 0.4

reward_scale = 10

vision_model = VisionModel(channel, height, width, dim_z,lr, device)
memory_model = MemoryModel(dim_z, hidden_size, lr, device)

q_main = Rainbow_DQN(control_state_dim, action_dim, lr, device)

train_episodic_reward = []
eval_episodic_reward = []


duju_utils.torch_network_load(vision_model, exp_dir + "/" + exp_title + "_vision_50000.torch")
duju_utils.torch_network_load(memory_model, exp_dir + "/" + exp_title + "_memory_25000.torch")
duju_utils.torch_network_load(q_main, exp_dir + "/" + exp_title + "_q_main_1000.torch")


info = env.reset(arenas_configurations = arena_config)["Learner"]
for epi_i in range(1, max_episode + 1):
    s1_frame = info.visual_observations[0][0] #[height, width, channel]
    end = info.local_done[0]

    ep_reward = 0.0
    ep_count = 0

    epsilon = 0.05

    s1_fl = []
    s2_fl = []
    recon_fl = []
    next_recon_fl = []


    with torch.no_grad():

        h = torch.zeros([1, 1, hidden_size]).to(device)
        c = torch.zeros([1, 1, hidden_size]).to(device)

        current_image = np.moveaxis(s1_frame, [0, 1, 2], [1, 2, 0])

        mus, sigmas = vision_model.encode(torch.FloatTensor(current_image).to(device).view(1, channel, height, width))
        z = vision_model.reparameterization(mus, sigmas)  # [1, dim_z]

        s = torch.cat([z, h.view([1, hidden_size]), c.view(1, hidden_size)], dim=1)  # [1, state_dim]

    while True:
        with torch.no_grad():
            a_category = q_main.epsilon_sample(
                s,
                epsilon
            )
            #a_category = 1
            a_deploy = action_dict[a_category]

            a_one_hot = memory_model.one_hot([a_category], 3)
            a_one_hot = torch.FloatTensor(a_one_hot).view([1,1,3]).to(device)

            z_in = z.view([1,1, dim_z])

            h0, c0 = h, c
            z0 = z

            h, c = memory_model.hidden_forward(z_in, a_one_hot, h, c)

            info = env.step(a_deploy)["Learner"]

            end = info.local_done[0]

            ep_count += 1
            r = info.rewards[0] * reward_scale
            s2_frame = info.visual_observations[0][0]

            current_image = np.moveaxis(s2_frame, [0, 1, 2], [1, 2, 0])

            mus, sigmas = vision_model.encode(torch.FloatTensor(current_image).to(device).view(1, channel, height, width))
            z = vision_model.reparameterization(mus, sigmas)  # [1, dim_z]

            s2 = torch.cat([z, h.view([1, hidden_size]), c.view(1, hidden_size)], dim=1)  # [1, state_dim]

            s = s2
            ep_reward += r

            if end:
                break

            # for visualization
            if True:#epi_i % 25 == 0:
                recon = vision_model.decode(z0)[0].detach().cpu().numpy()
                recon = np.moveaxis(recon, [0,1,2],[2,0,1])

                z_out, _, _, _, _, _ = memory_model.full_forward(z_in, a_one_hot, h0, c0)
                z_out = z_out.view([1, dim_z])

                next_recon = vision_model.decode(z_out)[0].detach().cpu().numpy()
                next_recon = np.moveaxis(next_recon, [0, 1, 2], [2, 0, 1])

                img_str = "R : " + str(r) + " A : " + action_semantic[a_category]
                img = cv2.putText(cv2.resize(s1_frame[:, :, [2,1,0]], (420, 420)), img_str, (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
                cv2.imshow("train", img)
                s1_fl.append(np.uint8(cv2.resize(s1_frame[:, :, [2,1,0]], (420, 420))*256))
                cv2.waitKey(1)

                cv2.imshow("recon", cv2.resize(recon[:, :, [2,1,0]], (420, 420)))
                recon_fl.append(np.uint8(cv2.resize(recon[:, :, [2,1,0]], (420, 420))*256))
                cv2.waitKey(1)

                cv2.imshow("s2_frame", cv2.resize(s2_frame[:, :, [2, 1, 0]], (420, 420)))
                s2_fl.append(np.uint8(cv2.resize(s2_frame[:, :, [2,1,0]], (420, 420))*256))
                cv2.waitKey(1)

                cv2.imshow("next_recon", cv2.resize(next_recon[:, :, [2, 1, 0]], (420, 420)))
                next_recon_fl.append(np.uint8(cv2.resize(next_recon[:, :, [2,1,0]], (420, 420))*256))
                cv2.waitKey(1)

            s1_frame = s2_frame

    if ep_reward > 0:
        tag = "_success"
    else:
        tag = "_fail"

    s1_video_path = os.path.join(video_dir,"s1_"+str(epi_i) + tag +".avi")
    s2_video_path = os.path.join(video_dir,"s2_"+ str(epi_i) + tag + ".avi")
    recon_video_path = os.path.join(video_dir, "recon_"+str(epi_i) + tag + ".avi")
    next_recon_video_path = os.path.join(video_dir, "next_recon_"+  tag + str(epi_i) + ".avi")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(s1_video_path, fourcc, 25, (420, 420))

    for i in s1_fl:
        out.write(i)
    out.release()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(s2_video_path, fourcc, 25, (420, 420))

    for i in s2_fl:
        out.write(i)
    out.release()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(recon_video_path, fourcc, 25, (420, 420))

    for i in recon_fl:
        out.write(i)
    out.release()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(next_recon_video_path, fourcc, 25, (420, 420))

    for i in next_recon_fl:
        out.write(i)
    out.release()


cv2.destroyAllWindows()
env.close()
