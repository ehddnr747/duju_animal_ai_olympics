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

env = UnityEnvironment(worker_id=int(parser.parse_args().env),
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

exp_title = "Animal_AI_World_Proposed_"+ env_file.split(".")[0]
print(exp_title)

exp_dir = "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/proposed_results/"+exp_title
os.makedirs(exp_dir)
txt_path = os.path.join(exp_dir,"results.txt")


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

max_episode = 1000
max_step = max_episode * 1000

model_data_collection_episode = 250

batch_size = 16
buffer_size = int(5e5)

alpha = 0.6
initial_beta = 0.4

reward_scale = 10

model_replay_buffer = ReplayBuffer(rb_state_dim, rb_action_dim, buffer_size)
control_replay_buffer = PERBuffer(control_state_dim, rb_action_dim, buffer_size, alpha, initial_beta, max_step)
image_buffer = ImageBuffer(height, width, step_size, channel, int(buffer_size * 1.1))

vision_model = VisionModel(channel, height, width, dim_z,lr, device)
memory_model = MemoryModel(dim_z, hidden_size, lr, device)

q_main = Rainbow_DQN(control_state_dim, action_dim, lr, device)
q_target = Rainbow_DQN(control_state_dim, action_dim, lr, device)

# duju_utils.torch_network_load(memory_model, "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/baseline_results/Animal_AI_World_Test2-Preferences/Animal_AI_World_Test2-Preferences_memory_250.torch")
# duju_utils.torch_network_load(vision_model, "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/baseline_results/Animal_AI_World_Test2-Preferences/Animal_AI_World_Test2-Preferences_vision_250.torch")

target_initialize(q_main, q_target)

with open(txt_path, "a") as f:
    f.write(str(q_main)+"\n")

with open(txt_path, "a") as f:
    f.write(str(vision_model)+"\n")

with open(txt_path, "a") as f:
    f.write(str(memory_model)+"\n")


train_episodic_reward = []
eval_episodic_reward = []

# Left Data collect
info = env.reset(arenas_configurations = arena_config)["Learner"]

for epi_i in range(1, int(model_data_collection_episode + 1)):
    initial_frame = info.visual_observations[0][0] #[height, width, channel]
    end = info.local_done[0]

    image_buffer.animal_add(initial_frame)

    s_idx = image_buffer.get_current_index()

    ep_reward = 0.0
    ep_count = 0

    h = torch.zeros([1, 1, hidden_size]).to(device)
    c = torch.zeros([1, 1, hidden_size]).to(device)

    while True:

        a_category = 1
        a_deploy = action_dict[a_category]

        info = env.step(a_deploy)["Learner"]

        end = info.local_done[0]

        ep_count += 1
        r = info.rewards[0] * reward_scale
        s2_frame = info.visual_observations[0][0]

        image_buffer.animal_add(s2_frame)
        s2_idx = image_buffer.get_current_index()

        model_replay_buffer.add( np.array([s_idx]),
                             np.array([a_category]),
                             np.array([r]),
                             np.array([end]),
                             np.array([s2_idx]))

        s_idx = s2_idx
        ep_reward += r

        if end:
            break

    # with open(txt_path, "a") as f:
    #     f.write(str(epi_i) + " *** " + str(int(ep_count)) + " *** " +
    #             str(float(ep_reward)) +"\n")
    # print(str(epi_i) + " *** " + str(int(ep_count)) + " *** " +
    #             str(float(ep_reward)))

# Right Data collect
info = env.reset(arenas_configurations = arena_config)["Learner"]

for epi_i in range(1, int(model_data_collection_episode + 1)):
    initial_frame = info.visual_observations[0][0] #[height, width, channel]
    end = info.local_done[0]

    image_buffer.animal_add(initial_frame)

    s_idx = image_buffer.get_current_index()

    ep_reward = 0.0
    ep_count = 0

    h = torch.zeros([1, 1, hidden_size]).to(device)
    c = torch.zeros([1, 1, hidden_size]).to(device)

    while True:

        a_category = 2
        a_deploy = action_dict[a_category]

        info = env.step(a_deploy)["Learner"]

        end = info.local_done[0]

        ep_count += 1
        r = info.rewards[0] * reward_scale
        s2_frame = info.visual_observations[0][0]

        image_buffer.animal_add(s2_frame)
        s2_idx = image_buffer.get_current_index()

        model_replay_buffer.add( np.array([s_idx]),
                             np.array([a_category]),
                             np.array([r]),
                             np.array([end]),
                             np.array([s2_idx]))

        s_idx = s2_idx
        ep_reward += r

        if end:
            break

    # with open(txt_path, "a") as f:
    #     f.write(str(epi_i) + " *** " + str(int(ep_count)) + " *** " +
    #             str(float(ep_reward)) +"\n")
    # print(str(epi_i) + " *** " + str(int(ep_count)) + " *** " +
    #             str(float(ep_reward)))


# Random Data collect
info = env.reset(arenas_configurations = arena_config)["Learner"]

for epi_i in range(1, model_data_collection_episode + 1):
    initial_frame = info.visual_observations[0][0] #[height, width, channel]
    end = info.local_done[0]

    image_buffer.animal_add(initial_frame)

    s_idx = image_buffer.get_current_index()

    ep_reward = 0.0
    ep_count = 0

    h = torch.zeros([1, 1, hidden_size]).to(device)
    c = torch.zeros([1, 1, hidden_size]).to(device)

    while True:

        a_category = np.random.choice(3)
        a_deploy = action_dict[a_category]

        info = env.step(a_deploy)["Learner"]

        end = info.local_done[0]

        ep_count += 1
        r = info.rewards[0] * reward_scale
        s2_frame = info.visual_observations[0][0]

        image_buffer.animal_add(s2_frame)
        s2_idx = image_buffer.get_current_index()

        model_replay_buffer.add( np.array([s_idx]),
                             np.array([a_category]),
                             np.array([r]),
                             np.array([end]),
                             np.array([s2_idx]))

        s_idx = s2_idx
        ep_reward += r

        if end:
            break

    # with open(txt_path, "a") as f:
    #     f.write(str(epi_i) + " *** " + str(int(ep_count)) + " *** " +
    #             str(float(ep_reward)) +"\n")
    # print(str(epi_i) + " *** " + str(int(ep_count)) + " *** " +
    #             str(float(ep_reward)))


#duju_utils.torch_network_load(vision_model, "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/proposed_results/Animal_AI_World_Proposed_1-Food_vision_42500.torch")

# Train Vision Model and Memory Model
for train_step in range(1,50000+1):
    vm_result = vision_model.trainer(model_replay_buffer, image_buffer, batch_size, device)

    if train_step % 2500 == 0:
        print(train_step, vm_result)
        with open(txt_path, "a") as f:
            f.write(str(train_step) + " *** " + str((vm_result)) + " *** " + "\n")
        duju_utils.torch_network_save(vision_model,
                                      os.path.join(exp_dir, exp_title + "_vision_" + str(train_step) + ".torch"))

for train_step in range(1,25000+1):
    mm_result = memory_model.train_batch(vision_model, model_replay_buffer, image_buffer, 4)
    if train_step % 2500 == 0:
        print(train_step, mm_result)
        with open(txt_path, "a") as f:
            f.write(str(train_step) + " *** " + str(float(mm_result)) + "\n")
    if train_step % 2500 == 0:
        duju_utils.torch_network_save(memory_model,
                                      os.path.join(exp_dir, exp_title + "_memory_" + str(train_step) + ".torch"))

del model_replay_buffer
del image_buffer

#duju_utils.torch_network_load(memory_model, "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/proposed_results/Animal_AI_World_Proposed_1-Food_memory_5000.torch")

# Train Controller

info = env.reset(arenas_configurations = arena_config)["Learner"]
for epi_i in range(1, max_episode + 1):
    s1_frame = info.visual_observations[0][0] #[height, width, channel]
    end = info.local_done[0]

    ep_reward = 0.0
    ep_count = 0

    epsilon = max(1.0 * (1 - epi_i/100),0.05*(1 - epi_i/1000))

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
            a_deploy = action_dict[a_category]

            a_one_hot = memory_model.one_hot([a_category], 3)
            a_one_hot = torch.FloatTensor(a_one_hot).view([1,1,3]).to(device)

            z_in = z.view([1,1, dim_z])

            # h0, c0 = h, c
            # z0 = z

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

            control_replay_buffer.store( s.detach().cpu().numpy()[0],
                                 np.array([a_category]),
                                 np.array([r]),
                                 np.array([end]),
                                 s2.detach().cpu().numpy()[0])

            s = s2
            ep_reward += r

            if end:
                break

            # for visualization
            if False:#epi_i % 25 == 0:
                recon = vision_model.decode(z0)[0].detach().cpu().numpy()
                recon = np.moveaxis(recon, [0,1,2],[2,0,1])

                z_out, _, _, _, _, _ = memory_model.full_forward(z_in, a_one_hot, h0, c0)
                z_out = z_out.view([1, dim_z])

                next_recon = vision_model.decode(z_out)[0].detach().cpu().numpy()
                next_recon = np.moveaxis(next_recon, [0, 1, 2], [2, 0, 1])

                img_str = "R : " + str(r) + " A : " + action_semantic[a_category]
                img = cv2.putText(cv2.resize(s1_frame[:, :, [2,1,0]], (420, 420)), img_str, (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
                cv2.imshow("train", img)
                cv2.waitKey(1)

                cv2.imshow("recon", cv2.resize(recon[:, :, [2,1,0]], (420, 420)))
                cv2.waitKey(1)

                cv2.imshow("s2_frame", cv2.resize(s2_frame[:, :, [2, 1, 0]], (420, 420)))
                cv2.waitKey(1)

                cv2.imshow("next_recon", cv2.resize(next_recon[:, :, [2, 1, 0]], (420, 420)))
                cv2.waitKey(1)

            s1_frame = s2_frame


    for _idx in range(250):
        mean_q1, min_q1, max_q1, mean_q2, mean_reward = train_rainbow_dqn(q_main,
                                                                               q_target,
                                                                               control_replay_buffer,
                                                                               batch_size,
                                                                               gamma)

    train_episodic_reward.append(ep_reward)

    with open(txt_path, "a") as f:
        f.write(str(epi_i) + " *** " + str(int(ep_count)) + " *** " +
                str(float(ep_reward)) + " *** " +
                str((float(mean_q1), float(min_q1), float(max_q1), float(mean_q2), float(mean_reward))) +"\n")
    print(str(epi_i) + " *** " + str(int(ep_count)) + " *** " +
                str(float(ep_reward)) + " *** " +
                str((float(mean_q1), float(min_q1), float(max_q1), float(mean_q2), float(mean_reward))))

    if (epi_i % 25) == 0:
         duju_utils.torch_network_save(q_main, os.path.join(exp_dir, exp_title + "_q_main_" + str(epi_i) + ".torch"))
         duju_utils.torch_network_save(q_target, os.path.join(exp_dir, exp_title + "_q_target_" + str(epi_i) + ".torch"))


plt.plot(train_episodic_reward, color = "tab:blue")
plt.savefig(os.path.join(exp_dir, exp_title+".jpg"),dpi=100)
plt.close()

with open(os.path.join(exp_dir, exp_title + ".txt"), "w") as f:
    f.write("Train\n" + str(train_episodic_reward) + "\n")

cv2.destroyAllWindows()
env.close()
