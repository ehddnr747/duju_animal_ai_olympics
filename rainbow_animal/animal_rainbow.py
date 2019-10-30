from Model.ReplayBuffer import PERBuffer
from Model.ImageBuffer import ImageBuffer
from Model.rainbow_conv import Rainbow_DQN_Conv
from Model.rainbow_conv import train_rainbow_dqn_conv
from Model.SAC_base import target_initialize
import lib_duju.utils as duju_utils
import matplotlib.pyplot as plt

import torch
import cv2

from animalai.envs import UnityEnvironment
from animalai.envs import ArenaConfig
import numpy as np

exp_title = "Rainbow Double-PER-Duel DQN -Animal Obstacles"
print(exp_title)

env = UnityEnvironment(
        file_name='/home/duju/animal_ai_olympics/AnimalAI-Olympics/env/AnimalAI.x86_64',   # Path to the environment
)

arena_config = ArenaConfig("/home/duju/animal_ai_olympics/duju_animal_ai_olympics/configs/3-Obstacles.yaml")

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

max_episode = 10000
max_step = max_episode * 1000

batch_size = 32
buffer_size = int(5e5)

alpha = 0.6
initial_beta = 0.4

reward_scale = 10

replay_buffer = PERBuffer(rb_state_dim, rb_action_dim, buffer_size, alpha, initial_beta, max_step)
image_buffer = ImageBuffer(height, width, step_size, channel, int(buffer_size * 1.1))

q_main = Rainbow_DQN_Conv(step_size, channel, height, width, action_dim, lr, device)
q_target = Rainbow_DQN_Conv(step_size, channel, height, width, action_dim, lr, device)

duju_utils.torch_network_load(q_main, "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/trained/Rainbow Double-PER-Duel DQN -Animal Food_q_main_1270.torch")
duju_utils.torch_network_load(q_target, "/home/duju/animal_ai_olympics/duju_animal_ai_olympics/trained/Rainbow Double-PER-Duel DQN -Animal Food_q_target_1270.torch")

target_initialize(q_main, q_target)

print(q_main)

train_episodic_reward = []
eval_episodic_reward = []

info = env.reset(arenas_configurations = arena_config)["Learner"]

for epi_i in range(1, max_episode + 1):
    print(epi_i, end = "\t")

    initial_frame = info.visual_observations[0][0] #[height, width, channel]
    end = info.local_done[0]

    for _ in range(step_size):
        image_buffer.animal_add(initial_frame)

    s_idx = image_buffer.get_current_index()
    input_state = image_buffer.get_state(s_idx)

    ep_reward = 0.0
    ep_count = 0

    epsilon = 0.05
        #max(0.05, (100 - epi_i) / 100)

    while not end:
        ep_count += 1

        a_category = q_main.epsilon_sample(
            torch.FloatTensor(input_state).to(device).view(1, input_channel_size, height, width),
            epsilon
        )
        a_deploy = action_dict[a_category]

        info = env.step(a_deploy)["Learner"]

        end = info.local_done[0]
        r = info.rewards[0] * reward_scale
        s2_frame = info.visual_observations[0][0]

        image_buffer.animal_add(s2_frame)
        s2_idx = image_buffer.get_current_index()
        input_state = image_buffer.get_state(s2_idx)

        replay_buffer.store( np.array([s_idx]),
                             np.array([a_category]),
                             np.array([r]),
                             np.array([end]),
                             np.array([s2_idx]))

        s_idx = s2_idx
        ep_reward += r

        img_str = "R : " + str(r) + " A : " + action_semantic[a_category]
        img = cv2.putText(cv2.resize(s2_frame[:, :, [2,1,0]], (420, 420)), img_str, (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
        cv2.imshow("train", img)
        cv2.waitKey(1)

    # For the purpose to proceed episode
    info = env.step([0,0])["Learner"]

    for _idx in range(min(500, max(100, ep_count))):
        #print(_idx)
        mean_q1, min_q1, max_q1, mean_q2, mean_reward = train_rainbow_dqn_conv(q_main,
                                                                               q_target,
                                                                               replay_buffer,
                                                                               image_buffer,
                                                                               batch_size,
                                                                               gamma)

    train_episodic_reward.append(ep_reward)

    print("***", int(ep_count), "***", float(ep_reward), "***", (float(mean_q1), float(min_q1), float(max_q1), float(mean_q2), float(mean_reward)), end="\n")

    if (epi_i % 10) == 0:
        print("Networks Saved!")
        duju_utils.torch_network_save(q_main, "../trained/" + exp_title + "_q_main_" + str(epi_i) + ".torch")
        duju_utils.torch_network_save(q_target, "../trained/" + exp_title + "_q_target_" + str(epi_i) + ".torch")

    # timestep = env.reset()
    # end, _, _, s = timestep
    # end = end.last()
    # s = duju_utils.state_1d_flat(s)

    # eval_ep_reward = 0.0
    # eval_action = []
    #
    # if (epi_i % 1) == 0 :
    #     while not end:
    #         a_category = q_main.epsilon_sample(
    #             torch.FloatTensor(s).to(device).view(1, -1),
    #             0.0
    #         )
    #         a_deploy = action_dict[a_category]
    #
    #         timestep = env.step(a_deploy)
    #         eval_action.append(a_deploy)
    #
    #         end, r, _, s2 = timestep
    #         end = end.last()
    #         s2 = duju_utils.state_1d_flat(s2)
    #
    #         s = s2
    #         eval_ep_reward += r
    #
    #         frame = env.physics.render(camera_id=0, width=640, height=480) #[height, width, channel]
    #         cv2.imshow("test", frame)
    #         cv2.waitKey(1)
    #
    #
    #     print("*** Eval! *** ", eval_ep_reward)
    #     eval_episodic_reward.append(eval_ep_reward)

plt.plot(train_episodic_reward, color = "tab:blue")
#plt.plot(eval_episodic_reward, color = "tab:red")
plt.savefig("../graph/"+exp_title+".jpg",dpi=100)
plt.close()

with open("../graph/" + exp_title + ".txt", "w") as f:
    f.write("Train\n" + str(train_episodic_reward) + "\n")


# with open("../graph/" + exp_title + ".txt", "w") as f:
#     f.write("Train\n" + str(train_episodic_reward) + "\nEval\n" + str(eval_episodic_reward) + "\n")

cv2.destroyAllWindows()
env.close()
