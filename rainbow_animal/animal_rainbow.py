from Model.ReplayBuffer import PERBuffer
from Model.ImageBuffer import ImageBuffer
from Model.rainbow import Rainbow_DQN
from Model.rainbow import train_rainbow_dqn
from Model.SAC_base import target_initialize
import lib_duju.utils as duju_utils
import matplotlib.pyplot as plt

import torch
import cv2

from animalai.envs import UnityEnvironment
from animalai.envs import ArenaConfig
import numpy as np

exp_title = "Rainbow Double-PER-Duel DQN -Animal 4"
print(exp_title)

env = UnityEnvironment(
        file_name='/home/duju/animal_ai_olympics/AnimalAI-Olympics/env/AnimalAI.x86_64',   # Path to the environment
)

arena_config = ArenaConfig("./configs/4-Avoidance.yaml")

action_dict = { 0 : np.array([1,0]), # forward
                1 : np.array([0,2]), # left
                2 : np.array([0,1]), # right
                }

state_dim = (84, 84, 3)

height = state_dim[0]
width = state_dim[1]
channel = state_dim[2]

step_size = 4

action_dim = len(action_dict)

rb_state_dim = 1
rb_action_dim = 1

lr = 3e-4
gamma = 0.99
device = torch.device("cuda")

max_episode = 250
max_step = max_episode * 1000

batch_size = 32
buffer_size = int(1e6)

alpha = 0.6
initial_beta = 0.4

replay_buffer = PERBuffer(rb_state_dim, rb_action_dim, buffer_size, alpha, initial_beta, max_step)

q_main = Rainbow_DQN(state_dim, action_dim, lr, device)
q_target = Rainbow_DQN(state_dim, action_dim, lr, device)

target_initialize(q_main, q_target)

print(q_main)

train_episodic_reward = []
eval_episodic_reward = []

for epi_i in range(1, max_episode + 1):
    print(epi_i, end = "\t")

    timestep = env.reset()
    ep_reward = 0.0

    # timestep, reward, discount, observation
    end, _, _, s = timestep
    end = end.last()
    s = duju_utils.state_1d_flat(s)

    epsilon = np.random.sample() * 0.2

    while not end:
        a_category = q_main.epsilon_sample(
            torch.FloatTensor(s).to(device).view(1,-1),
            epsilon
        )
        a_deploy = action_dict[a_category]

        timestep = env.step(a_deploy)

        end, r, _, s2 = timestep
        end = end.last()
        s2 = duju_utils.state_1d_flat(s2)

        replay_buffer.store(s, np.array([a_category]), np.array([r]),np.array([end]), s2)

        s = s2
        ep_reward += r

        frame = env.physics.render(camera_id=0, width=640, height=480)  # [height, width, channel]
        cv2.imshow("train", frame)
        cv2.waitKey(1)

    for _idx in range(1000):
        #print(_idx)
        mean_q1, min_q1, max_q1, mean_q2, mean_reward = train_rainbow_dqn(q_main, q_target, replay_buffer, batch_size, gamma)

    train_episodic_reward.append(ep_reward)

    print("***", int(ep_reward), "***", (float(mean_q1), float(min_q1), float(max_q1), float(mean_q2), float(mean_reward)), end="\t")

    timestep = env.reset()
    end, _, _, s = timestep
    end = end.last()
    s = duju_utils.state_1d_flat(s)

    eval_ep_reward = 0.0
    eval_action = []

    if (epi_i % 1) == 0 :
        while not end:
            a_category = q_main.epsilon_sample(
                torch.FloatTensor(s).to(device).view(1, -1),
                0.0
            )
            a_deploy = action_dict[a_category]

            timestep = env.step(a_deploy)
            eval_action.append(a_deploy)

            end, r, _, s2 = timestep
            end = end.last()
            s2 = duju_utils.state_1d_flat(s2)

            s = s2
            eval_ep_reward += r

            frame = env.physics.render(camera_id=0, width=640, height=480) #[height, width, channel]
            cv2.imshow("test", frame)
            cv2.waitKey(1)


        print("*** Eval! *** ", eval_ep_reward)
        eval_episodic_reward.append(eval_ep_reward)

plt.plot(train_episodic_reward, color = "tab:blue")
plt.plot(eval_episodic_reward, color = "tab:red")
plt.savefig("../graph/"+exp_title+".jpg",dpi=100)
plt.close()

with open("../graph/" + exp_title + ".txt", "w") as f:
    f.write("Train\n" + str(train_episodic_reward) + "\nEval\n" + str(eval_episodic_reward) + "\n")

cv2.destroyAllWindows()

