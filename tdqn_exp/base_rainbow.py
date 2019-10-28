from Model.ReplayBuffer import PERBuffer
from Model.rainbow import Rainbow_DQN
from Model.rainbow import train_rainbow_dqn
from Model.SAC_base import target_initialize
import lib_duju.utils as duju_utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

from dm_control import suite
import numpy as np

exp_title = "Rainbow Double-PER-Duel DQN -A2"
print(exp_title)

env = suite.load(domain_name="cartpole",task_name="swingup")

action_dict = { 0 : -1.0,
                1 : 1.0
                }

state_dim = duju_utils.state_1d_dim_calc(env)[-1]
action_dim = len(action_dict)

rb_state_dim = state_dim
rb_action_dim = 1

lr = 3e-4
gamma = 0.99
device = torch.device("cuda")

max_episode = 200
max_step = max_episode * 1000

batch_size = 128
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

