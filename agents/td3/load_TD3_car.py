import csv
import os
from collections import defaultdict

import ray
import torch.nn
import numpy as np
from ray.rllib.agents.ppo import ppo

from agents.ray_utils import *
from environment.stopping_car_continuous import StoppingCar
import ray.rllib.agents.ddpg as td3
from agents.td3.tune.tune_train_TD3_car import get_TD3_config


def nn1():
    layers = []
    l0 = torch.nn.Linear(2, 2)
    l0.weight = torch.nn.Parameter(torch.tensor([[0, -1], [1, 0]], dtype=torch.float32))
    l0.bias = torch.nn.Parameter(torch.tensor([+20, 0], dtype=torch.float32))
    layers.append(l0)
    # layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Hardtanh(min_val=-3, max_val=3))
    l1 = torch.nn.Linear(2, 1)
    l1.weight = torch.nn.Parameter(torch.tensor([[0.15, 1]], dtype=torch.float32))
    l1.bias = torch.nn.Parameter(torch.tensor([0], dtype=torch.float32))
    layers.append(l1)

    nn = torch.nn.Sequential(*layers)
    return nn


def nn2():
    layers = []
    l0 = torch.nn.Linear(2, 3)
    l0.weight = torch.nn.Parameter(torch.tensor([[0, 1], [1, 0], [-1, 0]], dtype=torch.float32))
    l0.bias = torch.nn.Parameter(torch.tensor([-20, 0, 0], dtype=torch.float32))
    layers.append(l0)
    layers.append(torch.nn.ReLU())
    l1 = torch.nn.Linear(3, 2)
    l1.weight = torch.nn.Parameter(torch.tensor([[-1, 0, 0], [0, 1, 1]], dtype=torch.float32))
    l1.bias = torch.nn.Parameter(torch.tensor([0, 0], dtype=torch.float32))
    layers.append(l1)
    l2 = torch.nn.Linear(2, 1)
    l2.weight = torch.nn.Parameter(torch.tensor([[0.15, 1]], dtype=torch.float32))
    l2.bias = torch.nn.Parameter(torch.tensor([0], dtype=torch.float32))
    layers.append(l2)
    layers.append(torch.nn.Hardtanh(min_val=-3, max_val=3))
    nn = torch.nn.Sequential(*layers)
    return nn


def nn3():
    layers = []
    l0 = torch.nn.Linear(2, 1)
    l0.weight = torch.nn.Parameter(torch.tensor([[0, 1]], dtype=torch.float32))
    l0.bias = torch.nn.Parameter(torch.tensor([-30], dtype=torch.float32))
    layers.append(l0)
    l0 = torch.nn.Linear(1, 1)
    l0.weight = torch.nn.Parameter(torch.tensor([[1.2]], dtype=torch.float32))
    l0.bias = torch.nn.Parameter(torch.tensor([0], dtype=torch.float32))
    layers.append(l0)
    # layers.append(torch.nn.Hardtanh(min_val=-3, max_val=3))
    nn = torch.nn.Sequential(*layers)
    return nn


ray.init()
# config, trainer = get_PPO_trainer(use_gpu=0)

# config = get_TD3_config(1234)
# trainer = ppo.PPOTrainer(config=config)
# # trainer.restore("/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/TD3_StoppingCar_0a03b_00000_0_cost_fn=2,epsilon_input=0_2021-02-27_17-12-58/checkpoint_680/checkpoint-680")
# # trainer.restore("/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/TD3_StoppingCar_47b16_00000_0_cost_fn=3,epsilon_input=0_2021-03-04_17-08-46/checkpoint_600/checkpoint-600")
# # trainer.restore("/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/PPO_StoppingCar_2f9f7_00000_0_cost_fn=3,epsilon_input=0_2021-03-07_16-00-06/checkpoint_150/checkpoint-150")
# # trainer.restore("/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/PPO_StoppingCar_28110_00000_0_cost_fn=0,epsilon_input=0_2021-03-07_17-40-07/checkpoint_1250/checkpoint-1250")
# trainer.restore("/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/PPO_StoppingCar_90786_00000_0_cost_fn=0,epsilon_input=0_2021-03-09_14-34-33/checkpoint_2870/checkpoint-2870")
# policy = trainer.get_policy()
# sequential_nn = convert_ray_policy_to_sequential2(policy)

sequential_nn = nn3()
# policy.model.cuda()
# l0 = torch.nn.Linear(6, 2, bias=False)
# l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]], dtype=torch.float32))
# layers = [l0]
# for l in sequential_nn:
#     layers.append(l)
#
# sequential_nn2 = torch.nn.Sequential(*layers)
y_index = 1
x_index = 0
y_list = defaultdict(list)  # []
x_list = defaultdict(list)
config = {"cost_fn": 0,
          "epsilon_input": 0,
          "reduced": True}
env = StoppingCar(config)
env.reset()
# env.x_lead = 30
# env.x_ego = 0
# env.v_lead = 28
# env.v_ego = 36
min_distance = 9999
state_np = env.get_state()
print(state_np)
for n in range(1000):
    cumulative_reward = 0
    env.reset()
    env.x_ego = 0  # env.np_random.uniform(0, 10)
    env.x_lead = env.np_random.uniform(30, 40)  # env.np_random.uniform(27.6, 32.3)
    env.v_lead = 28
    env.v_ego = env.np_random.uniform(env.v_lead - 2.5, env.v_lead + 2.5)  # 36
    state_np = env.get_state()
    # state_np = np.array(state)
    y_list[0].append(state_np[y_index])
    x_list[0].append(state_np[x_index])
    for i in range(10):
        state = torch.from_numpy(state_np).float().unsqueeze(0)
        # state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]
        action = sequential_nn(state).squeeze()
        # action2 = policy.compute_single_action([state_np], explore=False)
        # action_score2 = sequential_nn2(state)
        # action = torch.argmax(action_score).item()
        # action2 = torch.argmax(action_score2).item()
        # assert action == action2
        print(f"action: {action}")
        state_np, reward, done, _ = env.step(action)
        y_list[1 + i].append(state_np[y_index])
        x_list[1 + i].append(state_np[x_index])
        min_distance = min(state_np[1], min_distance)
        cumulative_reward += reward
        print(f"iteration: {i}, delta_x: {state_np[1]:.2f}, delta_v: {state_np[0]:.2f}, reward: {reward}")
        # if state_np[0] > 300:
        #     break
        if done:
            print("done")

            break
    print("-------")
    print(f"cumulative_reward:{cumulative_reward}")
# we want positive delta_x and delta_v close to 0
print("all good")
print(f"min_distance:{min_distance}")
# with open(os.path.join("/home/edoardo/ray_results/tune_TD3_stopping_car_continuous", "simulation.csv"), 'w',
#           newline='') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
#     for i, item in enumerate(position_list):
#         wr.writerow((x_list[i], item))
#     wr.writerow("")
import plotly.graph_objects as go
import plotly.io as pio

fig = go.Figure()
with open('/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/test/plot.json', 'r') as f:
    fig = pio.from_json(f.read())

# trace1 = go.Scatter(x=list(range(len(position_list))), y=position_list, mode='markers', )
for i in range(len(y_list)):
    trace1 = go.Scatter(x=x_list[i], y=y_list[i], mode='markers')
    fig.add_trace(trace1)
fig.update_layout(xaxis_title="delta v", yaxis_title="delta x")
fig.show()
save_dir = "/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/test"
fig.write_html(os.path.join(save_dir, "merged.html"), include_plotlyjs="cdn")

ray.shutdown()

# suggest 200 timesteps
