import csv
import os

import ray
import torch.nn
import numpy as np
from agents.ray_utils import convert_td3_policy_to_sequential
from environment.stopping_car_continuous import StoppingCar
import ray.rllib.agents.ddpg as td3
from agents.td3.tune.tune_train_TD3_car import get_TD3_config

ray.init()
# config, trainer = get_PPO_trainer(use_gpu=0)

config = get_TD3_config(1234)
trainer = td3.TD3Trainer(config=config)
trainer.restore("/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/TD3_StoppingCar_0a03b_00000_0_cost_fn=2,epsilon_input=0_2021-02-27_17-12-58/checkpoint_680/checkpoint-680")
policy = trainer.get_policy()
sequential_nn = convert_td3_policy_to_sequential(policy).cpu()
# l0 = torch.nn.Linear(6, 2, bias=False)
# l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]], dtype=torch.float32))
# layers = [l0]
# for l in sequential_nn:
#     layers.append(l)
#
# sequential_nn2 = torch.nn.Sequential(*layers)
plot_index = 1
x_index = 0
position_list = []
x_list = []
config = {"cost_fn": 2,
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
for n in range(1):
    cumulative_reward = 0
    env.reset()
    env.x_ego = env.np_random.uniform(0, 10)
    env.x_lead = env.np_random.uniform(30, 40)
    env.v_lead = 28
    env.v_ego = 36
    state_np = env.get_state()
    # state_np = np.array(state)
    position_list.append(state_np[plot_index])
    x_list.append(state_np[x_index])
    for i in range(1000):
        state = torch.from_numpy(state_np).float().unsqueeze(0)
        # state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]
        action = sequential_nn(state).squeeze()
        # action_score2 = sequential_nn2(state)
        # action = torch.argmax(action_score).item()
        # action2 = torch.argmax(action_score2).item()
        # assert action == action2
        print(f"action: {action}")
        state_np, reward, done, _ = env.step(action)
        position_list.append(state_np[plot_index])
        x_list.append(state_np[x_index])
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
#we want positive delta_x and delta_v close to 0
print("all good")
print(f"min_distance:{min_distance}")
# with open(os.path.join("/home/edoardo/ray_results/tune_TD3_stopping_car_continuous", "simulation.csv"), 'w',
#           newline='') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
#     for i, item in enumerate(position_list):
#         wr.writerow((x_list[i], item))
#     wr.writerow("")
import plotly.graph_objects as go

fig = go.Figure()
# trace1 = go.Scatter(x=list(range(len(position_list))), y=position_list, mode='markers', )
trace1 = go.Scatter(x=x_list, y=position_list, mode='markers', )
fig.add_trace(trace1)
fig.show()
ray.shutdown()

# suggest 200 timesteps
