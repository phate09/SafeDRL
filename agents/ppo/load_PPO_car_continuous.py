import csv
import os

import ray
import torch.nn
import numpy as np
from agents.ray_utils import convert_ray_policy_to_sequential
from environment.stopping_car_continuous import StoppingCar
import ray.rllib.agents.ppo as ppo
from agents.ppo.tune.tune_train_PPO_car import get_PPO_config
from polyhedra.runnable.experiment.run_experiment_stopping_car_continuous import StoppingCarContinuousExperiment

ray.init()
# config, trainer = get_PPO_trainer(use_gpu=0)

sequential_nn = StoppingCarContinuousExperiment.get_nn_static()
l0 = torch.nn.Linear(6, 2, bias=False)
l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]], dtype=torch.float32))
layers = [l0]
for l in sequential_nn:
    layers.append(l)

sequential_nn2 = torch.nn.Sequential(*layers)
plot_index = 7
x_index = 0
position_list = []
x_list = []
env = StoppingCar()
env.reset()
# env.x_lead = 30
# env.x_ego = 0
# env.v_lead = 28
# env.v_ego = 36
min_distance = 9999
state_np = np.array([env.x_lead, env.x_ego, env.v_lead, env.v_ego, env.y_lead, env.y_ego, env.v_lead - env.v_ego, env.x_lead - env.x_ego])
print(state_np)
for n in range(1):
    cumulative_reward = 0
    env.reset()
    env.x_ego = 0  # env.np_random.uniform(0, 10)
    env.x_lead = 30  # env.np_random.uniform(30, 40)
    env.v_lead = 28
    env.v_ego = 36
    state_np = np.array([env.x_lead, env.x_ego, env.v_lead, env.v_ego, env.y_lead, env.y_ego, env.v_lead - env.v_ego, env.x_lead - env.x_ego])
    # state_np = np.array(state)
    position_list.append(state_np[plot_index])
    x_list.append(state_np[x_index])
    for i in range(1000):
        state = torch.from_numpy(state_np).float().unsqueeze(0)[:, :-2]
        state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]
        # action_score = sequential_nn(state_reduced)
        action = sequential_nn2(state)
        state_np, reward, done, _ = env.step(action)
        position_list.append(state_np[plot_index])
        x_list.append(state_np[x_index])
        min_distance = min(state_np[7], min_distance)
        cumulative_reward += reward
        print(f"iteration: {i}, delta_x: {state_np[7]:.2f}, delta_v: {state_np[6]:.2f}, v_ego: {state_np[3]:.2f},v_lead: {state_np[2]:.2f} , y_ego: {state_np[5]:.2f}, reward: {reward}")
        # if state_np[0] > 300:
        #     break
        if done:
            print("done")

            break
    print("-------")
    print(f"cumulative_reward:{cumulative_reward}")
print("all good")
print(f"min_distance:{min_distance}")
# with open(os.path.join("/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49", "simulation.csv"), 'w', newline='') as myfile:
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
