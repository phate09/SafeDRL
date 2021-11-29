import numpy as np
import ray
import torch.nn
from ray.rllib.agents.ppo import ppo

from environment.cartpole_ray import CartPoleEnv
from training.ppo.tune.tune_train_PPO_cartpole import get_PPO_config
from training.ray_utils import convert_ray_policy_to_sequential

ray.init()
# config, trainer = get_PPO_trainer(use_gpu=0)
# # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-07_10-57-537gv8ekj4/checkpoint_18/checkpoint-18")
# # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-07_07-43-24zjknadb4/checkpoint_21/checkpoint-21")
# # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-07_12-49-16sn6s0bd0/checkpoint_19/checkpoint-19")
# trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-09_15-34-25f0ld3dex/checkpoint_30/checkpoint-30")
# policy = trainer.get_policy()
# # sequential_nn = convert_ray_simple_policy_to_sequential(policy).cpu()
# sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
# l0 = torch.nn.Linear(4, 2, bias=False)
# l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32))
# layers = [l0]
# for l in sequential_nn:
#     layers.append(l)
#
# nn = torch.nn.Sequential(*layers)
config = get_PPO_config(1234)
trainer = ppo.PPOTrainer(config=config)
# trainer.restore("/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00001_1_cost_fn=1,tau=0.001_2021-01-16_20-25-43/checkpoint_3090/checkpoint-3090")
trainer.restore("/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00000_0_cost_fn=0,tau=0.001_2021-01-16_20-25-43/checkpoint_40/checkpoint-40")

policy = trainer.get_policy()
# sequential_nn = convert_ray_simple_policy_to_sequential(policy).cpu()
sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
l0 = torch.nn.Linear(4, 2, bias=False)
l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32))
layers = [l0]
for l in sequential_nn:
    layers.append(l)
nn = torch.nn.Sequential(*layers)
env = CartPoleEnv(None)

plot_index = 3
position_list = []
# env.render()
n_trials = 30
cumulative_reward = 0
for i in range(n_trials):
    state = env.reset()
    # env.state[2] = 0.01
    # env.state[2] = 0.045
    # env.state[3] = -0.51
    state = np.array(env.state)
    state_np = np.array(state)
    print(state_np)
    position_list.append(state_np[plot_index])
    for i in range(2000):
        state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)
        # state = torch.from_numpy(state_np).float().unsqueeze(0)
        action_score = nn(state_reduced)
        # action_score2 = sequential_nn2(state)
        action = torch.argmax(action_score).item()
        # action2 = torch.argmax(action_score2).item()
        # assert action == action2
        print(f"action: {action}")
        state_np, reward, done, _ = env.step(action)
        env.render()
        position_list.append(state_np[plot_index])
        # min_distance = min(state_np[7], min_distance)
        cumulative_reward += reward
        print(f"iteration: {i}, state: {state_np}, reward: {reward}")
        print("-------")
        if done:
            print("done")

            break
env.close()
print("all good")
# print(f"min_distance:{min_distance}")
print(f"cumulative_reward:{cumulative_reward/n_trials}")
ray.shutdown()
import plotly.graph_objects as go

fig = go.Figure()
trace1 = go.Scatter(x=list(range(len(position_list))), y=position_list, mode='markers', )
fig.add_trace(trace1)
fig.show()
