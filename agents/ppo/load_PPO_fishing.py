import gym
import pygame
import ray
import torch.nn
import numpy as np
from ray.rllib.agents.ppo import ppo
from ray.tune import register_env

from agents.ppo.train_PPO_cartpole import get_PPO_trainer
from agents.ppo.tune.tune_train_PPO_fishing import get_PPO_config, env_creator
from agents.ray_utils import convert_ray_policy_to_sequential, convert_ray_simple_policy_to_sequential

ray.init()
register_env("fishing", env_creator)
config = get_PPO_config(1234)
trainer = ppo.PPOTrainer(config=config)
# trainer.restore("/home/edoardo/ray_results/tune_PPO_lunar_hover/PPO_LunarHover_7ba4e_00000_0_2021-04-02_19-01-43/checkpoint_990/checkpoint-990")
trainer.restore("/home/edoardo/ray_results/tune_PPO_fishing/PPO_fishing_b9dcd_00000_0_2021-04-03_16-46-16/checkpoint_340/checkpoint-340")

policy = trainer.get_policy()
# sequential_nn = convert_ray_simple_policy_to_sequential(policy).cpu()
sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
# l0 = torch.nn.Linear(4, 2, bias=False)
# l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32))
# layers = [l0]
# for l in sequential_nn:
#     layers.append(l)
# nn = torch.nn.Sequential(*layers)
nn = sequential_nn
env = gym.make("fishing-v0")
# env.render()
plot_index = 0
position_list = []
# env.render()
n_trials = 10
cumulative_reward = 0
clock = pygame.time.Clock()
for i in range(n_trials):
    state = env.reset()
    # env.state[2] = 0.01
    # env.state[2] = 0.045
    # env.state[3] = -0.51
    # state = np.array(env.state)
    state_np = np.array(state)
    print(state_np)
    position_list.append(state_np[plot_index])
    for i in range(1000):
        state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)
        # state = torch.from_numpy(state_np).float().unsqueeze(0)
        action_score = nn(state_reduced)
        # action_score2 = sequential_nn2(state)
        action = torch.argmax(action_score).item()
        # action2 = torch.argmax(action_score2).item()
        # assert action == action2
        print(f"action: {action}")
        state_np, reward, done, _ = env.step(action)
        # env.render()
        position_list.append(state_np[plot_index])
        # min_distance = min(state_np[7], min_distance)
        cumulative_reward += reward
        print(f"iteration: {i}, state: {state_np}, reward: {reward}")
        print("-------")
        # env.render()
        clock.tick(30)  # framerate
        if done:
            print("done")

            break
env.close()
print("all good")
# print(f"min_distance:{min_distance}")
print(f"cumulative_reward:{cumulative_reward / n_trials}")
# ray.shutdown()
# import plotly.graph_objects as go
#
# fig = go.Figure()
# trace1 = go.Scatter(x=list(range(len(position_list))), y=position_list, mode='markers', )
# fig.add_trace(trace1)
# fig.show()
