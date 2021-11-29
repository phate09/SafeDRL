import numpy as np
import ray
import torch.nn
from ray.rllib.agents.ppo import ppo
from ray.tune import register_env

from environment.water_tank import MonitoredWaterTank
from training.ppo.tune.tune_train_PPO_watertank import get_PPO_config, env_creator
from training.ray_utils import convert_ray_policy_to_sequential

ray.init()
register_env("fishing", env_creator)
config = get_PPO_config(1234)
trainer = ppo.PPOTrainer(config=config)
trainer.restore("/home/edoardo/ray_results/tune_PPO_watertank/PPO_MonitoredWaterTank_e219e_00000_0_2021-05-19_20-39-10/checkpoint_3334/checkpoint-3334")

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
env = MonitoredWaterTank()  # gym.make("fishing-v0")
# env.render()
plot_index = 0
position_list = []
# env.render()
n_trials = 1
cumulative_reward = 0
for i in range(n_trials):
    state = env.reset()
    # state = env.set_state(0.75)
    print(state)
    state_np = np.array(state)
    print(state_np)
    position_list.append(state_np[plot_index])
    for i in range(100):
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
        # clock.tick(30)  # framerate
        print(state)
        if done:
            print("done")

            break
env.close()
print("all good")
# print(f"min_distance:{min_distance}")
print(f"cumulative_reward:{cumulative_reward / n_trials}")
# ray.shutdown()
import plotly.graph_objects as go

fig = go.Figure()
trace1 = go.Scatter(x=list(range(len(position_list))), y=position_list, mode='lines+markers', )
fig.add_trace(trace1)
fig.show()
