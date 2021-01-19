import ray
import torch.nn
import numpy as np
from ray.rllib.agents.ppo import ppo

from agents.ppo.train_PPO_bouncingball import get_PPO_trainer
from agents.ppo.tune.tune_train_PPO_bouncing_ball import get_PPO_config
from agents.ray_utils import convert_ray_policy_to_sequential
from environment.bouncing_ball_old import BouncingBall

ray.init()
# config, trainer = get_PPO_trainer(use_gpu=0)
# trainer.restore("/home/edoardo/ray_results/PPO_BouncingBall_2021-01-04_18-58-32smp2ln1g/checkpoint_272/checkpoint-272")  # ~980 score
# policy = trainer.get_policy()
# sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
# layers = []
# for l in sequential_nn:
#     layers.append(l)
#
# sequential_nn2 = torch.nn.Sequential(*layers)
ray.init(ignore_reinit_error=True)
config = get_PPO_config(1234)
trainer = ppo.PPOTrainer(config=config)
trainer.restore("/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00004_4_2021-01-18_23-48-21/checkpoint_10/checkpoint-10")
policy = trainer.get_policy()
sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
layers = []
for l in sequential_nn:
    layers.append(l)
nn = torch.nn.Sequential(*layers)
env = BouncingBall()
state = env.reset()
env.p = 7
# env.x_lead = 30
# env.x_ego = 0
# env.v_lead = 28
# env.v_ego = 36
# min_distance = 9999
state = np.array([env.p, env.v])
state_np = np.array(state)
cumulative_reward = 0
print(state_np)
position_list = [state_np[0]]
for i in range(1000):
    state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)
    # state = torch.from_numpy(state_np).float().unsqueeze(0)
    action_score = sequential_nn(state_reduced)
    # action_score2 = sequential_nn2(state)
    action = torch.argmax(action_score).item()
    # action2 = torch.argmax(action_score2).item()
    # assert action == action2
    print(f"action: {action}")
    state_np, reward, done, _ = env.step(action)
    position_list.append(state_np[0])
    # min_distance = min(state_np[7], min_distance)
    cumulative_reward += reward
    print(f"iteration: {i}, state: {state_np}, reward: {reward}")
    print("-------")
    if done:
        print("done")

        break
print("all good")
# print(f"min_distance:{min_distance}")
print(f"cumulative_reward:{cumulative_reward}")
ray.shutdown()
import plotly.graph_objects as go

fig = go.Figure()
trace1 = go.Scatter(x=list(range(len(position_list))), y=position_list, mode='markers', )
fig.add_trace(trace1)
fig.show()
