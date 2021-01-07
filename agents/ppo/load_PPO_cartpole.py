import ray
import torch.nn
import numpy as np
from agents.ppo.train_PPO_cartpole import get_PPO_trainer
from agents.ray_utils import convert_ray_policy_to_sequential, convert_ray_simple_policy_to_sequential
from environment.bouncing_ball_old import BouncingBall
from environment.cartpole_ray import CartPoleEnv

ray.init()
config, trainer = get_PPO_trainer(use_gpu=0)
trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-07_07-43-24zjknadb4/checkpoint_21/checkpoint-21")
policy = trainer.get_policy()
sequential_nn = convert_ray_simple_policy_to_sequential(policy).cpu()
layers = []
for l in sequential_nn:
    layers.append(l)

sequential_nn2 = torch.nn.Sequential(*layers)
env = CartPoleEnv(None)
state = env.reset()
env.state[2] = 0.045
env.state[3] = -0.51
# env.x_lead = 30
# env.x_ego = 0
# env.v_lead = 28
# env.v_ego = 36
# min_distance = 9999
state = np.array(env.state)
state_np = np.array(state)
cumulative_reward = 0
print(state_np)
plot_index = 2
position_list = [state_np[plot_index]]
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
    position_list.append(state_np[plot_index])
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
