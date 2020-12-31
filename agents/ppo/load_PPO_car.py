import ray
import torch.nn
import numpy as np
from agents.ppo.train_PPO_car import get_PPO_trainer
from agents.ray_utils import convert_ray_policy_to_sequential
from environment.stopping_car import StoppingCar

ray.init()
config, trainer = get_PPO_trainer(use_gpu=0)
# trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-12-30_17-06-3265yz3d63/checkpoint_60/checkpoint-60")
trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-12-30_17-06-3265yz3d63/checkpoint_65/checkpoint-65")
policy = trainer.get_policy()
sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
l0 = torch.nn.Linear(6, 2, bias=False)
l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]], dtype=torch.float32))
layers = [l0]
for l in sequential_nn:
    layers.append(l)

sequential_nn2 = torch.nn.Sequential(*layers)
env = StoppingCar()
env.reset()
env.x_lead = 30
env.x_ego = 0
env.v_lead = 28
env.v_ego = 36
min_distance = 9999
state_np = np.array([env.x_lead, env.x_ego, env.v_lead, env.v_ego, env.y_lead, env.y_ego, env.v_lead - env.v_ego, env.x_lead - env.x_ego])
cumulative_reward = 0
print(state_np)
for i in range(1000):
    state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]
    state = torch.from_numpy(state_np).float().unsqueeze(0)[:, :-2]
    action_score = sequential_nn(state_reduced)
    action_score2 = sequential_nn2(state)
    action = torch.argmax(action_score).item()
    action2 = torch.argmax(action_score2).item()
    assert action == action2
    print(f"action: {action}")
    state_np, reward, done, _ = env.step(action)
    min_distance = min(state_np[7], min_distance)
    cumulative_reward += reward
    print(f"iteration: {i}, delta_x: {state_np[7]:.2f}, delta_v: {state_np[6]:.2f}, v_ego: {state_np[3]:.2f},v_lead: {state_np[2]:.2f} , y_ego: {state_np[5]:.2f}, reward: {reward}")
    print("-------")
    if done:
        print("done")

        break
print("all good")
print(f"min_distance:{min_distance}")
print(f"cumulative_reward:{cumulative_reward}")
ray.shutdown()
