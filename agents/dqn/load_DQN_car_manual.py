import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTorchPolicy
import torch.nn

from agents.dqn.train_DQN_car import get_dqn_car_trainer
from environment.pendulum_abstract import PendulumEnv
import ray
from agents.ray_utils import convert_ray_policy_to_sequential, load_sequential_from_ray, get_car_ppo_agent, convert_DQN_ray_policy_to_sequential
from environment.stopping_car import StoppingCar
from polyhedra.net_methods import generate_nn_torch

ray.init(local_mode=True)
sequential_nn = generate_nn_torch(six_dim=False,min_distance=20,max_distance=30).float()
env = StoppingCar()
state = env.reset()
print(state)
for i in range(2000):
    state_reduced = torch.from_numpy(state).float().unsqueeze(0)
    v_ego = state_reduced[:, 3]
    delta_x = state_reduced[:, -1]
    state_reduced = torch.stack([v_ego, delta_x], dim=1)
    action = torch.argmax(sequential_nn(state_reduced)).item()
    print(f"action: {action}")
    state, reward, done, _ = env.step(action)
    print(f"iteration: {i}, delta_x: {state[7]:.2f}, delta_v: {state[6]:.2f}, v_ego: {state[3]:.2f},v_lead: {state[2]:.2f} , y_ego: {state[5]:.2f}")
    print("-------")
    if done:
        print("done")
        break
ray.shutdown()
print("all good")
