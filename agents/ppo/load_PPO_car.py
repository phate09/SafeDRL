import ray
import torch.nn
import numpy as np
from agents.ray_utils import convert_ray_policy_to_sequential
from environment.stopping_car import StoppingCar
import ray.rllib.agents.ppo as ppo
from agents.ppo.tune.tune_train_PPO_car import get_PPO_config

ray.init()
# config, trainer = get_PPO_trainer(use_gpu=0)

config = get_PPO_config(1234)
trainer = ppo.PPOTrainer(config=config)
# trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-12-30_17-06-3265yz3d63/checkpoint_65/checkpoint-65") # 5e-2 ~19.8 delta x
trainer.restore("/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_e6ed1_00000_0_cost_fn=0_2021-01-15_19-57-40/checkpoint_440/checkpoint-440")
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
# env.x_lead = 30
# env.x_ego = 0
# env.v_lead = 28
# env.v_ego = 36
min_distance = 9999
state_np = np.array([env.x_lead, env.x_ego, env.v_lead, env.v_ego, env.y_lead, env.y_ego, env.v_lead - env.v_ego, env.x_lead - env.x_ego])
print(state_np)
for n in range(10):
    cumulative_reward = 0
    for i in range(1000):
        state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]
        state = torch.from_numpy(state_np).float().unsqueeze(0)[:, :-2]
        action_score = sequential_nn(state_reduced)
        action_score2 = sequential_nn2(state)
        action = torch.argmax(action_score).item()
        action2 = torch.argmax(action_score2).item()
        # assert action == action2
        print(f"action: {action2}")
        state_np, reward, done, _ = env.step(action2)
        min_distance = min(state_np[7], min_distance)
        cumulative_reward += reward
        print(f"iteration: {i}, delta_x: {state_np[7]:.2f}, delta_v: {state_np[6]:.2f}, v_ego: {state_np[3]:.2f},v_lead: {state_np[2]:.2f} , y_ego: {state_np[5]:.2f}, reward: {reward}")
        if done:
            print("done")

            break
    print("-------")
    print(f"cumulative_reward:{cumulative_reward}")
print("all good")
print(f"min_distance:{min_distance}")
ray.shutdown()
