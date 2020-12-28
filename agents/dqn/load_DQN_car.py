import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTorchPolicy
import torch.nn

from agents.dqn.train_DQN_car import get_dqn_car_trainer
from environment.pendulum_abstract import PendulumEnv
import ray
from agents.ray_utils import convert_ray_policy_to_sequential, load_sequential_from_ray, get_car_ppo_agent, convert_DQN_ray_policy_to_sequential
from environment.stopping_car import StoppingCar

ray.init(local_mode=True)
trainer, config = get_dqn_car_trainer()
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-28_14-15-09rwm2u8a4/checkpoint_239/checkpoint-239")
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-28_14-57-456gqilswb/checkpoint_8/checkpoint-8")
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-28_15-03-06e7kcr1ke/checkpoint_49/checkpoint-49")
trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-28_15-49-16c3ga4n0f/checkpoint_12/checkpoint-12")  # super safe
policy = trainer.get_policy()
sequential_nn = convert_DQN_ray_policy_to_sequential(policy).cpu()
env = StoppingCar()
state = env.reset()
print(state)
for i in range(1000):
    state_reduced = torch.from_numpy(state).float().unsqueeze(0)[:, -2:]
    action = torch.argmax(sequential_nn(state_reduced)).item()
    print(f"action: {action}")
    state, reward, done, _ = env.step(action)
    print(f"iteration: {i}, delta_v: {state[7]:.2f}, delta_x: {state[6]:.2f}, v_ego: {state[3]:.2f},v_lead: {state[2]:.2f} , y_ego: {state[5]:.2f}")
    print("-------")
    if done:
        print("done")
        break
ray.shutdown()
print("all good")
