import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTorchPolicy
import torch.nn
from environment.pendulum_abstract import PendulumEnv
import ray
from agents.ray_utils import convert_ray_policy_to_sequential, load_sequential_from_ray, get_car_ppo_agent

# config = {"env": PendulumEnv,  # or "corridor" if registered above "env_config": {"corridor_length": 5, },"custom_model": "my_model"
#           "model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},  # model config
#           "vf_share_layers": False,  # try different lrs
#           "num_workers": 8,  # parallelism
#           "num_envs_per_worker": 5, "train_batch_size": 2000, "framework": "torch", "horizon": 1000}  # "batch_mode":"complete_episodes"
from environment.stopping_car import StoppingCar

ray.init()
# trainer = ppo.PPOTrainer(config=config)
# trainer.restore("/home/edoardo/ray_results/PPO_PendulumEnv_2020-09-18_11-23-17wpwqe3zd/checkpoint_25/checkpoint-25")
# policy: PPOTorchPolicy = trainer.get_policy()
trainer = get_car_ppo_agent()
# trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-10-26_09-41-35rhlw7i97/checkpoint_24/checkpoint-24")
trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-10-26_13-05-462lj8dhid/checkpoint_113/checkpoint-113")
policy = trainer.get_policy()
sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
env = StoppingCar()
state = env.reset()
print(state)
for i in range(1000):
    action = torch.argmax(sequential_nn(torch.from_numpy(state).float().unsqueeze(0))).item()
    print(f"action: {action}")
    state, reward, done, _ = env.step(action)
    print(f"iteration: {i}, delta_v: {state[7]:.2f}, delta_x: {state[6]:.2f}, v_ego: {state[3]:.2f},v_lead: {state[2]:.2f} , y_ego: {state[5]:.2f}")
    print("-------")
    if done:
        print("done")
        break
ray.shutdown()
print("all good")
