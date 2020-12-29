import ray
from ray.rllib.agents.dqn import DQNTorchPolicy
from ray.rllib.agents.ppo import PPOTorchPolicy, ppo
import torch

from environment.pendulum_abstract import PendulumEnv
from environment.stopping_car import StoppingCar


def convert_ray_policy_to_sequential(policy: PPOTorchPolicy) -> torch.nn.Sequential:
    layers_list = []
    for seq_layer in policy.model._modules['_hidden_layers']:
        for layer in seq_layer._modules['_model']:
            print(layer)
            layers_list.append(layer)
    for layer in policy.model._modules['_logits']._modules['_model']:
        print(layer)
        layers_list.append(layer)
    sequential_nn = torch.nn.Sequential(*layers_list)
    return sequential_nn
def convert_DQN_ray_policy_to_sequential(policy: DQNTorchPolicy) -> torch.nn.Sequential:
    layers_list = []
    for seq_layer in policy.model._modules['torch_sub_model']._modules['_hidden_layers']:
        for layer in seq_layer._modules['_model']:
            print(layer)
            layers_list.append(layer)
    for seq_layer in policy.model._modules['advantage_module']:
        for layer in seq_layer._modules['_model']:
            print(layer)
            layers_list.append(layer)
    sequential_nn = torch.nn.Sequential(*layers_list)
    return sequential_nn

def load_sequential_from_ray(filename: str,trainer):
    trainer.restore(filename)
    return convert_ray_policy_to_sequential(trainer.get_policy())


def get_pendulum_ppo_agent():
    config = {"env": PendulumEnv,  #
              "model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},  # model config
              "vf_share_layers": False,  # try different lrs
              "num_workers": 8,  # parallelism
              "num_envs_per_worker": 5, "train_batch_size": 2000, "framework": "torch", "horizon": 1000}  # "batch_mode":"complete_episodes"
    trainer = ppo.PPOTrainer(config=config)
    return trainer

def get_car_ppo_agent():
    config = {"env": StoppingCar,  #
              "model": {"fcnet_hiddens": [20, 20, 20, 20], "fcnet_activation": "relu"},  # model config,"custom_model": "my_model",
              "vf_share_layers": False,  # try different lrs
              "num_workers": 8,  # parallelism
              # "batch_mode": "complete_episodes", "use_gae": False,  #
              "num_envs_per_worker": 5, "train_batch_size": 2000, "framework": "torch", "horizon": 1000}
    trainer = ppo.PPOTrainer(config=config)
    return trainer
