from datetime import datetime

import ray
import ray.rllib.agents.ppo as ppo
from gym.vector.utils import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
import numpy as np
from ray import tune
import random
from environment.bouncing_ball_old import BouncingBall
from environment.cartpole_ray import CartPoleEnv
from environment.stopping_car_continuous import StoppingCar
from ray.tune import Callback

torch, nn = try_import_torch()

custom_input_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, custom_input_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(custom_input_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()[:, -2:]
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


def get_TD3_config(seed, use_gpu=1):
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)
    config = {"env": StoppingCar,  #
              # "model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},  # model config," "custom_model": "my_model""custom_model": "my_model",
              # "critic_lr": 0.001,
              # "actor_lr": 0.001,
              # "use_huber": True,
              # "huber_threshold": 1.0,
              # "l2_reg": 0.000001,
              # "learning_starts": 500,
              # "rollout_fragment_length": 1,
              # "train_batch_size": 64,
              # "num_gpus": use_gpu,
              # "twin_q": True,
              # "gamma": 0.99,
              # "timesteps_per_iteration": 600,
              # "target_network_update_freq": 0,
              # "tau": 0.001,
              # "clip_rewards": 1000,
              "num_workers": 3,  # parallelism
              "num_envs_per_worker": 10,
              # # "batch_mode": "complete_episodes",
              "evaluation_interval": 5,
              "evaluation_num_episodes": 20,
              "actor_hiddens": [64, 64],
              "critic_hiddens": [64, 64],
              "learning_starts": 5000,
              "exploration_config":
                  {"random_timesteps": 5000},
              "framework": "torch",
              "horizon": 1000,
              "seed": seed,
              "evaluation_config": {
                  # Example: overriding env_config, exploration, etc:
                  # "env_config": {...},
                  "explore": False
              },
              "env_config": {"cost_fn": tune.grid_search([3]),
                             "epsilon_input": tune.grid_search([0]),
                             "reduced": True}  #
              }
    return config


if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ray.init(local_mode=False, include_dashboard=True, log_to_driver=False)
    config = get_TD3_config(use_gpu=0.5, seed=seed)
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tune.run(
        "TD3",
        stop={"info/num_steps_trained": 2e8, "episode_reward_mean": -2e1},  #
        config=config,
        name=f"tune_TD3_stopping_car_continuous",
        checkpoint_freq=10,
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=10,
        checkpoint_at_end=True,
        log_to_file=True,
        resume="PROMPT",
        verbose=1,
        num_samples=1
    )
    ray.shutdown()
