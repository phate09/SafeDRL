import random
from datetime import datetime

import numpy as np
import ray
from gym.vector.utils import spaces
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

from environment.stopping_car import StoppingCar

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


def get_PPO_config(seed, use_gpu=1):
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)
    config = {"env": StoppingCar,  #
              "model": {"custom_model": "my_model", "fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},  # model config," "custom_model": "my_model"
              "vf_share_layers": False,
              "lr": 5e-4,
              "num_gpus": use_gpu,
              "vf_clip_param": 100000,
              "grad_clip": 2500,
              "clip_rewards": 5,
              "num_workers": 3,  # parallelism
              "num_envs_per_worker": 10,
              "batch_mode": "complete_episodes",
              "evaluation_interval": 10,
              "evaluation_num_episodes": 20,
              "use_gae": True,  #
              "lambda": 0.95,  # gae lambda param
              "num_sgd_iter": 10,
              "train_batch_size": 4000,
              "sgd_minibatch_size": 1024,
              "rollout_fragment_length": 1000,
              "framework": "torch",
              "horizon": 1000,
              "seed": seed,
              "evaluation_config": {
                  # Example: overriding env_config, exploration, etc:
                  # "env_config": {...},
                  "explore": False
              },
              "env_config": {"cost_fn": tune.grid_search([0]),
                             "epsilon_input": tune.grid_search([0])}  #
              }
    return config


if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ray.init(local_mode=True, include_dashboard=True, log_to_driver=False)
    config = get_PPO_config(use_gpu=0.5, seed=seed)
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tune.run(
        "PPO",
        stop={"info/num_steps_trained": 2e8, "episode_reward_mean": -2e1},
        config=config,
        name=f"tune_PPO_stopping_car",
        checkpoint_freq=10,
        checkpoint_at_end=True,
        log_to_file=True,
        # resume="PROMPT",
        verbose=1,
        num_samples=10
    )
    ray.shutdown()
