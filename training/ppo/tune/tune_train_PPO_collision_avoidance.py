import random
from datetime import datetime

import numpy as np
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

from environment.collision_avoidance import ColAvoidEnvDiscrete

torch, nn = try_import_torch()


# custom_input_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        # input_dict["obs"] = input_dict["obs"].float()[:, -2:]
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


def stopper(trial_id, result):
    if "evaluation" in result:
        return result["evaluation"]["episode_reward_min"] > 800 and result["evaluation"]["episode_len_mean"] == 1000
    else:
        return False


def get_PPO_config(seed, use_gpu: float = 1):
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)
    config = {"env": ColAvoidEnvDiscrete,  #
              "model": {"custom_model": "my_model", "fcnet_hiddens": [32, 32], "fcnet_activation": "relu"},  # model config," "custom_model": "my_model"
              "vf_share_layers": False,
              "lr": 5e-4,
              "num_gpus": use_gpu,
              "vf_clip_param": 100000,
              "grad_clip": 300,
              # "clip_rewards": 5,
              "num_workers": 7,  # parallelism
              "num_envs_per_worker": 5,
              "batch_mode": "truncate_episodes",
              "evaluation_interval": 10,
              "evaluation_num_episodes": 10,
              "use_gae": True,  #
              "lambda": 0.95,  # gae lambda param
              "num_sgd_iter": 10,
              "train_batch_size": 4096,
              "rollout_fragment_length": 256,
              "framework": "torch",
              "horizon": 512,
              "seed": seed,
              "evaluation_config": {
                  # Example: overriding env_config, exploration, etc:
                  # "env_config": {...},
                  "explore": False
              },
              # "env_config": {"tau": tune.grid_search([0.1, 0.05])}
              }
    return config


if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ray.init(local_mode=False, include_dashboard=True, log_to_driver=False, num_gpus=1)
    config = get_PPO_config(use_gpu=0.5, seed=seed)
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tune.run(
        "PPO",
        stop=stopper,  # {"info/num_steps_trained": 5e6, "episode_reward_min": 850},
        config=config,
        name=f"tune_PPO_collision_avoidance",
        checkpoint_freq=10,
        checkpoint_at_end=True,
        keep_checkpoints_num=5,
        checkpoint_score_attr="episode_reward_mean",
        log_to_file=True,
        # resume="PROMPT",
        verbose=1,
        num_samples=1
    )
    ray.shutdown()
