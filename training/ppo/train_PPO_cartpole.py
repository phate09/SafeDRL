import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from gym.vector.utils import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

from environment.cartpole_ray import CartPoleEnv

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


def get_PPO_trainer(use_gpu=1):
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)
    config = {"env": CartPoleEnv,  #
              "model": {"custom_model": "my_model", "fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},  # model config," "custom_model": "my_model"
              "vf_share_layers": False,
              "lr": 5e-4,
              "num_gpus": use_gpu,
              "vf_clip_param": 100000,
              "grad_clip": 2500,
              "num_workers": 8,  # parallelism
              "batch_mode": "truncate_episodes",
              "evaluation_interval": 10,
              "use_gae": True,  #
              "lambda": 0.95,  # gae lambda param
              "num_envs_per_worker": 5,
              "num_sgd_iter": 10,
              "train_batch_size": 4000,
              "evaluation_num_episodes": 20,
              "rollout_fragment_length": 2000,
              "framework": "torch",
              "horizon": 8000,
              "evaluation_config": {
                  # Example: overriding env_config, exploration, etc:
                  # "env_config": {...},
                  "explore": False
              },
              }
    trainer = ppo.PPOTrainer(config=config)
    return config, trainer


if __name__ == "__main__":
    ray.init(local_mode=False, include_dashboard=True)
    config, trainer = get_PPO_trainer()
    trainer.load_checkpoint("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-09_15-34-25f0ld3dex/checkpoint_30/checkpoint-30")
    i = 0
    while True:
        train_result = trainer.train()
        print(
            f"i:{i} episode_reward_max:{train_result['episode_reward_max']:.2E}, episode_reward_min:{train_result['episode_reward_min']:.2E}, episode_reward_mean:{train_result['episode_reward_mean']:.2E}, episode_len_mean:{train_result['episode_len_mean']}")
        i += 1
        if train_result["episode_reward_mean"] > 7950:
            print("Termination condition satisfied")
            break
        if i % 10 == 0:
            checkpoint = trainer.save()
            print("\ncheckpoint saved at", checkpoint)
    checkpoint = trainer.save()
    print("checkpoint saved at", checkpoint)
    ray.shutdown()
