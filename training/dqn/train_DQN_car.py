import numpy as np
import ray
import ray.rllib.agents.dqn as dqn
from gym.vector.utils import spaces
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
        input_dict["obs"] = input_dict["obs"].float()[:, -2:]  # takes the last 2 values (delta_x, delta_v)
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


def get_dqn_car_trainer():
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)
    config = {"env": StoppingCar,  #
              "model": {"custom_model": "my_model", "fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},  # model config,

              # "vf_share_layers": False,  # try different lrs
              # "vf_clip_param": 100,
              "lr": 0.001,
              # "clip_rewards": False,  # 500*1000,
              "grad_clip": 250,
              # "worker_side_prioritization": True,
              "num_workers": 8,  # parallelism
              "batch_mode": "complete_episodes",
              "rollout_fragment_length": 1,
              "num_envs_per_worker": 10,
              "train_batch_size": 256,
              "hiddens": [32],
              "framework": "torch", "horizon": 200}
    trainer = dqn.DQNTrainer(config=config)
    return trainer, config


def get_apex_dqn_car_trainer():
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)
    config = {"env": StoppingCar,  #
              "model": {"custom_model": "my_model", "fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},  # model config,
              "n_step": 1,
              "lr": 0.0005,
              "grad_clip": 2500,
              "batch_mode": "complete_episodes",
              "num_workers": 7,  # parallelism
              "num_envs_per_worker": 10,
              "train_batch_size": 512,
              "hiddens": [32],
              "framework": "torch",
              "optimizer": {"num_replay_buffer_shards": 1},
              "horizon": 1000}
    trainer = dqn.ApexTrainer(config=config)
    return trainer, config


if __name__ == "__main__":
    ray.init(local_mode=False, include_dashboard=True)
    trainer, config = get_apex_dqn_car_trainer()
    # trainer.load_checkpoint("/home/edoardo/ray_results/APEX_StoppingCar_2020-12-29_17-10-24qjvbq7ew/checkpoint_42/checkpoint-42")
    i = 0
    while True:
        train_result = trainer.train()
        print(
            f"i:{i} episode_reward_max:{train_result['episode_reward_max']:.2E}, episode_reward_min:{train_result['episode_reward_min']:.2E}, episode_reward_mean:{train_result['episode_reward_mean']:.2E}, episode_len_mean:{train_result['episode_len_mean']}")
        i += 1
        if train_result["episode_reward_mean"] > -5e2:
            print("Termination condition satisfied")
            break
        if i % 10 == 0:
            checkpoint = trainer.save()
            print("\ncheckpoint saved at", checkpoint)
    checkpoint = trainer.save()
    print("checkpoint saved at", checkpoint)
    ray.shutdown()
