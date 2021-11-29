import numpy as np
import ray
import ray.rllib.agents.sac as sac
from gym.vector.utils import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
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
        prev_safe_layer_size = int(np.product(custom_input_space.shape))
        vf_layers = []
        activation = model_config.get("fcnet_activation")
        hiddens = [32]
        for size in hiddens:
            vf_layers.append(
                SlimFC(
                    in_size=prev_safe_layer_size,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(1.0)))
            prev_safe_layer_size = size
        vf_layers.append(SlimFC(
            in_size=prev_safe_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None))
        self.safe_branch_separate = nn.Sequential(*vf_layers)
        self.last_in =None

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()[:, -2:]  # takes the last 2 values (delta_x, delta_v)
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        self.last_in = input_dict["obs"]
        return fc_out, []

    def value_function(self):
        value = torch.reshape(self.torch_sub_model.value_function(), [-1])
        safety = torch.reshape(self.safe_branch_separate(self.last_in).squeeze(1), [-1])
        return value + safety


def get_PPO_trainer(use_gpu=1):
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)
    config = {"env": StoppingCar,  #
              "model": {"custom_model": "my_model", "fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},  # model config,"
              "lr": 5e-4,
              "num_gpus": use_gpu,
              # "vf_share_layers": False,
              # "vf_clip_param": 100000,
              "grad_clip": 2500,
              "num_workers": 8,  # parallelism
              "batch_mode": "complete_episodes",
              "evaluation_interval": 10,
              # "use_gae": True,  #
              # "lambda": 0.95,  # gae lambda param
              "num_envs_per_worker": 10,
              "train_batch_size": 4000,
              "evaluation_num_episodes": 20,
              "rollout_fragment_length": 1000,
              "framework": "torch",
              "horizon": 1000}
    # trainer = ppo.PPOTrainer(config=config)
    trainer = sac.SACTrainer(config=config)
    return config, trainer


if __name__ == "__main__":
    ray.init(local_mode=True, include_dashboard=True)
    config, trainer = get_PPO_trainer()
    i = 0
    while True:
        train_result = trainer.train()
        print(
            f"i:{i} episode_reward_max:{train_result['episode_reward_max']:.2E}, episode_reward_min:{train_result['episode_reward_min']:.2E}, episode_reward_mean:{train_result['episode_reward_mean']:.2E}, episode_len_mean:{train_result['episode_len_mean']}")
        i += 1
        if train_result["episode_reward_mean"] > -5e1:
            print("Termination condition satisfied")
            break
        if i % 10 == 0:
            checkpoint = trainer.save()
            print("\ncheckpoint saved at", checkpoint)
    checkpoint = trainer.save()
    print("checkpoint saved at", checkpoint)
    ray.shutdown()
