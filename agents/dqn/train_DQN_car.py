import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
from gym.vector.utils import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
import numpy as np
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
              "model": {"custom_model": "my_model", "fcnet_hiddens": [20, 20, 20], "fcnet_activation": "relu"},  # model config,

              # "vf_share_layers": False,  # try different lrs
              # "vf_clip_param": 100,
              "clip_rewards": False,
              "num_workers": 8,  # parallelism
              "batch_mode": "complete_episodes",  # "use_gae": False,  #
              "num_envs_per_worker": 5, "train_batch_size": 2000, "framework": "torch", "horizon": 1000}
    trainer = dqn.DQNTrainer(config=config)
    return trainer, config


if __name__ == "__main__":
    ray.init(local_mode=True, include_dashboard=True)
    trainer, config = get_dqn_car_trainer()
    while True:
        train_result = trainer.train()
        print(train_result)
        if train_result["episode_len_mean"] > 800:
            print("Termination condition satisfied")
            break
    checkpoint = trainer.save()
    print("checkpoint saved at", checkpoint)
    ray.shutdown()
