import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

from environment.stopping_car import StoppingCar

torch, nn = try_import_torch()


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == "__main__":
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)
    config = {"env": StoppingCar,  #
              "model": {"fcnet_hiddens": [20, 20, 20, 20], "fcnet_activation": "relu"},  # model config,"custom_model": "my_model",
              "vf_share_layers": False,  # try different lrs
              "vf_clip_param": 100, "num_workers": 8,  # parallelism
              # "batch_mode": "complete_episodes", "use_gae": False,  #
              "num_envs_per_worker": 5, "train_batch_size": 2000, "framework": "torch", "horizon": 1000}
    ray.init(local_mode=False, include_dashboard=True)
    trainer = ppo.PPOTrainer(config=config)
    while True:
        train_result = trainer.train()
        print(train_result)
        if train_result["episode_len_mean"] > 800:
            print("Termination condition satisfied")
            break
    checkpoint = trainer.save()
    print("checkpoint saved at", checkpoint)
    ray.shutdown()
