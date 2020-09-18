import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTorchPolicy
import torch.nn
from environment.pendulum_abstract import PendulumEnv
import ray

config = {"env": PendulumEnv,  # or "corridor" if registered above "env_config": {"corridor_length": 5, },"custom_model": "my_model"
          "model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},  # model config
          "vf_share_layers": False,  # try different lrs
          "num_workers": 8,  # parallelism
          "num_envs_per_worker": 5, "train_batch_size": 2000, "framework": "torch", "horizon": 1000}  # "batch_mode":"complete_episodes"
ray.init()
trainer = ppo.PPOTrainer(config=config)
trainer.restore("/home/edoardo/ray_results/PPO_PendulumEnv_2020-09-18_11-23-17wpwqe3zd/checkpoint_25/checkpoint-25")
policy: PPOTorchPolicy = trainer.get_policy()
layers_list = []
for seq_layer in policy.model._modules['_hidden_layers']:
    for layer in seq_layer._modules['_model']:
        print(layer)
        layers_list.append(layer)
for layer in policy.model._modules['_logits']._modules['_model']:
    print(layer)
    layers_list.append(layer)
sequential_nn = torch.nn.Sequential(*layers_list)
for i in range(1000):
    test_input = torch.randn(2)
    seq_output = sequential_nn(test_input)
    # trainer_output = trainer.compute_action(test_input.numpy())
    # policy.model({"obs_flat":test_input})
    trainer_output = policy.model._logits(policy.model._hidden_layers(test_input))
    assert torch.all(torch.eq(seq_output,trainer_output)).item()
ray.shutdown()
