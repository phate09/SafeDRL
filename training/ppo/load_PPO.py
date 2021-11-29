import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTorchPolicy
import torch.nn
from environment.pendulum_abstract import PendulumEnv
import ray
from training.ray_utils import convert_ray_policy_to_sequential, load_sequential_from_ray, get_pendulum_ppo_agent

# config = {"env": PendulumEnv,  # or "corridor" if registered above "env_config": {"corridor_length": 5, },"custom_model": "my_model"
#           "model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},  # model config
#           "vf_share_layers": False,  # try different lrs
#           "num_workers": 8,  # parallelism
#           "num_envs_per_worker": 5, "train_batch_size": 2000, "framework": "torch", "horizon": 1000}  # "batch_mode":"complete_episodes"
ray.init()
# trainer = ppo.PPOTrainer(config=config)
# trainer.restore("/home/edoardo/ray_results/PPO_PendulumEnv_2020-09-18_11-23-17wpwqe3zd/checkpoint_25/checkpoint-25")
# policy: PPOTorchPolicy = trainer.get_policy()
trainer = get_pendulum_ppo_agent()
trainer.restore("/home/edoardo/ray_results/PPO_PendulumEnv_2020-09-18_11-23-17wpwqe3zd/checkpoint_25/checkpoint-25")
policy = trainer.get_policy()
sequential_nn = convert_ray_policy_to_sequential(policy)
for i in range(1000):
    test_input = torch.randn(2)
    seq_output = sequential_nn(test_input)
    trainer_output = policy.model._logits(policy.model._hidden_layers(test_input))
    assert torch.all(torch.eq(seq_output, trainer_output)).item()
ray.shutdown()
print("all good")
