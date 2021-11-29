from datetime import datetime
from typing import List

import ray
import torch
import torch.nn as nn
import ray.rllib.agents.ppo as ppo
from gym.vector.utils import spaces
from ray.rllib.agents.ddpg import TD3Trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
import numpy as np
from ray import tune
import random
import ray.tune.checkpoint_manager
from ray.tune.trial import Trial

from training.ray_utils import convert_ray_policy_to_sequential2
from environment.bouncing_ball_old import BouncingBall
from environment.cartpole_ray import CartPoleEnv
from environment.stopping_car_continuous import StoppingCar
from ray.tune import Callback

# torch, nn = try_import_torch()
from runnables.runnable.experiment.run_experiment_stopping_car_continuous import StoppingCarContinuousExperiment

custom_input_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(custom_input_space, action_space, num_outputs, model_config, name)
        self.torch_sub_model._logits = torch.nn.Sequential(self.torch_sub_model._logits, torch.nn.Hardtanh(min_val=-3, max_val=3))

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()[:, -2:]
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


class CustomDDPGTorchModel(TorchModelV2, nn.Module):
    """Extension of standard TorchModelV2 for DDPG.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_policy_output() -> pi(s)
        model_out, actions -> get_q_values() -> Q(s, a)
        model_out, actions -> get_twin_q_values() -> Q_twin(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 actor_hidden_activation="relu",
                 actor_hiddens=(64, 64),
                 critic_hidden_activation="relu",
                 critic_hiddens=(64, 64),
                 twin_q=False,
                 add_layer_norm=False):
        """Initialize variables of this model.

        Extra model kwargs:
            actor_hidden_activation (str): activation for actor network
            actor_hiddens (list): hidden layers sizes for actor network
            critic_hidden_activation (str): activation for critic network
            critic_hiddens (list): hidden layers sizes for critic network
            twin_q (bool): build twin Q networks.
            add_layer_norm (bool): Enable layer norm (for param noise).

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the output heads. Those layers for
        forward() should be defined in subclasses of DDPGTorchModel.
        """
        nn.Module.__init__(self)
        super(CustomDDPGTorchModel, self).__init__(obs_space, action_space,
                                                   num_outputs, model_config, name)

        self.bounded = np.logical_and(self.action_space.bounded_above,
                                      self.action_space.bounded_below).any()
        low_action = nn.Parameter(
            torch.from_numpy(self.action_space.low).float())
        low_action.requires_grad = False
        self.register_parameter("low_action", low_action)
        action_range = nn.Parameter(
            torch.from_numpy(self.action_space.high -
                             self.action_space.low).float())
        action_range.requires_grad = False
        self.register_parameter("action_range", action_range)
        self.action_dim = np.product(self.action_space.shape)

        # Build the policy network.
        self.policy_model = nn.Sequential()
        ins = num_outputs
        self.obs_ins = ins
        # activation = get_activation_fn(
        #     actor_hidden_activation, framework="torch")
        activation = torch.nn.Hardtanh
        for i, n in enumerate(actor_hiddens):
            self.policy_model.add_module(
                "action_{}".format(i),
                SlimFC(
                    ins,
                    n,
                    initializer=torch.nn.init.xavier_uniform_,
                    activation_fn=activation))
            # Add LayerNorm after each Dense.
            if add_layer_norm:
                self.policy_model.add_module("LayerNorm_A_{}".format(i),
                                             nn.LayerNorm(n))
            ins = n

        self.policy_model.add_module(
            "action_out",
            SlimFC(
                ins,
                self.action_dim,
                initializer=torch.nn.init.xavier_uniform_,
                activation_fn=None))

        # Use sigmoid to scale to [0,1], but also double magnitude of input to
        # emulate behaviour of tanh activation used in DDPG and TD3 papers.
        # After sigmoid squashing, re-scale to env action space bounds.
        class _Lambda(nn.Module):
            def forward(self_, x):
                sigmoid_out = torch.nn.Hardtanh()(x)
                squashed = self.action_range * sigmoid_out / 2
                return squashed

        # Only squash if we have bounded actions.
        if self.bounded:
            self.policy_model.add_module("action_out_squashed", _Lambda())

        # Build the Q-net(s), including target Q-net(s).
        def build_q_net(name_):
            # activation = get_activation_fn(
            #     critic_hidden_activation, framework="torch")
            activation = torch.nn.ReLU
            # For continuous actions: Feed obs and actions (concatenated)
            # through the NN. For discrete actions, only obs.
            q_net = nn.Sequential()
            ins = self.obs_ins + self.action_dim
            for i, n in enumerate(critic_hiddens):
                q_net.add_module(
                    "{}_hidden_{}".format(name_, i),
                    SlimFC(
                        ins,
                        n,
                        initializer=torch.nn.init.xavier_uniform_,
                        activation_fn=activation))
                ins = n

            q_net.add_module(
                "{}_out".format(name_),
                SlimFC(
                    ins,
                    1,
                    initializer=torch.nn.init.xavier_uniform_,
                    activation_fn=None))
            return q_net

        self.q_model = build_q_net("q")
        if twin_q:
            self.twin_q_model = build_q_net("twin_q")
        else:
            self.twin_q_model = None

    def get_q_values(self, model_out, actions):
        """Return the Q estimates for the most recent forward pass.

        This implements Q(s, a).

        Args:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Tensor): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim].

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        return self.q_model(torch.cat([model_out, actions], -1))

    def get_twin_q_values(self, model_out, actions):
        """Same as get_q_values but using the twin Q net.

        This implements the twin Q(s, a).

        Args:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
            actions (Optional[Tensor]): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim].

        Returns:
            tensor of shape [BATCH_SIZE].
        """
        return self.twin_q_model(torch.cat([model_out, actions], -1))

    def get_policy_output(self, model_out):
        """Return the action output for the most recent forward pass.

        This outputs the support for pi(s). For continuous action spaces, this
        is the action directly. For discrete, is is the mean / std dev.

        Args:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].

        Returns:
            tensor of shape [BATCH_SIZE, action_out_size]
        """
        return self.policy_model(model_out)

    def policy_variables(self, as_dict=False):
        """Return the list of variables for the policy net."""
        if as_dict:
            return self.policy_model.state_dict()
        return list(self.policy_model.parameters())

    def q_variables(self, as_dict=False):
        """Return the list of variables for Q / twin Q nets."""
        if as_dict:
            return {
                **self.q_model.state_dict(),
                **(self.twin_q_model.state_dict() if self.twin_q_model else {})
            }
        return list(self.q_model.parameters()) + \
               (list(self.twin_q_model.parameters()) if self.twin_q_model else [])


def stopper(trial_id, result):
    # if "evaluation" in result:
    #     return result["evaluation"]["episode_reward_min"] > -20 and result["evaluation"]["episode_len_mean"] == 1000
    #
    # else:
    return False


def get_TD3_config(seed, use_gpu: float = 1):
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)
    activation = torch.nn.Hardtanh
    config = {"env": StoppingCar,  #

              "model": {"fcnet_hiddens": [32, 32], "fcnet_activation": "relu", "custom_model": "my_model"},
              # model config," "custom_model": "my_model" "fcnet_activation": "relu" "custom_model": "my_model",
              "lr": 5e-5,
              "num_gpus": use_gpu,
              # "clip_rewards": 1000,
              "vf_clip_param": 500.0,
              "num_workers": 3,  # parallelism
              "num_envs_per_worker": 10,
              "batch_mode": "truncate_episodes",
              "evaluation_interval": 10,
              "evaluation_num_episodes": 30,
              "num_sgd_iter": 10,
              "train_batch_size": 4000,
              "sgd_minibatch_size": 1024,
              "no_done_at_end": True,
              "rollout_fragment_length": 1000,
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
              "env_config": {"cost_fn": tune.grid_search([0]),
                             "epsilon_input": tune.grid_search([0]),
                             "reduced": True}  #
              }
    return config


class MyCallback(Callback):
    def on_train_result(self, iteration, trials, trial, result, **info):
        result = info["result"]
        # if result["episode_reward_mean"] > 6500:
        #     phase = 1
        # else:
        #     phase = 0
        # trainer = info["trainer"]
        # trainer.workers.foreach_worker(
        #     lambda ev: ev.foreach_env(
        #         lambda env: env.set_phase(phase)))
    def on_checkpoint(self, iteration: int, trials,
                      trial, checkpoint, **info):
        experiment = StoppingCarContinuousExperiment()
        experiment.nn_path = checkpoint.value
        experiment.save_dir = "/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/test"
        experiment.plotting_time_interval = 60
        experiment.show_progressbar = False
        experiment.show_progress_plot = False
        experiment.use_rounding = False
        # template = Experiment.octagon(experiment.env_input_size)
        _, template = StoppingCarContinuousExperiment.get_template(1)
        experiment.analysis_template = template  # standard
        input_boundaries = [40, -40, 10, -0, 28, -28, 36, -36, 0, -0, 0, 0, 0]
        experiment.input_boundaries = input_boundaries
        experiment.time_horizon = 150

        def get_nn():
            # from training.td3.tune.tune_train_TD3_car import get_TD3_config
            config = get_TD3_config(1234)
            trainer = ppo.PPOTrainer(config)
            trainer.restore(experiment.nn_path)
            policy = trainer.get_policy()
            sequential_nn = convert_ray_policy_to_sequential2(policy).cpu()
            # l0 = torch.nn.Linear(6, 2, bias=False)
            # l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]], dtype=torch.float32))
            # layers = [l0]
            # for l in sequential_nn:
            #     layers.append(l)
            #
            # nn = torch.nn.Sequential(*layers)
            nn = sequential_nn
            # ray.shutdown()
            return nn
        experiment.get_nn_fn = get_nn
        elapsed_seconds, safe, max_t = experiment.run_experiment()
        if safe:
            trial.set_status(Trial.TERMINATED)
        return safe

if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ray.init(local_mode=False, include_dashboard=True, log_to_driver=False)
    config = get_TD3_config(use_gpu=1, seed=seed)
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tune.run(
        "PPO",
        # stop={"info/num_steps_trained": 2e8, "episode_reward_mean": -2e1},  #
        stop=stopper,
        config=config,
        name=f"tune_TD3_stopping_car_continuous",
        checkpoint_freq=10,
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=10,
        checkpoint_at_end=True,
        log_to_file=True,
        callbacks=[MyCallback()],
        # restore="/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/PPO_StoppingCar_28110_00000_0_cost_fn=0,epsilon_input=0_2021-03-07_17-40-07/checkpoint_1250/checkpoint-1250",
        # restore="/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/PPO_StoppingCar_7bdde_00000_0_cost_fn=0,epsilon_input=0_2021-03-09_11-49-20/checkpoint_1810/checkpoint-1810",
        # restore="/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/PPO_StoppingCar_5e486_00000_0_cost_fn=0,epsilon_input=0_2021-03-09_13-50-11/checkpoint_2060/checkpoint-2060",
        # restore="/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/PPO_StoppingCar_afe33_00000_0_cost_fn=0,epsilon_input=0_2021-03-09_14-13-57/checkpoint_2230/checkpoint-2230",
        # resume="PROMPT",
        verbose=1,
        num_samples=1
    )
    # TD3Trainer
    ray.shutdown()
