from typing import List, Tuple

import gurobi as grb
import numpy as np
import ray
import torch
from ray.rllib.agents.ppo import ppo

from agents.ppo.tune.tune_train_PPO_fishing import get_PPO_config
from agents.ray_utils import convert_ray_policy_to_sequential
from polyhedra.experiments_nn_analysis import Experiment


class FishingExperiment(Experiment):
    def __init__(self):
        env_input_size: int = 1
        super().__init__(env_input_size)
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        self.template_2d: np.ndarray = np.array([[1]])
        self.input_boundaries: List = [0.75, -0.75]
        template = Experiment.box(1)
        self.input_template: np.ndarray = template
        self.analysis_template: np.ndarray = template
        self.time_horizon = 500
        self.rounding_value = 2 ** 8
        # self.plotting_time_interval = 10
        p = Experiment.e(self.env_input_size, 0)
        self.unsafe_zone: List[Tuple] = [([p], np.array([-0.25]))]
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_fishing/PPO_fishing_55f44_00000_0_2021-04-27_12-54-35/checkpoint_1460/checkpoint-1460"

    @ray.remote
    def post_milp(self, x, nn, output_flag, t, template):
        post = []
        for chosen_action in range(100):
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            gurobi_model.setParam('Threads', 2)
            input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
            feasible_action = FishingExperiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
            if feasible_action:
                # apply dynamic
                x_prime = self.apply_dynamic(input, gurobi_model, chosen_action, env_input_size=self.env_input_size)
                gurobi_model.update()
                gurobi_model.optimize()
                found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                if found_successor:
                    post.append(tuple(x_prime_results))
        return post

    @staticmethod
    def apply_dynamic(input, gurobi_model: grb.Model, action, env_input_size):
        '''

        :param input:
        :param gurobi_model:
        :param t:
        :return:
        '''
        n_actions = 100
        K = 1.0
        r = 0.3
        n_fish = input[0]
        quota = (action / n_actions) * K
        fish_population = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name=f"x_prime")
        gurobi_model.addConstr(fish_population[0] == (n_fish + 1) * K, name=f"dyna_constr_1")
        fish_population_prime = gurobi_model.addMVar(shape=(1,), lb=0, name=f"fish_population_prime")  # lb set to 0
        gurobi_model.addConstr(fish_population_prime[0] == (fish_population - quota), name=f"dyna_constr_2")
        fish_population_post = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name=f"fish_population_post")
        growth = fish_population_prime[0] + r * fish_population_prime[0]  # * (1.0 - (fish_population_prime[0] / K))
        second_growth = (1.0 - (fish_population_prime[0] / K))
        third_growth = growth * second_growth
        gurobi_model.addConstr(fish_population_post[0] == growth, name=f"dyna_constr_3")
        x_prime = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name=f"x_prime")
        gurobi_model.addConstr(x_prime[0] == fish_population_post / K - 1, name=f"dyna_constr_4")
        return x_prime

    def plot(self, vertices_list, template, template_2d):
        self.generic_plot1d("t", "population", vertices_list, template, template_2d)
        pass

    def get_nn(self):
        config = get_PPO_config(1234)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(self.nn_path)
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        layers = []
        for l in sequential_nn:
            layers.append(l)
        nn = torch.nn.Sequential(*layers)
        return nn


if __name__ == '__main__':
    ray.init(local_mode=True, log_to_driver=False)
    experiment = FishingExperiment()
    experiment.run_experiment()
