from typing import List, Tuple

import gurobi as grb
import numpy as np
import ray
import torch
from ray.rllib.agents.ppo import ppo
from interval import interval, imath
from training.ppo.tune.tune_train_PPO_fishing import get_PPO_config
from training.ray_utils import convert_ray_policy_to_sequential
from polyhedra.experiments_nn_analysis import Experiment


class FishingExperiment(Experiment):
    def __init__(self):
        env_input_size: int = 1
        super().__init__(env_input_size)
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        self.assign_lbl_fn = self.assign_label
        self.template_2d: np.ndarray = np.array([[1]])
        self.input_boundaries: List = [0.75 - 1, -0.70 + 1]
        template = Experiment.box(1)
        self.input_template: np.ndarray = template
        self.analysis_template: np.ndarray = template
        self.time_horizon = 500
        self.use_rounding = False
        self.rounding_value = 2 ** 4
        # self.plotting_time_interval = 10
        p = Experiment.e(self.env_input_size, 0)
        self.minimum_percentage_population = -0.55
        self.n_actions = 10
        self.unsafe_zone: List[Tuple] = [([p], np.array([self.minimum_percentage_population]))]  # -0.55
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_fishing/PPO_MonitoredFishingEnv_fc1cb_00000_0_2021-04-29_12-42-32/checkpoint_780/checkpoint-780"

    @ray.remote
    def post_milp(self, x, x_label, nn, output_flag, t, template):
        post = []
        for unsafe_check in [True, False]:
            for chosen_action in range(self.n_actions):
                gurobi_model = grb.Model()
                gurobi_model.setParam('OutputFlag', output_flag)
                gurobi_model.setParam('Threads', 2)
                gurobi_model.params.NonConvex = 2
                input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)

                feasible_action = FishingExperiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
                if feasible_action:
                    # apply dynamic
                    x_prime = self.apply_dynamic(input, gurobi_model, chosen_action, env_input_size=self.env_input_size)
                    for A, b in self.unsafe_zone:  # splits the input based on the decision boundary of the ltl property
                        if unsafe_check:
                            Experiment.generate_region_constraints(gurobi_model, A, x_prime, b, self.env_input_size)
                        else:
                            Experiment.generate_region_constraints(gurobi_model, A, x_prime, b, self.env_input_size, invert=True)
                    gurobi_model.update()
                    gurobi_model.optimize()
                    found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                    if found_successor:
                        post.append((tuple(x_prime_results), (x, x_label)))
        return post

    def check_unsafe(self, template, bnds, x_label):
        if x_label >= 5:
            return True
        else:
            return False

    def assign_label(self, x_prime, parent, parent_lbl) -> int:
        if x_prime[0] <= self.minimum_percentage_population:
            return parent_lbl + 1
        else:
            return 0

    @staticmethod
    def apply_dynamic(input, gurobi_model: grb.Model, action, env_input_size):
        '''

        :param input:
        :param gurobi_model:
        :param t:
        :return:
        '''
        n_actions = 10
        K = 1.0
        r = 0.3
        n_fish = input[0]
        quota = (action / n_actions) * K
        fish_population = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name=f"fish_population")
        gurobi_model.addConstr(fish_population[0] == (n_fish + 1) * K, name=f"dyna_constr_1")
        fish_population_prime = gurobi_model.addMVar(shape=(1,), lb=0, name=f"fish_population_prime")  # lb set to 0
        gurobi_model.addConstr(fish_population_prime[0] == (fish_population - quota), name=f"dyna_constr_2")
        # gurobi_model.setObjective(fish_population_prime[0].sum(), grb.GRB.MAXIMIZE)
        # gurobi_model.optimize()
        # max_fish_population_prime = fish_population_prime[0].X  # todo switch to fish_population_prime
        # gurobi_model.setObjective(fish_population_prime[0].sum(), grb.GRB.MINIMIZE)
        # gurobi_model.optimize()
        # min_fish_population_prime = fish_population_prime[0].X
        # step_fish_population_prime = 0.1
        # split_fish_population_prime = np.arange(min(min_fish_population_prime, 0), min(max_fish_population_prime, 0), step_fish_population_prime)
        # split = []
        # for fish_pop in split_fish_population_prime:
        #     lb = fish_pop
        #     ub = min(fish_pop + step_fish_population_prime, max_fish_population_prime)
        #     split.append((interval([lb, ub])))
        # fish_growth_table = []
        # while (len(split)):
        #     fish_pop_interval = split.pop()

        fish_population_post = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name=f"fish_population_post")
        growth = r * fish_population_prime  # * (1.0 - (fish_population_prime[0] / K))
        growth1 = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name=f"growth1")
        gurobi_model.addConstr(growth1 == growth, name=f"dyna_constr_3")
        second_growth = (1.0 - (fish_population_prime / K))
        growth2 = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name=f"growth2")
        gurobi_model.addConstr(growth2 == second_growth, name=f"dyna_constr_4")
        third_growth = growth1 @ growth2
        gurobi_model.addConstr(fish_population_post == fish_population_prime + third_growth, name=f"dyna_constr_5")
        x_prime = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name=f"x_prime")
        gurobi_model.addConstr(x_prime[0] == (fish_population_post / K) - 1, name=f"dyna_constr_6")
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
