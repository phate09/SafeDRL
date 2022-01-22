import os
from typing import List, Tuple

import gurobi as grb
import numpy as np
import ray
import torch
from ray.rllib.agents.ppo import ppo

from polyhedra.experiments_nn_analysis import Experiment
from polyhedra.probabilistic_experiments_nn_analysis import ProbabilisticExperiment
from training.ppo.train_PPO_bouncingball import get_PPO_trainer
from training.ppo.tune.tune_train_PPO_bouncing_ball import get_PPO_config
from training.ray_utils import convert_ray_policy_to_sequential


class BouncingBallExperimentProbabilistic(ProbabilisticExperiment):
    def __init__(self):
        env_input_size: int = 2
        super().__init__(env_input_size)
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        self.template_2d: np.ndarray = np.array([[0, 1], [1, 0]])
        self.template_index = 1
        input_boundaries = self.get_template(self.template_index)

        self.input_boundaries: List = input_boundaries
        self.input_template: np.ndarray = Experiment.box(self.env_input_size)
        self.analysis_template: np.ndarray = Experiment.box(self.env_input_size)
        self.time_horizon = 20
        self.rounding_value = 2 ** 8
        self.load_graph = False
        self.minimum_length = 0.2
        self.max_probability_split = 0.20
        p = Experiment.e(self.env_input_size, 0)
        v = Experiment.e(self.env_input_size, 1)
        self.unsafe_zone: List[Tuple] = [([p, -v, v], np.array([1, 7, 7]))]
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_c7326_00000_0_2021-01-16_05-43-36/checkpoint_36/checkpoint-36"
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00004_4_2021-01-18_23-48-21/checkpoint_10/checkpoint-10"
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_c7326_00000_0_2021-01-16_05-43-36/checkpoint_10/checkpoint-10"

    @ray.remote
    def post_milp(self, x, x_label, nn, output_flag, t, template) -> List[Experiment.SuccessorInfo]:
        """milp method"""
        ranges_probs = self.create_range_bounds_model(template, x, self.env_input_size, nn)

        def standard_op():
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            input = self.generate_input_region(gurobi_model, template, x, self.env_input_size)
            z = self.apply_dynamic(input, gurobi_model, self.env_input_size)
            return gurobi_model, z, input

        post = []
        # case 0
        gurobi_model, z, input = standard_op()
        feasible0 = self.generate_guard(gurobi_model, z, case=0)  # bounce
        if feasible0:  # action is irrelevant in this case
            # apply dynamic
            x_prime = self.apply_dynamic2(z, gurobi_model, case=0, env_input_size=self.env_input_size)
            found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
            if found_successor:
                successor_info = Experiment.SuccessorInfo()
                successor_info.successor = tuple(x_prime_results)
                successor_info.parent = x
                successor_info.parent_lbl = x_label
                successor_info.t = t + 1
                successor_info.action = "case0"  # doesn't matter
                successor_info.lb = 1.0
                successor_info.ub = 1.0
                post.append(successor_info)

        for chosen_action in range(2):
            if ranges_probs[chosen_action][1] <= 1e-6:  # ignore very small probabilities of happening
                # skip action
                continue

            # case 1 : ball going down and hit
            gurobi_model, z, input = standard_op()
            feasible11 = self.generate_guard(gurobi_model, z, case=1)
            if feasible11:
                feasible12 = chosen_action == 1  # check for action =1 over input (not z!)
                if feasible12:
                    # apply dynamic
                    x_prime = self.apply_dynamic2(z, gurobi_model, case=1, env_input_size=self.env_input_size)
                    found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                    if found_successor:
                        successor_info = Experiment.SuccessorInfo()
                        successor_info.successor = tuple(x_prime_results)
                        successor_info.parent = x
                        successor_info.parent_lbl = x_label
                        successor_info.t = t + 1
                        successor_info.action = "policy"
                        successor_info.lb = ranges_probs[chosen_action][0]
                        successor_info.ub = ranges_probs[chosen_action][1]
                        post.append(successor_info)
            # case 2 : ball going up and hit
            gurobi_model, z, input = standard_op()
            feasible21 = self.generate_guard(gurobi_model, z, case=2)
            if feasible21:
                feasible22 = chosen_action == 1  # check for action =1 over input (not z!)
                if feasible22:
                    # apply dynamic
                    x_prime = self.apply_dynamic2(z, gurobi_model, case=2, env_input_size=self.env_input_size)
                    found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                    if found_successor:
                        successor_info = Experiment.SuccessorInfo()
                        successor_info.successor = tuple(x_prime_results)
                        successor_info.parent = x
                        successor_info.parent_lbl = x_label
                        successor_info.t = t + 1
                        successor_info.action = "policy"
                        successor_info.lb = ranges_probs[chosen_action][0]
                        successor_info.ub = ranges_probs[chosen_action][1]
                        post.append(successor_info)
            # case 1 alt : ball going down and NO hit
            gurobi_model, z, input = standard_op()
            feasible11_alt = self.generate_guard(gurobi_model, z, case=1)
            if feasible11_alt:
                feasible12_alt = chosen_action == 0  # check for action = 0 over input (not z!)
                if feasible12_alt:
                    # apply dynamic
                    x_prime = self.apply_dynamic2(z, gurobi_model, case=3, env_input_size=self.env_input_size)  # normal dynamic
                    found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                    if found_successor:
                        successor_info = Experiment.SuccessorInfo()
                        successor_info.successor = tuple(x_prime_results)
                        successor_info.parent = x
                        successor_info.parent_lbl = x_label
                        successor_info.t = t + 1
                        successor_info.action = "policy"
                        successor_info.lb = ranges_probs[chosen_action][0]
                        successor_info.ub = ranges_probs[chosen_action][1]
                        post.append(successor_info)
            # case 2 alt : ball going up and NO hit
            gurobi_model, z, input = standard_op()
            feasible21_alt = self.generate_guard(gurobi_model, z, case=2)
            if feasible21_alt:
                feasible22_alt = chosen_action == 0  # check for action = 0 over input (not z!)
                if feasible22_alt:
                    # apply dynamic
                    x_prime = self.apply_dynamic2(z, gurobi_model, case=3, env_input_size=self.env_input_size)  # normal dynamic
                    found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                    if found_successor:
                        successor_info = Experiment.SuccessorInfo()
                        successor_info.successor = tuple(x_prime_results)
                        successor_info.parent = x
                        successor_info.parent_lbl = x_label
                        successor_info.t = t + 1
                        successor_info.action = "policy"
                        successor_info.lb = ranges_probs[chosen_action][0]
                        successor_info.ub = ranges_probs[chosen_action][1]
                        post.append(successor_info)
        # case 3 : ball out of reach and not bounce
        gurobi_model, z, input = standard_op()
        feasible3 = self.generate_guard(gurobi_model, z, case=3)  # out of reach
        if feasible3:  # action is irrelevant in this case
            # apply dynamic
            x_prime = self.apply_dynamic2(z, gurobi_model, case=3, env_input_size=self.env_input_size)  # normal dynamic
            found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
            if found_successor:
                successor_info = Experiment.SuccessorInfo()
                successor_info.successor = tuple(x_prime_results)
                successor_info.parent = x
                successor_info.parent_lbl = x_label
                successor_info.t = t + 1
                successor_info.action = "case3"  # doesn't matter
                successor_info.lb = 1.0
                successor_info.ub = 1.0
                post.append(successor_info)

        return post

    @staticmethod
    def generate_guard(gurobi_model: grb.Model, input, case=0):
        eps = 1e-6
        if case == 0:  # v <= 0 && p <= 0
            gurobi_model.addConstr(input[1] <= 0, name=f"cond1")
            gurobi_model.addConstr(input[0] <= 0, name=f"cond2")
        if case == 1:  # v_prime <= 0 and p_prime > 4
            gurobi_model.addConstr(input[1] <= 0, name=f"cond1")
            gurobi_model.addConstr(input[0] >= 4, name=f"cond2")
        if case == 2:  # v_prime > 0 and p_prime > 4
            gurobi_model.addConstr(input[1] >= 0, name=f"cond1")
            gurobi_model.addConstr(input[0] >= 4, name=f"cond2")
        if case == 3:  # ball out of reach and not bounce
            gurobi_model.addConstr(input[0] <= 4 - eps, name=f"cond1")
            gurobi_model.addConstr(input[0] >= 0 + eps, name=f"cond2")
        gurobi_model.update()
        gurobi_model.optimize()
        # assert gurobi_model.status == 2, "LP wasn't optimally solved"
        return gurobi_model.status == 2

    @staticmethod
    def apply_dynamic2(input_prime, gurobi_model: grb.Model, case, env_input_size):
        p_prime = input_prime[0]
        v_prime = input_prime[1]
        z = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name=f"x_prime")
        v_second = v_prime
        p_second = p_prime
        if case == 0:  # v <= 0 && p <= 0
            v_second = -(0.90) * v_prime
            p_second = 0
        if case == 1:  # v <= 0 && p >= 4 && action = 1
            v_second = v_prime - 4
            p_second = 4
        if case == 2:  # v >=0 && p >= 4 && action = 1
            v_second = -(0.9) * v_prime - 4
            p_second = 4
        if case == 3:  # p>=0
            v_second = v_prime
            p_second = p_prime

        gurobi_model.addConstr(z[1] == v_second, name=f"dyna_constr2_1")
        gurobi_model.addConstr(z[0] == p_second, name=f"dyna_constr2_2")
        return z

    def get_observation_variable(self, input, gurobi_model):
        return input

    @staticmethod
    def apply_dynamic(input, gurobi_model: grb.Model, env_input_size):
        '''

        :param input:
        :param gurobi_model:
        :param t:
        :return:
        '''

        p = input[0]
        v = input[1]
        dt = 0.1
        z = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name=f"x_prime")
        # pos_max = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name=f"pos_max")
        v_second = v - 9.81 * dt
        p_second = p + dt * v_second
        gurobi_model.addConstr(z[1] == v_second, name=f"dyna_constr_1")
        # gurobi_model.addConstr(pos_max == grb.max_([p_second, 0]), name=f"dyna_constr_2")
        # gurobi_model.addConstr(z[1] == p_second, name=f"dyna_constr_2")
        max_switch = gurobi_model.addMVar(lb=0, ub=1, shape=p_second.shape, vtype=grb.GRB.INTEGER, name=f"max_switch")
        M = 10e6
        # gurobi_model.addConstr(v == grb.max_(0, gurobi_vars[-1]))
        gurobi_model.addConstr(z[0] >= p_second)
        gurobi_model.addConstr(z[0] <= p_second + M * max_switch)
        gurobi_model.addConstr(z[0] >= 0)
        gurobi_model.addConstr(z[0] <= M - M * max_switch)

        return z

    def plot(self, vertices_list, template, template_2d):
        self.generic_plot("v", "p", vertices_list, template, template_2d)

    def get_template(self, mode=0):
        p = Experiment.e(self.env_input_size, 0)
        v = Experiment.e(self.env_input_size, 1)
        if mode == 0:  # large s0
            # input_boundaries = [0, 0, 10, 10]
            input_boundaries = [6, -3, 1, 1]
            return input_boundaries
        if mode == 1:  # small s0
            input_boundaries = [6, -3, 0, 0.1]
            return input_boundaries
        if mode == 2:
            input_boundaries = [9, -3, 0, 0.1]
            return input_boundaries
        if mode == 3:
            input_boundaries = [9, -5, 0, 0.1]
            return input_boundaries
        if mode == 4:
            input_boundaries = [9, -5, 1, 1]
            return input_boundaries

    def get_nn_old(self):
        config, trainer = get_PPO_trainer(use_gpu=0)
        trainer.restore("/home/edoardo/ray_results/PPO_BouncingBall_2021-01-04_18-58-32smp2ln1g/checkpoint_272/checkpoint-272")
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        layers = []
        for l in sequential_nn:
            layers.append(l)
        nn = torch.nn.Sequential(*layers)
        return nn

    def get_nn(self):
        pickled_path = self.nn_path + ".pickle"
        if os.path.exists(pickled_path):
            nn = torch.load(pickled_path, map_location=torch.device('cpu'))
            print("Restored")
            return nn
        config = get_PPO_config(1234, use_gpu=0)
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ray.init(log_to_driver=False, local_mode=False)
    experiment = BouncingBallExperimentProbabilistic()
    experiment.save_dir = "/home/edoardo/ray_results/tune_PPO_bouncing_ball/h20_box_t4_psi01_t1"
    experiment.plotting_time_interval = 60
    experiment.show_progressbar = True
    experiment.show_progress_plot = False
    experiment.max_probability_split = 0.1
    experiment.max_t_split = 1
    # experiment.analysis_template = Experiment.octagon(2)
    # experiment.use_contained = False
    # experiment.load_graph = True
    experiment.input_boundaries = experiment.get_template(3)
    experiment.run_experiment()
