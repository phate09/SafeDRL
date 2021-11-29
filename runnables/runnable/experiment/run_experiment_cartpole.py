import math
from typing import List, Tuple

import gurobi as grb
import numpy as np
import ray
import torch
from interval import interval, imath
from ray.rllib.agents.ppo import ppo

from environment.cartpole_ray import CartPoleEnv
from polyhedra.experiments_nn_analysis import Experiment
from training.ppo.train_PPO_cartpole import get_PPO_trainer
from training.ppo.tune.tune_train_PPO_cartpole import get_PPO_config
from training.ray_utils import convert_ray_policy_to_sequential


class CartpoleExperiment(Experiment):
    def __init__(self):
        env_input_size: int = 4
        super().__init__(env_input_size)
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        self.template_2d: np.ndarray = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        input_boundaries, input_template = self.get_template(0)
        self.input_boundaries: List = input_boundaries
        self.input_template: np.ndarray = input_template
        _, template = self.get_template(1)
        self.analysis_template: np.ndarray = template
        safe_angle = 12 * 2 * math.pi / 360
        theta = [self.e(4, 2)]
        neg_theta = [-self.e(4, 2)]
        self.unsafe_zone: List[Tuple] = [(theta, np.array([-safe_angle])), (neg_theta, np.array([-safe_angle]))]
        # self.use_rounding = False
        self.rounding_value = 1024
        self.time_horizon = 300
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00001_1_cost_fn=1,tau=0.001_2021-01-16_20-25-43/checkpoint_3090/checkpoint-3090"
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00002_2_cost_fn=2,tau=0.001_2021-01-16_20-33-36/checkpoint_3334/checkpoint-3334"
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00000_0_cost_fn=0,tau=0.001_2021-01-16_20-25-43/checkpoint_193/checkpoint-193"
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00003_3_cost_fn=0,tau=0.02_2021-01-16_23-08-42/checkpoint_190/checkpoint-190"
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00004_4_cost_fn=1,tau=0.02_2021-01-16_23-14-15/checkpoint_3334/checkpoint-3334" #not determined
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_cartpole/PPO_CartPoleEnv_0205e_00005_5_cost_fn=2,tau=0.02_2021-01-16_23-27-15/checkpoint_3334/checkpoint-3334" #not determined
        # self.tau = 0.001
        self.tau = 0.02

    @ray.remote
    def post_milp(self, x, nn, output_flag, t, template):
        """milp method"""
        post = []
        for chosen_action in range(2):
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            gurobi_model.setParam('Threads', 2)
            input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
            max_theta, min_theta, max_theta_dot, min_theta_dot = self.get_theta_bounds(gurobi_model, input)
            sin_cos_table = self.get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action=chosen_action)
            feasible_action = CartpoleExperiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
            if feasible_action:
                thetaacc, xacc = CartpoleExperiment.generate_angle_milp(gurobi_model, input, sin_cos_table)
                # apply dynamic
                x_prime = self.apply_dynamic(input, gurobi_model, thetaacc=thetaacc, xacc=xacc, env_input_size=self.env_input_size)
                gurobi_model.update()
                gurobi_model.optimize()
                found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                if found_successor:
                    post.append(tuple(x_prime_results))
        return post

    def apply_dynamic(self, input, gurobi_model: grb.Model, thetaacc, xacc, env_input_size):
        '''

        :param costheta: gurobi variable containing the range of costheta values
        :param sintheta: gurobi variable containin the range of sintheta values
        :param input:
        :param gurobi_model:
        :param t:
        :return:
        '''

        tau = self.tau  # 0.001  # seconds between state updates
        x = input[0]
        x_dot = input[1]
        theta = input[2]
        theta_dot = input[3]
        z = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name=f"x_prime")
        x_prime = x + tau * x_dot
        x_dot_prime = x_dot + tau * xacc
        theta_prime = theta + tau * theta_dot
        theta_dot_prime = theta_dot + tau * thetaacc
        gurobi_model.addConstr(z[0] == x_prime, name=f"dyna_constr_1")
        gurobi_model.addConstr(z[1] == x_dot_prime, name=f"dyna_constr_2")
        gurobi_model.addConstr(z[2] == theta_prime, name=f"dyna_constr_3")
        gurobi_model.addConstr(z[3] == theta_dot_prime, name=f"dyna_constr_4")
        return z

    @staticmethod
    def get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action):
        assert min_theta <= max_theta, f"min_theta = {min_theta},max_theta={max_theta}"
        assert min_theta_dot <= max_theta_dot, f"min_theta_dot = {min_theta_dot},max_theta_dot={max_theta_dot}"
        step_theta = 0.1
        step_theta_dot = 0.1
        step_thetaacc = 0.3
        min_theta = max(min_theta, -math.pi / 2)
        max_theta = min(max_theta, math.pi / 2)
        split_theta1 = np.arange(min(min_theta, 0), min(max_theta, 0), step_theta)
        split_theta2 = np.arange(max(min_theta, 0), max(max_theta, 0), step_theta)
        split_theta = np.concatenate([split_theta1, split_theta2])
        split_theta_dot1 = np.arange(min(min_theta_dot, 0), min(max_theta_dot, 0), step_theta)
        split_theta_dot2 = np.arange(max(min_theta_dot, 0), max(max_theta_dot, 0), step_theta)
        split_theta_dot = np.concatenate([split_theta_dot1, split_theta_dot2])
        env = CartPoleEnv(None)
        force = env.force_mag if action == 1 else -env.force_mag

        split = []
        for t_dot in split_theta_dot:
            for theta in split_theta:
                lb_theta_dot = t_dot
                ub_theta_dot = min(t_dot + step_theta_dot, max_theta_dot)
                lb = theta
                ub = min(theta + step_theta, max_theta)
                split.append((interval([lb_theta_dot, ub_theta_dot]), interval([lb, ub])))
        sin_cos_table = []
        while (len(split)):
            theta_dot, theta = split.pop()
            sintheta = imath.sin(theta)
            costheta = imath.cos(theta)
            temp = (force + env.polemass_length * theta_dot ** 2 * sintheta) / env.total_mass
            thetaacc: interval = (env.gravity * sintheta - costheta * temp) / (env.length * (4.0 / 3.0 - env.masspole * costheta ** 2 / env.total_mass))
            xacc = temp - env.polemass_length * thetaacc * costheta / env.total_mass
            if thetaacc[0].sup - thetaacc[0].inf > step_thetaacc:
                # split theta theta_dot
                mid_theta = (theta[0].sup + theta[0].inf) / 2
                mid_theta_dot = (theta_dot[0].sup + theta_dot[0].inf) / 2
                theta_1 = interval([theta[0].inf, mid_theta])
                theta_2 = interval([mid_theta, theta[0].sup])
                theta_dot_1 = interval([theta_dot[0].inf, mid_theta_dot])
                theta_dot_2 = interval([mid_theta_dot, theta_dot[0].sup])
                split.append((theta_1, theta_dot_1))
                split.append((theta_1, theta_dot_2))
                split.append((theta_2, theta_dot_1))
                split.append((theta_2, theta_dot_2))
            else:
                sin_cos_table.append((theta, theta_dot, thetaacc, xacc))
        return sin_cos_table

    @staticmethod
    def get_theta_bounds(gurobi_model, input):
        gurobi_model.setObjective(input[2].sum(), grb.GRB.MAXIMIZE)
        gurobi_model.optimize()
        max_theta = gurobi_model.getVars()[2].X

        gurobi_model.setObjective(input[2].sum(), grb.GRB.MINIMIZE)
        gurobi_model.optimize()
        min_theta = gurobi_model.getVars()[2].X

        gurobi_model.setObjective(input[3].sum(), grb.GRB.MAXIMIZE)
        gurobi_model.optimize()
        max_theta_dot = gurobi_model.getVars()[3].X

        gurobi_model.setObjective(input[3].sum(), grb.GRB.MINIMIZE)
        gurobi_model.optimize()
        min_theta_dot = gurobi_model.getVars()[3].X
        return max_theta, min_theta, max_theta_dot, min_theta_dot

    @staticmethod
    def generate_angle_milp(gurobi_model: grb.Model, input, sin_cos_table: List[Tuple]):
        """MILP method
        input: theta, thetadot
        output: thetadotdot, xdotdot (edited)
        l_{theta, i}, l_{thatdot,i}, l_{thetadotdot, i}, l_{xdotdot, i}, u_....
        sum_{i=1}^k l_{x,i} - l_{x,i}*z_i <= x <= sum_{i=1}^k u_{x,i} - u_{x,i}*z_i, for each variable x
        sum_{i=1}^k l_{theta,i} - l_{theta,i}*z_i <= theta <= sum_{i=1}^k u_{theta,i} - u_{theta,i}*z_i
        """
        theta = input[2]
        theta_dot = input[3]
        k = len(sin_cos_table)
        zs = []
        thetaacc = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name="thetaacc")
        xacc = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name="xacc")
        for i in range(k):
            z = gurobi_model.addMVar(lb=0, ub=1, shape=(1,), vtype=grb.GRB.INTEGER, name=f"part_{i}")
            zs.append(z)
        gurobi_model.addConstr(k - 1 == sum(zs), name=f"const_milp1")
        theta_lb = 0
        theta_ub = 0
        theta_dot_lb = 0
        theta_dot_ub = 0
        thetaacc_lb = 0
        thetaacc_ub = 0
        xacc_lb = 0
        xacc_ub = 0
        for i in range(k):
            theta_interval, theta_dot_interval, theta_acc_interval, xacc_interval = sin_cos_table[i]
            theta_lb += theta_interval[0].inf - theta_interval[0].inf * zs[i]
            theta_ub += theta_interval[0].sup - theta_interval[0].sup * zs[i]
            theta_dot_lb += theta_dot_interval[0].inf - theta_dot_interval[0].inf * zs[i]
            theta_dot_ub += theta_dot_interval[0].sup - theta_dot_interval[0].sup * zs[i]

            thetaacc_lb += theta_acc_interval[0].inf - theta_acc_interval[0].inf * zs[i]
            thetaacc_ub += theta_acc_interval[0].sup - theta_acc_interval[0].sup * zs[i]

            xacc_lb += xacc_interval[0].inf - xacc_interval[0].inf * zs[i]
            xacc_ub += xacc_interval[0].sup - xacc_interval[0].sup * zs[i]

        gurobi_model.addConstr(theta >= theta_lb, name=f"theta_guard1")
        gurobi_model.addConstr(theta <= theta_ub, name=f"theta_guard2")
        gurobi_model.addConstr(theta_dot >= theta_dot_lb, name=f"theta_dot_guard1")
        gurobi_model.addConstr(theta_dot <= theta_dot_ub, name=f"theta_dot_guard2")

        gurobi_model.addConstr(thetaacc >= thetaacc_lb, name=f"thetaacc_guard1")
        gurobi_model.addConstr(thetaacc <= thetaacc_ub, name=f"thetaacc_guard2")
        gurobi_model.addConstr(xacc >= xacc_lb, name=f"xacc_guard1")
        gurobi_model.addConstr(xacc <= xacc_ub, name=f"xacc_guard2")

        gurobi_model.update()
        gurobi_model.optimize()
        # assert gurobi_model.status == 2, "LP wasn't optimally solved"
        return thetaacc, xacc

    def plot(self, vertices_list, template, template_2d):
        self.generic_plot("theta", "theta_dot", vertices_list, template, template_2d)

    def get_template(self, mode=0):
        x = Experiment.e(self.env_input_size, 0)
        x_dot = Experiment.e(self.env_input_size, 1)
        theta = Experiment.e(self.env_input_size, 2)
        theta_dot = Experiment.e(self.env_input_size, 3)
        if mode == 0:  # box directions with intervals
            # input_boundaries = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
            input_boundaries = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
            # input_boundaries = [0.04373426, -0.04373426, -0.04980056, 0.04980056, 0.045, -0.045, -0.51, 0.51]
            # optimise in a direction
            template = []
            for dimension in range(self.env_input_size):
                template.append(Experiment.e(self.env_input_size, dimension))
                template.append(-Experiment.e(self.env_input_size, dimension))
            template = np.array(template)  # the 6 dimensions in 2 variables
            return input_boundaries, template
        if mode == 1:  # directions to easily find fixed point
            input_boundaries = None
            template = np.array([theta, -theta, theta_dot, -theta_dot, theta + theta_dot, -(theta + theta_dot), (theta - theta_dot), -(theta - theta_dot)])  # x_dot, -x_dot,theta_dot - theta
            return input_boundaries, template
        if mode == 2:
            input_boundaries = None
            template = np.array([theta, -theta, theta_dot, -theta_dot])
            return input_boundaries, template
        if mode == 3:
            input_boundaries = None
            template = np.array([theta, theta_dot, -theta_dot])
            return input_boundaries, template
        if mode == 4:
            input_boundaries = [0.09375, 0.625, 0.625, 0.0625, 0.1875]
            # input_boundaries = [0.09375, 0.5, 0.5, 0.0625, 0.09375]
            template = np.array([theta, theta_dot, -theta_dot, theta + theta_dot, (theta - theta_dot)])
            return input_boundaries, template
        if mode == 5:
            input_boundaries = [0.125, 0.0625, 0.1875]
            template = np.array([theta, theta + theta_dot, (theta - theta_dot)])
            return input_boundaries, template

    def get_nn_old(self):
        config, trainer = get_PPO_trainer(use_gpu=0)
        # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-07_12-49-16sn6s0bd0/checkpoint_19/checkpoint-19")
        # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-07_17-13-476oom2etf/checkpoint_20/checkpoint-20")
        # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-08_16-19-23tg3bxrcz/checkpoint_18/checkpoint-18")
        # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-09_10-42-12ad16ozkq/checkpoint_150/checkpoint-150")
        # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-09_10-42-12ad16ozkq/checkpoint_170/checkpoint-170")
        # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-09_10-42-12ad16ozkq/checkpoint_200/checkpoint-200")
        trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-09_15-34-25f0ld3dex/checkpoint_30/checkpoint-30")

        policy = trainer.get_policy()
        # sequential_nn = convert_ray_simple_policy_to_sequential(policy).cpu()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        l0 = torch.nn.Linear(4, 2, bias=False)
        l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32))
        layers = [l0]
        for l in sequential_nn:
            layers.append(l)
        # ray.shutdown()
        nn = torch.nn.Sequential(*layers)
        return nn

    def get_nn(self):
        config = get_PPO_config(1234)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(self.nn_path)

        policy = trainer.get_policy()
        # sequential_nn = convert_ray_simple_policy_to_sequential(policy).cpu()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        l0 = torch.nn.Linear(4, 2, bias=False)
        l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32))
        layers = [l0]
        for l in sequential_nn:
            layers.append(l)
        nn = torch.nn.Sequential(*layers)
        # ray.shutdown()
        return nn


if __name__ == '__main__':
    ray.init()
    experiment = CartpoleExperiment()
    experiment.run_experiment()
