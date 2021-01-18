from typing import List, Tuple

import gurobi as grb
import numpy as np
import ray
import torch
from ray.rllib.agents.ppo import ppo

from agents.ppo.train_PPO_car import get_PPO_trainer
from agents.ppo.tune.tune_train_PPO_car import get_PPO_config
from agents.ray_utils import convert_ray_policy_to_sequential
from polyhedra.experiments_nn_analysis import Experiment


class StoppingCarExperiment(Experiment):
    def __init__(self):
        env_input_size: int = 6
        super().__init__(env_input_size)
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        self.template_2d: np.ndarray = np.array([[1, 0, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0]])
        input_boundaries, input_template = self.get_template(0)
        self.input_boundaries: List = input_boundaries
        self.input_template: np.ndarray = input_template
        _, template = self.get_template(1)
        self.analysis_template: np.ndarray = template
        collision_distance = 0
        distance = [Experiment.e(6, 0) - Experiment.e(6, 1)]
        # self.use_bfs = True
        # self.n_workers = 1
        self.rounding_value = 2 ** 10
        self.use_rounding = False
        self.time_horizon = 400
        self.unsafe_zone: List[Tuple] = [(distance, np.array([collision_distance]))]
        self.input_epsilon = 0
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_14b68_00000_0_cost_fn=0,epsilon_input=0_2021-01-17_11-56-58/checkpoint_31/checkpoint-31" #safe both with and without epsilon of 0.1
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_14b68_00001_1_cost_fn=0,epsilon_input=0.1_2021-01-17_11-56-58/checkpoint_37/checkpoint-37" #not determined
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00000_0_cost_fn=0,epsilon_input=0_2021-01-17_12-37-27/checkpoint_24/checkpoint-24"  # safe at t=216
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00001_1_cost_fn=0,epsilon_input=0.1_2021-01-17_12-37-27/checkpoint_36/checkpoint-36"  # not determined
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00002_2_cost_fn=0,epsilon_input=0_2021-01-17_12-38-53/checkpoint_40/checkpoint-40"  # not determined

    @ray.remote
    def post_milp(self, x, nn, output_flag, t, template):
        """milp method"""
        post = []
        for chosen_action in range(2):
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            gurobi_model.setParam('Threads', 2)
            input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
            observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="input")
            gurobi_model.addConstr(observation[1] <= input[0] - input[1] + self.input_epsilon / 2, name=f"obs_constr21")
            gurobi_model.addConstr(observation[1] >= input[0] - input[1] - self.input_epsilon / 2, name=f"obs_constr22")
            gurobi_model.addConstr(observation[0] <= input[2] - input[3] + self.input_epsilon / 2, name=f"obs_constr11")
            gurobi_model.addConstr(observation[0] >= input[2] - input[3] - self.input_epsilon / 2, name=f"obs_constr12")
            feasible_action = Experiment.generate_nn_guard(gurobi_model, observation, nn, action_ego=chosen_action)
            # feasible_action = Experiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
            if feasible_action:
                # apply dynamic
                x_prime = StoppingCarExperiment.apply_dynamic(input, gurobi_model, action=chosen_action, env_input_size=self.env_input_size)
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
        :param action_ego:
        :param t:
        :return:

        lead 100km/h 28m/s
        ego 130km/h  36.1 m/s


        '''

        x_lead = input[0]
        x_ego = input[1]
        v_lead = input[2]
        v_ego = input[3]
        a_lead = input[4]
        a_ego = input[5]
        z = gurobi_model.addMVar(shape=(6,), lb=float("-inf"), name=f"x_prime")
        const_acc = 3
        dt = .1  # seconds
        if action == 0:
            acceleration = -const_acc
        elif action == 1:
            acceleration = const_acc
        else:
            acceleration = 0
        a_ego_prime = acceleration
        v_ego_prime = v_ego + a_ego * dt
        v_lead_prime = v_lead + a_lead * dt
        x_lead_prime = x_lead + v_lead_prime * dt
        x_ego_prime = x_ego + v_ego_prime * dt
        # delta_x_prime = (x_lead + (v_lead + (a_lead + 0) * dt) * dt) - (x_ego + (v_ego + (a_ego + acceleration) * dt) * dt)
        # delta_v_prime = (v_lead + (a_lead + 0) * dt) - (v_ego + (a_ego + acceleration) * dt)
        gurobi_model.addConstr(z[0] == x_lead_prime, name=f"dyna_constr_1")
        gurobi_model.addConstr(z[1] == x_ego_prime, name=f"dyna_constr_2")
        gurobi_model.addConstr(z[2] == v_lead_prime, name=f"dyna_constr_3")
        gurobi_model.addConstr(z[3] == v_ego_prime, name=f"dyna_constr_4")
        gurobi_model.addConstr(z[4] == a_lead, name=f"dyna_constr_5")  # no change in a_lead
        gurobi_model.addConstr(z[5] == a_ego_prime, name=f"dyna_constr_6")
        return z

    @staticmethod
    def plot(vertices_list, template, template_2d):
        Experiment.generic_plot("x_ego", "x_lead-x_ego", vertices_list, template, template_2d)
        pass

    def get_template(self, mode=0):
        x_lead = Experiment.e(6, 0)
        x_ego = Experiment.e(6, 1)
        v_lead = Experiment.e(6, 2)
        v_ego = Experiment.e(6, 3)
        a_lead = Experiment.e(6, 4)
        a_ego = Experiment.e(6, 5)
        if mode == 0:  # box directions with intervals
            input_boundaries = [50, -40, 10, -0, 28, -28, 36, -36, 0, -0, 0, -0, 0]
            # optimise in a direction
            template = []
            for dimension in range(6):
                template.append(Experiment.e(6, dimension))
                template.append(-Experiment.e(6, dimension))
            template = np.array(template)  # the 6 dimensions in 2 variables

            # t1 = [0] * 6
            # t1[0] = -1
            # t1[1] = 1
            # template = np.vstack([template, t1])
            return input_boundaries, template
        if mode == 1:  # directions to easily find fixed point

            input_boundaries = [20]

            template = np.array([a_lead, -a_lead, a_ego, -a_ego, -v_lead, v_lead, -(v_lead - v_ego), (v_lead - v_ego), -(x_lead - x_ego), (x_lead - x_ego)])
            return input_boundaries, template
        if mode == 2:
            input_boundaries = [0, -100, 30, -31, 20, -30, 0, -35, 0, -0, -10, -10, 20]
            # optimise in a direction
            template = []
            for dimension in range(6):
                t1 = [0] * 6
                t1[dimension] = 1
                t2 = [0] * 6
                t2[dimension] = -1
                template.append(t1)
                template.append(t2)
            # template = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])  # the 8 dimensions in 2 variables
            template = np.array(template)  # the 6 dimensions in 2 variables

            t1 = [0] * 6
            t1[0] = 1
            t1[1] = -1
            template = np.vstack([template, t1])
            return input_boundaries, template
        if mode == 3:  # single point box directions +diagonal
            input_boundaries = [30, -30, 0, -0, 28, -28, 36, -36, 0, -0, 0, -0, 0]
            # optimise in a direction
            template = []
            for dimension in range(6):
                t1 = [0] * 6
                t1[dimension] = 1
                t2 = [0] * 6
                t2[dimension] = -1
                template.append(t1)
                template.append(t2)
            # template = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])  # the 8 dimensions in 2 variables
            template = np.array(template)  # the 6 dimensions in 2 variables

            t1 = [0] * 6
            t1[0] = -1
            t1[1] = 1
            template = np.vstack([template, t1])
            return input_boundaries, template
        if mode == 4:  # octagon, every pair of variables
            input_boundaries = [20]
            template = []
            for dimension in range(6):
                t1 = [0] * 6
                t1[dimension] = 1
                t2 = [0] * 6
                t2[dimension] = -1
                template.append(t1)
                template.append(t2)
                for other_dimension in range(dimension + 1, 6):
                    t1 = [0] * 6
                    t1[dimension] = 1
                    t1[other_dimension] = -1
                    t2 = [0] * 6
                    t2[dimension] = -1
                    t2[other_dimension] = 1
                    t3 = [0] * 6
                    t3[dimension] = 1
                    t3[other_dimension] = 1
                    t4 = [0] * 6
                    t4[dimension] = -1
                    t4[other_dimension] = -1
                    template.append(t1)
                    template.append(t2)
                    template.append(t3)
                    template.append(t4)
            return input_boundaries, np.array(template)

    def get_nn_old(self):
        ray.init(ignore_reinit_error=True)
        config, trainer = get_PPO_trainer(use_gpu=0)
        trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-12-30_17-06-3265yz3d63/checkpoint_65/checkpoint-65")
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
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

    def get_nn(self):
        ray.init(ignore_reinit_error=True)
        config = get_PPO_config(1234)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(self.nn_path)
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
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


if __name__ == '__main__':
    experiment = StoppingCarExperiment()
    experiment.run_experiment()
