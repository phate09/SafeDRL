from typing import List, Tuple

import gurobi as grb
import numpy as np
import ray
import torch
from ray.rllib.agents.ddpg import td3
from ray.rllib.agents.ppo import ppo
import torch.nn

from agents.ray_utils import *
from polyhedra.experiments_nn_analysis import Experiment


class StoppingCarContinuousExperiment(Experiment):
    def __init__(self):
        env_input_size: int = 6
        super().__init__(env_input_size)
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        # self.template_2d: np.ndarray = np.array([[1, 0, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0]])
        self.template_2d: np.ndarray = np.array([[0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]])
        input_boundaries, input_template = self.get_template(0)
        self.input_boundaries: List = input_boundaries
        self.input_template: np.ndarray = input_template
        _, template = self.get_template(1)
        self.analysis_template: np.ndarray = template
        collision_distance = 0
        distance = [Experiment.e(env_input_size, 0) - Experiment.e(env_input_size, 1)]
        # self.use_bfs = True
        # self.n_workers = 1
        self.rounding_value = 2 ** 6
        self.use_rounding = False
        self.time_horizon = 400
        self.unsafe_zone: List[Tuple] = [(distance, np.array([collision_distance]))]
        self.input_epsilon = 0
        # self.nn_path = "/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/TD3_StoppingCar_0a03b_00000_0_cost_fn=2,epsilon_input=0_2021-02-27_17-12-58/checkpoint_680/checkpoint-680"
        # self.nn_path = "/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/TD3_StoppingCar_62310_00000_0_cost_fn=2,epsilon_input=0_2021-03-04_13-34-45/checkpoint_780/checkpoint-780"
        # self.nn_path = "/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/TD3_StoppingCar_ 3665a_00000_0_cost_fn=3,epsilon_input=0_2021-03-04_14-37-57/checkpoint_114/checkpoint-114"
        # self.nn_path = "/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/TD3_StoppingCar_47b16_00000_0_cost_fn=3,epsilon_input=0_2021-03-04_17-08-46/checkpoint_600/checkpoint-600"
        # self.nn_path = "/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/PPO_StoppingCar_28110_00000_0_cost_fn=0,epsilon_input=0_2021-03-07_17-40-07/checkpoint_1250/checkpoint-1250"
        self.nn_path = "/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/PPO_StoppingCar_7bdde_00000_0_cost_fn=0,epsilon_input=0_2021-03-09_11-49-20/checkpoint_1460/checkpoint-1460"

    @ray.remote
    def post_milp(self, x, nn, output_flag, t, template):
        """milp method"""
        post = []
        # for chosen_action in range(2):
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
        observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="input")
        gurobi_model.addConstr(observation[1] <= input[0] - input[1] + self.input_epsilon / 2, name=f"obs_constr21")
        gurobi_model.addConstr(observation[1] >= input[0] - input[1] - self.input_epsilon / 2, name=f"obs_constr22")
        gurobi_model.addConstr(observation[0] <= input[2] - input[3] + self.input_epsilon / 2, name=f"obs_constr11")
        gurobi_model.addConstr(observation[0] >= input[2] - input[3] - self.input_epsilon / 2, name=f"obs_constr12")
        nn_output = Experiment.generate_nn_guard_continuous(gurobi_model, observation, nn)
        # feasible_action = Experiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
        # apply dynamic
        x_prime = StoppingCarContinuousExperiment.apply_dynamic(input, gurobi_model, action=nn_output, env_input_size=self.env_input_size)
        gurobi_model.update()
        gurobi_model.optimize()
        found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
        # x_prime_results = x_prime_results.round(4)  # correct for rounding errors introduced by the conversion to h-repr
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
        # gurobi_model.addConstr(action[0] <= 12)  # cap the acceleration to +12 m/s*2
        # gurobi_model.addConstr(action[0] >= -12)  # cap the acceleration to -12 m/s*2
        z = gurobi_model.addMVar(shape=(6,), lb=float("-inf"), name=f"x_prime")
        dt = .1  # seconds
        a_ego_prime = 0  # action
        v_ego_prime = v_ego + action * dt
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
        gurobi_model.addConstr(z[5] == a_ego_prime, name=f"dyna_constr_6")  # use index 0 which is the action (as opposed to index 1 which is the standard deviation for exploration)
        return z

    def plot(self, vertices_list, template, template_2d):
        # try:
        self.generic_plot("v_lead-v_ego", "x_lead-x_ego", vertices_list, template, template_2d)
        # except:
        #     print("Error in plotting")

    @staticmethod
    def get_template(mode=0):
        env_input_size = 6
        x_lead = Experiment.e(env_input_size, 0)
        x_ego = Experiment.e(env_input_size, 1)
        v_lead = Experiment.e(env_input_size, 2)
        v_ego = Experiment.e(env_input_size, 3)
        a_lead = Experiment.e(env_input_size, 4)
        a_ego = Experiment.e(env_input_size, 5)
        if mode == 0:  # box directions with intervals
            input_boundaries = [50, -40, 10, -0, 28, -28, 36, -36, 0, -0, 0, -0, 0]
            # optimise in a direction
            template = []
            for dimension in range(env_input_size):
                template.append(Experiment.e(env_input_size, dimension))
                template.append(-Experiment.e(env_input_size, dimension))
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
            for dimension in range(env_input_size):
                t1 = [0] * env_input_size
                t1[dimension] = 1
                t2 = [0] * env_input_size
                t2[dimension] = -1
                template.append(t1)
                template.append(t2)
            # template = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])  # the 8 dimensions in 2 variables
            template = np.array(template)  # the 6 dimensions in 2 variables

            t1 = [0] * env_input_size
            t1[0] = 1
            t1[1] = -1
            template = np.vstack([template, t1])
            return input_boundaries, template
        if mode == 3:  # single point box directions +diagonal
            input_boundaries = [30, -30, 0, -0, 28, -28, 36, -36, 0, -0, 0, -0, 0]
            # optimise in a direction
            template = []
            for dimension in range(env_input_size):
                t1 = [0] * env_input_size
                t1[dimension] = 1
                t2 = [0] * env_input_size
                t2[dimension] = -1
                template.append(t1)
                template.append(t2)
            # template = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])  # the 8 dimensions in 2 variables
            template = np.array(template)  # the 6 dimensions in 2 variables

            t1 = [0] * env_input_size
            t1[0] = -1
            t1[1] = 1
            template = np.vstack([template, t1])
            return input_boundaries, template
        if mode == 4:  # octagon, every pair of variables
            input_boundaries = [20]
            template = []
            for dimension in range(env_input_size):
                t1 = [0] * env_input_size
                t1[dimension] = 1
                t2 = [0] * env_input_size
                t2[dimension] = -1
                template.append(t1)
                template.append(t2)
                for other_dimension in range(dimension + 1, env_input_size):
                    t1 = [0] * env_input_size
                    t1[dimension] = 1
                    t1[other_dimension] = -1
                    t2 = [0] * env_input_size
                    t2[dimension] = -1
                    t2[other_dimension] = 1
                    t3 = [0] * env_input_size
                    t3[dimension] = 1
                    t3[other_dimension] = 1
                    t4 = [0] * env_input_size
                    t4[dimension] = -1
                    t4[other_dimension] = -1
                    template.append(t1)
                    template.append(t2)
                    template.append(t3)
                    template.append(t4)
            return input_boundaries, np.array(template)
        if mode == 5:
            input_boundaries = [20]

            template = np.array([a_lead, -a_lead, -v_lead, v_lead, -(v_lead - v_ego), (v_lead - v_ego), -(x_lead - x_ego), (x_lead - x_ego)])
            return input_boundaries, template
        if mode == 6:  # directions to easily find fixed point

            input_boundaries = [20]
            delta_v = (v_lead - v_ego)
            delta_x = (x_lead - x_ego)
            template = np.array(
                [a_lead, -a_lead, a_ego, -a_ego, -v_lead, v_lead, -delta_v, delta_v, -delta_x, delta_x, (delta_v - delta_x), -(delta_v - delta_x),
                 (delta_v + delta_x), -(delta_v + delta_x), (delta_v - delta_x)])
            return input_boundaries, template

    def get_nn_old(self):
        pass

    def get_nn(self):
        # from agents.td3.tune.tune_train_TD3_car import get_TD3_config
        from agents.td3.tune.tune_train_TD3_car import get_TD3_config
        config = get_TD3_config(1234)
        trainer = ppo.PPOTrainer(config)
        trainer.restore(self.nn_path)
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

    def get_nn_static(self):
        layers = []
        l0 = torch.nn.Linear(2, 1)
        l0.weight = torch.nn.Parameter(torch.tensor([[0, 1]], dtype=torch.float32))
        l0.bias = torch.nn.Parameter(torch.tensor([-30], dtype=torch.float32))
        layers.append(l0)
        l0 = torch.nn.Linear(1, 1)
        l0.weight = torch.nn.Parameter(torch.tensor([[1.2]], dtype=torch.float32))
        l0.bias = torch.nn.Parameter(torch.tensor([0], dtype=torch.float32))
        layers.append(l0)
        # layers.append(torch.nn.Hardtanh(min_val=-3, max_val=3))
        nn = torch.nn.Sequential(*layers)
        return nn


if __name__ == '__main__':
    ray.init(log_to_driver=False, local_mode=True)
    experiment = StoppingCarContinuousExperiment()
    experiment.save_dir = "/home/edoardo/ray_results/tune_TD3_stopping_car_continuous/test"
    experiment.plotting_time_interval = 60
    experiment.show_progressbar = True
    experiment.show_progress_plot = True
    experiment.use_rounding = False
    experiment.get_nn_fn = experiment.get_nn_static
    # template = Experiment.octagon(experiment.env_input_size)
    _, template = StoppingCarContinuousExperiment.get_template(6)
    experiment.analysis_template = template  # standard
    input_boundaries = [40, -30, 0, -0, 28, -28, 28 + 2.5, -(28 - 2.5), 0, -0, 0, 0, 0]
    experiment.input_boundaries = input_boundaries
    experiment.time_horizon = 150
    experiment.run_experiment()
