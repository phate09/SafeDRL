import os
from typing import List, Tuple

import gurobi as grb
import numpy as np
import ray
from ray.rllib.agents.ppo import ppo

from polyhedra.experiments_nn_analysis import Experiment
from polyhedra.milp_methods import generate_input_region
from training.ppo.train_PPO_car import get_PPO_trainer
from training.ppo.tune.tune_train_PPO_car import get_PPO_config
from training.ray_utils import convert_ray_policy_to_sequential


class StoppingCarExperiment(Experiment):
    def __init__(self):
        env_input_size: int = 3
        super().__init__(env_input_size)
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        self.template_2d: np.ndarray = np.array([[1, 0, 0], [1, -1, 0]])
        input_boundaries, input_template = self.get_template(0)
        self.input_boundaries: List = input_boundaries
        self.input_template: np.ndarray = input_template
        _, template = self.get_template(1)
        self.analysis_template: np.ndarray = template
        collision_distance = 0
        distance = [Experiment.e(self.env_input_size, 0) - Experiment.e(self.env_input_size, 1)]
        # self.use_bfs = True
        # self.n_workers = 1
        self.rounding_value = 2 ** 10
        self.use_rounding = False
        self.time_horizon = 400
        self.unsafe_zone: List[Tuple] = [(distance, np.array([collision_distance]))]
        self.input_epsilon = 0
        self.v_lead = 28
        self.max_speed = 36
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_14b68_00000_0_cost_fn=0,epsilon_input=0_2021-01-17_11-56-58/checkpoint_31/checkpoint-31"  # safe both with and without epsilon of 0.1.
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_14b68_00001_1_cost_fn=0,epsilon_input=0.1_2021-01-17_11-56-58/checkpoint_37/checkpoint-37" #not determined
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00000_0_cost_fn=0,epsilon_input=0_2021-01-17_12-37-27/checkpoint_24/checkpoint-24"  # safe at t=216
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00001_1_cost_fn=0,epsilon_input=0.1_2021-01-17_12-37-27/checkpoint_36/checkpoint-36"  # not determined
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00002_2_cost_fn=0,epsilon_input=0_2021-01-17_12-38-53/checkpoint_40/checkpoint-40"  # not determined
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00006_6_cost_fn=0,epsilon_input=0_2021-01-17_12-44-54/checkpoint_41/checkpoint-41" #safe
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00000_0_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_39/checkpoint-39" #unsafe
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58"  # safe
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00005_5_cost_fn=0,epsilon_input=0.1_2021-01-17_12-41-27/checkpoint_10/checkpoint-10" #unsafe

    @ray.remote
    def post_milp(self, x, x_label, nn, output_flag, t, template):
        """milp method"""
        post = []
        for chosen_action in range(2):
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            input = generate_input_region(gurobi_model, template, x, self.env_input_size)
            # gurobi_model.addConstr(input[0] >= 0, name=f"input_base_constr1")
            # gurobi_model.addConstr(input[1] >= 0, name=f"input_base_constr2")
            # gurobi_model.addConstr(input[2] >= 20, name=f"input_base_constr3")
            observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="observation")
            gurobi_model.addConstr(observation[1] <= input[0] - input[1] + self.input_epsilon / 2, name=f"obs_constr21")
            gurobi_model.addConstr(observation[1] >= input[0] - input[1] - self.input_epsilon / 2, name=f"obs_constr22")
            gurobi_model.addConstr(observation[0] <= self.v_lead - input[2] + self.input_epsilon / 2, name=f"obs_constr11")
            gurobi_model.addConstr(observation[0] >= self.v_lead - input[2] - self.input_epsilon / 2, name=f"obs_constr12")
            # gurobi_model.addConstr(input[3] <= self.max_speed, name=f"v_constr_input")
            # gurobi_model.addConstr(input[3] >= -self.max_speed, name=f"v_constr_input")
            feasible_action = Experiment.generate_nn_guard(gurobi_model, observation, nn, action_ego=chosen_action)
            # feasible_action = Experiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
            if feasible_action:
                # apply dynamic
                x_prime = StoppingCarExperiment.apply_dynamic(input, gurobi_model, action=chosen_action, env_input_size=self.env_input_size)
                gurobi_model.update()
                gurobi_model.optimize()
                found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                if found_successor:
                    # post.append((tuple(x_prime_results),(x, x_label)))
                    successor_info = Experiment.SuccessorInfo()
                    successor_info.successor = tuple(x_prime_results)
                    successor_info.parent = x
                    successor_info.parent_lbl = x_label
                    successor_info.t = t + 1
                    successor_info.action = "policy"  # chosen_action
                    # successor_info.lb = ranges_probs[chosen_action][0]
                    # successor_info.ub = ranges_probs[chosen_action][1]
                    post.append(successor_info)
        return post

    @ray.remote
    def post_milp2(self, x, nn, output_flag, t, template):
        """milp method for a simple guard that keeps distance at 20 metres"""
        post = []
        for chosen_action in range(2):
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
            observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="input")
            gurobi_model.addConstr(observation[1] <= input[0] - input[1] + self.input_epsilon / 2, name=f"obs_constr21")
            gurobi_model.addConstr(observation[1] >= input[0] - input[1] - self.input_epsilon / 2, name=f"obs_constr22")
            gurobi_model.addConstr(observation[0] <= self.v_lead - input[2] + self.input_epsilon / 2, name=f"obs_constr11")
            gurobi_model.addConstr(observation[0] >= self.v_lead - input[2] - self.input_epsilon / 2, name=f"obs_constr12")
            if chosen_action == 0:
                gurobi_model.addConstr(input[0] - input[1] <= 20)
                gurobi_model.update()
                gurobi_model.optimize()
                feasible_action = gurobi_model.status == 2 or gurobi_model.status == 5
            else:
                gurobi_model.addConstr(input[0] - input[1] >= 20)
                gurobi_model.update()
                gurobi_model.optimize()
                feasible_action = gurobi_model.status == 2 or gurobi_model.status == 5
            # feasible_action = Experiment.generate_nn_guard(gurobi_model, observation, nn, action_ego=chosen_action)
            # feasible_action = Experiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
            # gurobi_model.addConstr(input[3] <= self.max_speed, name=f"v_constr_input")
            # gurobi_model.addConstr(input[3] >= -self.max_speed, name=f"v_constr_input")
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
        v_lead = 28
        max_speed = 36
        x_lead = input[0]
        x_ego = input[1]
        # v_lead = input[2]
        v_ego = input[2]
        # a_lead = input[4]
        # a_ego = input[5]
        z = gurobi_model.addMVar(shape=(3,), lb=float("-inf"), name=f"x_prime")
        const_acc = 3
        dt = .1  # seconds
        if action == 0:
            acceleration = -const_acc
        elif action == 1:
            acceleration = const_acc
        else:
            acceleration = 0
        v_ego_prime = v_ego + acceleration * dt
        gurobi_model.addConstr(v_ego_prime <= max_speed, name=f"v_constr")
        # gurobi_model.addConstr(v_ego_prime >= -max_speed, name=f"v_constr")
        # gurobi_model.addConstr(a_lead == 0, name="a_lead_constr")
        v_lead_prime = v_lead
        x_lead_prime = x_lead + v_lead_prime * dt
        x_ego_prime = x_ego + v_ego_prime * dt
        # delta_x_prime = (x_lead + (v_lead + (a_lead + 0) * dt) * dt) - (x_ego + (v_ego + (a_ego + acceleration) * dt) * dt)
        # delta_v_prime = (v_lead + (a_lead + 0) * dt) - (v_ego + (a_ego + acceleration) * dt)
        gurobi_model.addConstr(z[0] == x_lead_prime, name=f"dyna_constr_1")
        gurobi_model.addConstr(z[1] == x_ego_prime, name=f"dyna_constr_2")
        gurobi_model.addConstr(z[2] == v_ego_prime, name=f"dyna_constr_3")
        # gurobi_model.addConstr(z[3] == v_ego_prime, name=f"dyna_constr_4")
        # gurobi_model.addConstr(z[4] == 0, name=f"dyna_constr_5")  # no change in a_lead
        # gurobi_model.addConstr(z[5] == acceleration, name=f"dyna_constr_6")
        return z

    def plot(self, vertices_list, template, template_2d):
        # try:
        self.generic_plot("x_lead", "x_lead-x_ego", vertices_list, template, template_2d)
        # except:
        #     print("Error in plotting")

    @staticmethod
    def get_template(mode=0):
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
        if mode == 5:
            input_boundaries = [20]

            template = np.array([a_lead, -a_lead, -v_lead, v_lead, -(v_lead - v_ego), (v_lead - v_ego), -(x_lead - x_ego), (x_lead - x_ego)])
            return input_boundaries, template

    def get_nn_old(self):
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
        config = get_PPO_config(1234, use_gpu=0)
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


class StoppingCarExperiment2(StoppingCarExperiment):
    def __init__(self):
        env_input_size: int = 3
        super().__init__(env_input_size)
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        self.template_2d: np.ndarray = np.array([[1, 0, 0], [1, -1, 0]])
        input_boundaries, input_template = self.get_template(0)
        self.input_boundaries: List = input_boundaries
        self.input_template: np.ndarray = input_template
        _, template = self.get_template(1)
        self.analysis_template: np.ndarray = template
        collision_distance = 0
        distance = [Experiment.e(self.env_input_size, 0) - Experiment.e(self.env_input_size, 1)]
        # self.use_bfs = True
        # self.n_workers = 1
        self.rounding_value = 2 ** 10
        self.use_rounding = False
        self.time_horizon = 400
        self.unsafe_zone: List[Tuple] = [(distance, np.array([collision_distance]))]
        self.input_epsilon = 0
        self.v_lead = 28
        self.max_speed = 36
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_14b68_00000_0_cost_fn=0,epsilon_input=0_2021-01-17_11-56-58/checkpoint_31/checkpoint-31"  # safe both with and without epsilon of 0.1.
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_14b68_00001_1_cost_fn=0,epsilon_input=0.1_2021-01-17_11-56-58/checkpoint_37/checkpoint-37" #not determined
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00000_0_cost_fn=0,epsilon_input=0_2021-01-17_12-37-27/checkpoint_24/checkpoint-24"  # safe at t=216
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00001_1_cost_fn=0,epsilon_input=0.1_2021-01-17_12-37-27/checkpoint_36/checkpoint-36"  # not determined
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00002_2_cost_fn=0,epsilon_input=0_2021-01-17_12-38-53/checkpoint_40/checkpoint-40"  # not determined
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00006_6_cost_fn=0,epsilon_input=0_2021-01-17_12-44-54/checkpoint_41/checkpoint-41" #safe
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00000_0_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_39/checkpoint-39" #unsafe
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58"  # safe
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00005_5_cost_fn=0,epsilon_input=0.1_2021-01-17_12-41-27/checkpoint_10/checkpoint-10" #unsafe

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
        # v_lead = 28
        max_speed = 36
        x_lead = input[0]
        x_ego = input[1]
        v_lead = input[2]
        v_ego = input[3]
        z = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name=f"x_prime")
        const_acc = 3
        dt = .1  # seconds
        if action == 0:
            acceleration = -const_acc
        elif action == 1:
            acceleration = const_acc
        else:
            acceleration = 0
        a_ego_prime = acceleration
        v_ego_prime = v_ego + acceleration * dt
        gurobi_model.addConstr(v_ego_prime <= max_speed, name=f"v_constr")
        # gurobi_model.addConstr(v_ego_prime >= -max_speed, name=f"v_constr")
        # gurobi_model.addConstr(a_lead == 0, name="a_lead_constr")
        v_lead_prime = v_lead
        x_lead_prime = x_lead + v_lead_prime * dt
        x_ego_prime = x_ego + v_ego_prime * dt
        # delta_x_prime = (x_lead + (v_lead + (a_lead + 0) * dt) * dt) - (x_ego + (v_ego + (a_ego + acceleration) * dt) * dt)
        # delta_v_prime = (v_lead + (a_lead + 0) * dt) - (v_ego + (a_ego + acceleration) * dt)
        gurobi_model.addConstr(z[0] == x_lead_prime, name=f"dyna_constr_1")
        gurobi_model.addConstr(z[1] == x_ego_prime, name=f"dyna_constr_2")
        gurobi_model.addConstr(z[2] == v_lead_prime, name=f"dyna_constr_2")
        gurobi_model.addConstr(z[3] == v_ego_prime, name=f"dyna_constr_3")
        # gurobi_model.addConstr(z[3] == v_ego_prime, name=f"dyna_constr_4")
        # gurobi_model.addConstr(z[4] == 0, name=f"dyna_constr_5")  # no change in a_lead
        # gurobi_model.addConstr(z[5] == acceleration, name=f"dyna_constr_6")
        return z

    @ray.remote
    def post_milp(self, x, nn, output_flag, t, template):
        """milp method"""
        post = []
        for chosen_action in range(2):
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
            observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="input")
            gurobi_model.addConstr(observation[1] <= input[0] - input[1] + self.input_epsilon / 2, name=f"obs_constr21")
            gurobi_model.addConstr(observation[1] >= input[0] - input[1] - self.input_epsilon / 2, name=f"obs_constr22")
            gurobi_model.addConstr(observation[0] <= self.v_lead - input[2] + self.input_epsilon / 2, name=f"obs_constr11")
            gurobi_model.addConstr(observation[0] >= self.v_lead - input[2] - self.input_epsilon / 2, name=f"obs_constr12")
            feasible_action = Experiment.generate_nn_guard(gurobi_model, observation, nn, action_ego=chosen_action)
            # feasible_action = Experiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
            if feasible_action:
                # apply dynamic
                x_prime = StoppingCarExperiment2.apply_dynamic(input, gurobi_model, action=chosen_action, env_input_size=self.env_input_size)
                gurobi_model.update()
                gurobi_model.optimize()
                found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                if found_successor:
                    post.append(tuple(x_prime_results))
        return post


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ray.init(log_to_driver=False, local_mode=True)
    experiment = StoppingCarExperiment()
    experiment.save_dir = "/home/edoardo/ray_results/tune_PPO_stopping_car/test"
    # experiment.plotting_time_interval = 60
    experiment.show_progressbar = True
    # experiment.show_progress_plot = True
    x_lead = Experiment.e(experiment.env_input_size, 0)
    x_ego = Experiment.e(experiment.env_input_size, 1)
    v_ego = Experiment.e(experiment.env_input_size, 2)
    template = np.array([-(x_lead - x_ego), v_ego, -v_ego]) #(x_lead - x_ego),
    experiment.analysis_template = template  # standard
    experiment.input_template = Experiment.box(3)
    input_boundaries = [40, -30, 10, -0, 36, -28]
    experiment.input_boundaries = input_boundaries
    experiment.time_horizon = 150
    experiment.run_experiment()
