from typing import List, Tuple

import gurobi as grb
import numpy as np
import ray
import torch
from ray.rllib.agents.ppo import ppo

from training.ppo.train_PPO_car import get_PPO_trainer
from training.ppo.tune.tune_train_PPO_car import get_PPO_config
from training.ray_utils import convert_ray_policy_to_sequential
from polyhedra.experiments_nn_analysis import Experiment
from polyhedra.probabilistic_experiments_nn_analysis import ProbabilisticExperiment
from runnables.runnable.experiment.run_experiment_stopping_car import StoppingCarExperiment


class StoppingCarExperimentProbabilistic(ProbabilisticExperiment):
    def __init__(self):
        env_input_size: int = 2
        super().__init__(env_input_size)
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        self.template_2d: np.ndarray = np.array([[0, 1], [1, 0]])
        self.input_boundaries = tuple([10, -3, 32, -26])
        self.input_template = Experiment.box(env_input_size)
        # self.input_boundaries: List = input_boundaries
        # self.input_template: np.ndarray = input_template
        # _, template = self.get_template(1)
        delta_x = Experiment.e(env_input_size, 0)
        v_ego = Experiment.e(env_input_size, 1)
        # template = Experiment.combinations([delta_x, - v_ego])
        template = np.array([delta_x, -delta_x, v_ego, -v_ego, 1 / 4.5 * delta_x + v_ego, 1 / 4.5 * delta_x - v_ego, -1 / 4.5 * delta_x + v_ego,
                             -1 / 4.5 * delta_x - v_ego])
        # template = np.stack([x_lead - x_ego, -(x_lead - x_ego), - v_ego, v_ego])
        self.analysis_template: np.ndarray = template
        collision_distance = 0
        distance = [Experiment.e(self.env_input_size, 0)]
        # self.use_bfs = True
        # self.n_workers = 1
        self.rounding_value = 2 ** 10
        self.use_rounding = False
        self.time_horizon = 20
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
    def post_milp(self, x, x_label, nn, output_flag, t, template) -> List[Experiment.SuccessorInfo]:
        """milp method"""
        ranges_probs = self.create_range_bounds_model(template, x, self.env_input_size, nn)
        post = []
        for chosen_action in range(2):
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            gurobi_model.setParam('DualReductions', 0)
            input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
            observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="input")
            gurobi_model.addConstr(observation[1] <= input[0] - input[1] + self.input_epsilon / 2, name=f"obs_constr21")
            gurobi_model.addConstr(observation[1] >= input[0] - input[1] - self.input_epsilon / 2, name=f"obs_constr22")
            gurobi_model.addConstr(observation[0] <= self.v_lead - input[2] + self.input_epsilon / 2, name=f"obs_constr11")
            gurobi_model.addConstr(observation[0] >= self.v_lead - input[2] - self.input_epsilon / 2, name=f"obs_constr12")
            feasible_action = Experiment.generate_nn_guard(gurobi_model, observation, nn, action_ego=chosen_action)
            if feasible_action:
                x_prime = StoppingCarExperiment.apply_dynamic(input, gurobi_model, action=chosen_action, env_input_size=self.env_input_size)
                gurobi_model.update()
                gurobi_model.optimize()
                found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                if found_successor:
                    successor_info = Experiment.SuccessorInfo()
                    successor_info.successor = tuple(x_prime_results)
                    successor_info.parent = x
                    successor_info.parent_lbl = x_label
                    successor_info.t = t + 1
                    successor_info.action = "policy"  # chosen_action
                    successor_info.lb = ranges_probs[chosen_action][0]
                    successor_info.ub = ranges_probs[chosen_action][1]
                    post.append(successor_info)
        return post
        #     x_prime = StoppingCarExperimentProbabilistic.apply_dynamic(input, gurobi_model, action=chosen_action, env_input_size=self.env_input_size)
        #     gurobi_model.update()
        #     gurobi_model.optimize()
        #     x_prime_results = self.optimise(template, gurobi_model, x_prime)
        #     if x_prime_results is None:
        #         assert x_prime_results is not None
        #     successor_info = Experiment.SuccessorInfo()
        #     successor_info.successor = tuple(x_prime_results)
        #     successor_info.parent = x
        #     successor_info.parent_lbl = x_label
        #     successor_info.t = t + 1
        #     successor_info.action = "policy"#chosen_action
        #     successor_info.lb = ranges_probs[chosen_action][0]
        #     successor_info.ub = ranges_probs[chosen_action][1]
        #     post.append(successor_info)
        #
        # return post

    # def post_milp(self, x, nn, output_flag, t, template):
    #     """milp method"""
    #     post = []
    #     for chosen_action in range(2):
    #         gurobi_model = grb.Model()
    #         gurobi_model.setParam('OutputFlag', output_flag)
    #         input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
    #         observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="input")
    #         gurobi_model.addConstr(observation[1] <= input[0] - input[1] + self.input_epsilon / 2, name=f"obs_constr21")
    #         gurobi_model.addConstr(observation[1] >= input[0] - input[1] - self.input_epsilon / 2, name=f"obs_constr22")
    #         gurobi_model.addConstr(observation[0] <= self.v_lead - input[2] + self.input_epsilon / 2, name=f"obs_constr11")
    #         gurobi_model.addConstr(observation[0] >= self.v_lead - input[2] - self.input_epsilon / 2, name=f"obs_constr12")
    #         feasible_action = Experiment.generate_nn_guard(gurobi_model, observation, nn, action_ego=chosen_action)
    #         # feasible_action = Experiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
    #         if feasible_action:
    #             # apply dynamic
    #             x_prime = StoppingCarExperiment2.apply_dynamic(input, gurobi_model, action=chosen_action, env_input_size=self.env_input_size)
    #             gurobi_model.update()
    #             gurobi_model.optimize()
    #             found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
    #             if found_successor:
    #                 post.append(tuple(x_prime_results))
    #     return post

    def get_observation_variable(self, input, gurobi_model):
        observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="observation")
        gurobi_model.addConstr(observation[1] <= input[0] + self.input_epsilon / 2, name=f"obs_constr21")
        gurobi_model.addConstr(observation[1] >= input[0] - self.input_epsilon / 2, name=f"obs_constr22")
        gurobi_model.addConstr(observation[0] <= self.v_lead - input[1] + self.input_epsilon / 2, name=f"obs_constr11")
        gurobi_model.addConstr(observation[0] >= self.v_lead - input[1] - self.input_epsilon / 2, name=f"obs_constr12")
        gurobi_model.update()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        return observation

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
        max_speed = 36.0
        delta_x = input[0]
        v_ego = input[1]
        z = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name=f"x_prime")
        const_acc = 3
        dt = .1  # seconds
        if action == 0:
            acceleration = -const_acc
        elif action == 1:
            acceleration = const_acc
        else:
            acceleration = 0
        v_ego_prime_temp1 = gurobi_model.addVar()
        v_ego_prime_temp2 = gurobi_model.addVar()
        v_next = v_ego._vararr[0] + acceleration * dt

        gurobi_model.addConstr(v_ego_prime_temp1 == v_next, name=f"v_constr")
        gurobi_model.addConstr(v_ego_prime_temp2 == grb.min_(max_speed, v_ego_prime_temp1), name=f"v_constr")
        v_ego_prime = grb.MVar(v_ego_prime_temp2)  # convert from Var to MVar
        v_lead_prime = v_lead
        delta_prime_v = v_lead_prime-v_ego_prime
        delta_prime_v_temp = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name=f"delta_prime_v_temp")
        gurobi_model.addConstr(delta_prime_v_temp == delta_prime_v, name=f"delta_prime_v_constr")
        delta_x_prime = delta_x+delta_prime_v_temp*dt
        # x_lead_prime = x_lead + v_lead_prime * dt
        # x_ego_prime = x_ego + v_ego_prime * dt
        gurobi_model.addConstr(z[0] == delta_x_prime, name=f"dyna_constr_1")
        gurobi_model.addConstr(z[1] == v_ego_prime, name=f"dyna_constr_3")
        return z

    def plot(self, vertices_list, template, template_2d):
        # try:
        self.generic_plot("x_lead", "x_lead-x_ego", vertices_list, template, template_2d)
        # except:
        #     print("Error in plotting")

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
        config = get_PPO_config(1234,0)
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

    def get_pre_nn(self):
        l0 = torch.nn.Linear(self.env_input_size, 2, bias=False)
        l0.weight = torch.nn.Parameter(torch.tensor([[0, -1], [1, 0]], dtype=torch.float32))
        l0.bias = torch.nn.Parameter(torch.tensor([[28, 0]], dtype=torch.float32))
        layers = [l0]
        pre_nn = torch.nn.Sequential(*layers)
        return pre_nn


if __name__ == '__main__':
    ray.init(log_to_driver=False, local_mode=True)
    experiment = StoppingCarExperimentProbabilistic()
    experiment.save_dir = "/home/edoardo/ray_results/tune_PPO_stopping_car/template_h7_probabilistic"
    # experiment.save_dir = "/home/edoardo/ray_results/tune_PPO_stopping_car/box_h7_probabilistic_psi05_nocontain"
    experiment.plotting_time_interval = 60
    experiment.show_progressbar = True
    experiment.show_progress_plot = False
    # experiment.analysis_template = Experiment.box(2)
    # experiment.use_contained = False
    # experiment.max_probability_split = 0.5
    experiment.time_horizon = 7
    # experiment.load_graph = True
    experiment.run_experiment()
