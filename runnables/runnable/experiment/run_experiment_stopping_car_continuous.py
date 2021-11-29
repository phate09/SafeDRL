from typing import List, Tuple
import ray
import gurobi as grb
import numpy as np
import torch.nn

from polyhedra.continuous_experiments_nn_analysis import ContinuousExperiment
from polyhedra.experiments_nn_analysis import Experiment
from training.ray_utils import *

v_lead = 28
max_speed = 36


class StoppingCarContinuousExperiment(ContinuousExperiment):
    def __init__(self):
        env_input_size: int = 3
        super().__init__(env_input_size)
        self.post_fn_continuous = self.post_milp
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        self.template_2d: np.ndarray = np.array([[0, 0, 1], [1, -1, 0]])
        # self.input_boundaries: List = input_boundaries
        x_lead = Experiment.e(self.env_input_size, 0)
        x_ego = Experiment.e(self.env_input_size, 1)
        v_ego = Experiment.e(self.env_input_size, 2)
        template = np.array([-(x_lead - x_ego)])
        self.analysis_template = template  # standard
        self.input_template = Experiment.box(self.env_input_size)
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

    def before_start(self):
        self.internal_model = grb.Model()

    def post_milp(self, x, nn, output_flag, t, template):
        """milp method"""
        post = []
        gurobi_model = self.internal_model  # grb.Model()
        input = self.last_input  # Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
        observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="observation")
        gurobi_model.addConstr(observation[1] <= input[0] - input[1] + self.input_epsilon / 2, name=f"obs_constr21")
        gurobi_model.addConstr(observation[1] >= input[0] - input[1] - self.input_epsilon / 2, name=f"obs_constr22")
        gurobi_model.addConstr(observation[0] <= v_lead - input[2] + self.input_epsilon / 2, name=f"obs_constr11")
        gurobi_model.addConstr(observation[0] >= v_lead - input[2] - self.input_epsilon / 2, name=f"obs_constr12")
        nn_output, max_val, min_val = Experiment.generate_nn_guard_continuous(gurobi_model, observation, nn)
        is_equal = torch.isclose(nn(torch.from_numpy(observation.X).float()), torch.from_numpy(nn_output.X).float(), rtol=1e-3).all().item()
        assert is_equal
        # clipped_nn_output = gurobi_model.addMVar(lb=float("-inf"), shape=(len(nn_output)), name=f"clipped_nn_output")
        # gurobi_model.addConstr(nn_output[0] >= -12, name=f"clipped_out_constr1")
        # gurobi_model.addConstr(nn_output[0] <= 12, name=f"clipped_out_constr2")
        # feasible_action = Experiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
        # apply dynamic
        x_prime = StoppingCarContinuousExperiment.apply_dynamic(input, gurobi_model, action=nn_output, env_input_size=self.env_input_size)
        gurobi_model.update()
        gurobi_model.optimize()
        found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
        self.last_input = x_prime
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
        v_ego = input[2]
        # gurobi_model.addConstr(action[0] <= 12)  # cap the acceleration to +12 m/s*2
        # gurobi_model.addConstr(action[0] >= -12)  # cap the acceleration to -12 m/s*2
        z = gurobi_model.addMVar(shape=(3,), lb=float("-inf"), name=f"x_prime")
        dt = .1  # seconds
        acceleration = action[0]
        a_ego_prime = acceleration
        v_ego_prime = v_ego + acceleration * dt
        gurobi_model.addConstr(v_ego_prime <= max_speed, name=f"v_constr")
        # gurobi_model.addConstr(v_ego_prime >= -max_speed, name=f"v_constr")
        # gurobi_model.addConstr(a_lead == 0, name="a_lead_constr")
        v_lead_prime = v_lead
        x_lead_prime = x_lead + v_lead_prime * dt
        x_ego_prime = x_ego + v_ego_prime * dt
        gurobi_model.addConstr(z[0] == x_lead_prime, name=f"dyna_constr_1")
        gurobi_model.addConstr(z[1] == x_ego_prime, name=f"dyna_constr_2")
        gurobi_model.addConstr(z[2] == v_ego_prime, name=f"dyna_constr_3")
        return z

    def plot(self, vertices_list, template, template_2d):
        # try:
        self.generic_plot("v_lead-v_ego", "x_lead-x_ego", vertices_list, template, template_2d)
        # except:
        #     print("Error in plotting")

    def get_nn(self):
        # from training.td3.tune.tune_train_TD3_car import get_TD3_config
        from training.td3.tune.tune_train_TD3_car import get_TD3_config
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

    @staticmethod
    def get_nn_static():
        layers = []
        l0 = torch.nn.Linear(2, 1)
        l0.weight = torch.nn.Parameter(torch.tensor([[0, 1]], dtype=torch.float32))
        l0.bias = torch.nn.Parameter(torch.tensor([-30], dtype=torch.float32))
        layers.append(l0)
        l0 = torch.nn.Linear(1, 1)
        l0.weight = torch.nn.Parameter(torch.tensor([[1.2]], dtype=torch.float32))
        l0.bias = torch.nn.Parameter(torch.tensor([0], dtype=torch.float32))
        layers.append(l0)
        layers.append(torch.nn.Hardtanh(min_val=-3, max_val=3))
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
    # experiment.get_nn_fn = experiment.get_nn_static
    # template = Experiment.octagon(experiment.env_input_size)
    # _, template = StoppingCarContinuousExperiment.get_template(6)

    input_boundaries = [30, -29, 0, -0, 28, -0]
    # experiment.analysis_template = template  # standard
    # input_boundaries = [40, -30, 0, -0, 28, -28, 28 + 2.5, -(28 - 2.5), 0, -0, 0, 0, 0]
    experiment.input_boundaries = input_boundaries
    experiment.time_horizon = 150
    experiment.run_experiment()
