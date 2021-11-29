from typing import List, Tuple

import gurobi as grb
import numpy as np
import ray

from polyhedra.experiments_nn_analysis import Experiment
from training.dqn.safe_dqn_agent import InvariantAgent
from training.ppo.train_PPO_car import get_PPO_trainer
from training.ray_utils import convert_ray_policy_to_sequential


class StoppingCarExperiment(Experiment):
    '''Simply use the neural network provided to do verification'''
    def __init__(self):
        env_input_size: int = 3
        super().__init__(env_input_size)
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        self.template_2d: np.ndarray = np.array([[1, 0, 0], [1, -1, 0]])
        input_boundaries = [30, -25, 0, 0, 36, -28]
        input_template = Experiment.box(self.env_input_size)
        self.input_boundaries: List = input_boundaries
        self.input_template: np.ndarray = input_template
        x_lead = Experiment.e(self.env_input_size, 0)
        x_ego = Experiment.e(self.env_input_size, 1)
        v_ego = Experiment.e(self.env_input_size, 2)
        template = np.array([-(x_lead - x_ego),(x_lead - x_ego), v_ego, -v_ego])
        self.analysis_template: np.ndarray = template
        collision_distance = 0
        distance = [Experiment.e(self.env_input_size, 0) - Experiment.e(self.env_input_size, 1)]
        self.rounding_value = 2 ** 10
        self.use_rounding = False
        self.time_horizon = 400
        self.unsafe_zone: List[Tuple] = [(distance, np.array([collision_distance]))]
        self.input_epsilon = 0
        self.v_lead = 28
        self.max_speed = 36

    @ray.remote
    def post_milp(self, x, x_label, nn, output_flag, t, template):
        """milp method"""
        post = []
        for chosen_action in range(2):
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
            # gurobi_model.addConstr(input[0] >= 0, name=f"input_base_constr1")
            # gurobi_model.addConstr(input[1] >= 0, name=f"input_base_constr2")
            # gurobi_model.addConstr(input[2] >= 20, name=f"input_base_constr3")
            observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="observation")
            gurobi_model.addConstr(observation[1] <= input[0] - input[1] + self.input_epsilon / 2, name=f"obs_constr21")  # delta_x
            gurobi_model.addConstr(observation[1] >= input[0] - input[1] - self.input_epsilon / 2, name=f"obs_constr22")
            gurobi_model.addConstr(observation[0] <= self.v_lead - input[2] + self.input_epsilon / 2, name=f"obs_constr11")  # delta_v
            gurobi_model.addConstr(observation[0] >= self.v_lead - input[2] - self.input_epsilon / 2, name=f"obs_constr12")
            # gurobi_model.addConstr(input[3] <= self.max_speed, name=f"v_constr_input")
            # gurobi_model.addConstr(input[3] >= -self.max_speed, name=f"v_constr_input")
            feasible_action = Experiment.generate_nn_guard(gurobi_model, observation, nn, action_ego=chosen_action, M=1e6)
            # feasible_action = Experiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
            if feasible_action:
                # apply dynamic
                x_prime = StoppingCarExperiment.apply_dynamic(input, gurobi_model, action=chosen_action, env_input_size=self.env_input_size)
                gurobi_model.update()
                gurobi_model.optimize()
                found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                if found_successor:
                    post.append((tuple(x_prime_results), (x, x_label)))
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
        # config = get_PPO_config(1234)
        # trainer = ppo.PPOTrainer(config=config)
        # trainer.restore(self.nn_path)
        # policy = trainer.get_policy()
        # sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        # # l0 = torch.nn.Linear(6, 2, bias=False)
        # # l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]], dtype=torch.float32))
        # # layers = [l0]
        # # for l in sequential_nn:
        # #     layers.append(l)
        # #
        # # nn = torch.nn.Sequential(*layers)
        # nn = sequential_nn
        # # ray.shutdown()

        state_size = 2
        action_size = 2
        ALPHA = 0.6  # the higher the more aggressive the sampling towards high TD transitions
        agent = InvariantAgent(state_size=state_size, action_size=action_size, alpha=ALPHA)
        agent.load("/home/edoardo/Development/SafeDRL/runs/Aug05_14-55-31_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_1100.pth")
        nn = agent.qnetwork_local.sequential
        nn.cpu()
        return nn


if __name__ == '__main__':
    ray.init(log_to_driver=False, local_mode=False)
    experiment = StoppingCarExperiment()
    experiment.save_dir = "/home/edoardo/ray_results/tune_PPO_stopping_car/test"
    experiment.plotting_time_interval = 600
    experiment.show_progressbar = True
    experiment.show_progress_plot = True
    experiment.time_horizon = 150
    experiment.run_experiment()
