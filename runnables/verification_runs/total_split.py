import pickle
import sys
from typing import List

import cdd
import gurobi as grb
import numpy as np
import progressbar
import ray
import sympy
import torch
from numpy import array, hstack
from ray.rllib.agents import ppo
from scipy.stats import norm
from sympy import Line

from polyhedra.experiments_nn_analysis import Experiment
from polyhedra.partitioning import is_split_range, sample_and_split, find_inverted_dimension, split_polyhedron_milp
from polyhedra.plot_utils import show_polygons
from runnables.runnable.templates import polytope
from runnables.runnable.templates.dikin_walk_simplified import plot_points_and_prediction
from symbolic import unroll_methods
from training.ppo.tune.tune_train_PPO_car import get_PPO_config
from training.ray_utils import convert_ray_policy_to_sequential


class TotalSplit:
    def __init__(self):
        self.use_split = True
        self.env_input_size: int = 2
        self.max_probability_split = 0.33
        self.input_epsilon = 0
        self.use_entropy_split = True
        self.output_flag = False
        self.minimum_length = 0.2
        self.template_2d: np.ndarray = np.array([[0, 1], [1, 0]])
        self.input_boundaries = tuple([50, 0, 36, 36])
        self.input_template = Experiment.box(self.env_input_size)
        delta_x = Experiment.e(self.env_input_size, 0)
        # x_ego = Experiment.e(self.env_input_size, 1)
        v_ego = Experiment.e(self.env_input_size, 1)
        self.use_softmax = True
        self.use_milp_range_prob = True
        # template = Experiment.combinations([1 / 4.5*(x_lead - x_ego), - v_ego])
        template = np.array([delta_x, -delta_x, v_ego, -v_ego, 1 / 4.5 * delta_x + v_ego, 1 / 4.5 * delta_x - v_ego, -1 / 4.5 * delta_x + v_ego,
                             -1 / 4.5 * delta_x - v_ego])
        self.analysis_template: np.ndarray = template
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58"  # safe

    def get_pre_nn(self):
        raise NotImplementedError()

    def get_observation_variable(self, input, gurobi_model):
        raise NotImplementedError()

    def create_range_bounds_model(self, template, x, env_input_size, nn, round=-1):
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', False)
        input = Experiment.generate_input_region(gurobi_model, template, x, env_input_size)
        gurobi_model.update()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        observation = self.get_observation_variable(input, gurobi_model)  # get the observation from the input
        ranges = Experiment.get_range_bounds(observation, nn, gurobi_model)
        if self.use_softmax:
            ranges_probs = unroll_methods.softmax_interval(ranges)
        else:
            ranges_probs = ranges
        if round >= 0:
            pass
            # todo round the probabilities
        return ranges_probs

    def sample_probabilities(self, template, x, nn, pre_nn):
        samples = polytope.sample(10000, template, np.array(x))
        preprocessed = pre_nn(torch.tensor(samples).float())
        samples_ontput = nn(preprocessed)
        if self.use_softmax:
            samples_ontput = torch.softmax(samples_ontput, 1)
        predicted_label = samples_ontput.detach().numpy()[:, 0]
        min_prob = np.min(samples_ontput.detach().numpy(), 0)
        max_prob = np.max(samples_ontput.detach().numpy(), 0)
        result = list(zip(min_prob, max_prob))
        return result

    def find_direction_split(self, template, x, nn, pre_nn):
        samples = polytope.sample(10000, template, np.array(x))
        preprocessed = pre_nn(torch.tensor(samples).float())
        preprocessed_np = preprocessed.detach().numpy()
        samples_ontput = torch.softmax(nn(preprocessed), 1)
        predicted_label = samples_ontput.detach().numpy()[:, 0]
        y = np.clip(predicted_label, 1e-7, 1 - 1e-7)
        inv_sig_y = np.log(y / (1 - y))  # transform to log-odds-ratio space
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(samples, inv_sig_y)
        template_2d: np.ndarray = np.array([Experiment.e(3, 2), Experiment.e(3, 0) - Experiment.e(3, 1)])

        def sigmoid(x):
            ex = np.exp(x)
            return ex / (1 + ex)

        preds = sigmoid(lr.predict(samples))
        plot_points_and_prediction(samples @ template_2d.T, preds)
        plot_points_and_prediction(samples @ template_2d.T, predicted_label)
        coeff = lr.coef_
        intercept = lr.intercept_
        a = sympy.symbols('x')
        b = sympy.symbols('y')
        classif_line1 = Line(coeff[0].item() * a + coeff[1].item() * b + intercept)
        new_coeff = -coeff[0].item() / coeff[1].item()

    def check_split(self, x, nn, template, template_2d) -> List:
        # -------splitting
        pre_nn = self.get_pre_nn()
        # self.find_direction_split(template,x,nn,pre_nn)
        ranges_probs = self.sample_probabilities(template, x, nn, pre_nn)  # sampled version
        if not is_split_range(ranges_probs, self.max_probability_split) and self.use_milp_range_prob:  # refine only if the range is small
            ranges_probs = self.create_range_bounds_model(template, x, self.env_input_size, nn)
        new_frontier = []
        to_split = []
        n_splits = 0
        to_split.append((x, ranges_probs))
        widgets = [progressbar.Variable('splitting_queue'), ", ", progressbar.Variable('frontier_size'), ", ", progressbar.widgets.Timer()]
        with progressbar.ProgressBar(prefix=f"Splitting states: ", widgets=widgets, is_terminal=True, term_width=200, redirect_stdout=True).start() as bar_split:
            while len(to_split) != 0:
                bar_split.update(value=bar_split.value + 1, splitting_queue=len(to_split), frontier_size=len(new_frontier))
                to_analyse, ranges_probs = to_split.pop()
                split_flag = is_split_range(ranges_probs, self.max_probability_split)
                can_be_split = self.can_be_split(template, to_analyse)
                action_to_split = np.nanargmax([x[1] - x[0] for x in ranges_probs])
                if split_flag and can_be_split:
                    split1, split2 = sample_and_split(pre_nn, nn, template, np.array(to_analyse), self.env_input_size, template_2d, action=action_to_split,
                                                      minimum_length=self.minimum_length, use_softmax=self.use_softmax)
                    n_splits += 1
                    if split1 is None or split2 is None:
                        split1, split2 = sample_and_split(pre_nn, nn, template, np.array(to_analyse), self.env_input_size, template_2d, action=action_to_split,
                                                          minimum_length=self.minimum_length, use_softmax=self.use_softmax)
                    ranges_probs1 = self.sample_probabilities(template, split1, nn, pre_nn)  # sampled version
                    if not is_split_range(ranges_probs1, self.max_probability_split) and self.use_milp_range_prob:  # refine only if the range is small
                        ranges_probs1 = self.create_range_bounds_model(template, split1, self.env_input_size, nn)
                    ranges_probs2 = self.sample_probabilities(template, split2, nn, pre_nn)  # sampled version
                    if not is_split_range(ranges_probs2, self.max_probability_split) and self.use_milp_range_prob:  # refine only if the range is small
                        ranges_probs2 = self.create_range_bounds_model(template, split2, self.env_input_size, nn)
                    to_split.append((tuple(split1), ranges_probs1))
                    to_split.append((tuple(split2), ranges_probs2))

                else:
                    new_frontier.append((to_analyse, ranges_probs))
                    # plot_frontier(new_frontier)

        pickle.dump(new_frontier, open("new_frontier3.p", "wb"))
        self.plot_frontier(new_frontier + to_split)
        print("", file=sys.stderr)  # new line
        return new_frontier

    def plot_2d_sample(self, sample_size=10000):
        root = self.generate_root_polytope(self.input_boundaries)
        pre_nn = self.get_pre_nn()
        nn = self.get_nn()
        samples = polytope.sample(sample_size, self.analysis_template, np.array(root))
        preprocessed = pre_nn(torch.tensor(samples).float())
        samples_ontput = nn(preprocessed)
        if self.use_softmax:
            samples_ontput = torch.softmax(samples_ontput, 1)
        predicted_label = samples_ontput.detach().numpy()[:, 0]
        plot_points_and_prediction(samples @ self.template_2d.T, predicted_label)
        print("plot done")

    def can_be_split(self, template, x):
        at_least_one_valid_dimension = False
        dimension_lengths = []
        for i, dimension in enumerate(template):
            inverted_dimension = find_inverted_dimension(-dimension, template)
            dimension_length = x[i] + x[inverted_dimension]
            dimension_lengths.append(dimension_length)
            if dimension_length > self.minimum_length:
                at_least_one_valid_dimension = True
        return at_least_one_valid_dimension

    def plot_frontier(self, new_frontier, show_probs=True):
        colours = []
        for x, ranges_probs in new_frontier:
            colours.append(np.mean(ranges_probs[0]))
        print("", file=sys.stderr)  # new line
        fig = show_polygons(self.analysis_template, [x[0] for x in new_frontier], self.template_2d, colours if show_probs else None)

    def load_frontier(self):
        new_frontier = pickle.load(open("new_frontier3.p", "rb"))
        self.plot_frontier(new_frontier)
        return new_frontier

    def split_item(self, template):
        '''takes an abstract mapping saved previously and splits a new input boundaries according to it'''
        frontier = pickle.load(open("new_frontier3.p", "rb"))
        root = self.generate_root_polytope(self.input_boundaries)
        chosen_polytopes = []
        to_split = [root]
        processed = []
        new_frontier = []
        for x, probs in frontier:
            choose = self.check_contained(x, root)
            if choose:
                new_frontier.append((x, probs))
        frontier = new_frontier  # shrink the selection of polytopes to only the ones which are relevant

        # colours = []
        # for x, ranges_probs in frontier:
        #     colours.append(np.mean(ranges_probs[0]))
        # print("", file=sys.stderr)  # new line
        # fig = show_polygons(template, [x[0] for x in frontier], self.template_2d, colours)

        widgets = [progressbar.Variable('splitting_queue'), ", ", progressbar.Variable('frontier_size'), ", ", progressbar.widgets.Timer()]
        with progressbar.ProgressBar(prefix=f"Splitting states: ", widgets=widgets, is_terminal=True, term_width=200, redirect_stdout=True).start() as bar_split:
            while len(to_split) > 0:
                current_polytope = to_split.pop()
                split_happened = False
                for x, probs in frontier:
                    choose = self.check_contained(x, current_polytope)
                    if choose:
                        dimensions_volume = self.find_important_dimensions(current_polytope, x)
                        if len(dimensions_volume) > 0:
                            assert len(dimensions_volume) > 0
                            min_idx = dimensions_volume[np.argmin(np.array(dimensions_volume)[:, 1])][0]
                            splits = split_polyhedron_milp(self.analysis_template, current_polytope, min_idx, x[min_idx])
                            chosen_polytopes.append((x, probs))
                            to_split.append(tuple(splits[0]))
                            to_split.append(tuple(splits[1]))
                            split_happened = True
                            break
                    bar_split.update(splitting_queue=len(to_split), frontier_size=len(processed))
                if not split_happened:
                    processed.append(current_polytope)

        # colours = []
        # for x, ranges_probs in frontier:
        #     colours.append(np.mean(ranges_probs[0]))
        # print("", file=sys.stderr)  # new line
        # fig = show_polygons(template, [x[0] for x in frontier] + to_split + processed, self.template_2d, colours + [0.5] * len(to_split) + [0.5] * len(processed))
        return processed

    def check_contained(self, poly1, poly2):
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', self.output_flag)
        input1 = Experiment.generate_input_region(gurobi_model, self.analysis_template, poly1, self.env_input_size)
        Experiment.generate_region_constraints(gurobi_model, self.analysis_template, input1, poly2, self.env_input_size, eps=1e-5)  # use epsilon to prevent single points
        x_results = self.optimise(self.analysis_template, gurobi_model, input1)
        if x_results is None:
            # not contained
            return False
        else:
            return True

    def find_important_dimensions(self, poly1, poly2):
        '''assuming check_contained(poly1,poly2) returns true, we are interested in the halfspaces that matters
        poly1 = root, poly2 = candidate
        '''
        # #Binary Space Partitioning
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', self.output_flag)
        input1 = Experiment.generate_input_region(gurobi_model, self.analysis_template, poly1, self.env_input_size)
        relevant_directions = []
        for j, template in enumerate(self.analysis_template):
            multiplication = 0
            for i in range(self.env_input_size):
                multiplication += template[i] * input1[i]
            previous_constraint = gurobi_model.getConstrByName("check_contained_constraint")
            if previous_constraint is not None:
                gurobi_model.remove(previous_constraint)
                gurobi_model.update()
            gurobi_model.addConstr(multiplication <= poly2[j], name=f"check_contained_constraint")
            gurobi_model.update()
            x_results = self.optimise(self.analysis_template, gurobi_model, input1)
            if np.allclose(np.array(poly1), x_results) is False:
                vertices = np.stack(self.pypoman_compute_polytope_vertices(self.analysis_template, np.array(x_results)))
                samples = polytope.sample(1000, self.analysis_template, x_results)
                from scipy.spatial import ConvexHull
                hull = ConvexHull(samples)
                volume = hull.volume  # estimated volume
                relevant_directions.append((j, volume))
        return relevant_directions

    def start(self):
        root = self.generate_root_polytope(self.input_boundaries)
        nn = self.get_nn()
        self.check_split(root, nn, self.analysis_template, self.template_2d)
        # self.split_item(self.analysis_template)

    def pypoman_compute_polytope_vertices(self, A, b):
        b = b.reshape((b.shape[0], 1))
        mat = cdd.Matrix(hstack([b, -A]), number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY
        P = cdd.Polyhedron(mat)
        g = P.get_generators()
        V = array(g)
        vertices = []
        for i in range(V.shape[0]):
            if V[i, 0] != 1:  # 1 = vertex, 0 = ray
                # raise Exception("Polyhedron is not a polytope")
                pass
            elif i not in g.lin_set:
                vertices.append(V[i, 1:])
        return vertices

    def generate_root_polytope(self, input_boundaries):
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', self.output_flag)
        input = Experiment.generate_input_region(gurobi_model, self.input_template, input_boundaries, self.env_input_size)
        x_results = self.optimise(self.analysis_template, gurobi_model, input)
        if x_results is None:
            print("Model unsatisfiable")
            return None
        root = tuple(x_results)
        return root

    @staticmethod
    def optimise(templates: np.ndarray, gurobi_model: grb.Model, x_prime: tuple):
        results = []
        for template in templates:
            gurobi_model.update()
            gurobi_model.setObjective(sum((template[i] * x_prime[i]) for i in range(len(template))), grb.GRB.MAXIMIZE)
            gurobi_model.optimize()
            # print_model(gurobi_model)
            if gurobi_model.status == 5:
                result = float("inf")
                results.append(result)
                continue
            if gurobi_model.status == 4 or gurobi_model.status == 3:
                return None
            assert gurobi_model.status == 2, f"gurobi_model.status=={gurobi_model.status}"
            # if gurobi_model.status != 2:
            #     return None
            result = gurobi_model.ObjVal
            results.append(result)
        return np.array(results)

    def get_nn(self):
        raise NotImplementedError()


class TotalSplitStoppingCar(TotalSplit):
    def __init__(self):
        super().__init__()
        self.v_lead = 28
        self.env_input_size: int = 2
        self.template_2d: np.ndarray = np.array([[0, 1], [1, 0]])
        self.input_boundaries = tuple([10, -3, 32, -26])
        self.input_template = Experiment.box(self.env_input_size)
        delta_x = Experiment.e(self.env_input_size, 0)
        # x_ego = Experiment.e(self.env_input_size, 1)
        v_ego = Experiment.e(self.env_input_size, 1)
        # template = Experiment.combinations([1 / 4.5*(x_lead - x_ego), - v_ego])
        template = np.array([delta_x, -delta_x, v_ego, -v_ego, 1 / 4.5 * delta_x + v_ego, 1 / 4.5 * delta_x - v_ego, -1 / 4.5 * delta_x + v_ego,
                             -1 / 4.5 * delta_x - v_ego])
        self.max_probability_split = 0.33
        self.analysis_template: np.ndarray = template
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58"  # safe

    def get_pre_nn(self):
        l0 = torch.nn.Linear(self.env_input_size, 2, bias=False)
        l0.weight = torch.nn.Parameter(torch.tensor([[0, -1], [1, 0]], dtype=torch.float32))
        l0.bias = torch.nn.Parameter(torch.tensor([[28, 0]], dtype=torch.float32))
        layers = [l0]
        pre_nn = torch.nn.Sequential(*layers)
        torch.nn.Identity()
        return pre_nn

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

    def get_nn(self):
        config = get_PPO_config(1234, 0)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(self.nn_path)
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        nn = sequential_nn
        return nn


class TotalSplitBouncingBall(TotalSplit):
    def __init__(self):
        super().__init__()
        self.v_lead = 28
        self.max_probability_split = 0.15
        self.env_input_size: int = 2
        self.template_2d: np.ndarray = np.array([[0, 1], [1, 0]])
        self.input_boundaries = tuple([9, 0, 3, 3])
        self.input_template = Experiment.box(self.env_input_size)
        p = Experiment.e(self.env_input_size, 0)
        # x_ego = Experiment.e(self.env_input_size, 1)
        v = Experiment.e(self.env_input_size, 1)
        template = np.array([p, -p, v, -v, 1 / 3 * (p - v), -1 / 3 * (p - v)])  # , 1 / 6 * p - v, -1 / 6 * p - v
        # self.analysis_template: np.ndarray = Experiment.box(self.env_input_size)
        self.analysis_template: np.ndarray = template
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_c7326_00000_0_2021-01-16_05-43-36/checkpoint_36/checkpoint-36"
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00004_4_2021-01-18_23-48-21/checkpoint_10/checkpoint-10"

    def get_pre_nn(self):
        # l0 = torch.nn.Linear(self.env_input_size, 2, bias=False)
        # l0.weight = torch.nn.Parameter(torch.tensor([[0, -1], [1, 0]], dtype=torch.float32))
        # l0.bias = torch.nn.Parameter(torch.tensor([[28, 0]], dtype=torch.float32))
        # layers = [l0]
        # pre_nn = torch.nn.Sequential(*layers)
        # torch.nn.Identity()
        return torch.nn.Identity()

    def get_observation_variable(self, input, gurobi_model):
        # observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="observation")
        # gurobi_model.addConstr(observation[1] <= input[0] + self.input_epsilon / 2, name=f"obs_constr21")
        # gurobi_model.addConstr(observation[1] >= input[0] - self.input_epsilon / 2, name=f"obs_constr22")
        # gurobi_model.addConstr(observation[0] <= self.v_lead - input[1] + self.input_epsilon / 2, name=f"obs_constr11")
        # gurobi_model.addConstr(observation[0] >= self.v_lead - input[1] - self.input_epsilon / 2, name=f"obs_constr12")
        # gurobi_model.update()
        # gurobi_model.optimize()
        # assert gurobi_model.status == 2, "LP wasn't optimally solved"
        return input

    def get_nn(self):
        from training.ppo.tune.tune_train_PPO_bouncing_ball import get_PPO_config
        config = get_PPO_config(1234, 0)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(self.nn_path)
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        nn = sequential_nn
        return nn


class TotalSplitPendulum(TotalSplit):
    def __init__(self):
        super().__init__()
        self.max_probability_split = 0.5
        self.env_input_size: int = 2
        self.template_2d: np.ndarray = np.array([[0, 1], [1, 0]])
        self.input_boundaries = [np.pi / 5, np.pi / 5, 1 / 5, 1 / 5]
        self.input_template = Experiment.box(self.env_input_size)
        # theta = Experiment.e(self.env_input_size, 0)
        # theta_dot = Experiment.e(self.env_input_size, 1)
        # template = np.array([theta, -theta, theta_dot, -theta_dot])
        template = Experiment.box(self.env_input_size)
        self.analysis_template: np.ndarray = template
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_pendulum/PPO_MonitoredPendulum_035b5_00000_0_2021-05-11_11-59-52/checkpoint_3333/checkpoint-3333"

    def get_pre_nn(self):
        return torch.nn.Identity()

    def get_observation_variable(self, input, gurobi_model):
        return input

    def get_nn(self):
        from training.ppo.tune.tune_train_PPO_inverted_pendulum import get_PPO_config
        config = get_PPO_config(1234, 0)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(self.nn_path)
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        nn = sequential_nn
        return nn

    def plot_frontier(self, new_frontier):
        colours = []
        for x, ranges_probs in new_frontier:
            colours.append(np.mean(np.array(ranges_probs), 1))
        print("", file=sys.stderr)  # new line
        fig = show_polygons(self.analysis_template, [x[0] for x in new_frontier], self.template_2d, colours, rgb=True)


class GaussianLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = 0
        self.std = 1

    def forward(self, x):
        return torch.unsqueeze(torch.tensor(norm.pdf(x[:, 0], self.mean, self.std)) * torch.tensor(norm.pdf(x[:, 1], self.mean, self.std)), 1) / 0.159


class TotalSplitGaussian(TotalSplit):
    def __init__(self):
        super().__init__()
        self.max_probability_split = 0.3
        self.env_input_size: int = 2
        self.template_2d: np.ndarray = np.array([[0, 1], [1, 0]])
        self.input_boundaries = [5, 5, 5, 5]
        self.use_softmax = False
        self.use_milp_range_prob = False
        self.input_template = Experiment.box(self.env_input_size)
        template = Experiment.box(self.env_input_size)
        self.analysis_template: np.ndarray = template
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_pendulum/PPO_MonitoredPendulum_035b5_00000_0_2021-05-11_11-59-52/checkpoint_3333/checkpoint-3333"

    def get_nn(self):
        layers = []
        layers.append(GaussianLayer())
        return torch.nn.Sequential(*layers)

    def get_pre_nn(self):
        return torch.nn.Identity()

    def get_observation_variable(self, input, gurobi_model):
        return input


if __name__ == '__main__':
    ray.init(local_mode=False)
    agent = TotalSplitStoppingCar()
    agent.plot_2d_sample(sample_size=100000)
    # agent.load_frontier()
    polytopes = agent.start()
