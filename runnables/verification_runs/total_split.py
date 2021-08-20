import sys
from typing import List, Tuple
import gurobi as grb
import progressbar
import ray
import torch
import numpy as np
from ray.rllib.agents import ppo
import pickle
from agents.ppo.tune.tune_train_PPO_car import get_PPO_config
from agents.ray_utils import convert_ray_policy_to_sequential
from polyhedra.experiments_nn_analysis import Experiment
from polyhedra.partitioning import is_split_range, sample_and_split, pick_longest_dimension, split_polyhedron
from polyhedra.plot_utils import show_polygons
from polyhedra.runnable.templates import polytope
from symbolic import unroll_methods


class TotalSplit:
    def __init__(self):
        self.use_split = True
        self.env_input_size: int = 3
        self.max_probability_split = 0.33
        self.input_epsilon = 0
        self.v_lead = 28
        self.use_entropy_split = True
        self.output_flag = False

        self.template_2d: np.ndarray = np.array([[0, 0, 1], [1, -1, 0]])
        self.input_boundaries = tuple([50, 0, 0, 0, 36, 0])
        self.input_template = Experiment.box(self.env_input_size)
        x_lead = Experiment.e(self.env_input_size, 0)
        x_ego = Experiment.e(self.env_input_size, 1)
        v_ego = Experiment.e(self.env_input_size, 2)
        # template = Experiment.combinations([x_lead - x_ego, - v_ego])
        template = np.array([x_lead - x_ego,-(x_lead - x_ego),v_ego,-v_ego,1/4.5*(x_lead - x_ego)+v_ego,1/4.5*(x_lead - x_ego)-v_ego,-1/4.5*(x_lead - x_ego)+v_ego,-1/4.5*(x_lead - x_ego)-v_ego])
        self.analysis_template: np.ndarray = template
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58"  # safe

    def get_pre_nn(self):
        l0 = torch.nn.Linear(self.env_input_size, 2, bias=False)
        l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, -1], [1, -1, 0]], dtype=torch.float32))
        l0.bias = torch.nn.Parameter(torch.tensor([[28, 0]], dtype=torch.float32))
        layers = [l0]
        pre_nn = torch.nn.Sequential(*layers)
        return pre_nn

    def get_observation_variable(self, input, gurobi_model):
        observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="observation")
        gurobi_model.addConstr(observation[1] <= input[0] - input[1] + self.input_epsilon / 2, name=f"obs_constr21")
        gurobi_model.addConstr(observation[1] >= input[0] - input[1] - self.input_epsilon / 2, name=f"obs_constr22")
        gurobi_model.addConstr(observation[0] <= self.v_lead - input[2] + self.input_epsilon / 2, name=f"obs_constr11")
        gurobi_model.addConstr(observation[0] >= self.v_lead - input[2] - self.input_epsilon / 2, name=f"obs_constr12")
        gurobi_model.update()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        return observation

    def create_range_bounds_model(self, template, x, env_input_size, nn, round=-1):
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', False)
        input = Experiment.generate_input_region(gurobi_model, template, x, env_input_size)
        gurobi_model.update()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        observation = self.get_observation_variable(input, gurobi_model)  # get the observation from the input
        ranges = Experiment.get_range_bounds(observation, nn, gurobi_model)
        ranges_probs = unroll_methods.softmax_interval(ranges)
        if round >= 0:
            pass
            # todo round the probabilities
        return ranges_probs

    def sample_probabilities(self, template, x, nn, pre_nn):
        samples = polytope.sample(10000, template, np.array(x))
        preprocessed = pre_nn(torch.tensor(samples).float())
        samples_ontput = torch.softmax(nn(preprocessed), 1)
        predicted_label = samples_ontput.detach().numpy()[:, 0]
        min_prob = np.min(samples_ontput.detach().numpy(), 0)
        max_prob = np.max(samples_ontput.detach().numpy(), 0)
        result = [(min_prob[0], max_prob[0]), (min_prob[1], max_prob[1])]
        return result

    def check_split(self, x, nn, template, template_2d) -> List:
        # -------splitting
        # todo sample probabilities instead of verifying
        pre_nn = self.get_pre_nn()
        ranges_probs = self.sample_probabilities(template, x, nn, pre_nn)  # sampled version
        if not is_split_range(ranges_probs, self.max_probability_split):  # refine only if the range is small
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
                if split_flag:
                    split1, split2 = sample_and_split(self.get_pre_nn(), nn, template, np.array(to_analyse), self.env_input_size, template_2d)
                    n_splits += 1
                    if split1 is None or split2 is None:
                        split1, split2 = sample_and_split(self.get_pre_nn(), nn, template, np.array(to_analyse), self.env_input_size, template_2d)
                    # todo plot to_split and split1,split2
                    ranges_probs1 = self.sample_probabilities(template, split1, nn, pre_nn)  # sampled version
                    if not is_split_range(ranges_probs1, self.max_probability_split):  # refine only if the range is small
                        ranges_probs1 = self.create_range_bounds_model(template, split1, self.env_input_size, nn)
                    ranges_probs2 = self.sample_probabilities(template, split2, nn, pre_nn)  # sampled version
                    if not is_split_range(ranges_probs2, self.max_probability_split):  # refine only if the range is small
                        ranges_probs2 = self.create_range_bounds_model(template, split2, self.env_input_size, nn)
                    to_split.append((tuple(split1), ranges_probs1))
                    to_split.append((tuple(split2), ranges_probs2))

                else:
                    new_frontier.append((to_analyse, ranges_probs))
                    # plot_frontier(new_frontier)

        pickle.dump(new_frontier, open("new_frontier.p", "wb"))
        colours = []
        for x, ranges_probs in new_frontier + to_split:
            colours.append(np.mean(ranges_probs[0]))
        print("", file=sys.stderr)  # new line
        fig = show_polygons(template, [x[0] for x in new_frontier+ to_split], template_2d, colours)
        fig.write_html("new_frontier.html")
        fig.show()
        print("", file=sys.stderr)  # new line
        return new_frontier

    def start(self):
        root = self.generate_root_polytope()
        nn = self.get_nn()
        self.check_split(root, nn, self.analysis_template, self.template_2d)

    def generate_root_polytope(self):
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', self.output_flag)
        input = Experiment.generate_input_region(gurobi_model, self.input_template, self.input_boundaries, self.env_input_size)
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
        config = get_PPO_config(1234, 0)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(self.nn_path)
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        nn = sequential_nn
        return nn


if __name__ == '__main__':
    ray.init()
    agent = TotalSplit()
    polytopes = agent.start()
