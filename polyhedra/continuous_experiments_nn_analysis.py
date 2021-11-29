import math
import time
from collections import defaultdict
from contextlib import nullcontext

import gurobipy as grb
import numpy as np
import progressbar

from polyhedra.experiments_nn_analysis import Experiment, contained


class ContinuousExperiment(Experiment):
    def __init__(self, env_input_size: int):
        super().__init__(env_input_size)
        self.post_fn_continuous = None
        self.keep_model = False  # whether to keep the gurobi model for later timesteps
        self.last_input = None
        self.internal_model = None

    def main_loop(self, nn, template, template_2d):
        assert self.post_fn_continuous is not None
        root = self.generate_root_polytope()
        root_list = [root]
        vertices_list = defaultdict(list)
        seen = []
        frontier = [(0, x) for x in root_list]
        max_t = 0
        num_already_visited = 0
        widgets = [progressbar.Variable('n_workers'), ', ', progressbar.Variable('frontier'), ', ', progressbar.Variable('seen'), ', ', progressbar.Variable('num_already_visited'), ", ",
                   progressbar.Variable('max_t'), ", ", progressbar.Variable('last_visited_state')]
        last_time_plot = None
        if self.before_start_fn is not None:
            self.before_start_fn(nn)
        self.internal_model = grb.Model()
        self.internal_model.setParam('OutputFlag', self.output_flag)
        input = Experiment.generate_input_region(self.internal_model, self.input_template, self.input_boundaries, self.env_input_size)
        self.last_input = input
        with progressbar.ProgressBar(widgets=widgets) if self.show_progressbar else nullcontext() as bar:
            while len(frontier) != 0:
                t, x = frontier.pop(0) if self.use_bfs else frontier.pop()
                if max_t > self.time_horizon:
                    print(f"Reached horizon t={t}")
                    self.plot_fn(vertices_list, template, template_2d)
                    return max_t, num_already_visited, vertices_list, False
                contained_flag = False
                to_remove = []
                for s in seen:
                    if contained(x, s):
                        contained_flag = True
                        break
                    if contained(s, x):
                        to_remove.append(s)
                for rem in to_remove:
                    num_already_visited += 1
                    seen.remove(rem)
                if contained_flag:
                    num_already_visited += 1
                    continue
                max_t = max(max_t, t)
                vertices_list[t].append(np.array(x))
                if self.check_unsafe(template, x):
                    print(f"Unsafe state found at timestep t={t}")
                    print(x)
                    self.plot_fn(vertices_list, template, template_2d)
                    return max_t, num_already_visited, vertices_list, True
                seen.append(x)
                x_primes_list = self.post_fn_continuous(x, nn, self.output_flag, t, template)
                if last_time_plot is None or time.time() - last_time_plot >= self.plotting_time_interval:
                    if last_time_plot is not None:
                        self.plot_fn(vertices_list, template, template_2d)
                    last_time_plot = time.time()
                if self.update_progress_fn is not None:
                    self.update_progress_fn(seen=len(seen), frontier=len(frontier), num_already_visited=num_already_visited, max_t=max_t)
                if self.show_progressbar:
                    bar.update(value=bar.value + 1, seen=len(seen), frontier=len(frontier), num_already_visited=num_already_visited, last_visited_state=str(x), max_t=max_t)
                assert len(x_primes_list) != 0, "something is wrong with the calculation of the successor"
                # for x_primes in x_primes_list:
                for x_prime in x_primes_list:
                    if self.use_rounding:
                        # x_prime_rounded = tuple(np.trunc(np.array(x_prime) * self.rounding_value) / self.rounding_value)  # todo should we round to prevent numerical errors?
                        x_prime_rounded = self.round_tuple(x_prime, self.rounding_value)
                        # x_prime_rounded should always be bigger than x_prime
                        assert contained(x_prime, x_prime_rounded)
                        x_prime = x_prime_rounded
                    frontier = [(u, y) for u, y in frontier if not contained(y, x_prime)]
                    if not any([contained(x_prime, y) for u, y in frontier]):
                        frontier.append(((t + 1), x_prime))
                        # print(x_prime)
                    else:
                        num_already_visited += 1
        self.plot_fn(vertices_list, template, template_2d)
        return max_t, num_already_visited, vertices_list, False

    @staticmethod
    def round_tuple(x, rounding_value):
        rounded_x = []
        for val in x:
            if val < 0:
                rounded_x.append(-1 * math.floor(abs(val) * rounding_value) / rounding_value)
            else:
                rounded_x.append(math.ceil(abs(val) * rounding_value) / rounding_value)
        return tuple(rounded_x)

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

    def optimise(self, templates: np.ndarray, gurobi_model: grb.Model, x_prime: tuple):
        results = []
        for template in templates:
            gurobi_model.update()
            gurobi_model.setObjective(sum((template[i] * x_prime[i]) for i in range(self.env_input_size)), grb.GRB.MAXIMIZE)
            gurobi_model.optimize()
            # print_model(gurobi_model)
            if gurobi_model.status == 5 or gurobi_model.status == 4:
                result = float("inf")
                results.append(result)
                continue
            assert gurobi_model.status == 2, f"gurobi_model.status=={gurobi_model.status}"
            # if gurobi_model.status != 2:
            #     return None
            result = gurobi_model.ObjVal
            results.append(result)
        return np.array(results)

    def check_unsafe(self, template, bnds):
        for A, b in self.unsafe_zone:
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', False)
            input = gurobi_model.addMVar(shape=(self.env_input_size,), lb=float("-inf"), name="input")
            Experiment.generate_region_constraints(gurobi_model, template, input, bnds, self.env_input_size)
            Experiment.generate_region_constraints(gurobi_model, A, input, b, self.env_input_size)
            gurobi_model.update()
            gurobi_model.optimize()
            if gurobi_model.status == 2:
                return True
        return False
