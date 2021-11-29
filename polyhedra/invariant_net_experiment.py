import time
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import progressbar
import ray

from polyhedra.experiments_nn_analysis import Experiment, contained


class InvariantNetExperiment(Experiment):
    def __init__(self, env_input_size: int):
        super().__init__(env_input_size)
        self.get_invariant_nn_fn = None

    def main_loop(self, nn, template, template_2d):
        proc_ids = []
        proc_ids.append(self.post_fn_remote(nn, self.get_invariant_nn_fn(), self.output_flag))
        ray.wait(proc_ids)

        root = self.generate_root_polytope()
        root_pair = (root, 0)  # label for root is always 0
        root_list = [root_pair]
        vertices_list = defaultdict(list)
        seen = []
        if self.additional_seen_fn is not None:
            for extra in self.additional_seen_fn():
                seen.append(extra)
        frontier = [(0, x) for x in root_list]
        if self.graph is not None:
            self.graph.add_node(root_pair)
        max_t = 0
        num_already_visited = 0
        widgets = [progressbar.Variable('n_workers'), ', ', progressbar.Variable('frontier'), ', ', progressbar.Variable('seen'), ', ', progressbar.Variable('num_already_visited'), ", ",
                   progressbar.Variable('max_t'), ", ", progressbar.Variable('last_visited_state')]

        last_time_plot = None
        if self.before_start_fn is not None:
            self.before_start_fn(nn)
        with progressbar.ProgressBar(widgets=widgets) if self.show_progressbar else nullcontext() as bar:
            while len(frontier) != 0 or len(proc_ids) != 0:
                while len(proc_ids) < self.n_workers and len(frontier) != 0:
                    t, (x, x_label) = frontier.pop(0) if self.use_bfs else frontier.pop()
                    if max_t > self.time_horizon:
                        print(f"Reached horizon t={t}")
                        self.plot_fn(vertices_list, template, template_2d)
                        return max_t, num_already_visited, vertices_list, False
                    contained_flag = False
                    to_remove = []
                    for (s, s_label) in seen:
                        if s_label == x_label:
                            if contained(x, s):
                                contained_flag = True
                                break
                            if contained(s, x):
                                to_remove.append((s, s_label))
                    for rem in to_remove:
                        num_already_visited += 1
                        seen.remove(rem)
                    if contained_flag:
                        num_already_visited += 1
                        continue
                    max_t = max(max_t, t)
                    vertices_list[t].append(np.array(x))
                    if self.check_unsafe(template, x, x_label):
                        print(f"Unsafe state found at timestep t={t}")
                        print((x, x_label))
                        self.plot_fn(vertices_list, template, template_2d)
                        return max_t, num_already_visited, vertices_list, True
                    seen.append((x, x_label))
                    proc_ids.append(self.post_fn_remote.remote(self, x, x_label, nn, self.output_flag, t, template))
                if last_time_plot is None or time.time() - last_time_plot >= self.plotting_time_interval:
                    if last_time_plot is not None:
                        self.plot_fn(vertices_list, template, template_2d)
                    last_time_plot = time.time()
                if self.update_progress_fn is not None:
                    self.update_progress_fn(n_workers=len(proc_ids), seen=len(seen), frontier=len(frontier), num_already_visited=num_already_visited, max_t=max_t)
                if self.show_progressbar:
                    bar.update(value=bar.value + 1, n_workers=len(proc_ids), seen=len(seen), frontier=len(frontier), num_already_visited=num_already_visited, last_visited_state=str(x), max_t=max_t)
                ready_ids, proc_ids = ray.wait(proc_ids, num_returns=len(proc_ids), timeout=0.5)
                if len(ready_ids) != 0:
                    x_primes_list = ray.get(ready_ids)
                    assert len(x_primes_list) != 0, "something is wrong with the calculation of the successor"
                    for x_primes in x_primes_list:
                        for x_prime, (parent, parent_lbl) in x_primes:
                            x_prime_label = self.assign_lbl_fn(x_prime, parent, parent_lbl)
                            if self.use_rounding:
                                # x_prime_rounded = tuple(np.trunc(np.array(x_prime) * self.rounding_value) / self.rounding_value)  # todo should we round to prevent numerical errors?
                                x_prime_rounded = self.round_tuple(x_prime, self.rounding_value)
                                # x_prime_rounded should always be bigger than x_prime
                                assert contained(x_prime, x_prime_rounded), f"{x_prime} not contained in {x_prime_rounded}"
                                x_prime = x_prime_rounded
                            frontier = [(u, (y, y_label)) for u, (y, y_label) in frontier if not (y_label == x_prime_label and contained(y, x_prime))]
                            if not any([(y_label == x_prime_label and contained(x_prime, y)) for u, (y, y_label) in frontier]):

                                frontier.append(((t + 1), (x_prime, x_prime_label)))
                                if self.graph is not None:
                                    self.graph.add_edge((parent, parent_lbl), (x_prime, x_prime_label))
                                # print(x_prime)
                            else:
                                num_already_visited += 1
        self.plot_fn(vertices_list, template, template_2d)
        return max_t, num_already_visited, vertices_list, False
