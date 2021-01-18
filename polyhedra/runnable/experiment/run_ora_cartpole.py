import gurobi as grb
import ray

from polyhedra.experiments_nn_analysis import Experiment
from polyhedra.runnable.experiment.run_experiment_cartpole import CartpoleExperiment


class ORACartpoleExperiment(CartpoleExperiment):
    def __init__(self):
        super().__init__()
        self.post_fn_remote = self.post_milp

    @ray.remote
    def post_milp(self, x, nn, output_flag, t, template):
        """milp method"""
        post = []
        for chosen_action in range(2):
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            gurobi_model.setParam('Threads', 2)
            input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
            max_theta, min_theta, max_theta_dot, min_theta_dot = self.get_theta_bounds(gurobi_model, input)
            sin_cos_table = self.get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action=chosen_action)
            feasible_action = CartpoleExperiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
            if feasible_action:
                x_prime_results = self.optimise(template, gurobi_model, input)  # h representation
                x_prime = Experiment.generate_input_region(gurobi_model, template, x_prime_results, self.env_input_size)
                thetaacc, xacc = CartpoleExperiment.generate_angle_milp(gurobi_model, x_prime, sin_cos_table)
                # apply dynamic
                x_second = self.apply_dynamic(x_prime, gurobi_model, thetaacc=thetaacc, xacc=xacc, env_input_size=self.env_input_size)
                gurobi_model.update()
                gurobi_model.optimize()
                found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_second)
                if found_successor:
                    post.append(tuple(x_prime_results))
        return post

if __name__ == '__main__':
    experiment = ORACartpoleExperiment()
    experiment.run_experiment(local_mode=False)
