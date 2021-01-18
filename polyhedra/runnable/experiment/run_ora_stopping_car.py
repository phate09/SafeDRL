import gurobi as grb
import ray
import numpy as np
from polyhedra.experiments_nn_analysis import Experiment
from polyhedra.runnable.experiment.run_experiment_stopping_car import StoppingCarExperiment


class ORAStoppingCarExperiment(StoppingCarExperiment):
    def __init__(self):
        super().__init__()
        self.post_fn_remote = self.post_milp
        self.before_start_fn = self.before_start

    def generate_nn_polyhedral_guard(self, nn, chosen_action, output_flag):
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        gurobi_model.setParam('Threads', 2)
        observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="observation")
        Experiment.generate_nn_guard(gurobi_model, observation, nn, action_ego=chosen_action)
        observable_template = Experiment.octagon(2)
        self.env_input_size = 2
        observable_result = self.optimise(observable_template, gurobi_model, observation)  # todo fix this
        self.env_input_size = 6
        return observable_template, observable_result

    def before_start(self, nn):
        observable_templates = []
        observable_results = []
        for chosen_action in range(2):
            observable_template, observable_result = self.generate_nn_polyhedral_guard(nn, chosen_action, False)
            observable_templates.append(observable_template)
            observable_results.append(observable_result)
        self.observable_templates = observable_templates
        self.observable_results = observable_results

    @ray.remote
    def post_milp(self, x, nn, output_flag, t, template):
        """milp method"""
        post = []
        for chosen_action in range(2):
            observable_template = self.observable_templates[chosen_action]
            observable_result = self.observable_results[chosen_action]
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            gurobi_model.setParam('Threads', 2)
            input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)
            observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="input")
            gurobi_model.addConstr(observation[1] <= input[0] - input[1] + self.input_epsilon / 2, name=f"obs_constr21")
            gurobi_model.addConstr(observation[1] >= input[0] - input[1] - self.input_epsilon / 2, name=f"obs_constr22")
            gurobi_model.addConstr(observation[0] <= input[2] - input[3] + self.input_epsilon / 2, name=f"obs_constr11")
            gurobi_model.addConstr(observation[0] >= input[2] - input[3] - self.input_epsilon / 2, name=f"obs_constr12")
            # feasible_action = Experiment.generate_nn_guard(gurobi_model, observation, nn, action_ego=chosen_action)
            # feasible_action = Experiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)

            Experiment.generate_region_constraints(gurobi_model, observable_template, observation, observable_result, 2)
            gurobi_model.optimize()
            feasible_action = gurobi_model.status
            if feasible_action:
                # apply dynamic
                # x_prime_results = self.optimise(template, gurobi_model, input)  # h representation
                # x_prime = Experiment.generate_input_region(gurobi_model, template, x_prime_results, self.env_input_size)
                x_second = StoppingCarExperiment.apply_dynamic(input, gurobi_model, action=chosen_action, env_input_size=self.env_input_size)
                gurobi_model.update()
                gurobi_model.optimize()
                found_successor, x_second_results = self.h_repr_to_plot(gurobi_model, template, x_second)
                if found_successor:
                    post.append(tuple(x_second_results))
        return post


if __name__ == '__main__':
    experiment = ORAStoppingCarExperiment()
    experiment.run_experiment(local_mode=False)
