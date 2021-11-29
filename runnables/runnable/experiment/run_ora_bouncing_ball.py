import gurobi as grb
import ray

from polyhedra.experiments_nn_analysis import Experiment
from runnables.runnable.experiment.run_experiment_bouncing_ball import BouncingBallExperiment


class ORABouncingBallExperiment(BouncingBallExperiment):
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
        # self.env_input_size = 2
        observable_result = self.optimise(observable_template, gurobi_model, observation)
        # self.env_input_size = 6
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
        post = []
        observable_template_action1 = self.observable_templates[1]
        observable_result_action1 = self.observable_results[1]
        observable_template_action0 = self.observable_templates[0]
        observable_result_action0 = self.observable_results[0]

        def standard_op():
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            input = self.generate_input_region(gurobi_model, template, x, self.env_input_size)
            z = self.apply_dynamic(input, gurobi_model, self.env_input_size)
            return gurobi_model, z, input

        # case 0
        gurobi_model, z, input = standard_op()
        feasible0 = self.generate_guard(gurobi_model, z, case=0)  # bounce
        if feasible0:  # action is irrelevant in this case
            # apply dynamic
            x_prime_results = self.optimise(template, gurobi_model, z)
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            input2 = self.generate_input_region(gurobi_model, template, x_prime_results, self.env_input_size)
            x_second = self.apply_dynamic2(input2, gurobi_model, case=0, env_input_size=self.env_input_size)
            found_successor, x_second_results = self.h_repr_to_plot(gurobi_model, template, x_second)
            if found_successor:
                post.append(tuple(x_second_results))

        # case 1 : ball going down and hit
        gurobi_model, z, input = standard_op()
        feasible11 = self.generate_guard(gurobi_model, z, case=1)
        if feasible11:
            Experiment.generate_region_constraints(gurobi_model, observable_template_action1, input, observable_result_action1, 2)
            gurobi_model.optimize()
            feasible12 = gurobi_model.status
            # feasible12 = self.generate_nn_guard(gurobi_model, input, nn, action_ego=1)  # check for action =1 over input (not z!)
            if feasible12:
                # apply dynamic
                x_prime_results = self.optimise(template, gurobi_model, z)
                gurobi_model = grb.Model()
                gurobi_model.setParam('OutputFlag', output_flag)
                input2 = self.generate_input_region(gurobi_model, template, x_prime_results, self.env_input_size)
                x_second = self.apply_dynamic2(input2, gurobi_model, case=1, env_input_size=self.env_input_size)
                found_successor, x_second_results = self.h_repr_to_plot(gurobi_model, template, x_second)
                if found_successor:
                    post.append(tuple(x_second_results))
        # case 2 : ball going up and hit
        gurobi_model, z, input = standard_op()
        feasible21 = self.generate_guard(gurobi_model, z, case=2)
        if feasible21:
            Experiment.generate_region_constraints(gurobi_model, observable_template_action1, input, observable_result_action1, 2)
            gurobi_model.optimize()
            feasible22 = gurobi_model.status
            # feasible22 = self.generate_nn_guard(gurobi_model, input, nn, action_ego=1)  # check for action =1 over input (not z!)
            if feasible22:
                # apply dynamic
                x_prime_results = self.optimise(template, gurobi_model, z)
                gurobi_model = grb.Model()
                gurobi_model.setParam('OutputFlag', output_flag)
                input2 = self.generate_input_region(gurobi_model, template, x_prime_results, self.env_input_size)
                x_second = self.apply_dynamic2(input2, gurobi_model, case=2, env_input_size=self.env_input_size)
                found_successor, x_second_results = self.h_repr_to_plot(gurobi_model, template, x_second)
                if found_successor:
                    post.append(tuple(x_second_results))
        # case 1 alt : ball going down and NO hit
        gurobi_model, z, input = standard_op()
        feasible11_alt = self.generate_guard(gurobi_model, z, case=1)
        if feasible11_alt:
            Experiment.generate_region_constraints(gurobi_model, observable_template_action0, input, observable_result_action0, 2)
            gurobi_model.optimize()
            feasible12_alt = gurobi_model.status
            # feasible12_alt = self.generate_nn_guard(gurobi_model, input, nn, action_ego=0)  # check for action = 0 over input (not z!)
            if feasible12_alt:
                # apply dynamic
                x_prime_results = self.optimise(template, gurobi_model, z)
                gurobi_model = grb.Model()
                gurobi_model.setParam('OutputFlag', output_flag)
                input2 = self.generate_input_region(gurobi_model, template, x_prime_results, self.env_input_size)
                x_second = self.apply_dynamic2(input2, gurobi_model, case=3, env_input_size=self.env_input_size)
                found_successor, x_second_results = self.h_repr_to_plot(gurobi_model, template, x_second)

                if found_successor:
                    post.append(tuple(x_second_results))
        # case 2 alt : ball going up and NO hit
        gurobi_model, z, input = standard_op()
        feasible21_alt = self.generate_guard(gurobi_model, z, case=2)
        if feasible21_alt:
            Experiment.generate_region_constraints(gurobi_model, observable_template_action0, input, observable_result_action0, 2)
            gurobi_model.optimize()
            feasible22_alt = gurobi_model.status
            # feasible22_alt = self.generate_nn_guard(gurobi_model, input, nn, action_ego=0)  # check for action = 0 over input (not z!)
            if feasible22_alt:
                # apply dynamic
                x_prime_results = self.optimise(template, gurobi_model, z)
                gurobi_model = grb.Model()
                gurobi_model.setParam('OutputFlag', output_flag)
                input2 = self.generate_input_region(gurobi_model, template, x_prime_results, self.env_input_size)
                x_second = self.apply_dynamic2(input2, gurobi_model, case=3, env_input_size=self.env_input_size)
                found_successor, x_second_results = self.h_repr_to_plot(gurobi_model, template, x_second)
                if found_successor:
                    post.append(tuple(x_second_results))
        # case 3 : ball out of reach and not bounce
        gurobi_model, z, input = standard_op()
        feasible3 = self.generate_guard(gurobi_model, z, case=3)  # out of reach
        if feasible3:  # action is irrelevant in this case
            # apply dynamic
            x_prime_results = self.optimise(template, gurobi_model, z)
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            input2 = self.generate_input_region(gurobi_model, template, x_prime_results, self.env_input_size)
            x_second = self.apply_dynamic2(input2, gurobi_model, case=3, env_input_size=self.env_input_size)
            found_successor, x_second_results = self.h_repr_to_plot(gurobi_model, template, x_second)
            if found_successor:
                post.append(tuple(x_second_results))

        return post


if __name__ == '__main__':
    ray.init(log_to_driver=False)
    experiment = ORABouncingBallExperiment()
    experiment.run_experiment()
