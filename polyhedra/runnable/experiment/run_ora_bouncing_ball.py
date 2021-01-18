import gurobi as grb
import ray

from polyhedra.runnable.experiment.run_experiment_bouncing_ball import BouncingBallExperiment


class ORABouncingBallExperiment(BouncingBallExperiment):
    def __init__(self):
        super().__init__()
        self.post_fn_remote = self.post_milp

    @ray.remote
    def post_milp(self, x, nn, output_flag, t, template):
        post = []

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
            feasible12 = self.generate_nn_guard(gurobi_model, input, nn, action_ego=1)  # check for action =1 over input (not z!)
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
            feasible22 = self.generate_nn_guard(gurobi_model, input, nn, action_ego=1)  # check for action =1 over input (not z!)
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
            feasible12_alt = self.generate_nn_guard(gurobi_model, input, nn, action_ego=0)  # check for action = 0 over input (not z!)
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
            feasible22_alt = self.generate_nn_guard(gurobi_model, input, nn, action_ego=0)  # check for action = 0 over input (not z!)
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
    experiment = ORABouncingBallExperiment()
    experiment.run_experiment()
