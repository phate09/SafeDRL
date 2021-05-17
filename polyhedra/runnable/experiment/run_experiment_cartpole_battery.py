from typing import List, Tuple

from pyomo.opt import TerminationCondition, SolverStatus
from ray.rllib.agents.ppo import ppo

from agents.ppo.train_PPO_cartpole import get_PPO_trainer
from agents.ppo.tune.tune_train_PPO_cartpole_battery import get_PPO_config
from agents.ray_utils import convert_ray_policy_to_sequential
from polyhedra.experiments_nn_analysis import Experiment
import ray
import gurobi as grb
import math
import numpy as np
import torch
from interval import interval, imath
from environment.cartpole_battery import CartPoleBatteryEnv
import pyomo.environ as pyo

USE_GUROBI = False


class CartpoleBatteryExperiment(Experiment):
    def __init__(self):
        env_input_size: int = 5
        super().__init__(env_input_size)
        self.post_fn_remote = self.post_milp
        self.get_nn_fn = self.get_nn
        self.plot_fn = self.plot
        self.assign_lbl_fn = self.assign_label
        self.additional_seen_fn = self.additional_seen
        self.template_2d: np.ndarray = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
        self.rounding_value = 2 ** 20  # 1024
        input_boundaries, input_template = self.get_template(0)
        self.input_boundaries: List = input_boundaries
        self.input_template: np.ndarray = input_template
        _, template = self.get_template(1)
        self.analysis_template: np.ndarray = template
        safe_angle = 30 * 2 * math.pi / 360
        theta = [self.e(env_input_size, 2)]
        neg_theta = [-self.e(env_input_size, 2)]
        battery = [self.e(env_input_size, 4)]
        self.unsafe_zone: List[Tuple] = [(theta, np.array([-safe_angle])), (neg_theta, np.array([-safe_angle]))]
        self.battery_split: List[Tuple] = [(battery, np.array([0]))]
        # self.use_rounding = False

        self.time_horizon = 3000
        # self.nn_path = "/home/edoardo/ray_results/tune_PPO_cartpole_battery/PPO_CartPoleBatteryEnv_e4e96_00000_0_2021-05-06_10-17-35/checkpoint_1640/checkpoint-1640"
        self.nn_path = "/home/edoardo/ray_results/tune_PPO_cartpole_battery/PPO_CartPoleBatteryEnv_54127_00000_0_2021-05-08_16-37-42/checkpoint_1230/checkpoint-1230"
        self.tau = 0.02
        self.n_actions = 3
        self.show_progress_plot = False
        # self.use_bfs = False
        # self.tau = 0.02

    @ray.remote
    def post_milp(self, x, x_label, nn, output_flag, t, template):
        """milp method"""
        post = []
        for split_battery in [True, False]:  # split successor if battery if >0 or <0
            for chosen_action in range(self.n_actions):  # [2]:
                if (chosen_action == 2 or chosen_action == 1) and x_label == 1:  # skip actions when battery is dead
                    continue
                if USE_GUROBI:
                    gurobi_model = grb.Model()
                    gurobi_model.setParam('OutputFlag', output_flag)
                    gurobi_model.setParam('Threads', 2)
                    # gurobi_model.setParam('DualReductions', 1)
                    input = Experiment.generate_input_region(gurobi_model, template, x, self.env_input_size)  # todo check battery >0
                    # gurobi_model.addConstr(input[4] <= 30.0)  # max battery cap at 100%

                    feasible_action = CartpoleBatteryExperiment.generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action, M=1e04)
                    if feasible_action or x_label == 1:  # performs action 2 automatically when battery is dead
                        max_theta, min_theta, max_theta_dot, min_theta_dot = self.get_theta_bounds(gurobi_model, input)
                        sin_cos_table = self.get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action=chosen_action)
                        thetaacc, xacc = CartpoleBatteryExperiment.generate_angle_milp(gurobi_model, input, sin_cos_table)
                        if gurobi_model.status != 2:
                            assert gurobi_model.status == 2, f"LP wasn't optimally solved {x}"
                        # apply dynamic
                        x_prime = self.apply_dynamic(input, gurobi_model, thetaacc=thetaacc, xacc=xacc, env_input_size=self.env_input_size, action=chosen_action)
                        for A, b in self.battery_split:
                            Experiment.generate_region_constraints(gurobi_model, A, x_prime, b, self.env_input_size, invert=not split_battery)
                        gurobi_model.update()
                        gurobi_model.optimize()
                        if gurobi_model.status != 2:
                            continue
                        found_successor, x_prime_results = self.h_repr_to_plot(gurobi_model, template, x_prime)
                        if found_successor:
                            post.append((tuple(x_prime_results), (x, x_label)))
                else:
                    model = pyo.ConcreteModel()
                    input = Experiment.generate_input_region_pyo(model, template, x, self.env_input_size)
                    feasible_action = CartpoleBatteryExperiment.generate_nn_guard_pyo(model, input, nn, action_ego=chosen_action, M=1e04)
                    if feasible_action or x_label == 1:  # performs action 2 automatically when battery is dead
                        max_theta, min_theta, max_theta_dot, min_theta_dot = self.get_theta_bounds_pyo(model, input)
                        sin_cos_table = self.get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action=chosen_action)
                        thetaacc, xacc = CartpoleBatteryExperiment.generate_angle_milp_pyo(model, input, sin_cos_table)

                        model.del_component(model.obj)
                        model.obj = pyo.Objective(expr=thetaacc, sense=pyo.maximize)
                        result = pyo.SolverFactory('glpk').solve(model)
                        assert (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal), f"LP wasn't optimally solved {x}"
                        # apply dynamic
                        x_prime = self.apply_dynamic_pyo(input, model, thetaacc=thetaacc, xacc=xacc, env_input_size=self.env_input_size, action=chosen_action)
                        for A, b in self.battery_split:
                            Experiment.generate_region_constraints_pyo(model, A, x_prime, b, self.env_input_size, invert=not split_battery)

                        x_prime_results = self.optimise_pyo(template, model, x_prime)
                        found_successor = x_prime_results is not None
                        if found_successor:
                            post.append((tuple(x_prime_results), (x, x_label)))

        return post

    def check_unsafe(self, template, bnds, x_label):
        if x_label >= 2:
            return True
        else:
            return False

    def additional_seen(self):
        # adds an element that captures all the states where battery is <=0
        return [((float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), 0, float("inf")), 1)]

    def assign_label(self, x_prime, parent, parent_lbl) -> int:
        if parent_lbl == 2:
            return 2  # once unsafe always unsafe
        if parent_lbl == 1:
            return 1
        if x_prime[9] > 0 or x_prime[8] <= 0:  # battery runs out
            return 1
        assert x_prime[8] > 0 and x_prime[9] <= 0
        for A, b in self.unsafe_zone:
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', False)
            input = gurobi_model.addMVar(shape=(self.env_input_size,), lb=float("-inf"), name="input")
            Experiment.generate_region_constraints(gurobi_model, self.analysis_template, input, x_prime, self.env_input_size)
            Experiment.generate_region_constraints(gurobi_model, A, input, b, self.env_input_size)
            gurobi_model.update()
            gurobi_model.optimize()
            if gurobi_model.status != 2:
                return 0  # still balancing

        return 2  # unsafe

    def apply_dynamic(self, input, gurobi_model: grb.Model, thetaacc, xacc, env_input_size, action):
        '''

        :param costheta: gurobi variable containing the range of costheta values
        :param sintheta: gurobi variable containin the range of sintheta values
        :param input:
        :param gurobi_model:
        :param t:
        :return:
        '''

        tau = self.tau  # 0.001  # seconds between state updates
        x = input[0]
        x_dot = input[1]
        theta = input[2]
        theta_dot = input[3]
        battery = input[4]
        z = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name=f"x_prime")
        x_prime = x + tau * x_dot
        x_dot_prime = x_dot + tau * xacc
        theta_prime = theta + tau * theta_dot
        theta_dot_prime = theta_dot + tau * thetaacc
        action_cost = 0.1
        if action != 0:
            action_cost = 0.5
        gurobi_model.addConstr(z[0] == x_prime, name=f"dyna_constr_1")
        gurobi_model.addConstr(z[1] == x_dot_prime, name=f"dyna_constr_2")
        gurobi_model.addConstr(z[2] == theta_prime, name=f"dyna_constr_3")
        gurobi_model.addConstr(z[3] == theta_dot_prime, name=f"dyna_constr_4")
        gurobi_model.addConstr(z[4] == battery - action_cost, name=f"dyna_constr_5")
        return z

    def apply_dynamic_pyo(self, input, model: pyo.ConcreteModel, thetaacc, xacc, env_input_size, action):
        '''

        :param costheta: gurobi variable containing the range of costheta values
        :param sintheta: gurobi variable containin the range of sintheta values
        :param input:
        :param gurobi_model:
        :param t:
        :return:
        '''

        tau = self.tau  # 0.001  # seconds between state updates
        x = input[0]
        x_dot = input[1]
        theta = input[2]
        theta_dot = input[3]
        battery = input[4]
        z = pyo.Var(range(env_input_size), name=f"x_prime", within=pyo.Reals)
        model.add_component("x_prime", z)
        x_prime = x + tau * x_dot
        x_dot_prime = x_dot + tau * xacc
        theta_prime = theta + tau * theta_dot
        theta_dot_prime = theta_dot + tau * thetaacc
        action_cost = 0.1
        if action != 0:
            action_cost = 0.5
        model.dynamic_constraints = pyo.ConstraintList()
        model.dynamic_constraints.add(expr=z[0] == x_prime)
        model.dynamic_constraints.add(expr=z[1] == x_dot_prime)
        model.dynamic_constraints.add(expr=z[2] == theta_prime)
        model.dynamic_constraints.add(expr=z[3] == theta_dot_prime)
        model.dynamic_constraints.add(expr=z[4] == battery - action_cost)
        return z

    @staticmethod
    def get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action):
        assert min_theta <= max_theta, f"min_theta = {min_theta},max_theta={max_theta}"
        assert min_theta_dot <= max_theta_dot, f"min_theta_dot = {min_theta_dot},max_theta_dot={max_theta_dot}"
        step_theta = 0.1  # 0.1
        step_theta_dot = 0.1  # 0.1
        step_thetaacc = 100  # 0.3
        min_theta = max(min_theta, -math.pi / 2)
        max_theta = min(max_theta, math.pi / 2)
        split_theta1 = np.arange(min(min_theta, 0), min(max_theta, 0), step_theta)
        split_theta2 = np.arange(max(min_theta, 0), max(max_theta, 0), step_theta)
        split_theta = np.concatenate([split_theta1, split_theta2])
        split_theta_dot1 = np.arange(min(min_theta_dot, 0), min(max_theta_dot, 0), step_theta)
        split_theta_dot2 = np.arange(max(min_theta_dot, 0), max(max_theta_dot, 0), step_theta)
        split_theta_dot = np.concatenate([split_theta_dot1, split_theta_dot2])
        env = CartPoleBatteryEnv(None)
        force = env.force_mag if action == 1 else -env.force_mag if action == 2 else 0

        split = []
        for t_dot in split_theta_dot:
            for theta in split_theta:
                lb_theta_dot = t_dot
                ub_theta_dot = min(t_dot + step_theta_dot, max_theta_dot)
                lb = theta
                ub = min(theta + step_theta, max_theta) if theta >= 0 else min(min(theta + step_theta, max_theta), 0)  # split at 0
                split.append((interval([lb_theta_dot, ub_theta_dot]), interval([lb, ub])))
        sin_cos_table = []
        while (len(split)):
            theta_dot, theta = split.pop()
            sintheta = imath.sin(theta)
            costheta = imath.cos(theta)
            temp = (force + env.polemass_length * theta_dot ** 2 * sintheta) / env.total_mass
            thetaacc: interval = (env.gravity * sintheta - costheta * temp) / (env.length * (4.0 / 3.0 - env.masspole * costheta ** 2 / env.total_mass))
            xacc = temp - env.polemass_length * thetaacc * costheta / env.total_mass
            if abs(thetaacc[0].sup - thetaacc[0].inf) > step_thetaacc:
                # split theta theta_dot
                mid_theta = (theta[0].sup + theta[0].inf) / 2
                mid_theta_dot = (theta_dot[0].sup + theta_dot[0].inf) / 2
                theta_1 = interval([theta[0].inf, mid_theta])
                theta_2 = interval([mid_theta, theta[0].sup])
                theta_dot_1 = interval([theta_dot[0].inf, mid_theta_dot])
                theta_dot_2 = interval([mid_theta_dot, theta_dot[0].sup])
                split.append((theta_1, theta_dot_1))
                split.append((theta_1, theta_dot_2))
                split.append((theta_2, theta_dot_1))
                split.append((theta_2, theta_dot_2))
            else:
                sin_cos_table.append((theta, theta_dot, thetaacc, xacc))
        return sin_cos_table

    @staticmethod
    def get_theta_bounds(gurobi_model, input):
        gurobi_model.setObjective(input[2].sum(), grb.GRB.MAXIMIZE)
        gurobi_model.optimize()
        assert gurobi_model.status == 2
        max_theta = gurobi_model.getVars()[2].X

        gurobi_model.setObjective(input[2].sum(), grb.GRB.MINIMIZE)
        gurobi_model.optimize()
        assert gurobi_model.status == 2
        min_theta = gurobi_model.getVars()[2].X

        gurobi_model.setObjective(input[3].sum(), grb.GRB.MAXIMIZE)
        gurobi_model.optimize()
        assert gurobi_model.status == 2
        max_theta_dot = gurobi_model.getVars()[3].X

        gurobi_model.setObjective(input[3].sum(), grb.GRB.MINIMIZE)
        gurobi_model.optimize()
        assert gurobi_model.status == 2
        min_theta_dot = gurobi_model.getVars()[3].X
        return max_theta, min_theta, max_theta_dot, min_theta_dot

    @staticmethod
    def get_theta_bounds_pyo(model: pyo.ConcreteModel, input):
        # TransformationFactory('gdp.bigm').apply_to(model, bigM=M)
        model.del_component(model.obj)
        model.obj = pyo.Objective(expr=input[2], sense=pyo.maximize)
        result = pyo.SolverFactory('glpk').solve(model)
        assert (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal)
        max_theta = pyo.value(model.obj)

        model.del_component(model.obj)
        model.obj = pyo.Objective(expr=input[2], sense=pyo.minimize)
        result = pyo.SolverFactory('glpk').solve(model)
        assert (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal)
        min_theta = pyo.value(model.obj)

        model.del_component(model.obj)
        model.obj = pyo.Objective(expr=input[3], sense=pyo.maximize)
        result = pyo.SolverFactory('glpk').solve(model)
        assert (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal)
        max_theta_dot = pyo.value(model.obj)

        model.del_component(model.obj)
        model.obj = pyo.Objective(expr=input[3], sense=pyo.minimize)
        result = pyo.SolverFactory('glpk').solve(model)
        assert (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal)
        min_theta_dot = pyo.value(model.obj)
        return max_theta, min_theta, max_theta_dot, min_theta_dot

    @staticmethod
    def generate_angle_milp(gurobi_model: grb.Model, input, sin_cos_table: List[Tuple]):
        """MILP method
        input: theta, thetadot
        output: thetadotdot, xdotdot (edited)
        l_{theta, i}, l_{thatdot,i}, l_{thetadotdot, i}, l_{xdotdot, i}, u_....
        sum_{i=1}^k l_{x,i} - l_{x,i}*z_i <= x <= sum_{i=1}^k u_{x,i} - u_{x,i}*z_i, for each variable x
        sum_{i=1}^k l_{theta,i} - l_{theta,i}*z_i <= theta <= sum_{i=1}^k u_{theta,i} - u_{theta,i}*z_i
        """
        # gurobi_model.setParam('OptimalityTol', 1e-6)
        # gurobi_model.setParam('FeasibilityTol', 1e-6)
        # gurobi_model.setParam('IntFeasTol', 1e-9)
        theta = input[2]
        theta_dot = input[3]
        k = len(sin_cos_table)
        zs = []
        thetaacc = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name="thetaacc")
        xacc = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name="xacc")
        for i in range(k):
            z = gurobi_model.addMVar(lb=0, ub=1, shape=(1,), vtype=grb.GRB.BINARY, name=f"part_{i}")
            zs.append(z)
        gurobi_model.addConstr(k - 1 == sum(zs), name=f"const_milp1")
        theta_lb = 0
        theta_ub = 0
        theta_dot_lb = 0
        theta_dot_ub = 0
        thetaacc_lb = 0
        thetaacc_ub = 0
        xacc_lb = 0
        xacc_ub = 0
        round_value = 2 ** 20
        for i in range(k):
            theta_interval, theta_dot_interval, theta_acc_interval, xacc_interval = sin_cos_table[i]
            theta_inf = Experiment.round_single(theta_interval[0].inf, round_value)
            theta_sup = Experiment.round_single(theta_interval[0].sup, round_value)
            theta_dot_inf = Experiment.round_single(theta_dot_interval[0].inf, round_value)
            theta_dot_sup = Experiment.round_single(theta_dot_interval[0].sup, round_value)
            theta_lb += theta_inf - theta_inf * zs[i]
            theta_ub += theta_sup - theta_sup * zs[i]
            theta_dot_lb += theta_dot_inf - theta_dot_inf * zs[i]
            theta_dot_ub += theta_dot_sup - theta_dot_sup * zs[i]

            theta_acc_interval__inf = Experiment.round_single(theta_acc_interval[0].inf, round_value)
            theta_acc_interval__sup = Experiment.round_single(theta_acc_interval[0].sup, round_value)
            xacc_interval__inf = Experiment.round_single(xacc_interval[0].inf, round_value)
            xacc_interval__sup = Experiment.round_single(xacc_interval[0].sup, round_value)

            thetaacc_lb += theta_acc_interval__inf - theta_acc_interval__inf * zs[i]
            thetaacc_ub += theta_acc_interval__sup - theta_acc_interval__sup * zs[i]
            xacc_lb += xacc_interval__inf - xacc_interval__inf * zs[i]
            xacc_ub += xacc_interval__sup - xacc_interval__sup * zs[i]

        gurobi_model.addConstr(theta >= theta_lb, name=f"theta_guard1")
        gurobi_model.addConstr(theta <= theta_ub, name=f"theta_guard2")
        gurobi_model.addConstr(theta_dot >= theta_dot_lb, name=f"theta_dot_guard1")
        gurobi_model.addConstr(theta_dot <= theta_dot_ub, name=f"theta_dot_guard2")

        gurobi_model.addConstr(thetaacc >= thetaacc_lb, name=f"thetaacc_guard1")
        gurobi_model.addConstr(thetaacc <= thetaacc_ub, name=f"thetaacc_guard2")
        gurobi_model.addConstr(xacc >= xacc_lb, name=f"xacc_guard1")
        gurobi_model.addConstr(xacc <= xacc_ub, name=f"xacc_guard2")

        gurobi_model.update()
        gurobi_model.optimize()

        return thetaacc, xacc

    @staticmethod
    def generate_angle_milp_pyo(model: pyo.ConcreteModel, input, sin_cos_table: List[Tuple]):
        """MILP method
        input: theta, thetadot
        output: thetadotdot, xdotdot (edited)
        l_{theta, i}, l_{thatdot,i}, l_{thetadotdot, i}, l_{xdotdot, i}, u_....
        sum_{i=1}^k l_{x,i} - l_{x,i}*z_i <= x <= sum_{i=1}^k u_{x,i} - u_{x,i}*z_i, for each variable x
        sum_{i=1}^k l_{theta,i} - l_{theta,i}*z_i <= theta <= sum_{i=1}^k u_{theta,i} - u_{theta,i}*z_i
        """
        # gurobi_model.setParam('OptimalityTol', 1e-6)
        # gurobi_model.setParam('FeasibilityTol', 1e-6)
        # gurobi_model.setParam('IntFeasTol', 1e-9)
        theta = input[2]
        theta_dot = input[3]
        k = len(sin_cos_table)
        pyo.Var()
        thetaacc = pyo.Var(name=f"thetaacc", within=pyo.Reals)
        model.add_component("thetaacc", thetaacc)
        xacc = pyo.Var(name=f"xacc", within=pyo.Reals)
        model.add_component("xacc", xacc)
        zs = pyo.Var(range(k), name=f"z", within=pyo.Integers, bounds=(0, 1))
        model.add_component("angle_section", zs)
        model.angle_constraints = pyo.ConstraintList()
        model.angle_constraints.add(expr=k - 1 == sum([zs[i] for i in range(k)]))
        theta_lb = 0
        theta_ub = 0
        theta_dot_lb = 0
        theta_dot_ub = 0
        thetaacc_lb = 0
        thetaacc_ub = 0
        xacc_lb = 0
        xacc_ub = 0
        round_value = 2 ** 20
        for i in range(k):
            theta_interval, theta_dot_interval, theta_acc_interval, xacc_interval = sin_cos_table[i]
            theta_inf = theta_interval[0].inf
            theta_sup = theta_interval[0].sup
            theta_dot_inf = theta_dot_interval[0].inf
            theta_dot_sup = theta_dot_interval[0].sup
            theta_lb += theta_inf - theta_inf * zs[i]
            theta_ub += theta_sup - theta_sup * zs[i]
            theta_dot_lb += theta_dot_inf - theta_dot_inf * zs[i]
            theta_dot_ub += theta_dot_sup - theta_dot_sup * zs[i]

            theta_acc_interval__inf = theta_acc_interval[0].inf
            theta_acc_interval__sup = theta_acc_interval[0].sup
            xacc_interval__inf = xacc_interval[0].inf
            xacc_interval__sup = xacc_interval[0].sup

            thetaacc_lb += theta_acc_interval__inf - theta_acc_interval__inf * zs[i]
            thetaacc_ub += theta_acc_interval__sup - theta_acc_interval__sup * zs[i]
            xacc_lb += xacc_interval__inf - xacc_interval__inf * zs[i]
            xacc_ub += xacc_interval__sup - xacc_interval__sup * zs[i]

        model.angle_constraints.add(expr=theta >= theta_lb)
        model.angle_constraints.add(expr=theta <= theta_ub)
        model.angle_constraints.add(expr=theta_dot >= theta_dot_lb)
        model.angle_constraints.add(expr=theta_dot <= theta_dot_ub)

        model.angle_constraints.add(expr=thetaacc >= thetaacc_lb)
        model.angle_constraints.add(expr=thetaacc <= thetaacc_ub)
        model.angle_constraints.add(expr=xacc >= xacc_lb)
        model.angle_constraints.add(expr=xacc <= xacc_ub)

        return thetaacc, xacc

    @staticmethod
    def test_objective(model: pyo.ConcreteModel, variable, maximise=True):
        model.del_component(model.obj)
        model.obj = pyo.Objective(expr=variable, sense=pyo.maximize if maximise else pyo.minimize)
        result = pyo.SolverFactory('glpk').solve(model)
        result.write()

    def plot(self, vertices_list, template, template_2d):
        self.generic_plot("theta", "theta_dot", vertices_list, template, template_2d)

    def get_template(self, mode=0):
        x = Experiment.e(self.env_input_size, 0)
        x_dot = Experiment.e(self.env_input_size, 1)
        theta = Experiment.e(self.env_input_size, 2)
        theta_dot = Experiment.e(self.env_input_size, 3)
        battery = Experiment.e(self.env_input_size, 4)
        if mode == 0:  # box directions with intervals
            input_boundaries = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 5, -1]
            # input_boundaries = (0.0986328125, 0.0908203125, 0.12890625, 0.1181640625, 0.2275390625, 0.2080078125, 0.08984375, 0.099609375, 27.908203125, 0.0)
            # input_boundaries = (-0.13397, 0.134071, 0.119921, -0.115271, -0.014145, 0.018799, -0.249242, 0.253988, 26.008204, 0.0)
            # input_boundaries = (-0.124331, 0.176405, 0.039151, 0.071584, -0.101738, 0.235551, -0.081928, 0.189861, 3.608215, 0.0)
            # input_boundaries = (-0.10726, 0.193659, 0.127264, -0.037434, -0.010739, 0.156224, -0.171754, 0.288924, 3.108215, 0.0)
            self.round_tuple(input_boundaries, self.rounding_value)
            # input_boundaries = [0.04373426, -0.04373426, -0.04980056, 0.04980056, 0.045, -0.045, -0.51, 0.51]
            # optimise in a direction
            template = []
            for dimension in range(self.env_input_size):
                template.append(Experiment.e(self.env_input_size, dimension))
                template.append(-Experiment.e(self.env_input_size, dimension))
            template = np.array(template)  # the 6 dimensions in 2 variables
            return self.round_tuple(input_boundaries, self.rounding_value), template
        if mode == 1:  # directions to easily find fixed point
            input_boundaries = None
            template = np.array(
                [theta, -theta, theta_dot, -theta_dot, theta + theta_dot, -(theta + theta_dot), (theta - theta_dot), -(theta - theta_dot), battery, -battery])  # x_dot, -x_dot,theta_dot - theta
            return input_boundaries, template
        if mode == 2:
            input_boundaries = None
            template = np.array([theta, -theta, theta_dot, -theta_dot])
            return input_boundaries, template
        if mode == 3:
            input_boundaries = None
            template = np.array([theta, theta_dot, -theta_dot])
            return input_boundaries, template
        if mode == 4:
            input_boundaries = [0.09375, 0.625, 0.625, 0.0625, 0.1875]
            # input_boundaries = [0.09375, 0.5, 0.5, 0.0625, 0.09375]
            template = np.array([theta, theta_dot, -theta_dot, theta + theta_dot, (theta - theta_dot)])
            return input_boundaries, template
        if mode == 5:
            input_boundaries = [0.125, 0.0625, 0.1875]
            template = np.array([theta, theta + theta_dot, (theta - theta_dot)])
            return input_boundaries, template

    def get_nn(self):
        config = get_PPO_config(1234)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(self.nn_path)

        policy = trainer.get_policy()
        # sequential_nn = convert_ray_simple_policy_to_sequential(policy).cpu()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        l0 = torch.nn.Linear(5, 2, bias=False)
        # l0 = torch.nn.Linear(5, 3, bias=False)
        l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]], dtype=torch.float32))
        # l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=torch.float32))
        layers = [l0]
        for l in sequential_nn:
            layers.append(l)
        nn = torch.nn.Sequential(*layers)
        # ray.shutdown()
        return nn


if __name__ == '__main__':
    ray.init(local_mode=False, log_to_driver=False)
    experiment = CartpoleBatteryExperiment()
    # experiment.n_workers = 1
    experiment.run_experiment()
