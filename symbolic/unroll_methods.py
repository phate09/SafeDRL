"""
Collection of methods to use in unroll_abstract_env
"""
import jsonpickle
import numpy as np
import mpmath
import progressbar
from mpmath import iv
# from interval import interval,imath
import pandas as pd
from plnn.bab_explore import DomainExplorer
from plnn.verification_network import VerificationNetwork
from prism.state_storage import StateStorage
from symbolic.cartpole_abstract import CartPoleEnv_abstract
from verification_runs.cartpole_bab_load import generateCartpoleDomainExplorer
import os
from verification_runs.aggregate_abstract_domain import aggregate
import random
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import intervals as I
from typing import Tuple, List
import functools
import operator


def interval_unwrap(state: np.ndarray) -> Tuple[Tuple[float,float]]:
    """From array of intervals to tuple of floats"""
    unwrapped_state = tuple([(float(x.a), float(x.b)) for x in state])
    return unwrapped_state


def step_state(state: Tuple[Tuple], action, env) -> Tuple[Tuple]:
    # given a state and an action, calculate next state
    env.reset()
    env.state = tuple([iv.mpf([float(x[0]), float(x[1])]) for x in state])
    next_state, _, _, _ = env.step(action)
    return interval_unwrap(next_state)


def abstract_step(abstract_states: List[Tuple[Tuple]], action: int, env: CartPoleEnv_abstract) -> List[np.ndarray]:
    """
    Given some abstract states, compute the next abstract states taking the action passed as parameter
    :param env:
    :param abstract_states: the abstract states from which to start, list of tuples of intervals
    :param action: the action to take
    :return: the next abstract states after taking the action (array)
    """
    next_states = []
    bar = progressbar.ProgressBar(prefix="Performing abstract step...", max_value=len(abstract_states) + 1).start()
    for i, interval in enumerate(abstract_states):
        next_state = step_state(interval, action, env)
        # unwrapped_next_state = interval_unwrap(next_state)
        next_states.append(next_state)
        bar.update(i)
    # next_states_array = np.array(next_states, dtype=np.float32)  # turns the list in an array
    bar.finish()
    return next_states

def abstract_step_store(abstract_states: List[Tuple[Tuple]], action: int, env: CartPoleEnv_abstract, storage: StateStorage) -> List[np.ndarray]:
    """
    Given some abstract states, compute the next abstract states taking the action passed as parameter
    :param env:
    :param abstract_states: the abstract states from which to start, list of tuples of intervals
    :param action: the action to take
    :return: the next abstract states after taking the action (array)
    """
    next_states = []
    bar = progressbar.ProgressBar(prefix="Performing abstract step...", max_value=len(abstract_states) + 1).start()
    for i, interval in enumerate(abstract_states):
        parent_index = storage.dictionary.inverse[interval]
        next_state = step_state(interval, action, env)
        next_state = tuple(next_state)
        storage.store_successor(next_state,parent_index)
        # unwrapped_next_state = interval_unwrap(next_state)
        next_states.append(next_state)
        bar.update(i)
    # next_states_array = np.array(next_states, dtype=np.float32)  # turns the list in an array
    bar.finish()
    return next_states

def explore_step(states: List[Tuple[Tuple]], action: int, env: CartPoleEnv_abstract, explorer: DomainExplorer, verification_model: VerificationNetwork) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    next_states_array = abstract_step(states, action, env)
    explorer.reset()
    stats = explorer.explore(verification_model, next_states_array, min_area=1e-8, debug=True)
    print(f"#states: {stats['n_states']} [safe:{stats['safe_relative_percentage']:.3%}, unsafe:{stats['unsafe_relative_percentage']:.3%}, ignore:{stats['ignore_relative_percentage']:.3%}]")
    safe_next = [i.cpu().numpy() for i in explorer.safe_domains]
    unsafe_next = [i.cpu().numpy() for i in explorer.unsafe_domains]
    ignore_next = [i.cpu().numpy() for i in explorer.ignore_domains]
    return safe_next, unsafe_next, ignore_next


def iteration(t: int, t_states: List[List[List[Tuple[I.Interval]]]], env: CartPoleEnv_abstract, explorer: DomainExplorer, verification_model: VerificationNetwork):
    print(f"Iteration for time t={t}")
    safe_states, unsafe_states, ignore_states = t_states[t]
    safe_next_total = []
    unsafe_next_total = []
    ignore_next_total = []
    # safe states
    print(f"Safe states")
    safe_next, unsafe_next, ignore_next = explore_step(safe_states, 0, env, explorer, verification_model)  # takes 0 in safe states
    safe_next_total.extend(safe_next)
    unsafe_next_total.extend(unsafe_next)
    ignore_next_total.extend(ignore_next)
    # unsafe states
    print(f"Unsafe states")
    safe_next, unsafe_next, ignore_next = explore_step(unsafe_states, 1, env, explorer, verification_model)  # takes 1 in unsafe states
    safe_next_total.extend(safe_next)
    unsafe_next_total.extend(unsafe_next)
    ignore_next_total.extend(ignore_next)
    # aggregate together states
    safe_array = aggregate(np.stack(safe_next_total)) if len(safe_next_total) != 0 else []
    unsafe_array = aggregate(np.stack(unsafe_next_total)) if len(unsafe_next_total) != 0 else []
    t_states.append([safe_array, unsafe_array, []])  # np.stack(ignore_next + ignore_next2)
    print(f"Finished iteration t={t}, #safe states:{len(safe_next_total)}, #unsafe states:{len(unsafe_next_total)}, #ignored states:{len(ignore_next_total)}")
    return t_states


def generate_points_in_intervals(total_states: np.ndarray, n_points=100) -> np.ndarray:
    """
    :param total_states: 3 dimensional array (n,dimension,interval)
    :return:
    """
    generated_points = []
    for i in range(n_points):
        group = i % total_states.shape[0]
        f = lambda x: [random.uniform(v[0], v[1]) for v in x]
        random_point = f(total_states[group])
        generated_points.append(random_point)
    return np.stack(generated_points)


def assign_action(random_points: np.ndarray, t_states):
    """
    Given some points and a list of domains divided in 3 categories (safe,unsafe,ignore) check which action to pick.
    It also check for duplicates throwing an assertion error
    :param random_points: collection of random points
    :param t_states: collection of states at timestep t
    :return:
    """
    assigned_action = [""] * len(random_points)
    for i in range(len(random_points)):
        point = random_points[i]
        for r in range(3):  # safe,unsafe,ignore
            for g in range(len(t_states[r])):  # group
                matches = 0
                interval = t_states[r][g]
                # if interval[0][0] <= point[0] <= interval[0][1]:
                #     if interval[1][0] <= point[1] <= interval[1][1]:
                #         if interval[2][0] <= point[2] <= interval[2][1]:
                #             if interval[3][0] <= point[3] <= interval[3][1]:
                #                 matches=4
                for d in range(len(interval)):  # dimension
                    if interval[d][0] <= point[d] <= interval[d][1]:
                        matches += 1
                if matches == 4:
                    if r == 0:
                        if assigned_action[i] != "" and assigned_action[i] != "safe":
                            print("something's wrong")
                        assigned_action[i] = "safe"
                    elif r == 1:
                        if assigned_action[i] != "" and assigned_action[i] != "unsafe":
                            print("something's wrong 2")
                        assigned_action[i] = "unsafe"
                    elif r == 2:
                        assigned_action[i] = "ignore"
    return assigned_action


def generate_points(t_states, t):
    """
    Generate random points and determines the action of the agent given a collection of abstract states and a time step
    :param t_states: list of list of arrays
    :param t: timestep
    :return: random points and corresponding actions
    """
    # Generate random points and determines the action of the agent
    random_points = generate_points_in_intervals(t_states[t][0], 500) if len(t_states[t][0]) > 0 else np.array([]).reshape((0, 4))  # safe
    random_points2 = generate_points_in_intervals(t_states[t][1], 500) if len(t_states[t][1]) > 0 else np.array([]).reshape((0, 4))  # unsafe
    random_points3 = generate_points_in_intervals(t_states[t][2], 500) if len(t_states[t][2]) > 0 else np.array([]).reshape((0, 4))  # unsafe
    random_points = np.concatenate((random_points, random_points2, random_points3))
    assigned_actions = assign_action(random_points, t_states[t])
    return random_points, assigned_actions


def calculate_area(state: np.ndarray):
    """Given an interval state calculate the area of the interval"""
    dom_sides: np.ndarray = np.abs(state[:, 0] - state[:, 1])
    dom_area = dom_sides.prod()
    return dom_area


def generate_middle_points(t_states, t):
    """
    Generate the middle points of each inteval and determines the action of the agent given a collection of abstract states and a time step
    :param t_states: list of list of arrays
    :param t: timestep
    :return: random points and corresponding actions
    """
    # Generate random points and determines the action of the agent
    random_points = []
    areas = []
    for s in t_states[t][0]:  # safe
        random_points.append(np.average(s, axis=1))
        areas.append(calculate_area(s))
    for s in t_states[t][1]:  # unsafe
        random_points.append(np.average(s, axis=1))
        areas.append(calculate_area(s))
    for s in t_states[t][2]:  # ignore
        random_points.append(np.average(s, axis=1))
        areas.append(calculate_area(s))
    random_points = np.stack(random_points)
    assigned_actions = assign_action(random_points, t_states[t])
    return random_points, areas, assigned_actions


def assign_action_to_blank_intervals(s_array: List[Tuple[I.Interval]], precision=1e-6) -> Tuple[List[Tuple]]:
    # convert list of tuples to list of arrays
    s_array = [np.array(x) for x in s_array]
    explorer, verification_model = generateCartpoleDomainExplorer()
    # given the initial states calculate which intervals go left or right
    stats = explorer.explore(verification_model, s_array, debug=True)
    print(f"#states: {stats['n_states']} [safe:{stats['safe_relative_percentage']:.3%}, unsafe:{stats['unsafe_relative_percentage']:.3%}, ignore:{stats['ignore_relative_percentage']:.3%}]")
    safe_next = [i.cpu().numpy() for i in explorer.safe_domains]
    unsafe_next = [i.cpu().numpy() for i in explorer.unsafe_domains]
    ignore_next = [i.cpu().numpy() for i in explorer.ignore_domains]
    safe_next = np.stack(safe_next) if len(safe_next) != 0 else []
    unsafe_next = np.stack(unsafe_next) if len(unsafe_next) != 0 else []
    ignore_next = np.stack(ignore_next) if len(ignore_next) != 0 else []
    t_states = [tuple([(x.item(0), x.item(1)) for x in k]) for k in safe_next], [tuple([(x.item(0), x.item(1)) for x in k]) for k in unsafe_next], [tuple([(x.item(0), x.item(1)) for x in k]) for k in
                                                                                                                                                    ignore_next]
    return t_states


def discard_negligibles(intervals: List[Tuple[float]]) -> List[Tuple[float]]:
    """discards the intervals with area 0"""
    result = []
    for interval in intervals:
        sizes = [abs(interval[dimension][1] - interval[dimension][0]) for dimension in range(len(interval))]
        if all([x > 1e-6 for x in sizes]):
            # area = functools.reduce(operator.mul, sizes)  # multiplication of each side of the rectangle
            # if area > 1e-10:
            result.append(interval)
    return result
