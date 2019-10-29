"""
Collection of methods to use in unroll_abstract_env
"""
import jsonpickle
import numpy as np
import mpmath
from mpmath import iv

from plnn.bab_explore import DomainExplorer
from plnn.verification_network import VerificationNetwork
from symbolic.cartpole_abstract import CartPoleEnv_abstract
from verification_runs.cartpole_bab_load import generateCartpoleDomainExplorer
import os
from verification_runs.aggregate_abstract_domain import aggregate
import random
import plotly.graph_objs as go


def interval_unwrap(state):
    unwrapped_state = tuple([[float(x.a), float(x.b)] for x in state])
    return unwrapped_state


def step_state(state, action, env):
    # given a state and an action, calculate next state
    env.reset()
    env.state = state
    next_state, _, _, _ = env.step(action)
    return tuple(next_state)


def abstract_step(abstract_states: np.ndarray, action, env: CartPoleEnv_abstract):
    """
    Given some abstract states, compute the next abstract states taking the action passed as parameter
    :param abstract_states: the abstract states from which to start
    :param action: the action to take
    :return: the next abstract states after taking the action (array)
    """
    next_states = []
    for interval in abstract_states:
        state = tuple([iv.mpf([x.item(0), x.item(1)]) for x in interval])
        next_state = step_state(state, action, env)
        unwrapped_next_state = interval_unwrap(next_state)
        next_states.append(unwrapped_next_state)
    next_states_array = np.array(next_states, dtype=np.float32)  # turns the list in an array
    return next_states_array


def explore_step(states: np.ndarray, action, env: CartPoleEnv_abstract, explorer: DomainExplorer, verification_model: VerificationNetwork):
    next_states_array = abstract_step(states, action, env)
    explorer.reset()
    stats = explorer.explore(verification_model, next_states_array, min_area=1e-8, debug=True)
    print(f"#states: {stats['n_states']} [safe:{stats['safe_relative_percentage']:.3%}, unsafe:{stats['unsafe_relative_percentage']:.3%}, ignore:{stats['ignore_relative_percentage']:.3%}]")
    safe_next = [i.cpu().numpy() for i in explorer.safe_domains]
    unsafe_next = [i.cpu().numpy() for i in explorer.unsafe_domains]
    ignore_next = [i.cpu().numpy() for i in explorer.ignore_domains]
    return safe_next, unsafe_next, ignore_next


def iteration(t: int, t_states, env: CartPoleEnv_abstract, explorer: DomainExplorer, verification_model: VerificationNetwork):
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


def generate_points_in_intervals(total_states: np.ndarray, n_points=100):
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
    :return:
    """
    assigned_action = [-1] * len(random_points)
    for i in range(len(random_points)):
        # for t in range(len(t_states)):
        t = len(t_states) - 1
        for r in range(3):  # safe,unsafe,ignore
            matches = 0
            for g in range(len(t_states[t][r])):  # group
                for d in range(len(t_states[t][r][g])):  # dimension
                    if t_states[t][r][g][d][0] <= random_points[i][d] <= t_states[t][r][g][d][1]:
                        matches += 1
                if matches == 4:
                    if r == 0:
                        assert assigned_action[i] == -1 or assigned_action[i] == 1
                        assigned_action[i] = 1
                    elif r == 1:
                        assert assigned_action[i] == -1 or assigned_action[i] == 0
                        assigned_action[i] = 0
                    elif r == 2:
                        assigned_action[i] = -1
    return assigned_action
