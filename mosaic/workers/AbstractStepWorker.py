from typing import List, Tuple

import numpy as np
import ray
from mpmath import iv

from mosaic.utils import round_tuple
from prism.state_storage import get_storage, StateStorage
from symbolic.cartpole_abstract import CartPoleEnv_abstract


@ray.remote
class AbstractStepWorker:
    def __init__(self, t, rounding: int,env_init):
        self.env = env_init()
        self.t = t
        self.rounding = rounding

    def work(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]) -> Tuple[List[List[Tuple[Tuple[float, float]]]], List[List[Tuple[Tuple[float, float]]]]]:
        next_states_total: List[List[Tuple[Tuple[float, float]]]] = []
        terminal_states_total: List[List[Tuple[Tuple[float, float]]]] = []
        for interval in intervals:
            next_states: List[Tuple[Tuple[float, float]]] = []
            terminal_states: List[Tuple[Tuple[float, float]]] = []
            # parent_index = self.storage.get_inverse(interval[0])
            action = 1 if interval[1] else 0  # 1 if safe 0 if not
            next_state, done = step_state(interval[0], action, self.env, self.rounding)
            next_state_sticky, done_sticky = step_state(next_state, action, self.env, self.rounding)
            # successor_id, sticky_successor_id = self.storage.store_sticky_successors(next_state, next_state_sticky, parent_index)
            # self.storage.assign_t(successor_id, self.t + 1)
            # self.storage.assign_t(sticky_successor_id, self.t + 1)
            if done:
                terminal_states.append(next_state)
            else:
                next_states.append(next_state)

            if done_sticky:
                terminal_states.append(next_state_sticky)
            else:
                next_states.append(next_state_sticky)
            next_states_total.append(next_states)
            terminal_states_total.append(terminal_states)
        # self.storage.mark_as_fail(terminal_states_ids_total)  # mark the terminal states as failed
        return next_states_total, terminal_states_total


def step_state(state: Tuple[Tuple[float, float]], action, env, rounding: int) -> Tuple[Tuple[Tuple[float, float]], bool]:
    # given a state and an action, calculate next state
    env.reset()
    state = round_tuple(state, rounding)  # round the state
    env.set_state(state)
    # env.state = tuple([iv.mpf([x[0], x[1]]) for x in state])
    next_state, reward, done, _ = env.step(action)
    return round_tuple(next_state, rounding), done


# def interval_unwrap(state: np.ndarray, rounding: int) -> Tuple[Tuple[float, float]]:
#     """From array of intervals to tuple of floats"""
#     # unwrapped_state = tuple([(round(float(x.a), rounding), round(float(x.b), rounding)) for x in state])
#     unwrapped_state = tuple([(float(x.a), float(x.b)) for x in state])
#     return unwrapped_state
