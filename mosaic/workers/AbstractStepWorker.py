from typing import List, Tuple

import numpy as np
import ray
from mpmath import iv

from prism.state_storage import get_storage
from symbolic.cartpole_abstract import CartPoleEnv_abstract


@ray.remote
class AbstractStepWorker:
    def __init__(self, explorer, t):
        self.env = CartPoleEnv_abstract()  # todo find a way to define the initialiser for the environment
        self.explorer = explorer
        self.t = t
        self.storage = get_storage()

    def work(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]):
        next_states_total = []
        terminal_states_total = []
        for interval in intervals:
            next_states = []
            terminal_states = []
            parent_index = self.storage.get_inverse(interval[0])
            # denormalised_interval = self.explorer.denormalise(interval[0])
            action = 1 if interval[1] else 0  # 1 if safe 0 if not
            next_state, done = step_state(interval[0], action, self.env)
            next_state_sticky, done_sticky = step_state(next_state, action, self.env)
            # normalised_next_state = self.explorer.normalise(next_state)
            # normalised_next_state_sticky = self.explorer.normalise(next_state_sticky)
            successor_id, sticky_successor_id = self.storage.store_sticky_successors(next_state, next_state_sticky, self.t, parent_index)
            # unwrapped_next_state = interval_unwrap(next_state)
            if done:
                terminal_states.append(successor_id)
            if done_sticky:
                terminal_states.append(sticky_successor_id)
            next_states.append(next_state)
            next_states.append(next_state_sticky)
            next_states_total.extend(next_states)
            terminal_states_total.extend(terminal_states)
        return next_states_total, terminal_states_total


def step_state(state: Tuple[Tuple], action, env) -> Tuple[Tuple[Tuple], bool]:
    # given a state and an action, calculate next state
    env.reset()
    env.state = tuple([iv.mpf([float(x[0]), float(x[1])]) for x in state])
    next_state, reward, done, _ = env.step(action)
    return interval_unwrap(next_state), done


def interval_unwrap(state: np.ndarray) -> Tuple[Tuple[float, float]]:
    """From array of intervals to tuple of floats"""
    unwrapped_state = tuple([(float(x.a), float(x.b)) for x in state])
    return unwrapped_state