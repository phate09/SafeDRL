from collections import defaultdict
from typing import List, Tuple

import numpy as np
import ray
from mpmath import iv

from mosaic.utils import round_tuple
from prism.state_storage import StateStorage
from symbolic.cartpole_abstract import CartPoleEnv_abstract


@ray.remote
class AbstractStepWorker:
    def __init__(self, rounding: int, env_init):
        self.env = env_init()
        self.rounding = rounding

    def work(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]) -> Tuple[
        List[Tuple[Tuple[Tuple[Tuple[float, float]], bool], List[Tuple[Tuple[float, float]]]]], List[Tuple[Tuple[Tuple[Tuple[float, float]], bool], List[Tuple[Tuple[float, float]]]]], List[
            Tuple[Tuple[Tuple[Tuple[float, float]], bool], List[Tuple[Tuple[float, float]]]]]]:
        successors_dict = defaultdict(list)
        terminals_dict = defaultdict(list)
        half_terminals_dict = defaultdict(list)
        for interval in intervals:
            action = 1 if interval[1] else 0  # 1 if safe 0 if not
            next_state, half_done, done = step_state(interval[0], action, self.env, self.rounding)
            next_state_sticky, half_done_sticky, done_sticky = step_state(next_state, action, self.env, self.rounding)
            if done:
                terminals_dict[interval].append(next_state)
            if done_sticky:
                terminals_dict[interval].append(next_state)
            successors_dict[interval].append((next_state, next_state_sticky))
        successors_list = []
        terminals_list = []
        half_terminals_list = []
        for key in terminals_dict:
            if len(terminals_dict[key]) != 0:
                terminals_list.append((key, terminals_dict[key]))
        for key in half_terminals_dict:
            if len(half_terminals_dict[key]) != 0:
                half_terminals_list.append((key, half_terminals_dict[key]))
        for key in successors_dict:
            if len(successors_dict[key]) != 0:
                successors_list.append((key, successors_dict[key]))
        return successors_list, half_terminals_list, terminals_list


def step_state(state: Tuple[Tuple[float, float]], action, env, rounding: int) -> Tuple[Tuple[Tuple[float, float]], bool, bool]:
    # given a state and an action, calculate next state
    env.reset()
    state = round_tuple(state, rounding)  # round the state
    env.set_state(state)
    next_state, reward, done, half_done = env.step(action)
    return round_tuple(next_state, rounding), half_done, done
