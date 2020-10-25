from collections import defaultdict
from typing import List, Tuple

import ray

from mosaic.hyperrectangle import HyperRectangle_action, HyperRectangle


@ray.remote
class AbstractStepWorker:
    def __init__(self, rounding: int, env_init, probabilistic=False):
        self.env = env_init()
        self.rounding = rounding
        self.probabilistic = probabilistic

    def work(self, intervals: List[HyperRectangle_action]):
        successors_dict = defaultdict(list)
        terminals_dict = defaultdict(bool)
        half_terminals_dict = defaultdict(bool)
        for interval in intervals:
            if not self.probabilistic:
                action = 1 if interval.action else 0  # 1 if safe 0 if not
                next_state, half_done, done = step_state(interval, action, self.env, self.rounding)
                next_state_sticky, half_done_sticky, done_sticky = step_state(next_state, action, self.env, self.rounding)
                if done:
                    terminals_dict[next_state] = True
                if done_sticky:
                    terminals_dict[next_state_sticky] = True
                if half_done:
                    half_terminals_dict[next_state] = True
                if half_done_sticky:
                    half_terminals_dict[next_state_sticky] = True
                successors_dict[interval].append((next_state, next_state_sticky))  # todo move probability here
        return successors_dict, half_terminals_dict, terminals_dict


def step_state(state: HyperRectangle, action, env, rounding: int) -> Tuple[HyperRectangle, bool, bool]:
    # given a state and an action, calculate next state
    env.reset()
    state = state.round(rounding)  # round the state
    env.set_state(state)
    next_state, reward, done, half_done = env.step(action)
    return next_state.round(rounding), half_done, done
