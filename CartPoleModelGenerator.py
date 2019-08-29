from collections import deque

import numpy as np
from gym.envs.classic_control import CartPoleEnv
import torch
from dqn_agent import Agent

state_size = 4
action_size = 2
ALPHA = 0.4
class CartPoleModelGenerator:
    def __init__(self):
        self.p: float = 0.8  # probability of not sticky actions
        self.last_state: np.ndarray
        self.last_action: np.ndarray
        self.max_n: int = 12
        self.current_t: int = 0
        self.failed: bool = False
        self.terminal: bool = False
        self.env = CartPoleEnv()
        self.env.seed(0)
        self.agent = Agent(state_size=state_size, action_size=action_size, alpha=ALPHA)
        self.agent.qnetwork_local.load_state_dict(torch.load('model.pth'))

    def getVarNames(self):
        return ["x", "y", "z"]

    def getLabelNames(self):
        return ["failed"]

    def getInitialState(self):
        self.last_state = self.env.reset()
        self.current_t = 0
        return self.last_state, False, 0

    def exploreState(self, state: np.ndarray, t):
        # cache the state to consider as current
        self.last_state = state
        self.env.state = state  # loads the state in the environment
        self.current_t = t

    def getNumChoices(self):
        return 1

    def getNumTransitions(self):
        if self.current_t > self.max_n:
            return 1
        return 2

    def getTransitionActions(self):
        return None  # no action names needed

    def getTransitionProbabilities(self, i: int, offset: int):  # i is action
        if self.current_t > self.max_n:
            return 1.0
        return self.p if offset == 0 else 1 - self.p

    def computeTransitionTarget(self, i: int, offset: int):  # i is action
        if self.current_t >= self.max_n:
            return self.last_state, self.terminal, self.current_t
        else:
            action = self.agent.act(self.last_state,0)
            state, reward, done, _ = self.env.step(action)
            self.last_state = state
            self.terminal = done
            self.current_t = self.current_t + 1
            if offset == 1 and not done:
                state, reward, done, _ = self.env.step(action)
                self.last_state = state
                self.terminal = done
        return self.last_state, self.terminal, self.current_t

    def isLabelTrue(self, i: int):
        # todo check if the given label is true
        pass

    def getRewardStructNames(self):
        return ["r"]

    def getStateReward(self, r: int, state: np.ndarray):
        return 1.0  # as long as it's not over the agent receives 1

    def getStateActionReward(self, r: int, state: np.ndarray, action: float):
        return 0.0

    def generateTree(self):
        queue = deque()
        state, terminal, t = self.getInitialState()
        queue.append((state, terminal, 1.0, t))
        set = {(tuple(self.last_state), terminal, t)}
        max_t = 0
        while len(queue) != 0:
            (state, terminal, prob, t) = queue.popleft()
            for i in range(self.getNumChoices()):
                for offset in range(self.getNumTransitions()):
                    self.exploreState(state, t)  # loads the state in the environment
                    new_prob = self.getTransitionProbabilities(i, offset)
                    state, terminal, t_next = self.computeTransitionTarget(i, offset)
                    if (tuple(state), terminal,t_next) not in set:
                        queue.append((state, terminal, new_prob, t_next))
                    # else:
                    #     print("skipped")
                    max_t=max(max_t,t_next)
                    set.add((tuple(state), terminal, t_next))
        return set,max_t


if __name__ == '__main__':
    model = CartPoleModelGenerator()
    tree,max_t = model.generateTree()
    # print(tree)
    print(f"total length {len(tree)}, max_t {max_t}")
