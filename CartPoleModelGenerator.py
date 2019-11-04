import os

import numpy as np
import torch
import zmq
from gym.envs.classic_control import CartPoleEnv

from dqn.dqn_agent import Agent
from proto.state_request_pb2 import StringVarNames, CartPoleState

state_size = 4
action_size = 2
ALPHA = 0.4


class CartPoleModelGenerator:
    def __init__(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.last_state: np.ndarray
        self.last_action: np.ndarray
        self.p: float = 0.2  # probability of sticky actions
        self.max_n: int = 10
        self.current_t: int = 0
        self.failed: bool = False
        self.terminal: bool = False
        self.env = CartPoleEnv()
        self.env.seed(0)
        torch.manual_seed(0)
        self.agent = Agent(state_size=state_size, action_size=action_size, alpha=ALPHA)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.agent.load('./save/Sep19_12-42-51_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_final.pth')

    def start(self):
        while True:
            message: list = self.socket.recv_multipart()  # receives a multipart message
            decode = message[0].decode('utf-8')
            print(decode)
            method_name = decode
            method = getattr(self, method_name, self.invalid_method)
            # Call the method as we return it
            method(message)

    def invalid_method(self, message):
        print("Invalid method name")
        self.socket.send_string("Invalid")

    def getVarNames(self, message):
        string_var_names = StringVarNames()
        string_var_names.value.append("t")
        string_var_names.value.append("done")
        string_var_names.value.append("x1")
        string_var_names.value.append("x2")
        string_var_names.value.append("x3")
        string_var_names.value.append("x4")
        self.socket.send(string_var_names.SerializeToString())

    def getVarTypes(self, message):
        string_var_names = StringVarNames()
        for i in range(6):  # 4 variables + current_t + done
            string_var_names.value.append("TypeInt")
        self.socket.send(string_var_names.SerializeToString())

    def getLabelNames(self, message):
        label_names = ["failed", "done"]
        string_var_names = StringVarNames()
        for label in label_names:
            string_var_names.value.append(label)
        self.socket.send(string_var_names.SerializeToString())

    def createVarList(self, message):
        pass

    def getInitialState(self, message):
        self.env.seed(0)
        torch.manual_seed(0)
        self.last_state = self.env.reset()
        self.current_t = 0
        self.terminal = False
        state = self.currentStateToProtobuf()
        self.socket.send(state.SerializeToString())

    def exploreState(self, message):
        state = CartPoleState()
        state.ParseFromString(message[1])
        self.last_state = self.parseFromPrism(state)  # parse from StateInt
        self.env.state = self.last_state  # loads the state in the environment
        self.current_t = state.t
        self.terminal = state.done
        self.socket.send_string("OK")

    def getNumChoices(self, message):
        if self.current_t > self.max_n or self.terminal:  # if done
            self.socket.send_string(str(1))
        else:
            self.socket.send_string("1")  # returns only 1 choice

    def getNumTransitions(self, message):
        if self.current_t > self.max_n or self.terminal:  # if done
            self.socket.send_string(str(1))
        else:
            self.socket.send_string(str(2))

    def getTransitionAction(self, message):
        pass

    def getTransitionProbability(self, message):
        i = int(message[1].decode('utf-8'))
        offset = int(message[2].decode('utf-8'))
        prob = 1.0 if (self.current_t > self.max_n or self.terminal) else (1 - self.p if offset == 0 else self.p)
        self.socket.send_string(str(prob))

    def computeTransitionTarget(self, message):
        i = int(message[1].decode('utf-8'))
        offset = int(message[2].decode('utf-8'))
        if self.current_t >= self.max_n or self.terminal:
            state = self.currentStateToProtobuf()  # append the current state values
        else:
            self.env.reset()
            self.env.state = self.last_state
            action = self.agent.act(self.last_state, 0)
            state, reward, done, _ = self.env.step(action)
            current_t = self.current_t + 1
            if offset == 1 and not done:
                state, reward, done, _ = self.env.step(action)
            state = self.stateToProtobuf(current_t, state, done)
        self.socket.send(state.SerializeToString())

    def isLabelTrue(self, message):
        i = int(message[1].decode('utf-8'))
        if i == 0:
            x, x_dot, theta, theta_dot = self.last_state
            fail = self.terminal or x < -self.env.x_threshold \
                   or x > self.env.x_threshold \
                   or theta < -self.env.theta_threshold_radians \
                   or theta > self.env.theta_threshold_radians
            value = fail
        elif i == 1:
            value = self.current_t >= self.max_n
        else:
            # should never happen
            value = False
        self.socket.send_string(str(value))
        pass

    def getRewardStructNames(self, message):
        reward = StringVarNames()
        reward.value.append("r")
        self.socket.send(reward.SerializeToString())

    def getStateReward(self, message):
        state = CartPoleState()
        state.ParseFromString(message[1])
        last_state = self.parseFromPrism(state)  # parse from StateInt
        current_t = state.t
        terminal = state.done
        if not terminal:
            self.socket.send_string(str(1.0))
        else:
            self.socket.send_string(str(0.0))

    def getStateActionReward(self, message):
        self.socket.send_string(str(0.0))

    def parseFromPrism(self, state: CartPoleState):
        return np.asarray(state.value, dtype=float) / 1000

    def currentStateToProtobuf(self):
        return self.stateToProtobuf(self.current_t, self.last_state, self.terminal)

    def stateToProtobuf(self, t, env_state: np.ndarray, done: bool):
        state = CartPoleState()
        state.t = t
        state.done = done
        for x in env_state:
            state.value.append(int(x * 1000))  # multiply by 100 for rounding up integers
        return state


if __name__ == '__main__':
    os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
    model = CartPoleModelGenerator(5558)
    model.start()
