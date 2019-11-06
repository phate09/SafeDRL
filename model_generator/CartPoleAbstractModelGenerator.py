import os

import numpy as np
import torch
import zmq
from symbolic.cartpole_abstract import CartPoleEnv_abstract

from proto.state_request_pb2 import StringVarNames, CartPoleState
from mpmath import iv

from symbolic.unroll_methods import interval_unwrap
from verification_runs.aggregate_abstract_domain import aggregate
from verification_runs.cartpole_bab_load import generateCartpoleDomainExplorer

state_size = 4
action_size = 2
ALPHA = 0.4


# noinspection DuplicatedCode,PyMethodMayBeStatic,SpellCheckingInspection,PyUnusedLocal
class CartPoleAbstractModelGenerator:
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
        self.env = CartPoleEnv_abstract()
        self.env.seed(0)
        torch.manual_seed(0)
        self.explorer, self.verification_model = generateCartpoleDomainExplorer()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def start(self):
        while True:
            message: list = self.socket.recv_multipart()  # receives a multipart message
            decode = message[0].decode('utf-8')
            # print(decode)
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
        string_var_names.value.append("x1_0")
        string_var_names.value.append("x2_0")
        string_var_names.value.append("x3_0")
        string_var_names.value.append("x4_0")
        string_var_names.value.append("x1_1")
        string_var_names.value.append("x2_1")
        string_var_names.value.append("x3_1")
        string_var_names.value.append("x4_1")
        self.socket.send(string_var_names.SerializeToString())

    def getVarTypes(self, message):
        string_var_names = StringVarNames()
        for i in range(10):  # 4 variables + current_t + done
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
        self.env.reset()
        self.env.state = self.last_state  # loads the state in the environment
        self.current_t = state.t
        self.terminal = state.done
        self.t_states = self.assign_intervals(self.last_state, self.verification_model, self.explorer)
        print(f"Current t:{self.current_t}")
        self.socket.send_string("OK")

    def assign_intervals(self, state, verification_model, explorer):
        s_array = np.stack([interval_unwrap(state)])  # unwrap the intervals in an array representation
        stats = explorer.explore(verification_model, s_array, debug=False)
        safe_next = [i.cpu().numpy() for i in explorer.safe_domains]
        unsafe_next = [i.cpu().numpy() for i in explorer.unsafe_domains]
        ignore_next = [i.cpu().numpy() for i in explorer.ignore_domains]
        safe_next = np.stack(safe_next) if len(safe_next) != 0 else []
        unsafe_next = np.stack(unsafe_next) if len(unsafe_next) != 0 else []
        ignore_next = np.stack(ignore_next) if len(ignore_next) != 0 else []
        safe_next = aggregate(safe_next)
        unsafe_next = aggregate(unsafe_next)
        ignore_next = aggregate(ignore_next)
        return safe_next, unsafe_next, ignore_next

    def getNumChoices(self, message):
        if self.current_t > self.max_n or self.terminal:  # if done
            self.socket.send_string(str(1))
        else:
            safe_next, unsafe_next, ignore_next = self.t_states
            choices = 0
            if len(safe_next) > 0:
                choices += 1
            if len(unsafe_next) > 0:
                choices += 1
            if choices == 0:
                print("Warning: the choices value is 0")
            self.socket.send_string(str(choices))  # returns only 1 choice

    def getNumTransitions(self, message):
        i = int(message[1].decode('utf-8'))
        if self.current_t > self.max_n or self.terminal:  # if done
            self.socket.send_string(str(1))
        else:
            if len(self.t_states[i]) == 0:  # if it's not the first go to the second list
                i += 1
            self.socket.send_string(str(len(self.t_states[i])))

    def getTransitionAction(self, message):
        pass

    def getTransitionProbability(self, message):
        i = int(message[1].decode('utf-8'))
        if len(self.t_states[i]) == 0:  # if it's not the first go to the second list
            i += 1
        offset = int(message[2].decode('utf-8'))
        # prob = 1.0 if (self.current_t > self.max_n or self.terminal) else (1 - self.p if offset == 0 else self.p)
        # uniform probability
        prob = 1.0 / len(self.t_states[i])
        self.socket.send_string(str(prob))

    def computeTransitionTarget(self, message):
        i = int(message[1].decode('utf-8'))
        if len(self.t_states[i]) == 0:  # if it's not the first go to the second list
            i += 1
        offset = int(message[2].decode('utf-8'))
        if self.current_t >= self.max_n or self.terminal:
            state = self.currentStateToProtobuf()  # append the current state values
        else:
            self.env.reset()
            state = tuple([iv.mpf([x.item(0), x.item(1)]) for x in self.t_states[i][offset]])
            self.env.state = state  # self.last_state
            state, reward, done, _ = self.env.step(i)  # 0=safe, 1 = unsafe
            current_t = self.current_t + 1
            # if offset == 1 and not done:
            #     state, reward, done, _ = self.env.step(action)
            state = self.stateToProtobuf(current_t, state, done)
        self.socket.send(state.SerializeToString())

    def isLabelTrue(self, message):
        i = int(message[1].decode('utf-8'))
        if i == 0:  # fail
            x, x_dot, theta, theta_dot = self.last_state
            fail = self.terminal or x < -self.env.x_threshold or x > self.env.x_threshold or theta < -self.env.theta_threshold_radians or theta > self.env.theta_threshold_radians
            value = fail
        elif i == 1:  # horizon
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
        intervals = []
        for i in range(4):
            lb = state.value[i * 2] / 1000.0
            ub = state.value[i * 2 + 1] / 1000.0
            interval = iv.mpf([lb, ub])
            intervals.append(interval)

        return np.asarray(intervals)

    def currentStateToProtobuf(self):
        return self.stateToProtobuf(self.current_t, self.last_state, self.terminal)

    def stateToProtobuf(self, t, env_state: np.ndarray, done: bool):
        state = CartPoleState()
        state.t = t
        state.done = done
        for x in env_state:
            state.value.append(int(float(x.a) * 1000))  # multiply by 100 for rounding up integers
            state.value.append(int(float(x.b) * 1000))  # multiply by 100 for rounding up integers
        return state


if __name__ == '__main__':
    os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
    model = CartPoleAbstractModelGenerator(5558)
    model.start()
