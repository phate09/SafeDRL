import zmq

from model_generator.state_request_pb2 import StringVarNames, StateInt


class RandomWalkModelGenerator:
    def __init__(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.x = 0
        self.n = 10
        self.p = 0.5

    def start(self):
        while True:
            message: list = self.socket.recv_multipart()  # receives a multipart message
            decode = message[0].decode('utf-8')
            method_name = decode
            method = getattr(self, method_name, self.invalid_method)
            # Call the method as we return it
            method(message)

    def getVarNames(self, message):
        string_var_names = StringVarNames()
        string_var_names.value.append("x")
        self.socket.send(string_var_names.SerializeToString())

    def getVarTypes(self, message):
        string_var_names = StringVarNames()
        string_var_names.value.append("TypeInt")
        self.socket.send(string_var_names.SerializeToString())

    def getLabelNames(self, message):
        label_names = ["end", "left", "right"]
        string_var_names = StringVarNames()
        for label in label_names:
            string_var_names.value.append(label)
        self.socket.send(string_var_names.SerializeToString())

    def createVarList(self, message):
        pass

    def getInitialState(self, message):
        state = StateInt()
        state.value.append(0)  # initialise the state to 0
        self.socket.send(state.SerializeToString())

    def exploreState(self, message):
        state = StateInt()
        state.ParseFromString(message[1])
        self.x = int(state.value[0])
        self.socket.send_string("OK")

    def getNumChoices(self, message):
        self.socket.send_string("1")  # returns only 1 choice

    def getNumTransitions(self, message):
        transitions = 1 if (self.x == -self.n or self.x == self.n) else 2
        self.socket.send_string(str(transitions))

    def getTransitionAction(self, message):
        pass

    def getTransitionProbability(self, message):
        i = int(message[1].decode('utf-8'))
        offset = int(message[2].decode('utf-8'))
        prob = 1.0 if (self.x == -self.n or self.x == self.n) else (1 - self.p if offset == 0 else self.p)
        self.socket.send_string(str(prob))

    def computeTransitionTarget(self, message):
        state = StateInt()
        i = int(message[1].decode('utf-8'))
        offset = int(message[2].decode('utf-8'))
        if self.x == -self.n or self.x == self.n:
            # do nothing
            state.value.append(self.x)
        else:
            state.value.append(self.x - 1 if offset == 0 else self.x + 1)
        self.socket.send(state.SerializeToString())

    def isLabelTrue(self, message):
        i = int(message[1].decode('utf-8'))
        if i == 0:
            value = (self.x == -self.n or self.x == self.n)
        elif i == 1:
            value = self.x == -self.n
        elif i == 2:
            value = self.x == self.n
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
        self.socket.send_string(str(1.0))

    def getStateActionReward(self, message):
        self.socket.send_string(str(0.0))

    def invalid_method(self, message):
        print("Invalid method name")
        self.socket.send_string("Invalid")


if __name__ == '__main__':
    generator = RandomWalkModelGenerator(5558)
    generator.start()
