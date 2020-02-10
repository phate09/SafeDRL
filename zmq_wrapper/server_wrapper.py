import zmq


class ServerWrapper:
    def __init__(self, port):
        self.port = port
    def start(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        while True:
            #  Wait for next request from client
            message = socket.recv()
