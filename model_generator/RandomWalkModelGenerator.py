import zmq


class RandomWalkModelGenerator:
    def __init__(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

    def start(self):
        while True:
            message: list = self.socket.recv_multipart()  # receives a multipart message
            decode = message[0].decode('utf-8')
            method_name = decode
            method = getattr(self, method_name, lambda: print("Invalid method name"))
            # Call the method as we return it
            return method(message)
    def getVarNames(self,message):
        return ["x"]
    def getVarTypes(self,message):
        return ["TypeInt"]

if __name__ == '__main__':
    generator = RandomWalkModelGenerator(5558)
    generator.start()
