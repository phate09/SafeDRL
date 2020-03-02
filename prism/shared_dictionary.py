import zerorpc


class SharedDict():
    """A dictionary shared across multiple processes """

    def __init__(self):
        self.dictionary = dict()

    def reset(self):
        self.dictionary = dict()

    def get(self, key, default=None):
        key = tuple([tuple(x) for x in key])
        return self.dictionary.get(key, default)

    def set(self, key, value):
        key = tuple([tuple(x) for x in key])
        self.dictionary[key] = value


def get_shared_dictionary():
    c = zerorpc.Client(timeout=99999999, heartbeat=9999999)
    c.connect("ipc:///tmp/shared_dict")
    # c.connect("tcp://127.0.0.1:4242")
    return c


if __name__ == '__main__':
    s = zerorpc.Server(SharedDict())
    s.bind("ipc:///tmp/shared_dict")
    # s.bind("tcp://0.0.0.0:4242")
    print("SharedDict server started")
    s.run()
