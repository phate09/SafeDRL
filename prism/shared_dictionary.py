import Pyro5.api

@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="single")
class SharedDict():
    """A dictionary shared across multiple processes """

    def __init__(self):
        self.dictionary = dict()

    def reset(self):
        self.dictionary = dict()
        print("Resetting the SharedDict")

    def get(self, key, default=None):
        # key = tuple([tuple(x) for x in key])
        return self.dictionary.get(key, default)

    def set(self, key, value):
        # key = tuple([tuple(x) for x in key])
        self.dictionary[key] = value


def get_shared_dictionary()->SharedDict:
    # c = zerorpc.Client(timeout=99999999, heartbeat=9999999)
    # c.connect("ipc:///tmp/shared_dict")
    # # c.connect("tcp://127.0.0.1:4242")
    # return c
    Pyro5.api.config.SERIALIZER = "marshal"
    dictionary = Pyro5.api.Proxy("PYRONAME:prism.shareddict")
    return dictionary


if __name__ == '__main__':
    # s = zerorpc.Server(SharedDict())
    # s.bind("ipc:///tmp/shared_dict")
    # # s.bind("tcp://0.0.0.0:4242")
    # print("SharedDict server started")
    # s.run()
    Pyro5.api.config.SERIALIZER = "marshal"
    Pyro5.api.Daemon.serveSimple({SharedDict: "prism.shareddict"}, ns=True)
