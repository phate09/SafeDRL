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

    def set_multiple(self, keys, value):
        for key in keys:
            self.set(key, value)


def get_shared_dictionary() -> SharedDict:
    Pyro5.api.config.SERIALIZER = "marshal"
    dictionary = Pyro5.api.Proxy("PYRONAME:prism.shareddict")
    return dictionary


if __name__ == '__main__':
    Pyro5.api.config.SERIALIZER = "marshal"
    Pyro5.api.config.SERVERTYPE = "multiplex"
    Pyro5.api.Daemon.serveSimple({SharedDict: "prism.shareddict"}, ns=True)
