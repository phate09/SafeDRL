import zerorpc
from prism.state_storage import StateStorage

class HelloRPC(object):
    def hello(self, name):
        return "Hello, %s" % name


s = zerorpc.Server(StateStorage())
s.bind("ipc:///tmp/zerorpc")
s.run()
