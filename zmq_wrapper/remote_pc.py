import time
from pyRpc import PyRpc


def add(a, b):
    """ Returns a + b """
    return a + b


def favoriteColor():
    """ Tell you our favorite color """
    return "red"


myRpc = PyRpc("com.myCompany.MyApplication")
time.sleep(.1)

myRpc.publishService(add)
myRpc.publishService(favoriteColor)
myRpc.start()

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    myRpc.stop()
