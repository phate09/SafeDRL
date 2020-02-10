import time
from pyRpc import RpcConnection


def response(resp):
    print("Got response:", resp)
    print("1 + 2 =", resp.result)


remote = RpcConnection("com.myCompany.MyApplication")
time.sleep(.1)

# we can ask the remote server what services are available
resp = remote.availableServices()
for service in resp.result:
    print("\nService: %(service)s \nDescription: %(doc)s \nUsage: %(format)s\n" % service)

# calls remote add() and does not wait. Result will be returned to response()
remote.call("add", args=(1, 2), callback=response)

# blocks while waiting for a response
resp = remote.call("favoriteColor")
print("Favorite color:", resp.result)

time.sleep(1)

remote.close()

time.sleep(1)
