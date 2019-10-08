from mpmath import *
from symbolic.cartpole_abstract import CartPoleEnv_abstract
interval = mpi(3, 4)
print(interval)
print(interval+1)

env = CartPoleEnv_abstract()
state = env.reset()
next_state = env.step(1)
print(next_state)