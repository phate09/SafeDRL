#%%
import jsonpickle
import numpy as np
import mpmath
from mpmath import iv
from symbolic.cartpole_abstract import CartPoleEnv_abstract
from verification_runs.cartpole_bab_load import generateCartpoleDomainExplorer

#%%

with open("../save/aggregated_safe_domains.json", 'r') as f:
    frozen_safe = jsonpickle.decode(f.read())
# with open("../save/unsafe_domains.json", 'r') as f:
#     frozen_unsafe = jsonpickle.decode(f.read())
# with open("../save/ignore_domains.json", 'r') as f:
#     frozen_ignore = jsonpickle.decode(f.read())
frozen_safe = np.stack(frozen_safe)#.take(range(10), axis=0)
# frozen_unsafe = np.stack(frozen_unsafe)  # .take(range(10), axis=0)
# frozen_ignore = np.stack(frozen_ignore)  # .take(range(10), axis=0)

#%%
env = CartPoleEnv_abstract()

#%%
def interval_unwrap(state):
    unwrapped_state = tuple([[float(x.a), float(x.b)] for x in state])
    return unwrapped_state

def step_state(state, action):
    env.reset()
    env.state = state
    next_state, _, _, _ = env.step(action)
    return tuple(next_state)


#%%

next_states = []
for interval in frozen_safe:
    state = tuple([iv.mpf([x.item(0), x.item(1)]) for x in interval])
    next_state = step_state(state, 1)
    unwrapped_next_state = interval_unwrap(next_state)
    next_states.append(unwrapped_next_state)

next_states_array = np.array(next_states,dtype=np.float32) #turns the list in an array


#%%
explorer, verification_model = generateCartpoleDomainExplorer()


#%%
explorer.explore(verification_model, next_states_array, precision=1e-6, min_area=0)
safe_next = explorer.safe_domains
unsafe_next = explorer.unsafe_domains
ignore_next = explorer.ignore_domains

#%%
# frozen_safe = jsonpickle.encode([i for i in next_states_array])
# with open("../save/next_safe_domains.json", 'w+') as f:
#     f.write(frozen_safe)


#%%



