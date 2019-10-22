# %%
import jsonpickle
import numpy as np
import mpmath
from mpmath import iv
from symbolic.cartpole_abstract import CartPoleEnv_abstract
from verification_runs.cartpole_bab_load import generateCartpoleDomainExplorer
import os

# %%
os.chdir("/home/edoardo/Development/SafeDRL")
with open("./save/aggregated_safe_domains.json", 'r') as f:
    frozen_safe = jsonpickle.decode(f.read())
with open("./save/unsafe_domains.json", 'r') as f:
    frozen_unsafe = jsonpickle.decode(f.read())
with open("./save/ignore_domains.json", 'r') as f:
    frozen_ignore = jsonpickle.decode(f.read())
frozen_safe = np.stack(frozen_safe)  # .take(range(10), axis=0)
frozen_unsafe = np.stack(frozen_unsafe)  # .take(range(10), axis=0)
frozen_ignore = np.stack(frozen_ignore)  # .take(range(10), axis=0)

# %%
env = CartPoleEnv_abstract()
s = env.reset()


# %%
def interval_unwrap(state):
    unwrapped_state = tuple([[float(x.a), float(x.b)] for x in state])
    return unwrapped_state


def step_state(state, action):
    # given a state and an action, calculate next state
    env.reset()
    env.state = state
    next_state, _, _, _ = env.step(action)
    return tuple(next_state)


# %%
def abstract_step(abstract_states: np.ndarray, action=1):
    """
    Given some abstract states, compute the next abstract states taking the action passed as parameter
    :param abstract_states: the abstract states from which to start
    :param action: the action to take
    :return: the next abstract states after taking the action (array)
    """
    next_states = []
    for interval in abstract_states:
        state = tuple([iv.mpf([x.item(0), x.item(1)]) for x in interval])
        next_state = step_state(state, action)
        unwrapped_next_state = interval_unwrap(next_state)
        next_states.append(unwrapped_next_state)
    next_states_array = np.array(next_states, dtype=np.float32)  # turns the list in an array
    return next_states_array


# %%
explorer, verification_model = generateCartpoleDomainExplorer()
t_states = [(frozen_safe, frozen_unsafe, frozen_ignore)]

# %%
for t in range(3):
    print(f"Iteration for time t={t}")
    safe_states, unsafe_states, ignore_states = t_states[t]

    # safe states
    next_states_array = abstract_step(safe_states, 1)  # takes 1 in safe states
    explorer.reset()
    explorer.explore(verification_model, next_states_array, precision=1e-6, min_area=0)
    safe_next = explorer.safe_domains
    unsafe_next = explorer.unsafe_domains
    ignore_next = explorer.ignore_domains

    # unsafe states
    next_states_array2 = abstract_step(unsafe_states, 0)  # takes 0 in unsafe states
    explorer.reset()
    explorer.explore(verification_model, next_states_array2, precision=1e-6, min_area=0)
    safe_next2 = explorer.safe_domains
    unsafe_next2 = explorer.unsafe_domains
    ignore_next2 = explorer.ignore_domains

    t_states.append((np.stack(safe_next + safe_next2), np.stack(unsafe_next + unsafe_next2), []))  # np.stack(ignore_next + ignore_next2)

# %%
# frozen_safe = jsonpickle.encode([i for i in next_states_array])
# with open("../save/next_safe_domains.json", 'w+') as f:
#     f.write(frozen_safe)


# %%
