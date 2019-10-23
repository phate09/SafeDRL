# %%
import jsonpickle
import numpy as np
import mpmath
from mpmath import iv
from symbolic.cartpole_abstract import CartPoleEnv_abstract
from verification_runs.cartpole_bab_load import generateCartpoleDomainExplorer
import os
from verification_runs.aggregate_abstract_domain import aggregate


# %%
# os.chdir("/home/edoardo/Development/SafeDRL")
# with open("./save/aggregated_safe_domains.json", 'r') as f:
#     frozen_safe = jsonpickle.decode(f.read())
# with open("./save/unsafe_domains.json", 'r') as f:
#     frozen_unsafe = jsonpickle.decode(f.read())
# with open("./save/ignore_domains.json", 'r') as f:
#     frozen_ignore = jsonpickle.decode(f.read())
# frozen_safe = np.stack(frozen_safe)  # .take(range(10), axis=0)
# frozen_unsafe = np.stack(frozen_unsafe)  # .take(range(10), axis=0)
# frozen_ignore = np.stack(frozen_ignore)  # .take(range(10), axis=0)
# t_states = [(frozen_safe, frozen_unsafe, frozen_ignore)]

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
env = CartPoleEnv_abstract()
s = env.reset()
s_array = np.stack([interval_unwrap(s)])


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
# given the initial states calculate which intervals go left or right
explorer.explore(verification_model, s_array)
safe_next = [i.cpu().numpy() for i in explorer.safe_domains]
unsafe_next = [i.cpu().numpy() for i in explorer.unsafe_domains]
ignore_next = [i.cpu().numpy() for i in explorer.ignore_domains]
t_states = [(safe_next, unsafe_next, ignore_next)]


# %%
def explore_step(states: np.ndarray, action):
    next_states_array = abstract_step(states, action)
    explorer.reset()
    stats = explorer.explore(verification_model, next_states_array, min_area=1e-8, debug=True)
    print(f"#states: {stats['n_states']} [safe:{stats['safe_relative_percentage']:.3%}, unsafe:{stats['unsafe_relative_percentage']:.3%}, ignore:{stats['ignore_relative_percentage']:.3%}]")
    safe_next = [i.cpu().numpy() for i in explorer.safe_domains]
    unsafe_next = [i.cpu().numpy() for i in explorer.unsafe_domains]
    ignore_next = [i.cpu().numpy() for i in explorer.ignore_domains]
    return safe_next, unsafe_next, ignore_next


# %%
for t in range(30):
    print(f"Iteration for time t={t}")
    safe_states, unsafe_states, ignore_states = t_states[t]
    safe_next_total = []
    unsafe_next_total = []
    ignore_next_total = []
    # safe states
    print(f"Safe states")
    safe_next, unsafe_next, ignore_next = explore_step(safe_states, 0)  # takes 0 in safe states
    safe_next_total.extend(safe_next)
    unsafe_next_total.extend(unsafe_next)
    ignore_next_total.extend(ignore_next)
    # unsafe states
    print(f"Unsafe states")
    safe_next, unsafe_next, ignore_next = explore_step(unsafe_states, 1)  # takes 1 in unsafe states
    safe_next_total.extend(safe_next)
    unsafe_next_total.extend(unsafe_next)
    ignore_next_total.extend(ignore_next)

    # aggregate together states
    safe_array = aggregate(np.stack(safe_next_total)) if len(safe_next_total) != 0 else []
    unsafe_array = aggregate(np.stack(unsafe_next_total)) if len(unsafe_next_total) != 0 else []
    t_states.append((safe_array, unsafe_array, []))  # np.stack(ignore_next + ignore_next2)
    print(f"Finished iteration t={t}, #safe states:{len(safe_next_total)}, #unsafe states:{len(unsafe_next_total)}, #ignored states:{len(ignore_next_total)}")

# %%
# frozen_safe = jsonpickle.encode([i for i in next_states_array])
# with open("../save/next_safe_domains.json", 'w+') as f:
#     f.write(frozen_safe)


# %%
