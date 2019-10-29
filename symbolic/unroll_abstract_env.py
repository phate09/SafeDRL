# %%
import jsonpickle
import numpy as np
import mpmath
from mpmath import iv
from symbolic.cartpole_abstract import CartPoleEnv_abstract
from verification_runs.cartpole_bab_load import generateCartpoleDomainExplorer
import os
from verification_runs.aggregate_abstract_domain import aggregate
import random


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
def iteration(t: int):
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
    t_states.append([safe_array, unsafe_array, []])  # np.stack(ignore_next + ignore_next2)
    print(f"Finished iteration t={t}, #safe states:{len(safe_next_total)}, #unsafe states:{len(unsafe_next_total)}, #ignored states:{len(ignore_next_total)}")


# %%
explorer, verification_model = generateCartpoleDomainExplorer()
# given the initial states calculate which intervals go left or right
stats = explorer.explore(verification_model, s_array, debug=False)
print(f"#states: {stats['n_states']} [safe:{stats['safe_relative_percentage']:.3%}, unsafe:{stats['unsafe_relative_percentage']:.3%}, ignore:{stats['ignore_relative_percentage']:.3%}]")
safe_next = [i.cpu().numpy() for i in explorer.safe_domains]
unsafe_next = [i.cpu().numpy() for i in explorer.unsafe_domains]
ignore_next = [i.cpu().numpy() for i in explorer.ignore_domains]
safe_next = np.stack(safe_next) if len(safe_next) != 0 else []
unsafe_next = np.stack(unsafe_next) if len(unsafe_next) != 0 else []
ignore_next = np.stack(ignore_next) if len(ignore_next) != 0 else []
t_states = [[safe_next, unsafe_next, ignore_next]]

# %%
for t in range(3):
    iteration(t)

# %%
import plotly.graph_objects as go

fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
fig.write_html('first_figure.html', auto_open=True)

# %%
total_states = []
for t in range(len(t_states)):
    for i in range(3):
        if isinstance(t_states[t][i], np.ndarray):
            total_states.append(t_states[t][i])
total_states = np.concatenate(total_states)


# %%
def generate_points_in_intervals(total_states: np.ndarray):
    """

    :param total_states: 3 dimensional array (n,dimension,interval)
    :return:
    """
    generated_points = []
    for i in range(100):
        group = i % total_states.shape[0]
        f = lambda x: [random.uniform(v[0], v[1]) for v in x]
        random_point = f(total_states[group])
        generated_points.append(random_point)
    return np.stack(generated_points)


# %%
random_points = generate_points_in_intervals(total_states)
# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(random_points)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

#%%
fig = go.Figure(data=go.Scatter(x=x[:,0], y=x[:,1], mode='markers'))
fig.write_html('first_figure.html', auto_open=True)
