# %%
from symbolic.unroll_methods import *

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


# %%
env = CartPoleEnv_abstract()
s = env.reset()
s_array = np.stack([interval_unwrap(s)])

# %%


# %%


# %%


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
for t in range(4):
    iteration(t, t_states, env, explorer, verification_model)

# %%
total_states = []
# for t in range(len(t_states)):
t = len(t_states) - 1
for i in range(3):
    if isinstance(t_states[t][i], np.ndarray):
        total_states.append(t_states[t][i])
total_states = np.concatenate(total_states)
# %%
with open("../save/t_states.json", 'w+') as f:
    f.write(jsonpickle.encode(t_states))
# %%
with open("../save/t_states.json", 'r') as f:
    t_states = jsonpickle.decode(f.read())

# %%


# %%
random_points = generate_points_in_intervals(total_states, 1000)

# %%


# %%
assigned_actions = assign_action(random_points, t_states)
# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(random_points)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

# %%
import plotly.graph_objs

fig = plotly.graph_objs.Figure(data=go.Scatter(x=x[:, 0], y=x[:, 1], mode='markers'))
fig.write_html('first_figure.html', auto_open=True)

# %%
import pandas as pd

frame = pd.DataFrame(principalComponents, columns=list("AB"))
frame["action"] = assigned_actions
# %%
import plotly.express as px

fig = px.scatter(frame, x="A", y="B", color="action")
fig.write_html('first_figure.html', auto_open=True)
