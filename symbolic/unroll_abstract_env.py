# %%
from symbolic.unroll_methods import *

# %%
os.chdir("/home/edoardo/Development/SafeDRL")

# %%
env = CartPoleEnv_abstract()
s = env.reset()
s_array = np.stack([interval_unwrap(s)])

# %%
explorer, verification_model = generateCartpoleDomainExplorer()
# %%
# # given the initial states calculate which intervals go left or right
# stats = explorer.explore(verification_model, s_array, debug=False)
# print(f"#states: {stats['n_states']} [safe:{stats['safe_relative_percentage']:.3%}, unsafe:{stats['unsafe_relative_percentage']:.3%}, ignore:{stats['ignore_relative_percentage']:.3%}]")
# safe_next = [i.cpu().numpy() for i in explorer.safe_domains]
# unsafe_next = [i.cpu().numpy() for i in explorer.unsafe_domains]
# ignore_next = [i.cpu().numpy() for i in explorer.ignore_domains]
# safe_next = np.stack(safe_next) if len(safe_next) != 0 else []
# unsafe_next = np.stack(unsafe_next) if len(unsafe_next) != 0 else []
# ignore_next = np.stack(ignore_next) if len(ignore_next) != 0 else []
# t_states = [[safe_next, unsafe_next, ignore_next]]

# %%
# for t in range(4):
#     iteration(t, t_states, env, explorer, verification_model)
# %%
# with open("../save/t_states.json", 'w+') as f:
#     f.write(jsonpickle.encode(t_states))
# # %%
# with open("./save/t_states.json", 'r') as f:
#     t_states = jsonpickle.decode(f.read())
#
# # %% Generate random points
# random_points = generate_points_in_intervals(t_states[4][0], 500)
# random_points2 = generate_points_in_intervals(t_states[4][1], 500)
# random_points = np.concatenate((random_points, random_points2))
# del random_points2
# # %%
# assigned_actions = assign_action(random_points, t_states)
# # %%
# original_frame = pd.DataFrame(random_points)
# original_frame["action"] = assigned_actions
# #%%
# original_frame.to_json("./save/original_dataframe.json")
# %%
original_frame = pd.read_json("./save/original_dataframe.json")
# %%
x = StandardScaler().fit_transform(original_frame.iloc[:, 0:4])
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

# %%

pca_frame = pd.DataFrame(principalComponents, columns=["A", "B"])
pca_result = pd.concat([original_frame, pca_frame], axis=1, sort=False)
pca_result.head()
# %%
import plotly.express as px

fig = px.scatter(pca_result, x="A", y="B", color="action")
fig.write_html('first_figure.html', auto_open=True)
# # %%
# lca = LinearDiscriminantAnalysis(n_components=None)
# x = StandardScaler().fit_transform(original_frame.iloc[:, 0:4])
# linearDiscriminants = lca.fit_transform(x,original_frame["action"])
# lca_frame = pd.DataFrame(linearDiscriminants,columns=["C","D"])
# lca_result = pd.concat([pca_result, lca_frame], axis=1, sort=False)
# lca_result.head()
# #%%
# fig = px.scatter(pca_result, x="C", y="D", color="action")
# fig.write_html('first_figure.html', auto_open=True)
#%%
