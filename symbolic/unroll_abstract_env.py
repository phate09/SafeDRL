# %%
from symbolic.unroll_methods import *

# %%
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
# %%
env = CartPoleEnv_abstract()
s = env.reset()
s_array = np.stack([interval_unwrap(s)])

# %%
explorer, verification_model = generateCartpoleDomainExplorer()
# %%
# given the initial states calculate which intervals go left or right
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
# with open("./save/t_states.json", 'w+') as f:
#     f.write(jsonpickle.encode(t_states))
# %%
with open("./save/t_states.json", 'r') as f:
    t_states = jsonpickle.decode(f.read())

# %%
frames = []
for t in range(len(t_states)):
    random_points, areas, assigned_actions = generate_middle_points(t_states, t)
    new_frame = pd.DataFrame(random_points)
    new_frame["action"] = assigned_actions
    new_frame["area"] = areas
    new_frame["t"] = t
    frames.append(new_frame)
main_frame = pd.concat(frames, ignore_index=True)
# %%
main_frame.to_pickle("./save/original_dataframe.pickle")
# %%
main_frame: pd.DataFrame = pd.read_pickle("./save/original_dataframe.pickle")
# append A and B column for PCA dimensions
main_frame["A"] = np.nan
main_frame["B"] = np.nan
main_frame.head()
# %%
x = StandardScaler().fit_transform(main_frame.iloc[:, 0:4])
pca = PCA(n_components=2)
pca.fit(x)
principalComponents = pca.transform(x)
# %%
main_frame.loc[:, ["A", "B"]] = principalComponents  # (0, 0):(0, 500)
main_frame.at[0,"action"] = "safe" # fix to plot not showing all the labels unless they both appear at t=0
main_frame.head()
# %%
import plotly.express as px

fig = px.scatter(main_frame, x="A", y="B",animation_frame="t",size="area", color="action",size_max=500,height=1200)
fig.write_html('first_figure.html', auto_open=True)  # # %%  # lca = LinearDiscriminantAnalysis(n_components=None)  # x = StandardScaler().fit_transform(original_frame.iloc[:, 0:4])  # linearDiscriminants = lca.fit_transform(x,original_frame["action"])  # lca_frame = pd.DataFrame(linearDiscriminants,columns=["C","D"])  # lca_result = pd.concat([pca_result, lca_frame], axis=1, sort=False)  # lca_result.head()  # #%%  # fig = px.scatter(pca_result, x="C", y="D", color="action")


# fig.write_html('first_figure.html', auto_open=True)
# %%
def mklbl(prefix, n):
    return ["%s%s" % (prefix, i) for i in range(n)]


miindex = pd.MultiIndex.from_product([mklbl('A', 2), mklbl('B', 2), mklbl('C', 2), mklbl('D', 2)])

micolumns = pd.MultiIndex.from_tuples([('a', 'foo'), ('a', 'bar'), ('b', 'foo'), ('b', 'bah')], names=['lvl0', 'lvl1'])

dfmi = pd.DataFrame(np.arange(len(miindex) * len(micolumns)).reshape((len(miindex), len(micolumns))), index=miindex, columns=micolumns).sort_index().sort_index(axis=1)

dfmi.loc[:, ('b', 'foo')] = np.arange(0, 16)
# %%
import plotly.express as px

gapminder = px.data.gapminder()
