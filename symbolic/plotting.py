import prism.state_storage
from mosaic.utils import pca_map
import mosaic.utils as utils
import symbolic.unroll_methods as unroll_methods


def plot(environment_name, storage: prism.state_storage.StateStorage, *, plot_type="lb", folder_path, file_name_save: str = None):
    state_size = len(storage.root[0])
    # states = unroll_methods.get_n_states(storage,horizon)
    # shortest_path_abstract = nx.shortest_path(storage.graph, source=storage.root)
    # layers = unroll_methods.get_layers(storage.graph,storage.root)
    save_path = f"{folder_path}/{file_name_save}" if file_name_save is not None else None
    if plot_type == "lb":
        results = [(x, prob) for (x, action), prob in unroll_methods.get_property_at_timestep(storage, 1, ["lb"])]
        # if state_size > 2:
        # results = [(x, prob) for x, prob in results if x[3][0] <= 0 <= x[3][1] and x[2][0] <= 0 <= x[2][1]]
        utils.show_heatmap(results, save_to=save_path, rounding=3)  # , title="Heatmap of the lower bound measured at the initial state"
    elif plot_type == "ub":
        results = [(x, prob) for (x, action), prob in unroll_methods.get_property_at_timestep(storage, 1, ["ub"])]
        if state_size > 2:
            results = [(x, prob) for x, prob in results if x[3][0] <= 0 <= x[3][1] and x[2][0] <= 0 <= x[2][1]]
        utils.show_heatmap(results, save_to=save_path, rounding=2)  # , title="Heatmap of the upper bound measured at the initial state"
    elif plot_type == "error":
        results = [(x, ub - lb) for (x, action), lb, ub in unroll_methods.get_property_at_timestep(storage, 1, ["lb", "ub"])]  # difference between boundaries
        # if state_size > 2:
        #     results = [(x, prob) for x, prob in results if x[3][0] <= 0 <= x[3][1] and x[2][0] <= 0 <= x[2][1]]
        utils.show_heatmap(results, save_to=save_path, rounding=3)  # title="Heatmap of the probability error measured at the initial state"
    elif plot_type == "safe_unsafe":
        results = [(x, lb, ub) for (x, action), lb, ub in unroll_methods.get_property_at_timestep(storage, 1, ["lb", "ub"])]
        safe = []
        unsafe = []
        undecidable = []
        for x, lb, ub in results:
            if lb >= 0.8:
                unsafe.append(x)
            elif ub <= 0.2:
                safe.append(x)
            else:
                undecidable.append(x)
        utils.show_plot(safe, unsafe, undecidable, legend=["Safe", "Unsafe", "Undecidable"])
    elif plot_type == "p_chart":
        results = [(x, prob) for (x, action), prob in unroll_methods.get_property_at_timestep(storage, 1, ["ub"])]
        utils.p_chart(results, save_to=save_path, rounding=2)#, title="Barchart of volumes in the initial state grouped by the upper bound probability "
    elif plot_type == "pca":
        results = [(x, utils.centre_tuple(x), utils.area_tuple(x), action, prob) for (x, action), prob in unroll_methods.get_property_at_timestep(storage, 1, ["ub"])]
        pca_map(results, save_path, state_size)
    else:
        raise Exception("Option for plot_type not recognised")


if __name__ == '__main__':
    folder_path = "/home/edoardo/Development/SafeDRL/save"
    horizon = 7
    rounding = 3
    environment_name = "cartpole"
    abstract = True
    env_type = "concrete" if not abstract else "abstract"
    storage = prism.state_storage.StateStorage()
    storage.reset()
    storage.load_state(f"{folder_path}/nx_graph_{environment_name}_e{rounding}_{env_type}.p")
    for x, successors in storage.graph.adjacency():
        storage.graph.nodes[x]["root"] = True
        storage.root = x
        break
    storage.recreate_prism(horizon)
    plot(environment_name, storage, folder_path=folder_path, plot_type="pca", file_name_save=f"{environment_name}_e{rounding}_h{horizon}_pca.svg")
    plot(environment_name, storage, folder_path=folder_path, plot_type="p_chart", file_name_save=f"{environment_name}_e{rounding}_h{horizon}_p_chart.svg")
