import networkx as nx

import prism.state_storage
from prism.shared_rtree import SharedRtree
import mosaic.utils as utils
import symbolic.unroll_methods as unroll_methods


def plot(environment_name="cartpole", horizon: int = 8, abstract: bool = True, rounding: int = 3, *, plot_type="lb", folder_path):
    rtree = SharedRtree()
    storage = prism.state_storage.StateStorage()
    storage.reset()
    env_type = "concrete" if not abstract else "abstract"
    rtree.load_from_file(f"{folder_path}/union_states_total_{environment_name}_e{rounding}_{env_type}.p", rounding)
    storage.load_state(f"{folder_path}/nx_graph_{environment_name}_e{rounding}_{env_type}.p")
    # states = unroll_methods.get_n_states(storage,horizon)
    # shortest_path_abstract = nx.shortest_path(storage.graph, source=storage.root)
    # layers = unroll_methods.get_layers(storage.graph,storage.root)
    if plot_type == "lb":
        results = [(x, prob) for (x, action), prob in unroll_methods.get_property_at_timestep(storage, 1, ["lb"])]
        utils.show_heatmap(results)
    elif plot_type == "ub":
        results = [(x, prob) for (x, action), prob in unroll_methods.get_property_at_timestep(storage, 1, ["ub"])]
        utils.show_heatmap(results)
    elif plot_type == "error":
        results = [(x, ub - lb) for (x, action), lb, ub in unroll_methods.get_property_at_timestep(storage, 1, ["lb", "ub"])]  # difference between boundaries
        utils.show_heatmap(results)
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
    else:
        raise Exception("Option for plot_type not recognised")
