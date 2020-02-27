#%%
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([(0,1),(1, 2), (1, 3), (2,4),(4,5),(5,6),(5,7)])
#%%
to_remove =2
G.remove_node(to_remove)
for component in nx.connected_components(G.to_undirected()):
    if 0 not in component:
        G.remove_nodes_from(component)
print(G.nodes)
