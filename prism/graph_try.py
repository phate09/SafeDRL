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

#%%
for (u, v,wt) in G.out_edges.data('p'):
    print(f"u:{u} v:{v} wt:{wt}")
#%%
for n, nbrs in G.adjacency():
    for nbr, eattr in nbrs.items():
        print(f"u:{n} v:{nbr} wt:{eattr.get('p',0)}")
#%%
nodes = [0]
while len(nodes)!=0:
    node = nodes.pop()
    G.adjacency()
