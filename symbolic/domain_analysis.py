# %%
import scipy.spatial
from rtree import index

from symbolic.unroll_methods import *
from mosaic.utils import compute_remaining_intervals2, compute_remaining_intervals2_multi, truncate_multi, beep, compute_remaining_intervals3_multi, shelve_variables2, unshelve_variables, \
    bulk_load_rtree_helper
from py4j.java_gateway import JavaGateway

from verification_runs.aggregate_abstract_domain import merge_list_tuple
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
gateway = JavaGateway()
storage = StateStorage()
env = CartPoleEnv_abstract()
s = env.reset()
current_interval = s
explorer, verification_model = generateCartpoleDomainExplorer()
with open("./save/t_states.json", 'r') as f:
    t_states = jsonpickle.decode(f.read())
safe_states = t_states[0][0]
unsafe_states = t_states[0][1]
# reshape with tuples
current_interval = tuple(
    [((float(x.a) - float(explorer.domain_lb[i].item())) / float(explorer.domain_width[i].item()), (float(x.b) - float(explorer.domain_lb[i].item())) / float(explorer.domain_width[i].item())) for i, x
     in enumerate(current_interval)])
safe_states_total = [tuple([(float(x[0]), float(x[1])) for x in k]) for k in safe_states]
unsafe_states_total = [tuple([(float(x[0]), float(x[1])) for x in k]) for k in unsafe_states]
union_states_total = [(x, True) for x in safe_states_total] + [(x, False) for x in unsafe_states_total]
p = index.Property(dimension=4)

if os.path.exists('save/rtree.dat') and os.path.exists('save/rtree.idx'):
    print("Loading the tree")
    rtree = index.Index('save/rtree', interleaved=False, properties=p)
    print("Finished loading the tree")
else:
    print("Building the tree")
    helper = bulk_load_rtree_helper(union_states_total)
    rtree = index.Index('save/rtree', helper, interleaved=False, properties=p)
    rtree.flush()
    print("Finished building the tree")
remainings = [current_interval]
parent_id = storage.store(current_interval)
#
last_time_remaining_number = -1
precision = 1e-6
t = 0
failed = []
failed_area = 0
terminal_states = []
# %%

for i in range(5):
    remainings, safe_intervals_union, unsafe_intervals_union = compute_remaining_intervals3_multi(remainings, union_states_total, rtree)  # checks areas not covered by total intervals
    assigned_action_intervals = [(x, True) for x in safe_intervals_union] + [(x, False) for x in unsafe_intervals_union]
    assigned_action_intervals = merge_list_tuple(assigned_action_intervals) #aggregate intervals
    # todo states should have an action that leads to aggregated states
    print(f"Remainings before negligibles: {len(remainings)}")
    remainings = discard_negligibles(remainings)  # discard intervals with area 0
    area = sum([calculate_area(np.array(remaining)) for remaining in remainings])
    failed.extend(remainings)
    failed_area += area
    print(f"Remainings : {len(remainings)} Area:{area} Total Area:{failed_area}")
    remainings = []  # reset remainings

    for successor in assigned_action_intervals:
        storage.store_successor(successor[0], parent_id)

    next_states_array, terminal_states_id = abstract_step_store2(assigned_action_intervals, env, storage, explorer)  # performs a step in the environment with the assigned action and retrieve the result
    terminal_states.extend(terminal_states_id)
    remainings = next_states_array
    print(f"Sucessors : {len(remainings)}")
    t = t + 1
    print(f"t:{t}")
storage.save_state("/home/edoardo/Development/SafeDRL/save")
# %%
solution = gateway.entry_point.check_property(1188255)
print(solution[1188255])
# solution = gateway.entry_point.check_state_list(terminal_states)
# gateway.entry_point.export_to_dot_file()
# %%

# storage.load_state("/home/phate09/Development/SafeDRL/save")
