# %%
import scipy.spatial
from symbolic.unroll_methods import *
from mosaic.utils import compute_remaining_intervals2, compute_remaining_intervals2_multi, truncate_multi, beep, compute_remaining_intervals3_multi
from py4j.java_gateway import JavaGateway

os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
gateway = JavaGateway()
mdp = gateway.entry_point.getMdpSimple()
dictionary = dict()
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
safe_states_total = [tuple([(x[0], x[1]) for x in k]) for k in safe_states]
unsafe_states_total = [tuple([(x[0], x[1]) for x in k]) for k in unsafe_states]
remainings = [current_interval]
#
last_time_remaining_number = -1
precision = 1e-6
t = 0
failed = []
failed_area = 0
print("Building the tree")
safetree = scipy.spatial.cKDTree(data=safe_states[:, :, 0])
safetree2 = scipy.spatial.cKDTree(data=safe_states[:, :, 1])
unsafetree = scipy.spatial.cKDTree(data=unsafe_states[:, :, 0])
unsafetree2 = scipy.spatial.cKDTree(data=unsafe_states[:, :, 1])
print("Finished building the tree")
# %%
for i in range(10):
    remainings, safe_intervals_union = compute_remaining_intervals3_multi(remainings, safe_states_total, safetree)  # checks areas not covered by total safe intervals
    remainings, safe_intervals_union2 = compute_remaining_intervals3_multi(remainings, safe_states_total, safetree2)  # checks areas not covered by total safe intervals
    remainings, unsafe_intervals_union = compute_remaining_intervals3_multi(remainings, unsafe_states_total, unsafetree)  # checks areas not covered by total unsafe intervals
    remainings, unsafe_intervals_union2 = compute_remaining_intervals3_multi(remainings, unsafe_states_total, unsafetree2)  # checks areas not covered by total unsafe intervals
    print(f"Remainings before negligibles: {len(remainings)}")
    remainings = discard_negligibles(remainings)  # discard intervals with area 0
    area = sum([calculate_area(np.array(remaining)) for remaining in remainings])
    failed.extend(remainings)
    failed_area += area
    print(f"Remainings : {len(remainings)} Area:{area} Total Area:{failed_area}")
    remainings = []  # reset remainings

    safe_states_assigned = []  # only current iteration
    unsafe_states_assigned = []
    safe_states_assigned.extend(safe_intervals_union)
    safe_states_assigned.extend(safe_intervals_union2)
    unsafe_states_assigned.extend(unsafe_intervals_union)
    unsafe_states_assigned.extend(unsafe_intervals_union2)

    safe_states_denormalised = [explorer.denormalise(x) for x in safe_states_assigned]
    unsafe_states_denormalised = [explorer.denormalise(x) for x in unsafe_states_assigned]

    next_states_array = abstract_step(safe_states_denormalised, 1, env)  # performs a step in the environment with the assigned action and retrieve the result
    next_states_array2 = abstract_step(unsafe_states_denormalised, 0, env)  # performs a step in the environment with the assigned action and retrieve the result
    # the result is appended to the list of remaining intervals to verify
    remainings.extend([tuple([(float(x[dimension].item(0)), float(x[dimension].item(1))) for dimension in range(len(x))]) for x in next_states_array])
    remainings.extend([tuple([(float(x[dimension].item(0)), float(x[dimension].item(1))) for dimension in range(len(x))]) for x in next_states_array2])
    remainings = [explorer.normalise(x) for x in remainings]
    print(f"Remainings : {len(remainings)}")
    t = t + 1
    print(f"t:{t}")
