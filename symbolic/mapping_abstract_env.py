# %%
import scipy.spatial
from bidict import bidict

from prism.state_storage import StateStorage
from symbolic.unroll_methods import *
from mosaic.utils import compute_remaining_intervals2, compute_remaining_intervals2_multi, truncate_multi, beep
from symbolic.unroll_methods import compute_remaining_intervals3_multi

storage = StateStorage()
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
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
parent_id = storage.store(current_interval)
#
last_time_remaining_number = -1
precision = 1e-6
t = 0
safe_states_assigned = []  # only current iteration
unsafe_states_assigned = []
with open("./save/safetree.json", 'r') as f:
    safetree = jsonpickle.decode(f.read())
with open("./save/safetree2.json", 'r') as f:
    safetree2 = jsonpickle.decode(f.read())
with open("./save/unsafetree.json", 'r') as f:
    unsafetree = jsonpickle.decode(f.read())
with open("./save/unsafetree2.json", 'r') as f:
    unsafetree2 = jsonpickle.decode(f.read())
# safetree = scipy.spatial.cKDTree(data=safe_states[:, :, 0])
# safetree2 = scipy.spatial.cKDTree(data=safe_states[:, :, 1])
# unsafetree = scipy.spatial.cKDTree(data=unsafe_states[:, :, 0])
# unsafetree2 = scipy.spatial.cKDTree(data=unsafe_states[:, :, 1])
# %%
remainings, safe_intervals_union = compute_remaining_intervals3_multi(remainings)  # checks areas not covered by total safe intervals
remainings, safe_intervals_union2 = compute_remaining_intervals3_multi(remainings)  # checks areas not covered by total safe intervals
remainings, unsafe_intervals_union = compute_remaining_intervals3_multi(remainings)  # checks areas not covered by total unsafe intervals
remainings, unsafe_intervals_union2 = compute_remaining_intervals3_multi(remainings)  # checks areas not covered by total unsafe intervals
print(f"Remainings before negligibles: {len(remainings)}")
remainings = discard_negligibles(remainings)  # discard intervals with area 0
print(f"Remainings : {len(remainings)}")
successors = safe_intervals_union + safe_intervals_union2 + unsafe_intervals_union + unsafe_intervals_union2
for successor in successors:
    storage.store_successor(successor, parent_id)
safe_states_assigned = safe_intervals_union + safe_intervals_union2
unsafe_states_assigned = unsafe_intervals_union + unsafe_intervals_union2
# %%
safe_states_current, unsafe_states_current, ignored = assign_action_to_blank_intervals(remainings)  # finds the action for intervals which are blanks
safe_states_assigned.extend(safe_states_current)
safe_states_assigned.extend(safe_intervals_union)
unsafe_states_assigned.extend(unsafe_states_current)
unsafe_states_assigned.extend(unsafe_intervals_union)
remainings, ignored_union = compute_remaining_intervals2_multi(remainings, ignored)  # checks areas not covered by ignored intervals calculated in this iteration
remainings, safe_intervals_union = compute_remaining_intervals2_multi(remainings, safe_states_current)  # checks areas not covered by safe intervals calculated in this iteration
remainings, unsafe_intervals_union = compute_remaining_intervals2_multi(remainings, unsafe_states_current)  # checks areas not covered by unsafe intervals calculated in this iteration
# beep()
safe_states_assigned.extend(safe_intervals_union)
unsafe_states_assigned.extend(unsafe_intervals_union)
safe_states_total.extend(safe_states_assigned)  # appends the intervals that got discovered in this iteration to the global list
unsafe_states_total.extend(unsafe_states_assigned)  # appends the intervals that got discovered in this iteration to the global list
remainings = discard_negligibles(remainings)  # discard intervals with area 0
assert len(remainings) == 0
# %%
next_states_array = abstract_step_store(safe_states_assigned, 0, env,storage)  # performs a step in the environment with the assigned action and retrieve the result
next_states_array2 = abstract_step_store(unsafe_states_assigned, 1, env,storage)  # performs a step in the environment with the assigned action and retrieve the result
# the result is appended to the list of remaining intervals to verify
remainings.extend([tuple([(float(x[dimension].item(0)), float(x[dimension].item(1))) for dimension in range(len(x))]) for x in next_states_array])
remainings.extend([tuple([(float(x[dimension].item(0)), float(x[dimension].item(1))) for dimension in range(len(x))]) for x in next_states_array2])
print(f"Remainings : {len(remainings)}")
t = t + 1
# #%%
# with open("./save/remain_list_unsafe.json", 'w+') as f:
#     f.write(jsonpickle.encode(remain_list_unsafe))
# %%
with open("./save/remain_list_unsafe.json", 'r') as f:
    remain_list_unsafe = jsonpickle.decode(f.read())
# %%
assigned_list = assign_action_to_blank_intervals(remain_list_unsafe)
safe_states.extend(assigned_list[0])  # updates the list of safe states
unsafe_states.extend(assigned_list[1])  # updates the list of unsafe states
remaining_list_safe = compute_remaining_intervals2_multi(remain_list_unsafe, assigned_list[0])
remain_list_unsafe = compute_remaining_intervals2_multi(remaining_list_safe, assigned_list[1])
safe_next, unsafe_next, ignore_next = explore_step(safe_states, 0, env, explorer, verification_model)  # takes 0 in safe states
safe_next, unsafe_next, ignore_next = explore_step(unsafe_states, 1, env, explorer, verification_model)  # takes 1 in unsafe states
# %%
with open("./save/hard_remainings.json", 'w+') as f:
    f.write(jsonpickle.encode(remainings))
# %% Load
with open("./save/remainings.json", 'r') as f:
    remainings = jsonpickle.decode(f.read())
with open("./save/safe_states_total.json", 'r') as f:
    safe_states_total = jsonpickle.decode(f.read())
with open("./save/unsafe_states_total.json", 'r') as f:
    unsafe_states_total = jsonpickle.decode(f.read())
# %% Save -- CAUTION!
with open("./save/remainings.json", 'w+') as f:
    f.write(jsonpickle.encode(remainings))
with open("./save/safe_states_total.json", 'w+') as f:
    f.write(jsonpickle.encode(safe_states_total))
with open("./save/unsafe_states_total.json", 'w+') as f:
    f.write(jsonpickle.encode(unsafe_states_total))
# %% Save Tree -- CAUTION!
with open("./save/safetree.json", 'w+') as f:
    f.write(jsonpickle.encode(safetree))
with open("./save/safetree2.json", 'w+') as f:
    f.write(jsonpickle.encode(safetree2))
with open("./save/unsafetree.json", 'w+') as f:
    f.write(jsonpickle.encode(unsafetree))
with open("./save/unsafetree2.json", 'w+') as f:
        f.write(jsonpickle.encode(unsafetree2))