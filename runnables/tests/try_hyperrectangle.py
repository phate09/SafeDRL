import ray

from mosaic.hyperrectangle import HyperRectangle
from prism.shared_rtree import SharedRtree
from symbolic import unroll_methods


def try1():
    local_mode = True
    if not ray.is_initialized():
        ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
    rtree = SharedRtree()
    rtree.reset(2)
    rounding = 3
    rtree.load_from_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p", rounding)
    # interval = HyperRectangle.from_tuple(((-0.386, -0.385), (1.11, 1.125)))
    interval = HyperRectangle.from_tuple(((-0.785, 0.785), (-2.0, 2.0)))
    remainings, intersection = unroll_methods.compute_remaining_intervals4_multi([interval], rtree.tree, rounding)
    # remainings, intersection = unroll_methods.compute_remaining_intervals3(interval,rtree.tree_intervals(),debug=False)
    print(remainings)
    # if len(remainings) != 0:
    #     assigned_intervals, ignore_intervals = unroll_methods.assign_action_to_blank_intervals(remainings, explorer, verification_model, n_workers, rounding)


if __name__ == '__main__':
    try1()
