import math

import jsonpickle
import numpy as np
import torch.nn

from dqn.dqn_agent import Agent
from models.model_critic_sequential import QNetwork
from plnn.bab_explore import DomainExplorer
from plnn.branch_and_bound import bab
from plnn.verification_network import VerificationNetwork


def merge_list(frozen_safe, frozen_safe_mean, sorted_indices) -> np.ndarray:
    shrank = [frozen_safe[sorted_indices[0]]]
    for i in sorted_indices[1:]:
        last_merged = shrank[-1]
        last_merged_mean = np.mean(last_merged, axis=1)
        equal_dim_count = 0
        n_fields = frozen_safe_mean.shape[1]  # the number of entries across the second dimension
        for dim in range(n_fields):  # todo vectorise?
            current_mean = frozen_safe_mean[i]
            if last_merged_mean[dim] == current_mean[dim]:
                equal_dim_count += 1

        current_safe = frozen_safe[i]
        if equal_dim_count >= n_fields - 1:  # >3 fields have the same mean
            output = merge_single(last_merged, current_safe, n_fields)
            shrank[-1] = output  # replace last one
        else:
            output = current_safe
            shrank.append(output)  # add to last one
    return np.stack(shrank)


def merge_single(first, second, n_dims) -> np.ndarray:
    output = []
    for dim in range(n_dims):
        if not (first[dim] == second[dim]).all():
            if first[dim][0] == second[dim][1] or first[dim][1] == second[dim][0]:
                lb = min(first[dim][0], second[dim][0])
                ub = max(first[dim][1], second[dim][1])
                output.append([lb, ub])
                continue
            else:
                # don't know
                raise Exception("unspecified behaviour")
        else:
            output.append(first[dim])
    return np.array(output)


def main():
    with open("../runs/safe_domains.json", 'r') as f:
        frozen_safe = jsonpickle.decode(f.read())
    # with open("../runs/unsafe_domains.json", 'r') as f:
    #     frozen_unsafe=jsonpickle.decode(f.read())
    # with open("../runs/ignore_domains.json", 'r') as f:
    #     frozen_ignore=jsonpickle.decode(f.read())
    frozen_safe = np.stack(frozen_safe).take(range(10), axis=0)
    new_list = frozen_safe
    new_list_mean = np.mean(new_list, axis=2)
    new_list_sorted_indices = np.lexsort((new_list_mean[:, 3], new_list_mean[:, 2], new_list_mean[:, 1], new_list_mean[:, 0]), axis=0)
    for i in range(5):
        new_list = merge_list(new_list, new_list_mean, new_list_sorted_indices)
        new_list_mean = np.mean(new_list, axis=2)
        new_list_sorted_indices = np.lexsort((new_list_mean[:, 3], new_list_mean[:, 2], new_list_mean[:, 1], new_list_mean[:, 0]), axis=0)
    print(new_list.shape)


if __name__ == '__main__':
    main()
