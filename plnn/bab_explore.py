import bisect
import time
from typing import List, Tuple

import jsonpickle
import numpy as np
import ray
import torch

import mosaic.utils


class DomainExplorer:
    def __init__(self, safe_property_index: int, domain: torch.Tensor, device: torch.device, precision, rounding: int):
        """

        :param domain: the domain to explore for abstract interpretations
        :param safe_property_index: the index of the property. This property corresponds to "always on"
        The algorithm will check for lumps in the state space where the given property is always true
        """
        self.device = device
        # self.initial_domain = domain
        self.safe_property_index = safe_property_index
        # self.domains = []
        self.safe_domains = []
        self.safe_area = 0
        self.unsafe_domains = []
        self.unsafe_area = 0
        self.ignore_domains = []
        self.ignore_area = 0
        # self.nb_input_var = domain.size()[0]  # size of the domain, second dimension is [lb,ub]
        # self.domain_lb = domain.select(-1, 0)
        # self.domain_width = domain.select(-1, 1) - domain.select(-1, 0)
        self.precision_constraints = [precision, precision, precision, precision]  # DomainExplorer.generate_precision(self.domain_width, precision)
        self.rounding = rounding

    def explore(self, net, domains: List[np.ndarray], n_workers: int, debug=True):
        # eps = 1e-3
        # precision = 1e-3  # does not allow precision of any dimension to go under this amount
        # min_area = 1e-5  # minimum area of the domain for it to be considered
        global_min_area = float("inf")
        shortest_dimension = float("inf")
        message_queue = []
        queue = []  # queue of domains to explore
        total_area = 0
        self.reset()  # reset statistics
        for domain in domains:
            # normed_domain = np.copy(domain)
            tensor = torch.tensor(domain, dtype=torch.float32)
            queue.append(tensor)
            total_area += mosaic.utils.area_tensor(tensor)
        last_save = time.time()
        while len(queue) > 0:
            while len(queue) > 0:
                for i in range(min(len(queue), n_workers)):
                    global_min_area, shortest_dimension = self.start_explore_one_domain(global_min_area, message_queue, net, queue, shortest_dimension)
                while len(message_queue) > n_workers:
                    message_queue = self.process_one_queue_element(message_queue, queue)
                if debug:
                    print(f"\rqueue length : {len(queue)}, # safe domains: {len(self.safe_domains)}, # unsafe domains: {len(self.unsafe_domains)}, abstract areas: [unknown:"
                          f"{1 - (self.safe_area + self.unsafe_area + self.ignore_area) / total_area:.3%} --> safe:{self.safe_area / total_area:.3%}, unsafe:{self.unsafe_area / total_area:.3%}, ignore:{self.ignore_area / total_area:.3%}]",
                          end="")
                if time.time() - last_save > 60 * 10:  # every 5 minutes
                    # save the queue and the safe/unsafe domains
                    with open("./save/safe_domains.json", 'w+') as f:
                        f.write(jsonpickle.encode(self.safe_domains))
                    with open("./save/unsafe_domains.json", 'w+') as f:
                        f.write(jsonpickle.encode(self.unsafe_domains))
                    with open("./save/queue.json", 'w+') as f:
                        f.write(jsonpickle.encode(queue))
                    # print(f"Saved at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                    last_save = time.time()
            while len(message_queue) > 0:
                message_queue = self.process_one_queue_element(message_queue, queue)
        if debug:
            print("\n")
        # save the queue and the safe/unsafe domains
        with open("./save/safe_domains.json", 'w+') as f:
            f.write(jsonpickle.encode(self.safe_domains))
        with open("./save/unsafe_domains.json", 'w+') as f:
            f.write(jsonpickle.encode(self.unsafe_domains))
        with open("./save/queue.json", 'w+') as f:
            f.write(jsonpickle.encode(queue))
        print(f"Saved at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        # prepare stats
        stats = dict()
        # total_area = (self.safe_area + self.unsafe_area + self.ignore_area)
        # if total_area == 0:
        #     total_area = 1
        stats["safe_relative_percentage"] = self.safe_area / total_area
        stats["unsafe_relative_percentage"] = self.unsafe_area / total_area
        stats["ignore_relative_percentage"] = self.ignore_area / total_area
        stats["n_states"] = len(self.safe_domains) + len(self.unsafe_domains) + len(self.ignore_domains)
        stats["n_safe"] = len(self.safe_domains)
        stats["n_unsafe"] = len(self.unsafe_domains)
        stats["n_ignore"] = len(self.ignore_domains)
        return stats

    hasIgnored = False

    def start_explore_one_domain(self, global_min_area, message_queue, net, queue, shortest_dimension):
        normed_domain = queue.pop(0)
        # Genearate new, smaller (normalized) domains using box split.
        ndoms = self.box_split(normed_domain, self.rounding)
        for i, ndom_i in enumerate(ndoms):
            area = mosaic.utils.area_tensor(ndom_i)
            length, dim = self.max_length(ndom_i)
            global_min_area = min(area, global_min_area)
            shortest_dimension = min(length, shortest_dimension)
            if length < self.precision_constraints[dim]:  # or area < min_area:
                # too short or too small
                # approximate the interval to the closest datapoint and determine if it is safe or not
                action_values = self.assign_approximate_action(net, ndom_i)
                value, index = torch.max(action_values, dim=0)
                if index == self.safe_property_index:
                    self.safe_domains.append(ndom_i)
                    self.safe_area += area
                else:
                    self.unsafe_domains.append(ndom_i)
                    self.unsafe_area += area
                if not self.hasIgnored:
                    # print("The value has been ignored succesfully")
                    self.hasIgnored = True
            else:
                # queue up the next split at the beginning of the list
                message_queue.insert(0, bab_remote.remote(ndom_i, self.safe_property_index, net))
        return global_min_area, shortest_dimension

    def process_one_queue_element(self, message_queue, queue):
        message_ready, message_queue = ray.wait(message_queue)  # retrieve the next ready item
        explore, safe, unsafe = ray.get(message_ready[0])
        if safe is not None:
            self.safe_domains.append(safe)
            self.safe_area += mosaic.utils.area_tensor(safe)
        if unsafe is not None:
            self.unsafe_domains.append(unsafe)
            self.unsafe_area += mosaic.utils.area_tensor(unsafe)
        if explore is not None:
            queue.insert(0, explore)
        return message_queue

    def assign_approximate_action(self, net: torch.nn.Module, normed_domain) -> torch.Tensor:
        """Used for assigning an action value to an interval that is so small it gets approximated to the closest single datapoint according to #precision field"""
        approximate_domain = DomainExplorer.approximate_to_single_datapoint(normed_domain, self.precision_constraints)
        approximate_domain = approximate_domain.to(self.device)
        outcome = net(approximate_domain)
        return outcome

    @staticmethod
    def approximate_to_single_datapoint(normed_domain: torch.Tensor, precisions: List[float]) -> torch.Tensor:
        # dom_i = domain_lb.unsqueeze(dim=1) + domain_width.unsqueeze(dim=1) * normed_domain
        mean_dom = torch.mean(normed_domain, dim=1)
        results = []
        for i, value in enumerate(mean_dom):
            # result = mosaic.utils.custom_rounding(value.item(), 3, precisions[i])
            result = float(value.item())
            results.append(result)
        return torch.tensor(results)

    def reset(self):
        self.safe_domains = []
        self.safe_area = 0
        self.unsafe_domains = []
        self.unsafe_area = 0
        self.ignore_domains = []
        self.ignore_area = 0

    def save(self, path):
        f = open(path, 'w')
        frozen = jsonpickle.encode(self)
        f.write(frozen)

    @staticmethod
    def load(path):
        f = open(path)
        json_str = f.read()
        thawed: DomainExplorer = jsonpickle.decode(json_str)
        return thawed

    @staticmethod
    def generate_precision(domain_width: torch.Tensor, precision_constraint=1e-3):
        """Generate the dictionary of normalised precision to use to constrain the splitting of intervals"""
        precisions = []
        for value in domain_width:
            precision = precision_constraint / value.item()
            precisions.append(precision)
        return precisions

    def find_groups(self, net):
        pass

    @staticmethod
    def max_length(domain):
        diff = domain[:, 1] - domain[:, 0]
        edgelength, dim = torch.max(diff, 0)

        # Unwrap from tensor containers
        edgelength = edgelength.item()
        dim = dim.item()
        return edgelength, dim

    @staticmethod
    def check_min_area(domain, min_area=float("inf")):
        return mosaic.utils.area_tensor(domain) < min_area

    @staticmethod
    def box_split(domain, rounding: int):
        """
        Use box-constraints to split the input domain.
        Split by dividing the domain into two from its longest edge.
        Assumes a rectangular domain, which is aligned with the cartesian
        coordinate frame.

        `domain`:  A 2d tensor whose rows contain lower and upper limits
                   of the corresponding dimension.
        Returns: A list of sub-domains represented as 2d tensors.
        """
        # Find the longest edge by checking the difference of lower and upper
        # limits in each dimension.
        diff = domain[:, 1] - domain[:, 0]
        edgelength, dim = torch.max(diff, 0)

        # Unwrap from tensor containers
        edgelength = edgelength.item()
        dim = dim.item()

        # Now split over dimension dim:
        half_length = edgelength / 2

        # dom1: Upper bound in the 'dim'th dimension is now at halfway point.
        dom1 = domain.clone()
        dom1[dim, 1] -= half_length

        # dom2: Lower bound in 'dim'th dimension is now at haflway point.
        dom2 = domain.clone()
        dom2[dim, 0] += half_length

        sub_domains = [dom1, dom2]

        return sub_domains

    @staticmethod
    def box_split_tuple(domain: Tuple[Tuple[float, float]], rounding: int) -> List[Tuple[Tuple[float, float]]]:
        domain_array = np.array(domain)
        diff = domain_array[:, 1] - domain_array[:, 0]
        edgelength = np.max(diff, 0).item()
        dim = np.argmax(diff, 0).item()
        half_length = edgelength / 2
        dom1 = domain_array.copy()
        dom1[dim, 1] -= half_length
        dom2 = domain_array.copy()
        dom2[dim, 0] += half_length
        sub_domains = [mosaic.utils.array_to_tuple(dom1), mosaic.utils.array_to_tuple(dom2)]
        return sub_domains

    @staticmethod
    def add_domain(candidate, domains):
        """
        Use binary search to add the new domain `candidate`
        to the candidate list `domains` so that `domains` remains a sorted list.
        """
        bisect.insort_left(domains, candidate)


@ray.remote
def bab_remote(normed_domain, safe_property_index, net):
    eps = 1e-6
    dom_ub, dom_lb = net.get_boundaries(normed_domain, safe_property_index, False)
    assert dom_lb <= dom_ub, "lb must be lower than ub"
    if dom_ub < 0:
        # discard
        return None, None, normed_domain
    if dom_lb >= 0:
        # keep
        return None, normed_domain, None
    # if dom_ub - dom_lb < eps:
    #     # ignore
    #     return None, None, None
    if dom_lb <= 0 <= dom_ub:
        # explore
        return normed_domain, None, None


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper
