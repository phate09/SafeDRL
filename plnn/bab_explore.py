import ray
import bisect
import time

import jsonpickle
import numpy as np
import torch
from jsonpickle import json
from itertools import cycle

from plnn.branch_and_bound import CandidateDomain


class DomainExplorer():
    def __init__(self, safe_property_index: int, domain: torch.Tensor):
        """

        :param domain: the domain to explore for abstract interpretations
        :param safe_property_index: the index of the property. This property corresponds to "always on"
        The algorithm will check for lumps in the state space where the given property is always true
        """
        self.initial_domain = domain
        self.safe_property_index = safe_property_index
        # self.domains = []
        self.safe_domains = []
        self.safe_area = 0
        self.unsafe_domains = []
        self.unsafe_area = 0
        self.ignore_domains = []
        self.ignore_area = 0
        self.nb_input_var = domain.size()[0]  # size of the domain, second dimension is [lb,ub]
        self.domain_lb = domain.select(-1, 0)
        self.domain_width = domain.select(-1, 1) - domain.select(-1, 0)
        normed_domain = torch.stack((torch.zeros(self.nb_input_var), torch.ones(self.nb_input_var)), 1)  # self.domains = [normed_domain]

    def explore(self, net, domains=None, n_workers=8, precision=1e-3, min_area=1e-6, debug=True):
        # eps = 1e-3
        # precision = 1e-3  # does not allow precision of any dimension to go under this amount
        # min_area = 1e-5  # minimum area of the domain for it to be considered
        global_min_area = float("inf")
        shortest_dimension = float("inf")
        if not ray.is_initialized():
            ray.init()

        message_queue = []
        if domains is None:
            normed_domain = torch.stack((torch.zeros(self.nb_input_var), torch.ones(self.nb_input_var)), 1)
            domain_id = ray.put((normed_domain, None, None))
            message_queue.append(domain_id)
        else:  # if given a list of normed domains then prefills the queue
            for domain in domains:
                normed_domain = np.copy(domain)
                # widths = normed_domain[:,1]-normed_domain[:,0]
                # normed_domain[:,0] = normed_domain[:,0]-self.domain_lb.cpu().numpy()
                # normed_domain[:,1] = widths/self.domain_width.cpu().numpy()
                domain_id = ray.put((torch.from_numpy(normed_domain).float(), None, None))
                message_queue.append(domain_id)
        # a=ParallelExplorer.remote(self.domain_lb, self.domain_width, self.nb_input_var, net)
        explorers = cycle([ParallelExplorer.remote(self.domain_lb, self.domain_width, self.nb_input_var, net) for i in range(n_workers)])
        while len(message_queue) > 0:
            explore, safe, unsafe = ray.get(message_queue.pop(0))
            if safe is not None:
                self.safe_domains.append(safe)
                self.safe_area += self.area(safe)
            if unsafe is not None:
                self.unsafe_domains.append(unsafe)
                self.unsafe_area += self.area(unsafe)
            if explore is not None:
                # Genearate new, smaller (normalized) domains using box split.
                ndoms = self.box_split(explore)
                for i, ndom_i in enumerate(ndoms):
                    area = self.area(ndom_i)
                    length = self.max_length(ndom_i)
                    global_min_area = min(area, global_min_area)
                    shortest_dimension = min(length, shortest_dimension)
                    if length < precision or area < min_area:
                        # too short or too small
                        self.ignore_domains.append(ndom_i)
                        self.ignore_area += self.area(ndom_i)
                    else:
                        message_queue.append(next(explorers).bab.remote(ndom_i, self.safe_property_index))  # starts on the next available explorer
            if debug:
                print(f"\rqueue length : {len(message_queue)}, # safe domains: {len(self.safe_domains)}, # unsafe domains: {len(self.unsafe_domains)}, abstract areas: [unknown:{1 - (self.safe_area + self.unsafe_area + self.ignore_area):.3%} --> safe:{self.safe_area:.3%}, unsafe:{self.unsafe_area:.3%}, ignore:{self.ignore_area:.3%}], shortest_dim:{shortest_dimension}, min_area:{global_min_area:.8f}", end="")
        if debug:
            print("\n")
        # prepare stats
        stats = {}
        total_area = (self.safe_area + self.unsafe_area + self.ignore_area)
        if total_area == 0:
            total_area = 1
        stats["safe_relative_percentage"] = self.safe_area / total_area
        stats["unsafe_relative_percentage"] = self.unsafe_area / total_area
        stats["ignore_relative_percentage"] = self.ignore_area / total_area
        stats["n_states"] = len(self.safe_domains) + len(self.unsafe_domains) + len(self.ignore_domains)
        stats["n_safe"] = len(self.safe_domains)
        stats["n_unsafe"] = len(self.unsafe_domains)
        stats["n_ignore"] = len(self.ignore_domains)
        return stats

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
    def area(domain: torch.Tensor):
        '''
        Compute the area of the domain
        '''
        dom_sides = domain.select(1, 1) - domain.select(1, 0)
        dom_area = dom_sides.prod()
        return dom_area

    def find_groups(self, net):
        pass

    def bab(self):
        with torch.no_grad():
            eps = 1e-3
            decision_bound = 0
            # This counter is used to decide when to prune domains
            while len(self.domains) > 0:
                selected_candidate_domain: torch.Tensor = self.domains.pop(0)
                print(f'Splitting domain')
                # Genearate new, smaller (normalized) domains using box split.
                ndoms = self.box_split(selected_candidate_domain)
                print(f"# domains : {len(self.domains)}, # abstract domains: {len(self.safe_domains)}, abstract area: {self.safe_area:.3%}")
                for i, ndom_i in enumerate(ndoms):
                    # Find the upper and lower bounds on the minimum in dom_i
                    dom_i = self.domain_lb.unsqueeze(dim=1) + self.domain_width.unsqueeze(dim=1) * ndom_i
                    dom_ub, dom_lb = self.net.get_boundaries(dom_i, self.safe_property_index, False)

                    assert dom_lb <= dom_ub, "lb must be lower than ub"
                    if dom_ub < decision_bound:
                        # discard
                        # print("discard")
                        continue
                    if dom_ub - dom_lb < eps:  # todo what do in this case?
                        continue
                    if dom_lb >= decision_bound:
                        # keep
                        self.safe_domains.append(ndom_i)
                        self.safe_area += self.area(ndom_i).item()
                        pass
                    if dom_lb <= decision_bound <= dom_ub:
                        # explore
                        print(f'dom_ub:{dom_ub}')
                        print(f'dom_lb:{dom_lb}')
                        # candidate_domain_to_add = CandidateDomain(lb=dom_lb, ub=dom_ub, dm=ndom_i)
                        # self.add_domain(candidate_domain_to_add, self.domains)
                        self.domains.append(ndom_i)  # O(1)
                        # self.domains.insert(0,candidate_domain_to_add) #O(1)
                        pass
            print(self.safe_domains)
        return self.safe_domains

    @staticmethod
    def max_length(domain):
        diff = domain[:, 1] - domain[:, 0]
        edgelength, dim = torch.max(diff, 0)

        # Unwrap from tensor containers
        edgelength = edgelength.item()
        return edgelength

    @staticmethod
    def check_min_area(domain, min_area=float("inf")):
        return DomainExplorer.area(domain) < min_area

    @staticmethod
    def box_split(domain):
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
    def add_domain(candidate, domains):
        """
        Use binary search to add the new domain `candidate`
        to the candidate list `domains` so that `domains` remains a sorted list.
        """
        bisect.insort_left(domains, candidate)


@ray.remote
class ParallelExplorer:
    def __init__(self, domain_lb, domain_width, nb_input_var, net):
        self.net = net
        self.domain_width = domain_width
        self.domain_lb = domain_lb
        self.nb_input_var = nb_input_var

    def bab(self, normed_domain, safe_property_index):
        eps = 1e-3
        dom_i = self.domain_lb.unsqueeze(dim=1) + self.domain_width.unsqueeze(dim=1) * normed_domain
        dom_ub, dom_lb = self.net.get_boundaries(dom_i, safe_property_index, False)
        assert dom_lb <= dom_ub, "lb must be lower than ub"
        if dom_ub < 0:
            # discard
            return None, None, normed_domain
        if dom_ub - dom_lb < eps:
            # ignore
            return None, None, None
        if dom_lb >= 0:
            # keep
            return None, normed_domain, None
        if dom_lb <= 0 <= dom_ub:
            # explore
            return normed_domain, None, None
