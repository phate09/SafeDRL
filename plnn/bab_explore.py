import bisect

import numpy as np
import torch

from plnn.branch_and_bound import CandidateDomain


class DomainExplorer():
    def __init__(self, domain: torch.Tensor,net, safe_property_index: int):
        """

        :param domain: the domain to explore for abstract interpretations
        :param safe_property_index: the index of the property. This property corresponds to "always on"
        The algorithm will check for lumps in the state space where the given property is always true
        """
        self.initial_domain = domain
        self.safe_property_index = safe_property_index
        self.domains = []
        self.nb_input_var = domain.size()[0]  # size of the domain, second dimension is [lb,ub]
        self.domain_lb = domain.select(-1, 0)
        self.net = net
        self.domain_width = domain.select(-1, 1) - domain.select(-1, 0)
        global_ub = self.net.get_upper_bound2(self.initial_domain, self.safe_property_index,False)
        global_lb = self.net.get_lower_bound(self.initial_domain, self.safe_property_index, False)
        assert global_lb <= global_ub, "lb must be lower than ub"
        print(f'global_ub:{global_ub}')
        print(f'global_lb:{global_lb}')
        self.global_ub = global_ub
        self.global_lb = global_lb
        normed_domain = torch.stack((torch.zeros(self.nb_input_var),torch.ones(self.nb_input_var)), 1)
        # Use objects of type CandidateDomain to store domains with their bounds.
        candidate_domain = CandidateDomain(lb=global_lb, ub=global_ub, dm=normed_domain)
        self.domains = [candidate_domain]

    def _pick_next_domain(self,eps):
        threshold = self.global_ub - eps
        return self._pick_out(self.domains, threshold)

    def find_groups(self, net):
        pass

    def bab(self):
        eps = 1e-3
        decision_bound = 0
        # This counter is used to decide when to prune domains
        prune_counter = 0
        while self.global_ub - self.global_lb > eps:
            selected_candidate_domain = self._pick_next_domain(eps)
            print(f'Splitting domain')
            # Genearate new, smaller (normalized) domains using box split.
            ndoms = self.box_split(selected_candidate_domain.domain)

            for i, ndom_i in enumerate(ndoms):
                print(f'Domain #{i}')
                # Find the upper and lower bounds on the minimum in dom_i
                dom_i = self.domain_lb.unsqueeze(dim=1) + self.domain_width.unsqueeze(dim=1) * ndom_i
                dom_ub = self.net.get_upper_bound2(dom_i, self.safe_property_index,False)
                dom_lb = self.net.get_lower_bound(dom_i, self.safe_property_index, False)
                print(f'dom_ub:{dom_ub}')
                print(f'dom_lb:{dom_lb}')
                assert dom_lb <= dom_ub, "lb must be lower than ub"
                # Update the global upper if the new upper bound found is lower.
                if dom_ub < self.global_ub:
                    self.global_ub = dom_ub
                    print(f'updated global_ub to {self.global_ub}')
                # Add the domain to our current list of domains if its lowerbound
                # is less than the global upperbound.
                if dom_lb < self.global_ub:
                    candidate_domain_to_add = CandidateDomain(lb=dom_lb,ub=dom_ub,dm=ndom_i)
                    self.add_domain(candidate_domain_to_add, self.domains)
                    prune_counter += 1
            # Prune domains whose lowerbounds are larger than or equal to the
            # global upperbound.
            # If domains list is larger than 100 items and if prune_counter has
            # reached a threshold, prune domains that we no longer need.
            if prune_counter >= 100 and len(self.domains) >= 100:
                # Remove domains with dom_lb >= global_ub
                self.domains = self.prune_domains(self.domains, self.global_ub - eps)
                prune_counter = 0

                # Do a pass over all the remaining domains to evaluate how much of
                # the input is there left to prune
                # print_remaining_domain(domains)
                # print(f"Current: lb: {global_lb}\t ub: {global_ub}")
            # Update the global lower bound with a lower bound that belongs
            # to a domain in the updated list "domains" .
            # TODO: This current implementation is only a global lower bound
            #       if we sort domains by lower_bound.
            if len(self.domains) > 0:
                print(f'----------------------------------')
                print(f'remaining domains: {len(self.domains)}')
                # print(f'lbs:{[i.lower_bound for i in self.domains]}')
                self.global_lb = self.domains[0].lower_bound
                # print(f'first domain lb: {domains[0].lower_bound}')
                # global_lb = max([i.lower_bound for i in domains])
                print(f'global_ub:{self.global_ub}')
                print(f'global_lb:{self.global_lb}')
            else:
                # if there is no more domains, we have pruned them all.
                self.global_lb = self.global_ub - eps

            # Stopping criterion
            if self.global_lb >= decision_bound:
                break
            elif self.global_ub < decision_bound:
                break
        print(f'Reached a decision')
        return self.global_lb, self.global_ub

    @staticmethod
    def _box_split():
        pass

    @staticmethod
    def _pick_out(domains, threshold):
        """
        Pick the first domain in the `domains` sequence
        that has a lower bound lower than `threshold`.

        Any domain appearing before the chosen one but having a lower_bound greater
        than the threshold is discarded.

        Returns: Non prunable CandidateDomain with the lowest reference_value.
        """
        assert len(domains) > 0, "The given domains list is empty."
        while True:
            assert len(domains) > 0, "No domain left to pick from."
            selected_candidate_domain = domains.pop(0)
            if selected_candidate_domain.lower_bound < threshold:
                break

        return selected_candidate_domain

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
