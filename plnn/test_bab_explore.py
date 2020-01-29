import math
from unittest import TestCase

import torch

from mosaic.utils import custom_rounding
from plnn.bab_explore import DomainExplorer
from plnn.verification_network import VerificationNetwork
from verification_runs.cartpole_bab_load import generateCartpoleDomainExplorer


class TestDomainExplorer(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.verification_model: VerificationNetwork
        cls.explorer, cls.verification_model = generateCartpoleDomainExplorer()

    def test_precision(self):
        domain_width = self.explorer.domain_width
        precisions = DomainExplorer.generate_precision(domain_width)
        print()
        print(domain_width)
        print(precisions)
        assert math.isclose(precisions[0], 0.00020833332505491078)
        assert math.isclose(precisions[1], 0.00020833332505491078)
        assert math.isclose(precisions[2], 0.0023873240799450854)
        assert math.isclose(precisions[3], 0.0023873240799450854)

    def test_precision2(self):
        precisions = DomainExplorer.generate_precision(self.explorer.domain_width)
        normed_domain = torch.tensor([[1, 1 - precisions[0]], [1, 1 - precisions[1]], [1, 1 - precisions[2]], [1, 1 - precisions[3]]])
        output = DomainExplorer.approximate_to_single_datapoint(normed_domain, self.explorer.domain_lb, self.explorer.domain_width, precisions)

    def test_precision3(self):
        normed_domain = torch.tensor([[1, 1 - self.explorer.precision_constraints[0]], [1, 1 - self.explorer.precision_constraints[1]], [1, 1 - self.explorer.precision_constraints[2]], [1, 1 - self.explorer.precision_constraints[3]]])
        net: torch.nn.Module = self.verification_model.base_network
        action = self.explorer.assign_approximate_action(net, normed_domain)

    def test_custom_rounding(self):
        result = custom_rounding(0.9998, 3, 0.00020833332505491078)
        print(result)
        result = custom_rounding(0.99, 3, 0.00020833332505491078)
        print(result)