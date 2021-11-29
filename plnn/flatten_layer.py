import torch
from torch import nn as nn


class Flatten(nn.Module):
    """Flatten operation converted as a module to use with Sequential class in pytorch"""

    def forward(self, input: torch.Tensor):
        return input.view(input.size(0), -1)
