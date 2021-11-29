import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(state_size, 64))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.BatchNorm1d(64))
        self.layers.append(nn.Linear(64, 64))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.BatchNorm1d(64))
        self.layers.append(nn.Linear(64, action_size))
        # self.layers.append(nn.BatchNorm1d(action_size))
        self.sequential = nn.Sequential(*self.layers)

    #         print(f'parameters={self.parameters()}')

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # output: torch.Tensor = F.relu(self.bn1(self.fc1(state)))  # self.bn1(
        # output: torch.Tensor = F.relu(self.bn2(self.fc2(output)))  # self.bn2(
        # output: torch.Tensor = self.bn3(self.fc3(output))  # self.bn2(
        output = self.sequential(state)
        return output


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.head_v = nn.Linear(32, 1)
        self.head_a = nn.Linear(32, action_size)

    #         print(f'parameters={self.parameters()}')

    def forward(self, state):
        """Build a network that maps state -> action values."""
        foundation = F.relu(self.fc2(F.relu(self.fc1(state))))
        head_a_stream = F.relu(self.head_a(foundation))
        head_v_stream = F.relu(self.head_v(foundation))
        mean_a = head_a_stream.mean()
        Q_sa = head_v_stream + (head_a_stream - mean_a)
        return Q_sa


class TestNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self):
        super(TestNetwork, self).__init__()
        self.layers = []
        linear1 = nn.Linear(2, 2)
        linear1.weight = torch.nn.Parameter(torch.tensor([[2, 3], [1, 1]], dtype=torch.float64), requires_grad=True)
        linear1.bias = None
        self.layers.append(linear1)
        self.layers.append(nn.ReLU())
        linear2 = nn.Linear(2, 1)
        linear2.weight = torch.nn.Parameter(torch.tensor([1, -1], dtype=torch.float64), requires_grad=True)
        linear2.bias = None
        self.layers.append(linear2)
        self.sequential = nn.Sequential(*self.layers)

    def forward(self, state):
        output = self.sequential(state)
        return output


class TestNetwork2(nn.Module):
    """Actor (Policy) Model with 2 outputs"""

    def __init__(self):
        super(TestNetwork2, self).__init__()
        self.layers = []
        linear1 = nn.Linear(2, 2)
        linear1.weight = torch.nn.Parameter(torch.tensor([[2, 3], [1, 1]], dtype=torch.float64), requires_grad=True)
        linear1.bias = None
        self.layers.append(linear1)
        self.layers.append(nn.ReLU())
        linear2 = nn.Linear(2, 2)
        linear2.weight = torch.nn.Parameter(torch.tensor([[1, -1], [1, 1]], dtype=torch.float64), requires_grad=True)
        linear2.bias = None
        self.layers.append(linear2)
        self.sequential = nn.Sequential(*self.layers)

    def forward(self, state):
        output = self.sequential(state)
        return output
