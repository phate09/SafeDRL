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
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    #         print(f'parameters={self.parameters()}')

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(state)))))


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
