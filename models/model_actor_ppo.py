import torch
import torchtest
from torch import nn as nn
from torch.nn import functional as F


class Policy_actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        output: torch.Tensor = F.relu(self.bn1(self.fc1(x)))  # self.bn1(
        output: torch.Tensor = F.relu(self.bn2(self.fc2(output)))  # self.bn2(
        output: torch.Tensor = torch.tanh(self.bn3(self.fc3(output)))  # self.bn2(
        output: torch.Tensor = torch.sigmoid(output)
        # output = self.fc3(output)
        dist = torch.distributions.Normal(output, F.softplus(self.std))
        actions = dist.sample()
        log_prob = dist.log_prob(actions)
        # log_prob = torch.sum(log_prob, dim=-1)
        entropy = dist.entropy()  # torch.sum(dist.entropy(), dim=-1)
        return actions, log_prob, entropy

    def test(self, device='cpu'):
        input = torch.randn(10, 2, requires_grad=False)
        targets = torch.rand(10, 1, requires_grad=False)
        torchtest.test_suite(self, mse_surrogate, torch.optim.Adam(self.parameters()), batch=[input, targets], test_vars_change=True, test_inf_vals=False, test_nan_vals=False, device=device)
        print('All tests passed')


def mse_surrogate(input, target):
    """just for testing that the weights get updated"""
    adv_PPO, entropy = input[1], input[2]
    loss_actor = -torch.mean(adv_PPO) - 0.01 * entropy.mean()
    return loss_actor

if __name__ == '__main__':
    model = Policy_actor(33,4)
    torch.rand()