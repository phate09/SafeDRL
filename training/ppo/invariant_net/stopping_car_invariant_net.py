import torch
import numpy as np


class StoppingCarInvariantNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.env_size = 3
        weights = np.array([[1, -1, 0]])  # x_lead-x_ego
        linear = torch.nn.Linear(self.env_size, 1)
        linear.weight.data = torch.tensor(weights).float()
        self.net = torch.nn.Sequential(linear, )

    def forward(self, input: torch.Tensor):
        return self.net(input)


if __name__ == '__main__':
    nn = StoppingCarInvariantNet()
    print(nn(torch.tensor([10, 1, 0, 0]).float()))
