import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.torch.examples.train_example import LinearDataset
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from polyhedra.runnable.toy_2d.train_2d_classifier import Net
import mosaic.utils as utils

net = Net()
net.load_state_dict(torch.load("model.pt"))
net.eval()
points = np.random.random((10000, 2)) * 5
result = torch.argmax(net(torch.from_numpy(points).float()),dim=1)
mask_positives = result.bool().detach().numpy()
positives = points[mask_positives]
negatives = points[np.invert(mask_positives)]
fig = utils.scatter_plot(positives,negatives)
fig.show()
