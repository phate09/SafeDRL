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

net = Net()
net.load_state_dict(torch.load("model.pt"))
net.eval()

