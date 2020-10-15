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
from polyhedra.polyhedra_distance import is_separable
import pypoman
import cdd

polyhedron_vertices = [(2, 2), (4, 2), (2, 4)]
# polyhedron_vertices = [np.array([2, 2]), np.array([4, 2]), np.array([2, 4])]
A, b = pypoman.duality.compute_polytope_halfspaces(polyhedron_vertices)
net = Net()
net.load_state_dict(torch.load("model.pt"))
net.eval()
points = np.random.random((10000, 2)) * 5
# points2 = np.random.random((1000,2))*5
result = torch.argmax(net(torch.from_numpy(points).float()), dim=1)
mask_positives = result.bool().detach().numpy()
positives = points[mask_positives]
negatives = points[np.invert(mask_positives)]
# positives=np.concatenate((positives,np.array([(3,4)])))
positives=np.concatenate((positives,np.array([(1,4)])))
fig = utils.scatter_plot(positives, negatives)
fig.add_trace(utils.compute_trace_polygon(polyhedron_vertices))
fig.show()
separable = is_separable(positives, A, b)
print(separable)
