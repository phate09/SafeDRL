import os
import time
import gym
import ray
import sympy
from sklearn.linear_model import LogisticRegression
from sympy import Line
from sympy.geometry import Point
import mosaic.hyperrectangle_serialisation as serialisation
import mosaic.utils as utils
import prism.state_storage
import symbolic.unroll_methods as unroll_methods
import utility.domain_explorers_load
from mosaic.hyperrectangle import HyperRectangle_action, HyperRectangle
from prism.shared_rtree import SharedRtree
import numpy as np
import torch

gym.logger.set_level(40)
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
local_mode = False
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_dashboard=True, log_to_driver=False)
serialisation.register_serialisers()
n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
rounding = 2
precision = 10 ** (-rounding)
explorer, verification_model, env, current_interval, state_size, env_class = utility.domain_explorers_load.generatePendulumDomainExplorer(precision, rounding, sym=True)
storage = prism.state_storage.StateStorage()
storage.reset()
rtree = SharedRtree()
rtree.reset(state_size)
current_interval: HyperRectangle = HyperRectangle.from_tuple(((0.35, 0.79), (-1, 1)))

# %% show (approximated) true decision boundary
unroll_methods.check_tree_coverage(True, True, explorer, [current_interval], 8, rounding, rtree, verification_model)
intervals = rtree.tree_intervals()
heatmap = utils.show_plot([x.to_tuple() for x in intervals if x.action], [x.to_tuple() for x in intervals if not x.action])
heatmap.show()

# %% sample 10k points from the area and feed to the nn
samples = np.stack([current_interval.sample() for i in range(10000)])
actions = torch.argmax(verification_model.base_network(torch.from_numpy(samples)), dim=1).detach().numpy()
scatter = utils.scatter_plot([tuple(x) for i,x in enumerate(samples) if bool(actions[i].item())],[tuple(x) for i,x in enumerate(samples) if not bool(actions[i].item())])
scatter.show()
# %% Logistic regression
clf1 = LogisticRegression(random_state=0).fit(samples, actions)
coeff = clf1.coef_
intercept = clf1.intercept_
a, b = coeff[0]
c = intercept[0]
x = sympy.symbols('x')
y = sympy.symbols('y')
classif_line = Line(a.item() * x + b.item() * y + c.item())
bounding_box = np.stack([np.min(samples, 0), np.max(samples, 0)]).transpose()
ymax = (-a * bounding_box[0][1] - c) / b
ymin = (-a * bounding_box[0][0] - c) / b
scatter.add_scatter(x=[bounding_box[0][0],bounding_box[0][1]], y=[ymin,ymax], mode="lines", hoveron="points")
scatter.show()
#%% padding
print(ymax)
perp_line = classif_line.perpendicular_line(classif_line.p1)
p1 = np.array([float(perp_line.p1[0]),float(perp_line.p1[1])])
p2 = np.array([float(perp_line.p2[0]),float(perp_line.p2[1])])
v = p1-p2
u = v/np.linalg.norm(v,ord=1)
next_point = p1+2*u
parallel_line1 = classif_line.parallel_line(classif_line.p1+Point(0,0.2))
a1,b1,c1 = parallel_line1.coefficients
ymax1 = float((-a1 * bounding_box[0][1] - c1) / b1)
ymin1 =float( (-a1 * bounding_box[0][0] - c1) / b1)
scatter.add_scatter(x=[bounding_box[0][0],bounding_box[0][1]], y=[ymin1,ymax1], mode="lines", hoveron="points")
parallel_line2 = classif_line.parallel_line(classif_line.p1+Point(0,-0.2))
a2,b2,c2 = parallel_line2.coefficients
ymax2 = float((-a2 * bounding_box[0][1] - c2) / b2)
ymin2 =float( (-a2 * bounding_box[0][0] - c2) / b2)
scatter.add_scatter(x=[bounding_box[0][0],bounding_box[0][1]], y=[ymin2,ymax2], mode="lines", hoveron="points")
scatter.show()
# classif_line.parallel_line()

# %% Custom loss function
from scipy.optimize import minimize
from scipy.stats import logistic


def mean_absolute_percentage_error(y_pred, y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
    return (-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)).mean()


loss_function = mean_absolute_percentage_error


def objective_function(beta, X, Y):
    bias = np.ones((len(X), 1), dtype=X.dtype)
    error = loss_function(logistic.cdf(np.matmul(np.append(X, bias, axis=1), beta)), Y) + 0.001 * sum(np.array(beta) ** 2)
    return (error)


# You must provide a starting point at which to initialize
# the parameter search space
beta_init = np.array([1] * (samples.shape[1] + 1))
intercept_init = 0
result = minimize(objective_function, beta_init, args=(samples, actions), method='BFGS', options={'maxiter': 5000})

# The optimal values for the input parameters are stored
# in result.x
beta_hat = result.x
a, b, c = beta_hat
scatter = utils.scatter_plot([tuple(x) for i, x in enumerate(samples) if bool(actions[i].item())], [tuple(x) for i, x in enumerate(samples) if not bool(actions[i].item())])
bounding_box = np.stack([np.min(samples, 0), np.max(samples, 0)]).transpose()
ymax = (-a * bounding_box[0][1] - c) / b
ymin = (-a * bounding_box[0][0] - c) / b
scatter.add_scatter(x=[bounding_box[0][0], bounding_box[0][1]], y=[ymin, ymax], mode="lines", hoveron="points")
scatter.show()
print(beta_hat)
