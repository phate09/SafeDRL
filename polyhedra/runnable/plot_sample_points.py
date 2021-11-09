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
nn_path = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58"  # safe