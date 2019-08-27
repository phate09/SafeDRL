import datetime
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.envs.classic_control import CartPoleEnv
from tensorboardX import SummaryWriter

from dqn_agent import Agent
from utility.Scheduler import Scheduler
env = CartPoleEnv()  # gym.make("CartPole-v0")


