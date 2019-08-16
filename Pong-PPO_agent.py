#!/usr/bin/env python
# coding: utf-8
import json
import os
from datetime import datetime

import gym
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import constants
import pong_utils
from agents.Agent_PPO import AgentPPO
from networks.Policy import Policy2
from parallelEnv import parallelEnv

COST = "cost"
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
current_time = now.strftime('%b%d_%H-%M-%S')
device = pong_utils.device
print("using device: ", device)
env = gym.make('PongDeterministic-v4')

# plt.show()
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

policy = Policy2().to(device)
policy.test(device)
# policy = pong_utils.Policy().to(device)
# policy = torch.load('PPO2.policy')
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
ending_condition = lambda result: result['mean'] >= 30.0
config = {constants.optimiser: optimizer,
          constants.model: policy,
          constants.n_episodes: 2000,
          constants.max_t: 200,
          constants.epsilon: 0.1,
          constants.beta: 0.01,
          constants.input_dim: (1, 1),
          constants.output_dim: (1, 1),
          constants.discount: 0.99,
          constants.device: device,
          constants.sgd_iterations: 4,
          constants.ending_condition: ending_condition
          }
agent = AgentPPO(config)
envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)
comment = f"PPO PongDeterministic"
log_dir = os.path.join('runs', current_time + '_' + comment)
os.mkdir(log_dir)
config_file = open(os.path.join(log_dir, "config.json"), "w+")
# magic happens here to make it pretty-printed
config_file.write(json.dumps(config, indent=4, sort_keys=True, default=lambda o: '<not serializable>'))
config_file.close()
writer = SummaryWriter(log_dir=log_dir)

# # Training
agent.train_OpenAI(envs, writer)

pong_utils.play(env, policy, time=200)

# save your policy!
torch.save(policy, 'PPO2.policy')

# load policy if needed
# policy = torch.load('PPO.policy')

# try and test out the solution
# make sure GPU is enabled, otherwise loading will fail
# (the PPO verion can win more often than not)!
#
# policy_solution = torch.load('PPO_solution.policy')
# pong_utils.play(env, policy_solution, time=2000)
