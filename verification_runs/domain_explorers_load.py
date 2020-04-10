import os
import numpy as np
import torch
from dqn.dqn_agent import Agent
from plnn.bab_explore import DomainExplorer
from plnn.verification_network import VerificationNetwork
from symbolic.cartpole_abstract import CartPoleEnv_abstract
from symbolic.pendulum_abstract import PendulumEnv_abstract


def generateCartpoleDomainExplorer(precision=1e-2, rounding=6):
    use_cuda = False
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
    env_class = CartPoleEnv_abstract
    env = env_class()
    s = env.reset()
    state_size = 4
    agent = Agent(state_size, 2)
    agent.load(os.path.expanduser("~/Development") + "/SafeDRL/save/Sep19_12-42-51_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_5223.pth")
    verification_model = VerificationNetwork(agent.qnetwork_local).to(device)
    explorer = DomainExplorer(1, device, precision=precision, rounding=rounding)
    return explorer, verification_model, env, s, state_size, env_class


def generatePendulumDomainExplorer(precision=1e-2, rounding=6):
    use_cuda = False
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
    env_class = PendulumEnv_abstract
    env = env_class()
    s = env.reset()
    state_size = 2
    agent = Agent(state_size, 2)
    agent.load(os.path.expanduser("~/Development") + "/SafeDRL/save/Pendulum_Apr07_12-17-45_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_final.pth")
    verification_model = VerificationNetwork(agent.qnetwork_local).to(device)
    explorer = DomainExplorer(1, device, precision=precision, rounding=rounding)
    return explorer, verification_model, env, s, state_size, env_class
