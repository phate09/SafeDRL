import os
import numpy as np
import torch
from agents.dqn.dqn_agent import Agent
from agents.ray_utils import load_sequential_from_ray, get_pendulum_ppo_agent
from plnn.bab_explore import DomainExplorer
from plnn.bab_explore_sym import SymbolicDomainExplorer
from plnn.verification_network import VerificationNetwork
from environment.cartpole_abstract import CartPoleEnv_abstract
from environment.pendulum_abstract import PendulumEnv_abstract
from plnn.verification_network_sym import SymVerificationNetwork


def generateCartpoleDomainExplorer(precision=1e-2, rounding=6, sym=False):
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
    if not sym:
        verification_model = VerificationNetwork(agent.qnetwork_local).to(device)
        explorer = DomainExplorer(1, device, precision=precision, rounding=rounding)
    else:
        verification_model = SymVerificationNetwork(agent.qnetwork_local.sequential.cpu().double())
        explorer = SymbolicDomainExplorer(1, device, precision=precision, rounding=rounding)
    return explorer, verification_model, env, s, state_size, env_class


def generatePendulumDomainExplorer(file_name,precision=1e-2, rounding=6, sym=False):
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
    # agent.load(os.path.expanduser("~/Development") + "/SafeDRL/save/Pendulum_Apr07_12-17-45_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_final.pth")
    agent.load(file_name)
    if not sym:
        verification_model = VerificationNetwork(agent.qnetwork_local.sequential.cpu().double()).to(device)
        explorer = DomainExplorer(1, device, precision=precision, rounding=rounding)
    else:
        verification_model = SymVerificationNetwork(agent.qnetwork_local.sequential.cpu().double())
        explorer = SymbolicDomainExplorer(1, device, precision=precision, rounding=rounding)
    return explorer, verification_model, env, s, state_size, env_class


def generatePendulumDomainExplorerPPO(file_name,precision=1e-2, rounding=6, sym=False):
    use_cuda = False
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
    env_class = PendulumEnv_abstract
    env = env_class()
    s = env.reset()
    state_size = 2
    #os.path.expanduser("~/Development") + "/SafeDRL/save/PPO_PendulumEnv_2020-09-18_11-23-17wpwqe3zd/checkpoint_25/checkpoint-25"
    sequential_nn = load_sequential_from_ray(file_name, get_pendulum_ppo_agent())
    sequential_nn.add_module("softmax", torch.nn.Softmax())  # adds the softmax at the end
    if not sym:
        verification_model = VerificationNetwork(sequential_nn).to(device)
        explorer = DomainExplorer(1, device, precision=precision, rounding=rounding)
    else:
        verification_model = SymVerificationNetwork(sequential_nn).to(device)
        explorer = SymbolicDomainExplorer(1, device, precision=precision, rounding=rounding)
    return explorer, verification_model, env, s, state_size, env_class
