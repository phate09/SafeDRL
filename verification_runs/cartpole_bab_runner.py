import math

import jsonpickle
import numpy as np
import torch.nn

from dqn.dqn_agent import Agent
from models.model_critic_sequential import QNetwork
from plnn.bab_explore import DomainExplorer
from plnn.branch_and_bound import bab
from plnn.verification_network import VerificationNetwork

use_cuda = False
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

theta_threshold_radians = 12 * 2 * math.pi / 360  # maximum angle allowed
x_threshold = 2.4  # maximum distance allowed
domain_raw = np.array([[-x_threshold, x_threshold], [-x_threshold, x_threshold], [-theta_threshold_radians, theta_threshold_radians], [-theta_threshold_radians, theta_threshold_radians]])
agent = Agent(4, 2)
agent.load("/home/edoardo/Development/SafeDRL/runs/Sep19_12-42-51_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_5223.pth")
verification_model = VerificationNetwork(agent.qnetwork_local).to(device)

epsilon = 1e-5
decision_bound = 0
successes = 0
attempts = 0
last_result = ""
domain = torch.from_numpy(domain_raw).float().to(device)
explorer = DomainExplorer(1, domain)
explorer.explore(verification_model)

frozen_safe = jsonpickle.encode([i.cpu().numpy() for i in explorer.safe_domains])
frozen_unsafe = jsonpickle.encode([i.cpu().numpy() for i in explorer.unsafe_domains])
frozen_ignore = jsonpickle.encode([i.cpu().numpy() for i in explorer.ignore_domains])
with open("../runs/safe_domains.json", 'w+') as f:
    f.write(frozen_safe)
with open("../runs/unsafe_domains.json", 'w+') as f:
    f.write(frozen_unsafe)
with open("../runs/ignore_domains.json", 'w+') as f:
    f.write(frozen_ignore)
