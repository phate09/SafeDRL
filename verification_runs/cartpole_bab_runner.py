import math

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
explorer = DomainExplorer(domain,verification_model, 0)
min_lb0, min_ub0, ub_point0 = explorer.bab()
# min_lb0, min_ub0, ub_point0 = bab(verification_model, domain, 0, epsilon, decision_bound, save=False)
# min_lb1, min_ub1, ub_point1 = bab(verification_model, domain, 1, epsilon, decision_bound, save=False)
first_true = False
second_true = False
neither_true = not first_true and not second_true
if neither_true:
    # split both
    pass
elif first_true:
    # split second
    pass
elif second_true:
    # split first
    pass
else:  # both true
    # terminate
    pass
attempts += 1
if min_lb0 >= 0:
    successes += 1
    last_result = "UNSAT"
elif min_ub0 < 0:
    last_result = "SAT"
    # print(ub_point)
else:
    print("Unknown")  # 18
print(f'\rRunning percentage: {successes / attempts:.02%}, last result:{last_result}', end="")
print(f'Final percentage: {successes / attempts:.02%}')
