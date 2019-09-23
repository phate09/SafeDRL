import math

import jsonpickle
import numpy as np
import torch

from plnn.bab_explore import DomainExplorer
from plnn.verification_network import VerificationNetwork
from verification_runs.aggregate_abstract_domain import aggregate

from dqn.dqn_agent import Agent


def main():
    with open("../runs/safe_domains.json", 'r') as f:
        frozen_safe = jsonpickle.decode(f.read())
    with open("../runs/unsafe_domains.json", 'r') as f:
        frozen_unsafe = jsonpickle.decode(f.read())
    with open("../runs/ignore_domains.json", 'r') as f:
        frozen_ignore = jsonpickle.decode(f.read())
    frozen_safe = np.stack(frozen_safe)  # .take(range(10), axis=0)
    frozen_unsafe = np.stack(frozen_unsafe)  # .take(range(10), axis=0)
    frozen_ignore = np.stack(frozen_ignore)  # .take(range(10), axis=0)
    aggregated_safe = aggregate(frozen_safe)
    print(f"safe {{{frozen_safe.shape} --> {aggregated_safe.shape}}}")
    aggregated_unsafe = aggregate(frozen_unsafe)
    print(f"unsafe {{{frozen_unsafe.shape} --> {aggregated_unsafe.shape}}}")
    aggregated_ignore = aggregate(frozen_ignore)
    print(f"ignore {{{frozen_ignore.shape} --> {aggregated_ignore.shape}}}")
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

    domain = torch.from_numpy(domain_raw).float().to(device)
    explorer = DomainExplorer(1, domain)
    print("\n---------------checking safe domains, everything should be safe")
    explorer.explore(verification_model, aggregated_safe, precision=1e-6, min_area=0)  # double check for no mistakes
    print("\n---------------checking unsafe domains, everything should be unsafe")
    explorer.explore(verification_model, aggregated_unsafe, precision=1e-6, min_area=0)  # double check for no mistakes
    print("\n---------------checking ignore domains, might improve on identified domains")
    explorer.explore(verification_model, aggregated_ignore, precision=1e-6, min_area=0)  # unknown behaviour


if __name__ == '__main__':
    main()
