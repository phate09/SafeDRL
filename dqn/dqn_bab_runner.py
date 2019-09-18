from plnn.branch_and_bound import bab
from models.model_critic import QNetwork
from plnn.conv_net_convert import convert
from plnn.verification_network import VerificationNetwork
import torch
import torch.nn
import numpy as np


use_cuda = False
device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

def generate_domain(input_tensor, eps_size):
    return torch.stack((input_tensor - eps_size, input_tensor + eps_size), -1)

data = torch.tensor([0,0,0,0],dtype=torch.float,device=device)
domain_raw = generate_domain(data, 0.001)
model = QNetwork(4,2)
converted_layers = convert(model.layers)
verification_model = VerificationNetwork(model).to(device)

epsilon = 1e-5
decision_bound = 0
successes = 0
attempts = 0
last_result = ""
for data, target in iter(test_loader):
    data= data.view(-1)
    domain_raw = generate_domain(data, 1e-4)
    domain = domain_raw.to(device)  # at this point is (batch channel, width, height, bound)
    min_lb, min_ub, ub_point = bab(verification_model, domain, target.item(), epsilon, decision_bound)
    attempts += 1
    if min_lb >= 0:
        successes += 1
        last_result = "UNSAT"
    elif min_ub < 0:
        last_result = "SAT"
        # print(ub_point)
    else:
        print("Unknown")  # 18
    print(f'\rRunning percentage: {successes / attempts:.02%}, {attempts}/{len(test_loader)}, last result:{last_result}', end="")
print(f'Final percentage: {successes / attempts:.02%}')