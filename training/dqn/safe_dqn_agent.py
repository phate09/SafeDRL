import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn.modules.loss import _Loss

from mosaic.utils import chunks
from training.dqn.dqn_sequential import QNetwork
from utility.ExperienceReplay import ExperienceReplayBuffer
from utility.PrioritisedExperienceReplayBuffer import PrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-5  # learning rate
UPDATE_EVERY = 10  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
prioritised = True


class SafetyLoss(_Loss):
    def __init__(self, model, target):
        super().__init__()
        self.model = model
        self.target_net = target

    def forward(self, states: Tensor, actions: Tensor, successors: Tensor, flags: Tensor) -> Tensor:
        mean = torch.tensor([0], device=self.model[0].weight.device).float()
        safe_states = states[(flags == 1).nonzero().squeeze()]
        if len(safe_states.size()) < 2:
            safe_states = safe_states.unsqueeze(0)
        safe_action = actions[(flags == 1).nonzero().squeeze()]
        if len(safe_action.size()) < 1:
            safe_action = safe_action.unsqueeze(0)
        mean += ((1 - self.model(safe_states).gather(1, safe_action.unsqueeze(-1))) ** 2).sum()
        # mean += torch.relu(-self.model(safe_states)).sum()
        unsafe_states = states[(flags == -1).nonzero().squeeze()]
        if len(unsafe_states.size()) < 2:
            unsafe_states = unsafe_states.unsqueeze(0)
        unsafe_action = actions[(flags == -1).nonzero().squeeze()]
        if len(unsafe_action.size()) < 1:
            unsafe_action = unsafe_action.unsqueeze(0)
        mean += ((-1 - self.model(unsafe_states).gather(1, unsafe_action.unsqueeze(-1))) ** 2).sum()
        # mean += torch.relu(self.model(unsafe_states)).sum()
        undefined_states = states[(flags == 0).nonzero().squeeze()]
        if len(undefined_states.size()) < 2:
            undefined_states = undefined_states.unsqueeze(0)
        undefined_action = actions[(flags == 0).nonzero().squeeze()]
        if len(undefined_action.size()) < 1:
            undefined_action = undefined_action.unsqueeze(0)
        undefined_successors = successors[(flags == 0).nonzero().squeeze()]
        if len(undefined_successors.size()) < 2:
            undefined_successors = undefined_successors.unsqueeze(0)
        next_actions = self.target_net(undefined_successors).max(1)[1]
        mean += ((self.model(undefined_states).gather(1, undefined_action.unsqueeze(-1)) - self.target_net(undefined_successors).gather(1, next_actions.unsqueeze(-1))) ** 2).sum()
        # mean += (torch.relu(self.model(undefined_states)) * torch.relu(-self.model(undefined_successors))).sum()
        return mean / len(states)

    def set_target_net(self):
        self.target_net.load_state_dict(self.model.state_dict())


# noinspection SpellCheckingInspection,PyMethodMayBeStatic
class InvariantAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, alpha=0.6):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = pickle.loads(pickle.dumps(self.qnetwork_local)).to(device)  # clones the critic
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # I-Net
        self.inetwork_local = nn.Sequential(*[nn.Linear(state_size, 50),nn.ReLU(), nn.Linear(50, action_size), nn.Tanh()]).to(device)
        self.inetwork_target = nn.Sequential(*[nn.Linear(state_size, 50),nn.ReLU(), nn.Linear(50, action_size), nn.Tanh()]).to(device)
        self.inetwork_target.load_state_dict(self.inetwork_local.state_dict())  # clones the weigths
        self.ioptimizer = optim.Adam(self.inetwork_local.parameters(), 1e-3)

        # Replay memory
        if prioritised:
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=alpha)
        else:
            self.memory = ExperienceReplayBuffer(BUFFER_SIZE)
        self.local_memory = []
        self.temp_dataset = []  # keeps track of a trajectory
        self.invariant_dataset = []  # keeps track of the past trajectories
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.past_episodes = 0

    # self.t_update_target_step = 0

    def step(self, state, action, reward, next_state, done, beta):
        # Save experience in replay memory
        self.local_memory.append((state, action, reward, next_state, done))
        self.temp_dataset.append((state, action, next_state))
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:

            with torch.no_grad():
                self.qnetwork_local.eval()
                states, actions, rewards, next_states, dones = [list(tup) for tup in zip(*self.local_memory)]
                states = torch.tensor(states, dtype=torch.float).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float).to(device)
                dones = torch.tensor(dones, dtype=torch.float).to(device)
                actions = torch.tensor(actions, dtype=torch.float).to(device)

                Qa_s = self.qnetwork_local(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)  # state_action_values
                next_actions = self.qnetwork_local(next_states).max(1)[1]
                Qa_next_s = self.qnetwork_target(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)  # the maximum Q at the next state
                target_Q = rewards + GAMMA * Qa_next_s * (torch.ones_like(dones) - dones)
                # calculate td-errors
                td_errors = torch.abs(target_Q - Qa_s)
                # store the local memory in the PER
                for memory, error in zip(self.local_memory, td_errors):
                    if prioritised:
                        self.memory.add(memory, error.item())  # td_errors
                    else:
                        self.memory.add(memory)
            # empty the memory
            self.local_memory = []
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                if prioritised:
                    experiences, is_values, indexes = self.memory.sample(BATCH_SIZE, beta=beta)
                else:
                    experiences, is_values, indexes = self.memory.sample(BATCH_SIZE)
                error = self.learn(experiences, indexes, is_values)
                # ------------------- update target network ------------------- #
                self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
                return error

    def episode_end(self, done: bool):
        """Actions to be performed at the end of an episode"""
        self.past_episodes = (self.past_episodes + 1) % UPDATE_EVERY
        if done:
            for state_np, action, next_state_np in self.temp_dataset:
                self.invariant_dataset.append((state_np.astype(dtype=np.float32), action, next_state_np.astype(dtype=np.float32), -1))
        else:
            for state_np, action, next_state_np in self.temp_dataset:
                self.invariant_dataset.append((state_np.astype(dtype=np.float32), action, next_state_np.astype(dtype=np.float32), 0))
        self.temp_dataset = []
        if self.past_episodes == 0:  # updates the invariant
            self.ioptimizer.zero_grad()  # resets the gradient
            losses = []
            for chunk in chunks(self.invariant_dataset, 256):
                states, actions, successors, flags = zip(*chunk)
                loss = SafetyLoss(self.inetwork_local, self.inetwork_target)
                loss_value = loss(torch.tensor(states).to(device), torch.tensor(actions).to(device), torch.tensor(successors).to(device), torch.tensor(flags, dtype=torch.float32).to(device))
                loss_value.backward()
                self.ioptimizer.step()  # updates the invariant
                self.soft_update(self.inetwork_local, self.inetwork_target, TAU)
                losses.append(loss_value.item())
            self.invariant_dataset = []
            return np.array(losses).mean()

    def act(self, state: np.ndarray, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            # action_values2 = action_values.clone()
            # invariant_values = self.inetwork_local(state)
            # action_values2 *= invariant_values
            # final_action_values = action_values * eps + (1 - eps) * action_values2
        # self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, indexes, is_values):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.tensor(states, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        if prioritised:
            is_values = torch.tensor(is_values, dtype=torch.float).to(device)
        else:
            is_values = torch.ones(len(experiences)).to(device)

        self.optimizer.zero_grad()  # resets the gradient
        self.qnetwork_local.train()
        Qa_s = self.qnetwork_local(states).gather(1, actions.long().unsqueeze(-1)).squeeze(
            -1)  # state_action_values
        next_actions = self.qnetwork_local(next_states).max(1)[1]
        Qa_next_s = self.qnetwork_target(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(
            -1)  # the maximum Q at the next state
        target_Q = rewards + GAMMA * Qa_next_s * (torch.ones_like(dones) - dones)
        td_errors = torch.abs(target_Q - Qa_s) + 1e-5
        if prioritised:
            # updates the priority by using the newly computed td_errors
            self.memory.update_priorities(indexes, td_errors.detach().cpu().numpy())
        # Notice that detach for the target_Q
        error = 0.5 * (target_Q.detach() - Qa_s).pow(2) * is_values.detach()  # * td_errors
        error = error.mean()
        error.backward()
        nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1, norm_type=2)  # float("inf")
        self.optimizer.step()
        self.qnetwork_local.eval()
        return error

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, path, global_step):
        torch.save({
            "global_step": global_step,
            "critic": self.qnetwork_local.state_dict(),
            "invariant_net": self.inetwork_local.state_dict(),
            "target_critic": self.qnetwork_target.state_dict(),
            "target_invariant_net": self.inetwork_target.state_dict(),
            "optimiser_critic": self.optimizer.state_dict(),
        }, path)

    def load(self, path,agent=True,invariant=True):
        checkpoint = torch.load(path, map_location=device)
        if agent:
            self.qnetwork_local.load_state_dict(checkpoint["critic"])
            self.qnetwork_target.load_state_dict(checkpoint["target_critic"])
        self.optimizer.load_state_dict(checkpoint["optimiser_critic"])
        if invariant:
            self.inetwork_local.load_state_dict(checkpoint["invariant_net"])
            self.inetwork_target.load_state_dict(checkpoint["target_invariant_net"])
        self.global_step = checkpoint["global_step"]
        print(f'Loading complete')
