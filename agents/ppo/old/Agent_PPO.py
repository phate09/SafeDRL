import numpy as np
import progressbar as pb
import torch
from scipy import stats
import torch.nn
import torch.optim.optimizer
import constants
import pong_utils
from agents.GenericAgent import GenericAgent

ACTION_LEFT = 0


class AgentPPO(GenericAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, config):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(config)
        self.input_dim: int = config[constants.input_dim]
        self.output_dim: int = config[constants.output_dim]
        # self.seed: int = random.seed(config[constants.seed])
        self.max_t: int = config[constants.max_t]
        self.sgd_iterations: int = config[constants.sgd_iterations]
        self.n_episodes: int = config[constants.n_episodes]
        self.discount: float = config[constants.discount]
        self.epsilon: float = config[constants.epsilon]
        self.beta: float = config[constants.beta]
        self.device = config[constants.device]
        self.model: torch.nn.Module = config[constants.model]
        self.optimiser: torch.optim.optimizer.Optimizer = config[constants.optimiser]
        self.ending_condition = config[constants.ending_condition]

        self.state_list = []
        self.action_list = []
        self.prob_list = []
        self.reward_list = []

    # self.t_update_target_step = 0
    def required_properties(self):
        return [constants.input_dim,
                constants.output_dim,
                constants.max_t,
                constants.sgd_iterations,
                constants.n_episodes,
                constants.discount,
                constants.beta,
                constants.device,
                constants.model,
                constants.optimiser,
                constants.ending_condition]

    def collect(self, state, action, probs, reward):
        self.state_list.append(state)
        self.action_list.append(action)
        self.prob_list.append(probs)
        self.reward_list.append(reward)
        pass

    def act(self, state=None):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        return np.random.rand(self.action_size)

    def learn(self):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        rewards = self.reward_list.copy()
        rewards.reverse()
        previous_rewards = 0
        for i in range(len(rewards)):
            rewards[i] = rewards[i] + self.discount * previous_rewards
            previous_rewards = rewards[i]
        rewards.reverse()
        rewards_standardised = stats.zscore(rewards, axis=1)
        rewards_standardised = np.nan_to_num(rewards_standardised, False)
        assert not np.isnan(rewards_standardised).any()
        rewards_standardised = torch.tensor(rewards_standardised, dtype=torch.float, device=self.device)
        states = torch.stack(self.state_list)
        output = self.model(states)

        pass

    def reset(self):
        """Reset the memory of the agent"""
        self.state_list = []
        self.action_list = []
        self.prob_list = []
        self.reward_list = []

    def train_OpenAI(self, env, writer):
        epsilon = self.epsilon
        beta = self.beta

        # keep track of progress
        mean_rewards = []
        # widget = ['training loop: ', pb.Percentage(), ' ',
        #           pb.Bar(), ' ', pb.ETA()]
        timer = pb.ProgressBar(maxval=self.n_episodes).start()
        for i_episode in range(self.n_episodes):

            # collect trajectories
            old_probs, states, actions, rewards = pong_utils.collect_trajectories(env, self.model, tmax=self.max_t)

            total_rewards = np.sum(rewards, axis=0)

            # gradient ascent step
            for _ in range(self.sgd_iterations):
                # uncomment to utilize your own clipped function!
                L = -self.clipped_surrogate(self.model, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)
                # L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)
                self.optimiser.zero_grad()
                L.backward()
                self.optimiser.step()
                del L

            # the clipping parameter reduces as time goes on
            epsilon *= .999

            # the regulation term also reduces
            # this reduces exploration in later runs
            beta *= .995

            # get the average reward of the parallel environments
            mean = np.mean(total_rewards)
            writer.add_scalar('data/score', np.max(total_rewards), i_episode)
            writer.add_scalar('data/score_average', np.mean(total_rewards), i_episode)
            result = {"mean": mean}
            mean_rewards.append(mean)
            if self.ending_condition(result):
                print("ending condition met!")
                break
            # display some progress every 20 iterations
            # if (i_episode + 1) % 20 == 0:
            #     print("Episode: {0:d}, score: {1:f}".format(i_episode + 1, np.mean(total_rewards)))
            #     print(total_rewards)

            # update progress widget bar
            timer.update(i_episode + 1)

        timer.finish()

    def clipped_surrogate(self, policy, old_probs, states, actions, rewards,
                          discount=0.995, epsilon=0.1, beta=0.01):
        actions = torch.tensor(actions, dtype=torch.int8, device=self.device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=self.device)
        rewards.reverse()
        previous_rewards = 0
        for i in range(len(rewards)):
            rewards[i] = rewards[i] + discount * previous_rewards
            previous_rewards = rewards[i]
        rewards.reverse()
        rewards_standardised = stats.zscore(rewards, axis=1)
        rewards_standardised = np.nan_to_num(rewards_standardised, False)
        assert not np.isnan(rewards_standardised).any()
        rewards_standardised = torch.tensor(rewards_standardised, dtype=torch.float, device=self.device)

        # convert states to policy (or probability)
        new_probs = pong_utils.states_to_prob(policy, states)
        new_probs = torch.where(actions == ACTION_LEFT, new_probs, 1.0 - new_probs)

        # cost = torch.log(new_probs) * rewards_standardised
        ratio = new_probs / old_probs
        cost = torch.min(ratio, torch.clamp(ratio, 1 - epsilon, 1 + epsilon)) * rewards_standardised
        # include a regularization term
        # this steers new_policy towards 0.5
        # which prevents policy to become exactly 0 or 1
        # this helps with exploration
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

        my_surrogate = torch.mean(cost + beta * entropy)
        return my_surrogate
