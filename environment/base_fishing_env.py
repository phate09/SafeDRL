import gym
import numpy as np
from gym import spaces

from gym_fishing.envs.shared_env import (
    csv_entry,
    estimate_policyfn,
    plot_mdp,
    plot_policyfn,
    simulate_mdp,
)


# consider adding support for gym logger, error, and seeding


class BaseFishingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            params={"r": 0.3, "K": 1, "sigma": 0.0, "x0": 0.75},
            Tmax=100,
            file=None,
    ):

        # parameters
        self.K = params["K"]
        self.r = params["r"]
        self.sigma = params["sigma"]
        self.init_state = params["x0"]
        self.params = params

        # Preserve these for reset
        self.fish_population = self.init_state
        self.reward = 0
        self.harvest = 0
        self.years_passed = 0
        self.Tmax = Tmax
        self.file = file

        # for render() method only
        if file is not None:
            self.write_obj = open(file, "w+")

        # Initial state
        self.state = np.array([self.init_state / self.K - 1])

        # Best if cts actions / observations are normalized to a [-1, 1] domain
        self.action_space = spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )

    def step(self, action):

        # Map from re-normalized model space to [0,2K] real space
        quota = self.get_quota(action)
        self.fish_population = self.get_fish_population(self.state)

        # Apply harvest and population growth
        self.harvest = min(self.fish_population, quota)
        self.fish_population = max(self.fish_population - self.harvest, 0.0)
        self.fish_population = self.population_draw()

        # Map population back to system state (normalized space):
        self.state = self.get_state(self.fish_population)

        # should be the instanteous reward, not discounted
        self.reward = max(self.harvest, 0.0)
        self.years_passed += 1
        done = bool(self.years_passed > self.Tmax)

        if self.fish_population <= 0.0:
            done = True

        return self.state, self.reward, done, {}

    def reset(self):
        self.state = np.array([self.init_state / self.K - 1])
        self.fish_population = self.init_state
        self.years_passed = 0

        # for tracking only
        self.reward = 0
        self.harvest = 0
        return self.state

    def render(self, mode="human"):
        return csv_entry(self)

    def close(self):
        if self.file is not None:
            self.write_obj.close()

    def simulate(env, model, reps=1):
        return simulate_mdp(env, model, reps)

    def plot(self, df, output="results.png"):
        return plot_mdp(self, df, output)

    def policyfn(env, model, reps=1):
        return estimate_policyfn(env, model, reps)

    def plot_policy(self, df, output="results.png"):
        return plot_policyfn(self, df, output)

    def population_draw(self):
        """
        Select a value for population to grow or decrease at each time step.
        """
        growth = self.r * self.fish_population * (1.0 - self.fish_population / self.K)
        random_variation = self.fish_population * self.sigma * np.random.normal(0, 1)
        return np.maximum(
            self.fish_population
            + growth
            + random_variation,
            0.0,
        )

    def get_quota(self, action):
        """
        Convert action into quota
        """
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            quota = (action / self.n_actions) * self.K
        # Continuous Actions
        else:
            action = np.clip(
                action, self.action_space.low, self.action_space.high
            )[0]
            quota = (action + 1) * self.K
        return quota

    def get_action(self, quota):
        """
        Convert quota into action
        """
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            return round(quota * self.n_actions / self.K)
        else:
            return quota / self.K - 1

    def get_fish_population(self, state):
        return (state[0] + 1) * self.K

    def get_state(self, fish_population):
        return np.array([fish_population / self.K - 1])
