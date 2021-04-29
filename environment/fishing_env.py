from gym import spaces

# from gym_fishing.envs.base_fishing_env import BaseFishingEnv
from environment.base_fishing_env import BaseFishingEnv


class FishingEnv(BaseFishingEnv):
    def __init__(
            self,
            r=0.3,
            K=1,
            sigma=0.0,
            n_actions=10,
            init_state_ub=0.75,
            init_state_lb=0.70,
            Tmax=100,
            file=None,
    ):
        super().__init__(
            params={"r": r, "K": K, "sigma": sigma, "x0_lb": init_state_lb, "x0_ub": init_state_ub},
            Tmax=Tmax,
            file=file,
        )
        # override to use discrete actions
        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)
        self.current_violation_step = 0


class MonitoredFishingEnv(FishingEnv):
    def __init__(self,
                 config=None):
        super().__init__()
        self.n_steps = 5
        self.lower_bound = 0.25
        self.reward_penalty = 1000

    def step(self, action):
        state, reward, done, _ = super().step(action)
        population = self.get_fish_population(state)
        if population < self.lower_bound:
            self.current_violation_step += 1
            if self.current_violation_step > self.n_steps or population <= 0:
                # failure
                reward -= 10
                done = True
        else:
            self.current_violation_step = 0
        return state, reward, done, {}


if __name__ == '__main__':
    env = MonitoredFishingEnv()
    state = env.reset()
    # env.render()
    print(state)
    for i in range(1000):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        print(state)
        if done:
            break
    env.close()
