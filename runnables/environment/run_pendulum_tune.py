import os
from ray import tune

from training.dqn.dqn_agent import Agent
from environment.pendulum_abstract import PendulumEnv
import numpy as np

state_size = 2
agent = Agent(state_size, 2)
agent.load(os.path.expanduser("~/Development") + "/SafeDRL/save/Pendulum_Apr07_12-17-45_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_final.pth")


def trainable(config):
    n_trials = config["n_trials"]
    horizon = config["horizon"]
    x = config["x"]
    y = config["y"]
    start_state = np.array([x, y])
    env = PendulumEnv()
    n_fail = 0
    for trial in range(n_trials):
        env.reset()
        env.state = start_state
        for i in range(horizon):
            action = agent.act(env.state)
            next_state, reward, done, _ = env.step(action)
            if np.random.rand() > 0.8:
                next_state, reward, done, _ = env.step(action)  # sticky action
            # if render:
            #     env.render()
            if done:
                n_fail += 1
                # print(f"exiting at timestep {i}")
                break

    tune.report(score=n_fail / n_trials)  # This sends the score to Tune.


if __name__ == '__main__':
    space = {"x": tune.uniform(0.31, 0.4), "y": tune.uniform(-1, -0.88), "n_trials": 100, "horizon": 4}
    analysis = tune.run(trainable, num_samples=100,resources_per_trial={'gpu': 1},config=space)
    # Get the best hyperparameters
    results = [(trial.metric_analysis["score"]["max"],trial.config) for trial in analysis.trials]
    max_hyperparameters = max(results,key=lambda x:x[0])
    min_hyperparameters = min(results,key=lambda x:x[0])
    print(f"max hyperparameters {max_hyperparameters}")
    print(f"min hyperparameters {min_hyperparameters}")