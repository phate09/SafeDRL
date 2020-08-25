import os

import progressbar

from dqn.dqn_agent import Agent
from environment.pendulum_abstract import PendulumEnv
import numpy as np

from utility.standard_progressbar import StandardProgressBar

env = PendulumEnv()
# env.state = np.array([-0.57,0.21]) #guaranteed fail
# env.state = np.array([-0.57,0.21]) #guaranteed fail
# env.state = np.array([-0.22,0.05]) #success right
# env.state = np.array([0.25, -1])  # success left
# env.state = np.array([0.39,0.9]) # fail left
# env.state = np.array([0.834,0.497]) # fail
# start_state = np.array([-0.12, 0.33]) #100% success
# start_state = np.array([0.35,-0.83]) # should be 83% but appears 100%
# start_state = np.array([-0.48, -0.44])  #100% fail
start_state = np.array([0.06,0.45])  #min 33% max 100%

state_size = 2
agent = Agent(state_size, 2)
agent.load(os.path.expanduser("~/Development") + "/SafeDRL/save/Pendulum_Apr07_12-17-45_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_final.pth")
action = None

render = False
n_trials = 10000
n_fail = 0
widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Variable('fails'), ', ', progressbar.Variable('trials'), ]
with progressbar.ProgressBar(max_value=n_trials, widgets=widgets) as bar:
    for trial in range(n_trials):
        env.reset()
        env.state = start_state
        for i in range(4):
            # if action is None or np.random.rand() < 0.8:
            action = agent.act(env.state)
            next_state, reward, done, _ = env.step(action)
            if np.random.rand() > 0.8:
                next_state, reward, done, _ = env.step(action)  # sticky action
            if render:
                env.render()
            if done:
                n_fail += 1
                # print(f"exiting at timestep {i}")
                break
        bar.update(trial, fails=n_fail, trials=trial+1)
    print("finished")
print(f"# failures: {n_fail}/{n_trials} = {n_fail / n_trials:.0%}")
