import ray
import torch.nn

from environment.stopping_car import StoppingCar
from polyhedra.net_methods import generate_nn_torch

ray.init(local_mode=True)
sequential_nn = generate_nn_torch(six_dim=False, min_distance=20, max_distance=30).float()
env = StoppingCar()
state = env.reset()
print(state)
for i in range(1000):
    state_reduced = torch.from_numpy(state).float().unsqueeze(0)
    v_ego = state_reduced[:, 3]
    delta_x = state_reduced[:, -1]
    state_reduced = torch.stack([v_ego, delta_x], dim=1)
    action = torch.argmax(sequential_nn(state_reduced)).item()
    print(f"action: {action}")
    state, reward, done, _ = env.step(action)
    print(f"iteration: {i}, delta_x: {state[7]:.2f}, delta_v: {state[6]:.2f}, v_ego: {state[3]:.2f},v_lead: {state[2]:.2f} , y_ego: {state[5]:.2f}")
    print("-------")
    if done:
        print("done")
        break
ray.shutdown()
print("all good")
