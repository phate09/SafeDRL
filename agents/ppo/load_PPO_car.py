import ray
import torch.nn

from agents.ray_utils import convert_ray_policy_to_sequential, get_car_ppo_agent
from environment.stopping_car import StoppingCar

ray.init()
trainer = get_car_ppo_agent()
trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-12-28_14-44-17_y3io75q/checkpoint_12/checkpoint-12")
policy = trainer.get_policy()
sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
env = StoppingCar()
state = env.reset()
print(state)
for i in range(1000):
    action = torch.argmax(sequential_nn(torch.from_numpy(state).float().unsqueeze(0))).item()
    print(f"action: {action}")
    state, reward, done, _ = env.step(action)
    print(f"iteration: {i}, delta_v: {state[7]:.2f}, delta_x: {state[6]:.2f}, v_ego: {state[3]:.2f},v_lead: {state[2]:.2f} , y_ego: {state[5]:.2f}")
    print("-------")
    if done:
        print("done")
        break
ray.shutdown()
print("all good")
