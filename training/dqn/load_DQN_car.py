import ray
import torch.nn

from environment.stopping_car import StoppingCar
from training.dqn.train_DQN_car import get_apex_dqn_car_trainer
from training.ray_utils import convert_DQN_ray_policy_to_sequential

ray.init(local_mode=True)
trainer, config = get_apex_dqn_car_trainer()
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-28_14-15-09rwm2u8a4/checkpoint_239/checkpoint-239")
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-28_14-57-456gqilswb/checkpoint_8/checkpoint-8")
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-28_15-03-06e7kcr1ke/checkpoint_49/checkpoint-49")
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-28_15-49-16c3ga4n0f/checkpoint_12/checkpoint-12")  # super safe
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-28_16-46-205jtg2ce8/checkpoint_8/checkpoint-8")
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-28_16-48-24_ovepj_6/checkpoint_27/checkpoint-27")
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-29_14-48-03xgehld_5/checkpoint_100/checkpoint-100") #target 0 distance
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-29_14-52-57j5qb7ovs/checkpoint_200/checkpoint-200") #target 20mt but keeps >0
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-29_15-12-59onuvlhtv/checkpoint_400/checkpoint-400")
# trainer.restore("/home/edoardo/ray_results/DQN_StoppingCar_2020-12-29_15-12-59onuvlhtv/checkpoint_560/checkpoint-560")
# trainer.restore("/home/edoardo/ray_results/APEX_StoppingCar_2020-12-29_17-10-24qjvbq7ew/checkpoint_42/checkpoint-42")
trainer.restore("/home/edoardo/ray_results/APEX_StoppingCar_2020-12-30_07-08-19vc1f79qh/checkpoint_264/checkpoint-264")


policy = trainer.get_policy()
trainer.cleanup()
sequential_nn = convert_DQN_ray_policy_to_sequential(policy).cpu()
env = StoppingCar()
state = env.reset()
min_distance = 9999
cumulative_reward = 0
print(state)
for i in range(1000):
    state_reduced = torch.from_numpy(state).float().unsqueeze(0)[:, -2:]
    action_score = sequential_nn(state_reduced)
    action = torch.argmax(action_score).item()
    print(f"action: {action}")
    state, reward, done, _ = env.step(action)
    min_distance = min(state[7], min_distance)
    cumulative_reward += reward
    print(f"iteration: {i}, delta_x: {state[7]:.2f}, delta_v: {state[6]:.2f}, v_ego: {state[3]:.2f},v_lead: {state[2]:.2f} , y_ego: {state[5]:.2f}, reward: {reward}")
    print("-------")
    if done:
        print("done")

        break
print("all good")
print(f"min_distance:{min_distance}")
print(f"cumulative_reward:{cumulative_reward}")
ray.shutdown()
