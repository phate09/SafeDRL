import csv
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import ray
import torch.nn
import numpy as np
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.torch.training_operator import amp
from ray.util.sgd.utils import NUM_SAMPLES
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from training.ray_utils import convert_ray_policy_to_sequential
from environment.stopping_car import StoppingCar
import ray.rllib.agents.ppo as ppo
from training.ppo.tune.tune_train_PPO_car import get_PPO_config
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import random
import pickle

print(torch.cuda.is_available())
ray.init(local_mode=True)


class TrainedPolicyDataset(torch.utils.data.Dataset):
    def __init__(self, path, size=(500, 100), seed=1234, traces=True):
        self.size = size
        config = get_PPO_config(1234, use_gpu=0)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(path)
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        config = {"cost_fn": 1, "simplified": True}
        self.env = StoppingCar(config)
        self.env.seed(seed)
        load_dataset = True
        file_name = "dataset_new.p"
        if load_dataset and traces and os.path.exists(file_name):
            dataset = pickle.load(open(file_name, "rb"))
        else:
            dataset = []
            while len(dataset) < size[0]:
                state_np = self.env.reset()  # only starting states
                state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]  # pick just delta_x and delta_v
                action = torch.argmax(sequential_nn(state_reduced)).item()
                next_state_np, reward, done, _ = self.env.step(action)
                dataset.append((state_np.astype(dtype=np.float32), next_state_np.astype(dtype=np.float32), 1))
            param_grid = {'delta_v': np.arange(-30, 30, 0.5), 'delta_x': np.arange(-10, 40, 0.5)}
            for parameters in ParameterGrid(param_grid):
                delta_v = parameters["delta_v"]
                delta_x = parameters["delta_x"]
                self.env.reset()
                self.env.x_lead = delta_x
                self.env.x_ego = 0
                self.env.v_lead = delta_v
                self.env.v_ego = 0
                done = False
                temp_dataset = []
                state_np = np.array([delta_v, delta_x])
                state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]  # pick just delta_x and delta_v
                for i in (range(100) if traces else range(1)):
                    # action = torch.argmax(sequential_nn(state_reduced)).item()
                    action = self.env.perfect_action()
                    next_state_np, reward, done, _ = self.env.step(action)
                    temp_dataset.append((state_np, next_state_np))
                    state_np = next_state_np
                    if next_state_np[1] < 0.5 and not done:
                        done = True
                    if done is True:  # only unsafe states
                        break
                if done:
                    for state_np, next_state_np in temp_dataset:
                        dataset.append((state_np.astype(dtype=np.float32), next_state_np.astype(dtype=np.float32), -1))
                else:
                    for state_np, next_state_np in temp_dataset:
                        dataset.append((state_np.astype(dtype=np.float32), next_state_np.astype(dtype=np.float32), 0))
            if traces:
                pickle.dump(dataset, open(file_name, "wb+"))
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
        # state_np = self.env.random_sample()
        # state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]  # pick just delta_x and delta_v
        # action = torch.argmax(sequential_nn(state_reduced)).item()
        # next_state_np, reward, done, _ = self.env.step(action)
        # return torch.from_numpy(state_np).float(), torch.from_numpy(next_state_np).float()


class SafetyLoss(_Loss):
    def __init__(self, model, target):
        super().__init__()
        self.model = model
        self.target_net = target

    def forward(self, states: Tensor, successors: Tensor, flags: Tensor) -> Tensor:
        mean = torch.tensor([0], device=self.model[0].weight.device).float()
        safe_states = states[(flags == 1).nonzero().squeeze()]
        # mean += ((1 - self.model(safe_states)) ** 2).sum()
        mean += torch.relu(-self.model(safe_states)).sum()
        unsafe_states = states[(flags == -1).nonzero().squeeze()]
        # mean += ((-1 - self.model(unsafe_states)) ** 2).sum()
        mean += torch.relu(self.model(unsafe_states)).sum()
        undefined_states = states[(flags == 0).nonzero().squeeze()]
        undefined_successors = successors[(flags == 0).nonzero().squeeze()]
        mean += ((self.model(undefined_states) - self.target_net(undefined_successors)) ** 2).sum()
        # mean += (torch.relu(self.model(undefined_states)) * torch.relu(-self.model(undefined_successors))).sum()
        return mean / len(states)

    def set_target_net(self):
        self.target_net.load_state_dict(self.model.state_dict())


class SafetyTrainingOperator(TrainingOperator):

    def setup(self, config):
        path1 = config["path"]
        batch_size = config["batch_size"]
        train_data = TrainedPolicyDataset(path1, size=(500, 10), seed=3451)
        val_data = TrainedPolicyDataset(path1, size=(0, 0), seed=4567)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        model = torch.nn.Sequential(torch.nn.Linear(train_data.env.observation_space.shape[0], 50), torch.nn.ReLU(), torch.nn.Linear(50, 1), torch.nn.Tanh())
        target = torch.nn.Sequential(torch.nn.Linear(train_data.env.observation_space.shape[0], 50), torch.nn.ReLU(), torch.nn.Linear(50, 1), torch.nn.Tanh())
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.get("lr", 1e-3))
        loss = SafetyLoss(model, target)  # torch.nn.MSELoss()

        self.model, self.optimizer, self.criterion = self.register(
            models=model,
            optimizers=optimizer,
            criterion=loss)

        self.register_data(
            train_loader=train_loader,
            validation_loader=val_loader)

    def train_epoch(self, iterator, info):
        self.criterion.set_target_net()  # updates the target_network
        return super(SafetyTrainingOperator, self).train_epoch(iterator, info)

    def train_batch(self, batch, batch_info):
        """Computes loss and updates the model over one batch.

        This method is responsible for computing the loss and gradient and
        updating the model.

        By default, this method implementation assumes that batches
        are in (\\*features, labels) format. So we also support multiple inputs
        model. If using amp/fp16 training, it will also scale the loss
        automatically.

        You can provide custom loss metrics and training operations if you
        override this method.

        You do not need to override this method if you plan to
        override ``train_epoch``.

        Args:
            batch: One item of the validation iterator.
            batch_info (dict): Information dict passed in from ``train_epoch``.

        Returns:
            A dictionary of metrics.
                By default, this dictionary contains "loss" and "num_samples".
                "num_samples" corresponds to number of datapoints in the batch.
                However, you can provide any number of other values.
                Consider returning "num_samples" in the metrics because
                by default, ``train_epoch`` uses "num_samples" to
                calculate averages.

        """
        if not hasattr(self, "model"):
            raise RuntimeError("Either set self.model in setup function or "
                               "override this method to implement a custom "
                               "training loop.")
        if not hasattr(self, "optimizer"):
            raise RuntimeError("Either set self.optimizer in setup function "
                               "or override this method to implement a custom "
                               "training loop.")
        if not hasattr(self, "criterion"):
            raise RuntimeError("Either set self.criterion in setup function "
                               "or override this method to implement a custom "
                               "training loop.")
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        # unpack features into list to support multiple inputs model
        features, targets, flags = batch
        # Create non_blocking tensors for distributed training
        if self.use_gpu:
            features = features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            flags = flags.cuda(non_blocking=True)
            # features = [
            #     feature.cuda(non_blocking=True) for feature in features
            # ]
            # targets = [
            #     target.cuda(non_blocking=True) for target in targets
            # ]
        # Compute output.
        with self.timers.record("fwd"):
            # output = model(features)

            loss = self.criterion(features, targets, flags)

        # Compute gradients in a backward pass.
        with self.timers.record("grad"):
            optimizer.zero_grad()
            if self.use_fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

        # Call step of optimizer to update model params.
        with self.timers.record("apply"):
            optimizer.step()

        return {"train_loss": loss.item(), NUM_SAMPLES: features.size(0)}

    def validate_batch(self, batch, batch_info):
        """Calcuates the loss and accuracy over a given batch.

        You can override this method to provide arbitrary metrics.

        Same as ``train_batch``, this method implementation assumes that
        batches are in (\\*features, labels) format by default. So we also
        support multiple inputs model.

        Args:
            batch: One item of the validation iterator.
            batch_info (dict): Contains information per batch from
                ``validate()``.

        Returns:
            A dict of metrics.
                By default, returns "val_loss", "val_accuracy", and
                "num_samples". When overriding, consider returning
                "num_samples" in the metrics because
                by default, ``validate`` uses "num_samples" to
                calculate averages.
        """
        if not hasattr(self, "model"):
            raise RuntimeError("Either set self.model in setup function or "
                               "override this method to implement a custom "
                               "training loop.")
        if not hasattr(self, "criterion"):
            raise RuntimeError("Either set self.criterion in setup function "
                               "or override this method to implement a custom "
                               "training loop.")
        model = self.model
        criterion = self.criterion
        # unpack features into list to support multiple inputs model
        features, targets, flags = batch
        if self.use_gpu:
            features = features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            flags = flags.cuda(non_blocking=True)

        # compute output

        with self.timers.record("eval_fwd"):
            loss = self.criterion(features, targets, flags)
            # loss = criterion(output, target)
            # _, predicted = torch.max(output.data, 1)

        num_samples = targets.size(0)
        num_correct = 0  # todo find a good value (predicted == target).sum().item()
        return {
            "val_loss": loss.item(),
            "val_accuracy": num_correct / num_samples,
            NUM_SAMPLES: num_samples
        }


enable_training = True
path1 = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58"
# path1 = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00005_5_cost_fn=0,epsilon_input=0.1_2021-01-17_12-41-27/checkpoint_10/checkpoint-10"
val_data = TrainedPolicyDataset(path1, size=(0, 0), seed=4567, traces=False)
config = get_PPO_config(1234, use_gpu=0)
trainer = ppo.PPOTrainer(config=config)
trainer.restore(path1)
policy = trainer.get_policy()
sequential_nn = convert_ray_policy_to_sequential(policy).cpu()

if enable_training:
    trainer1 = TorchTrainer(
        training_operator_cls=SafetyTrainingOperator,
        num_workers=1,
        use_gpu=True,
        config={
            "lr": 1e-2,  # used in optimizer_creator
            "hidden_size": 1,  # used in model_creator
            "batch_size": 1024,  # used in data_creator
            "path": path1,  # path to load the agent nn
        },
        backend="auto",
        scheduler_step_freq="epoch")
    for i in range(100):
        stats = trainer1.train()
        print(stats)

    print(trainer1.validate())
    torch.save(trainer1.state_dict(), "checkpoint.pt")
    torch.save(trainer1.get_model().state_dict(), "invariant_checkpoint.pt")
    m = trainer1.get_model()
    print(f"trained weight: torch.tensor([[{m[0].weight.data.cpu().numpy()[0][0]},{m[0].weight.data.cpu().numpy()[0][1]}]]), bias: torch.tensor({m[0].bias.data.cpu().numpy()})")
    # trainer1.shutdown()
    print("success!")
else:
    m = torch.nn.Sequential(torch.nn.Linear(2, 50), torch.nn.ReLU(), torch.nn.Linear(50, 1), torch.nn.Tanh())
    checkpoint = torch.load("invariant_checkpoint.pt", torch.device("cpu"))
    m.load_state_dict(checkpoint)
    # trained weight:  [[0.0018693  0.05228069]], bias: [-0.5533147] , train_loss = 0.0
    # trained weight:  [[-0.01369903  0.03511396]], bias: [-0.6535952] , train_loss = 0.0
    # trained weight:  [[0.00687088  0.26634103]], bias: [-0.6658108] , train_loss = 0.0
    # trained weight: torch.tensor([[0.038166143000125885,0.16197167336940765]]), bias: torch.tensor([-2.3122551])
# %%
m.cpu()
random.seed(0)
x_data = []
xprime_data = []
y_data = []
for data in random.sample(val_data.dataset, k=9000):
    value = min(max(m(torch.from_numpy(data[0])).item(), -1.0), 1.0)
    # value = 1 if data[2] >= 0 else -1
    # print(f"(delta_v,delta_x) {val_data[i][0].numpy()} value:{value}")
    x_data.append(data[0])
    xprime_data.append(data[1])
    y_data.append(value)
x_data = np.array(x_data)
xprime_data = np.array(xprime_data)
y_data = np.array(y_data)
import plotly.figure_factory as ff

# fig = ff.create_quiver(x_data[:, 0], x_data[:, 1], xprime_data[:, 0] - x_data[:, 0], xprime_data[:, 1] - x_data[:, 1], scale=1)
# fig.show()


x = x_data[:, 0]
y = x_data[:, 1]
u = xprime_data[:, 0] - x_data[:, 0]
v = xprime_data[:, 1] - x_data[:, 1]
colors = y_data

norm = Normalize(vmax=1.0, vmin=-1.0)
norm.autoscale(colors)
# we need to normalize our colors array to match it colormap domain
# which is [0, 1]

colormap = cm.bwr
plt.figure(figsize=(18, 16), dpi=80)
line_x = np.linspace(-30, 30, 1000)
plt.plot(line_x, np.array([0] * 1000))
plt.plot(np.array([0] * 1000), np.linspace(-10, 40, 1000))
plt.quiver(x, y, u, v, angles='xy', color=colormap(colors),
           scale_units='xy', scale=1, pivot='mid')  # colormap(norm(colors))
plt.xlim([-30, 30])
plt.ylim([-10, 40])
plt.show()
# %%
# m(torch.tensor([-20, 15]).float())
# # %%
# env = StoppingCar()
# env.reset()
# env.x_lead = 20
# env.x_ego = 0
# env.v_lead = -15
# env.v_ego = 0
# state_np = np.array([env.v_lead - env.v_ego, env.x_lead - env.x_ego])
# state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]
# action = torch.argmax(sequential_nn(state_reduced.float())).item()
# next_state_np, reward, done, _ = env.step(action)
