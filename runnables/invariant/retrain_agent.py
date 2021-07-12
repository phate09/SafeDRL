import csv
import os
import random

import ray
import torch.nn
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.torch.training_operator import amp
from ray.util.sgd.utils import NUM_SAMPLES
from sklearn.model_selection import ParameterGrid
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from agents.ray_utils import convert_ray_policy_to_sequential
from environment.stopping_car import StoppingCar
import ray.rllib.agents.ppo as ppo
from agents.ppo.tune.tune_train_PPO_car import get_PPO_config
import matplotlib.pyplot as plt
import torch.nn.functional as F

# config, trainer = get_PPO_trainer(use_gpu=0)
print(torch.cuda.is_available())


class GridSearchDataset(torch.utils.data.Dataset):
    def __init__(self, size=16000):
        dataset = []
        param_grid = {'delta_v': np.arange(-30, 30, 0.5), 'delta_x': np.arange(-10, 40, 0.5)}
        for parameters in ParameterGrid(param_grid):
            delta_v = parameters["delta_v"]
            delta_x = parameters["delta_x"]
            state_np = np.array([delta_v, delta_x])
            dataset.append((torch.from_numpy(state_np).float()))
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class RetrainLoss(_Loss):
    def __init__(self, model, invariant_model):
        super().__init__()
        self.model = model
        self.invariant_model = invariant_model

    def forward(self, states: Tensor) -> Tensor:
        device = self.model[0].weight.device
        mean = torch.tensor([0], device=device).float()
        for state in states:
            action = torch.argmax(self.model(state)).item()
            next_state_np, reward, done, _ = StoppingCar.compute_successor(state.cpu().numpy(), action)
            next_state = torch.from_numpy(next_state_np).float().to(device)
            A = torch.relu(self.invariant_model(state))  # we punish if the initial state is positive and the successor is negative
            B = torch.relu(-self.invariant_model(next_state))
            mean += A * B
        return mean  # torch.mean(torch.relu(self.model(input)) * torch.relu(-self.model(target)))
        # return F.mse_loss(input, target, reduction=self.reduction)


class SafetyRetrainingOperator(TrainingOperator):

    def setup(self, config):
        path1 = config["path"]
        path_invariant = config["path_invariant"]
        batch_size = config["batch_size"]
        train_data = GridSearchDataset()
        val_data = GridSearchDataset()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        invariant_model = torch.nn.Sequential(torch.nn.Linear(2, 50), torch.nn.ReLU(), torch.nn.Linear(50, 1), torch.nn.Tanh())
        invariant_model.load_state_dict(torch.load(path_invariant, map_location=torch.device('cpu')))  # load the invariant model
        invariant_model.cuda()
        config = get_PPO_config(1234)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(path1)
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cuda()  # load the agent model
        model = sequential_nn
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.get("lr", 1e-3))
        loss = RetrainLoss(model, invariant_model)  # torch.nn.MSELoss()

        (self.model, self.invariant_model), self.optimizer, self.criterion = self.register(
            models=[model, invariant_model],
            optimizers=optimizer,
            criterion=loss)

        self.register_data(
            train_loader=train_loader,
            validation_loader=val_loader)

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
        # model = self.model
        # invariant_model = self.invariant_model
        optimizer = self.optimizer
        criterion = self.criterion
        # unpack features into list to support multiple inputs model
        features = batch
        # Create non_blocking tensors for distributed training
        if self.use_gpu:
            features = features.cuda(non_blocking=True)
        # Compute output.
        with self.timers.record("fwd"):
            loss = self.criterion(features)

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
        # model = self.model
        # criterion = self.criterion
        # unpack features into list to support multiple inputs model
        features = batch
        if self.use_gpu:
            features = features.cuda(non_blocking=True)

        # compute output

        with self.timers.record("eval_fwd"):
            loss = self.criterion(features)
            # loss = criterion(output, target)
            # _, predicted = torch.max(output.data, 1)

        num_samples = features.size(0)
        num_correct = 0  # todo find a good value (predicted == target).sum().item()
        return {
            "val_loss": loss.item(),
            "val_accuracy": num_correct / num_samples,
            NUM_SAMPLES: num_samples
        }


ray.init(local_mode=True)
path1 = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58"
path_invariant = "/home/edoardo/Development/SafeDRL/runnables/invariant/invariant_checkpoint.pt"
config = get_PPO_config(1234, use_gpu=0)
trainer = ppo.PPOTrainer(config=config)
trainer.restore(path1)
policy = trainer.get_policy()
old_agent_model = convert_ray_policy_to_sequential(policy).cpu()




enable_training = True
if enable_training:
    trainer1 = TorchTrainer(
        training_operator_cls=SafetyRetrainingOperator,
        num_workers=1,
        use_gpu=True,
        config={
            "lr": 1e-2,  # used in optimizer_creator
            "hidden_size": 1,  # used in model_creator
            "batch_size": 128,  # used in data_creator
            "path": path1,  # path to load the agent nn
            "path_invariant": path_invariant,  # the path to the invariant network
        },
        backend="auto",
        scheduler_step_freq="epoch")
    for i in range(10):
        stats = trainer1.train()
        print(stats)

    print(trainer1.validate())
    torch.save(trainer1.state_dict(), "checkpoint.pt")
    torch.save(trainer1.get_model()[0].state_dict(), "retrained_agent.pt")
    agent_model, invariant_model = trainer1.get_model()
else:
    sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
    sequential_nn.load_state_dict(torch.load("/home/edoardo/Development/SafeDRL/runnables/invariant/retrained_agent.pt"))
    agent_model = sequential_nn
    invariant_model = torch.nn.Sequential(torch.nn.Linear(2, 50), torch.nn.ReLU(), torch.nn.Linear(50, 1), torch.nn.Tanh())
    invariant_model.load_state_dict(torch.load(path_invariant, map_location=torch.device('cpu')))  # load the invariant model
# %%
agent_model.cpu()
invariant_model.cpu()
old_agent_model.cpu()
val_data = GridSearchDataset()
random.seed(0)
x_data = []
xprime_data = []
old_xprime_data = []
y_data = []
for data in random.sample(val_data.dataset, k=1500):
    value = torch.tanh(invariant_model(data)).item()
    x_data.append(data.numpy())
    action = torch.argmax(agent_model(data)).item()
    next_state_np, reward, done, _ = StoppingCar.compute_successor(data.numpy(), action)
    xprime_data.append(next_state_np)
    y_data.append(value)

    action = torch.argmax(old_agent_model(data)).item()
    next_state_np, _, _, _ = StoppingCar.compute_successor(data.numpy(), action)
    old_xprime_data.append(next_state_np)
x_data = np.array(x_data)
xprime_data = np.array(xprime_data)
old_xprime_data = np.array(old_xprime_data)

x = x_data[:, 0]
y = x_data[:, 1]
u = xprime_data[:, 0] - x_data[:, 0]
v = xprime_data[:, 1] - x_data[:, 1]
old_u = old_xprime_data[:, 0] - x_data[:, 0]
old_v = old_xprime_data[:, 1] - x_data[:, 1]
colors = y_data

norm = Normalize(vmax=1.0, vmin=-1.0)
norm.autoscale(colors)
# we need to normalize our colors array to match it colormap domain
# which is [0, 1]

colormap = cm.bwr
plt.figure()
plt.quiver(x, y, old_u, old_v, color="yellow", angles='xy',
           scale_units='xy', scale=1, pivot='mid')
plt.quiver(x, y, u, v, color=colormap(norm(colors)), angles='xy',
           scale_units='xy', scale=1, pivot='mid')  # colormap(norm(colors))
'''
0.00687088x+ 0.26634103y-0.6658108=0
z=0
y=-0.00687088x/0.26634103+0.6658108
'''
# w1 = m.weight.data[0][0].item()
# w2 = m.weight.data[0][1].item()
# b = m.bias.data[0].item()
# plt.plot(x, -w1 / w2 * x - b)
plt.show()
