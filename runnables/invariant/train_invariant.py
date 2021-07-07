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

from agents.ray_utils import convert_ray_policy_to_sequential
from environment.stopping_car import StoppingCar
import ray.rllib.agents.ppo as ppo
from agents.ppo.tune.tune_train_PPO_car import get_PPO_config

print(torch.cuda.is_available())
ray.init(local_mode=True)
# config, trainer = get_PPO_trainer(use_gpu=0)

config = get_PPO_config(1234)
trainer = ppo.PPOTrainer(config=config)
# trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-12-30_17-06-3265yz3d63/checkpoint_65/checkpoint-65") # 5e-2 ~19.8 delta x
# trainer.restore("/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_e6ed1_00000_0_cost_fn=0_2021-01-15_19-57-40/checkpoint_440/checkpoint-440")
# trainer.restore("/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00006_6_cost_fn=0,epsilon_input=0_2021-01-17_12-44-54/checkpoint_41/checkpoint-41")
# trainer.restore("/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58")
trainer.restore("/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_c1c7e_00005_5_cost_fn=0,epsilon_input=0.1_2021-01-17_12-41-27/checkpoint_10/checkpoint-10")
policy = trainer.get_policy()
sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
del trainer


class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, a, b, size=1000):
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(a * x + b)

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


class TrainedPolicyDataset(torch.utils.data.Dataset):
    def __init__(self, path, size=(1000, 500, 10), seed=1234):
        self.size = size
        config = get_PPO_config(1234)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(path)
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        config = {"cost_fn": 1}
        self.env = StoppingCar(config)
        self.env.seed(seed)
        dataset = []
        while len(dataset) < size[0]:
            state_np = self.env.reset()  # only starting states
            state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]  # pick just delta_x and delta_v
            action = torch.argmax(sequential_nn(state_reduced)).item()
            next_state_np, reward, done, _ = self.env.step(action)
            dataset.append((1, torch.from_numpy(state_np).float()))

        while len(dataset) < size[0] + size[1]:
            state_np = self.env.random_sample()
            state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]  # pick just delta_x and delta_v
            action = torch.argmax(sequential_nn(state_reduced)).item()
            next_state_np, reward, done, _ = self.env.step(action)
            dataset.append((torch.from_numpy(state_np).float(), torch.from_numpy(next_state_np).float()))
        while len(dataset) < size[0] + size[1] + size[2]:
            state_np = self.env.random_sample()
            state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]  # pick just delta_x and delta_v
            action = torch.argmax(sequential_nn(state_reduced)).item()
            next_state_np, reward, done, _ = self.env.step(action)
            if done is True:  # only unsafe states
                dataset.append((torch.from_numpy(state_np).float(), 1))
        self.dataset = dataset

    def __len__(self):
        return sum(self.size)

    def __getitem__(self, index):
        return self.dataset[index]
        # state_np = self.env.random_sample()
        # state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]  # pick just delta_x and delta_v
        # action = torch.argmax(sequential_nn(state_reduced)).item()
        # next_state_np, reward, done, _ = self.env.step(action)
        # return torch.from_numpy(state_np).float(), torch.from_numpy(next_state_np).float()


class SafetyLoss(_Loss):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mean = torch.tensor([0]).float()
        for x, y in zip(input, target):
            A = (torch.relu(self.model(x)) if x.size().numel() != 1 else torch.tensor([1]).float())
            B = (torch.relu(self.model(y)) if y.size().numel() != 1 else torch.tensor([1]).float())
            mean += A * B / len(input)
        return mean  # torch.mean(torch.relu(self.model(input)) * torch.relu(-self.model(target)))
        # return F.mse_loss(input, target, reduction=self.reduction)


def model_creator(config):
    """Returns a torch.nn.Module object."""
    return torch.nn.Linear(1, config.get("hidden_size", 1))


def data_creator(config):
    """Returns training dataloader, validation dataloader."""
    train_dataset = LinearDataset(2, 5, size=config.get("data_size", 1000))
    val_dataset = LinearDataset(2, 5, size=config.get("val_size", 400))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32))
    return train_loader, validation_loader


def scheduler_creator(optimizer, config):
    """Returns a learning rate scheduler wrapping the optimizer.
    You will need to set ``TorchTrainer(scheduler_step_freq="epoch")``
    for the scheduler to be incremented correctly.
    If using a scheduler for validation loss, be sure to call
    ``trainer.update_scheduler(validation_loss)``.
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-2))


CustomTrainingOperator = TrainingOperator.from_creators(
    model_creator=model_creator, optimizer_creator=optimizer_creator,
    data_creator=data_creator, scheduler_creator=scheduler_creator,
    loss_creator=torch.nn.MSELoss)


class MyTrainingOperator(TrainingOperator):

    def setup(self, config):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.get("lr", 1e-4))
        loss = torch.nn.MSELoss()

        batch_size = config["batch_size"]
        train_data, val_data = LinearDataset(2, 5), LinearDataset(2, 5)
        train_loader = DataLoader(train_data, batch_size=batch_size)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        self.model, self.optimizer, self.criterion = self.register(
            models=model,
            optimizers=optimizer,
            criterion=loss)

        self.register_data(
            train_loader=train_loader,
            validation_loader=val_loader)


class SafetyTrainingOperator(TrainingOperator):

    def setup(self, config):
        model = torch.nn.Linear(8, 1)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.get("lr", 1e-4))
        loss = SafetyLoss(model)  # torch.nn.MSELoss()

        batch_size = config["batch_size"]
        path1 = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58"

        train_data, val_data = TrainedPolicyDataset(path1, seed=3451), TrainedPolicyDataset(path1, size=(100, 50, 0), seed=4567)
        train_loader = DataLoader(train_data, batch_size=batch_size)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        self.model, self.optimizer, self.criterion = self.register(
            models=model,
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
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        # unpack features into list to support multiple inputs model
        features, targets = batch
        # Create non_blocking tensors for distributed training
        if self.use_gpu:
            features = features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            # features = [
            #     feature.cuda(non_blocking=True) for feature in features
            # ]
            # targets = [
            #     target.cuda(non_blocking=True) for target in targets
            # ]
        # Compute output.
        with self.timers.record("fwd"):
            # output = model(features)

            loss = self.criterion(features, targets)

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


trainer1 = TorchTrainer(
    training_operator_cls=SafetyTrainingOperator,
    num_workers=1,
    use_gpu=True,
    config={
        "lr": 1e-2,  # used in optimizer_creator
        "hidden_size": 1,  # used in model_creator
        "batch_size": 4,  # used in data_creator
    },
    backend="auto",
    scheduler_step_freq="epoch")
for i in range(50):
    stats = trainer1.train()
    print(stats)

print(trainer1.validate())
torch.save(trainer1.state_dict(), "checkpoint.pt")
# trainer1.shutdown()
# print("success!")
# If using Ray Client, make sure to force model onto CPU.
# import ray

m = trainer1.get_model()
print("trained weight: % .2f, bias: % .2f" % (
    m.weight.item(), m.bias.item()))
trainer1.shutdown()
print("success!")
