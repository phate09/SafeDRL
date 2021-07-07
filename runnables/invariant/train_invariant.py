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


class TrainedPolicyDataset(torch.utils.data.Dataset):
    def __init__(self, path, size=(500, 1000, 100), seed=1234):
        self.size = size
        config = get_PPO_config(1234)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(path)
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        config = {"cost_fn": 1, "simplified": True}
        self.env = StoppingCar(config)
        self.env.seed(seed)
        dataset = []
        while len(dataset) < size[0]:
            state_np = self.env.reset()  # only starting states
            state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]  # pick just delta_x and delta_v
            action = torch.argmax(sequential_nn(state_reduced)).item()
            next_state_np, reward, done, _ = self.env.step(action)
            dataset.append((torch.zeros(state_np.size).float(), torch.from_numpy(state_np).float()))

        while len(dataset) < size[0] + size[1]:
            state_np = self.env.random_sample()
            state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]  # pick just delta_x and delta_v
            action = torch.argmax(sequential_nn(state_reduced)).item()
            next_state_np, reward, done, _ = self.env.step(action)
            if done is True:  # only unsafe states
                dataset.append((torch.from_numpy(state_np).float(), torch.zeros(state_np.size).float()))
            else:
                dataset.append((torch.from_numpy(state_np).float(), torch.from_numpy(next_state_np).float()))
        while len(dataset) < size[0] + size[1] + size[2]:
            state_np = self.env.random_sample()
            state_reduced = torch.from_numpy(state_np).float().unsqueeze(0)[:, -2:]  # pick just delta_x and delta_v
            action = torch.argmax(sequential_nn(state_reduced)).item()
            next_state_np, reward, done, _ = self.env.step(action)
            if done is True:  # only unsafe states
                dataset.append((torch.from_numpy(state_np).float(), torch.zeros(state_np.size).float()))
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
            A = (torch.relu(self.model(x)) if x.sum().value() != 0 else torch.tensor([1]).float())
            B = (torch.relu(-self.model(y)) if y.sum().value() != 0 else torch.tensor([1]).float())
            mean += A * B / len(input)
        return mean  # torch.mean(torch.relu(self.model(input)) * torch.relu(-self.model(target)))
        # return F.mse_loss(input, target, reduction=self.reduction)


class SafetyTrainingOperator(TrainingOperator):

    def setup(self, config):
        path1 = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58"
        batch_size = config["batch_size"]
        train_data = TrainedPolicyDataset(path1, size=(500, 10000, 10), seed=3451)
        val_data = TrainedPolicyDataset(path1, size=(0, 1000, 0), seed=4567)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        model = torch.nn.Linear(train_data.env.observation_space.shape[0], 1)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.get("lr", 1e-4))
        loss = SafetyLoss(model)  # torch.nn.MSELoss()

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

        return {"train_loss": loss.value(), NUM_SAMPLES: features.size(0)}

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
        features, targets = batch
        if self.use_gpu:
            features = features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        # compute output

        with self.timers.record("eval_fwd"):
            loss = self.criterion(features, targets)
            # loss = criterion(output, target)
            # _, predicted = torch.max(output.data, 1)

        num_samples = targets.size(0)
        num_correct = 0  # todo find a good value (predicted == target).sum().item()
        return {
            "val_loss": loss.value(),
            "val_accuracy": num_correct / num_samples,
            NUM_SAMPLES: num_samples
        }

enable_training = False
if enable_training:
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
    for i in range(20):
        stats = trainer1.train()
        print(stats)

    print(trainer1.validate())
    torch.save(trainer1.state_dict(), "checkpoint.pt")

    m = trainer1.get_model()
    print(f"trained weight: {m.weight.data.numpy()}, bias: {m.bias.data.numpy()}")
    trainer1.shutdown()
    print("success!")

m = torch.nn.Linear(2,1)
m.weight.data = torch.tensor([[0.00687088, 0.26634103]])
m.bias.data = torch.tensor([-0.6658108])
path1 = "/home/edoardo/ray_results/tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58"
val_data = TrainedPolicyDataset(path1, size=(0, 1000, 0), seed=4567)
for i in range(len(val_data)):
    value = m(val_data[i][0]).item()
    print(f"(delta_v,delta_x) {val_data[i][0].numpy()} value:{value}")
