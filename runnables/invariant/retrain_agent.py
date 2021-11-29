import os
import random

import matplotlib.pyplot as plt
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
import torch.nn
import torch.nn.functional as F
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

import utils
from environment.stopping_car import StoppingCar
from training.ppo.tune.tune_train_PPO_car import get_PPO_config
from training.ray_utils import convert_ray_policy_to_sequential

# config, trainer = get_PPO_trainer(use_gpu=0)
print(torch.cuda.is_available())


class GridSearchDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=False, size=16000):
        dataset = []
        param_grid = {'delta_v': np.arange(-30, 30, 0.2), 'delta_x': np.arange(-10, 40, 0.2)}
        for parameters in ParameterGrid(param_grid):
            delta_v = parameters["delta_v"]
            delta_x = parameters["delta_x"]
            state_np = np.array([delta_v, delta_x])
            dataset.append((torch.from_numpy(state_np).float()))
        self.dataset = dataset
        if shuffle:
            random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class RetrainLoss(_Loss):
    def __init__(self, invariant_model):
        super().__init__()
        self.invariant_model = invariant_model

    def forward(self, states: Tensor, actions_prob: Tensor) -> Tensor:
        # device = states.device
        accelerations = torch.tensor([[-3.0, 3.0]], device=states.get_device())  # (actions - 0.5) * 6
        next_delta_v = states[:, 0].unsqueeze(1).repeat(1, 2) + accelerations * 0.1
        next_delta_x = states[:, 1].unsqueeze(1).repeat(1, 2) + next_delta_v * 0.1
        next_states = torch.stack([next_delta_v.flatten(), next_delta_x.flatten()], dim=1)
        A = F.relu(self.invariant_model(states.repeat(2, 1)))
        B = F.relu(self.invariant_model(next_states) * actions_prob.flatten().unsqueeze(1))
        return (A * B).mean()


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
        sequential_nn = convert_ray_policy_to_sequential(policy)  # load the agent model
        sequential_nn.cuda()

        model = sequential_nn
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.get("lr", 1e-3))
        loss = RetrainLoss(invariant_model)  # torch.nn.MSELoss()

        self.models, self.optimizer, self.criterion = self.register(
            models=[model, invariant_model],
            optimizers=optimizer,
            criterion=loss)
        self.model = self.models[0]
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
        model = self.model
        # unpack features into list to support multiple inputs model
        features = batch
        # Create non_blocking tensors for distributed training
        if self.use_gpu:
            features = features.cuda(non_blocking=True)
        # Compute output.
        with self.timers.record("fwd"):
            action_prob = torch.softmax(model(features), dim=1)
            eps = torch.finfo(action_prob.dtype).eps
            action_prob = action_prob.clamp(min=eps, max=1 - eps)
            log_probs = torch.log(action_prob)
            # actions = torch.argmax(action_prob, dim=1)
            # log_probs = Categorical(action_prob).log_prob(actions)
            loss = criterion(features, log_probs)

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
        features = batch
        if self.use_gpu:
            features = features.cuda(non_blocking=True)

        # compute output

        with self.timers.record("eval_fwd"):
            action_prob = torch.softmax(model(features), dim=1)
            eps = torch.finfo(action_prob.dtype).eps
            action_prob = action_prob.clamp(min=eps, max=1 - eps)
            log_probs = torch.log(action_prob)
            # actions = torch.argmax(action_prob, dim=1)
            # log_probs = Categorical(action_prob).log_prob(actions)
            loss = criterion(features, log_probs)
            # loss = criterion(output, target)
            # _, predicted = torch.max(output.data, 1)

        num_samples = features.size(0)
        num_correct = 0  # todo find a good value (predicted == target).sum().item()
        return {
            "val_loss": loss.item(),
            "val_accuracy": num_correct / num_samples,
            NUM_SAMPLES: num_samples
        }


if __name__ == '__main__':

    ray.init(local_mode=True)
    path1 = os.path.join(utils.get_save_dir(),"tune_PPO_stopping_car/PPO_StoppingCar_acc24_00001_1_cost_fn=0,epsilon_input=0_2021-01-21_02-30-49/checkpoint_58/checkpoint-58")
    path_invariant = os.path.join(utils.get_save_dir(),"invariant_checkpoint_old.pt")
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
                "batch_size": 1024,  # used in data_creator
                "path": path1,  # path to load the agent nn
                "path_invariant": path_invariant,  # the path to the invariant network
            },
            backend="auto",
            scheduler_step_freq="epoch")
        for i in range(50):
            stats = trainer1.train()
            print(stats)

        print(trainer1.validate())
        torch.save(trainer1.state_dict(), os.path.join(utils.get_save_dir(),"checkpoint.pt"))
        torch.save(trainer1.get_model()[0].state_dict(), os.path.join(utils.get_save_dir(),"retrained_agent.pt"))
        agent_model, invariant_model = trainer1.get_model()
    else:
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        sequential_nn.load_state_dict(torch.load(os.path.join(utils.get_save_dir(),"retrained_agent.pt")))
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
    changed_indices = []
    for i, data in enumerate(random.sample(val_data.dataset, k=7000)):
        value = torch.tanh(invariant_model(data)).item()
        action = torch.argmax(agent_model(data)).item()
        next_state_np, reward, done, _ = StoppingCar.compute_successor(data.numpy(), action)
        old_action = torch.argmax(old_agent_model(data)).item()
        next_state_np_old, _, _, _ = StoppingCar.compute_successor(data.numpy(), old_action)
        x_data.append(data.numpy())
        xprime_data.append(next_state_np)
        y_data.append(value)
        old_xprime_data.append(next_state_np_old)
        if action != old_action:
            changed_indices.append(i)

    x_data = np.array(x_data)
    xprime_data = np.array(xprime_data)
    old_xprime_data = np.array(old_xprime_data)
    changed_indices = np.array(changed_indices)
    y_data = np.array(y_data)
    x = x_data[:, 0][changed_indices]
    y = x_data[:, 1][changed_indices]
    u = xprime_data[:, 0][changed_indices] - x_data[:, 0][changed_indices]
    v = xprime_data[:, 1][changed_indices] - x_data[:, 1][changed_indices]
    x_full = x_data[:, 0]
    y_full = x_data[:, 1]
    u_full = xprime_data[:, 0] - x_data[:, 0]
    v_full = xprime_data[:, 1] - x_data[:, 1]
    u_old_full = old_xprime_data[:, 0] - x_data[:, 0]
    v_old_full = old_xprime_data[:, 1] - x_data[:, 1]

    colors = y_data[changed_indices]
    colors_full = y_data

    norm = Normalize(vmax=1.0, vmin=-1.0)
    norm.autoscale(colors)
    # we need to normalize our colors array to match it colormap domain
    # which is [0, 1]

    colormap = cm.bwr
    plt.figure(figsize=(18, 16), dpi=80)
    # plt.quiver(x, y, old_u, old_v, color="yellow", angles='xy',
    #            scale_units='xy', scale=1, pivot='mid', zorder=1)
    plt.quiver(x, y, u, v, color=colormap(norm(colors)), angles='xy',
               scale_units='xy', scale=1, pivot='mid', zorder=0)  # colormap(norm(colors))
    plt.title("Diff")
    plt.xlim([-30, 30])
    plt.ylim([-10, 40])
    plt.show()
    plt.figure(figsize=(18, 16), dpi=80)
    # plt.quiver(x, y, old_u, old_v, color="yellow", angles='xy',
    #            scale_units='xy', scale=1, pivot='mid', zorder=1)
    plt.quiver(x_full, y_full, u_full, v_full, color=colormap(norm(colors_full)), angles='xy',
               scale_units='xy', scale=1, pivot='mid', zorder=0)  # colormap(norm(colors))
    plt.title("New")
    plt.xlim([-30, 30])
    plt.ylim([-10, 40])
    plt.show()
    plt.figure(figsize=(18, 16), dpi=80)
    plt.quiver(x_full, y_full, u_old_full, v_old_full, color=colormap(norm(colors_full)), angles='xy',
               scale_units='xy', scale=1, pivot='mid', zorder=0)  # colormap(norm(colors))
    plt.title("Old")
    plt.xlim([-30, 30])
    plt.ylim([-10, 40])
    plt.show()
