import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.torch.examples.train_example import LinearDataset
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ToyDataset(torch.utils.data.Dataset):
    """generates 2d points around -10 + 10 with true value between range -1 +1"""

    def __init__(self, size=10000):
        x = np.random.random((size, 2)) * 10 - 5
        self.x = torch.from_numpy(x).float()
        y = np.array([1 if 1 > x[0] > -1 and 1 > x[1] > -1 else 0 for x in x])
        self.y = torch.from_numpy(y).long()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class CustomTrainingOperator(TrainingOperator):
    def setup(self, config):
        # Load data.
        train_loader = DataLoader(ToyDataset(), config["batch_size"])
        val_loader = DataLoader(ToyDataset(), config["batch_size"])

        # Create model.
        model = Net()

        # Create optimizer.
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        # Create loss.
        loss = torch.nn.CrossEntropyLoss()

        # Register model, optimizer, and loss.
        self.model, self.optimizer, self.criterion = self.register(models=model, optimizers=optimizer, criterion=loss)

        # Register data loaders.
        self.register_data(train_loader=train_loader, validation_loader=val_loader)


ray.init(local_mode=True)

trainer1 = TorchTrainer(training_operator_cls=CustomTrainingOperator, num_workers=1, use_gpu=False, config={"batch_size": 64})

for i in range(10):
    stats = trainer1.train()
    print(stats)
    val_stats = trainer1.validate()
    print(val_stats)
model = trainer1.get_model()
torch.save(model.state_dict(), "model.pt")  # save the model in the current folder
trainer1.shutdown()
print("success!")
