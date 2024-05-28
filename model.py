import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.optim import Adam
from tqdm.auto import tqdm
import os

# Number of elements in the group
GROUP_ORDER = 4

class LiftingConvolution(nn.Conv2d):
    group_order = GROUP_ORDER
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        torch.nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        weight = torch.cat([
            torch.rot90(self.weight, k=i, dims=[-2, -1])
            for i in range(self.group_order)
        ], dim=0)
        x = self._conv_forward(input=x, weight=weight, bias=None)
        assert x.shape == (x.shape[0], self.out_channels * self.group_order, x.shape[2], x.shape[3])
        return x


class GroupConvolution(nn.Conv2d):
    group_order = GROUP_ORDER
    
    def __init__(self, in_channels, **kwargs):
        super().__init__(in_channels=self.group_order * in_channels, **kwargs)
        torch.nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        weight = self.weight
        assert weight.shape == (self.out_channels, self.in_channels, weight.shape[-2], weight.shape[-1])
        
        weight = weight.reshape(
            self.out_channels, 
            self.group_order,
            self.in_channels // self.group_order, 
            weight.shape[-2], weight.shape[-1])

        weight = torch.cat([
            torch.roll(
                torch.rot90(weight, k=i, dims=[-2, -1]), 
                shifts=i, dims=1
            )
            for i in range(self.group_order)
        ], dim=0)

        weight = weight.reshape(
            self.out_channels * self.group_order, 
            self.in_channels,
            weight.shape[-2], weight.shape[-1]
        )
        x = self._conv_forward(input=x, weight=weight, bias=None)
        assert x.shape == (x.shape[0], self.out_channels * self.group_order, x.shape[2], x.shape[3])
        return x

class RandomRot90:
    rng = np.random.default_rng(seed=0)

    def __call__(self, x):
        return torch.rot90(x, self.rng.integers(0, 3), dims=(-2, -1))


class GroupCNN(nn.Module):
    group_order = GROUP_ORDER

    def __init__(self):
        super().__init__()
        self.hidden_channel_number = 16
        self.hidden_layer_number = 5
        self.out_channels = 10
        self.kernel_size = 3
        self.padding = 1
        self.image_size = 28

        self.convs = nn.Sequential(
            LiftingConvolution(in_channels=1, out_channels=self.hidden_channel_number, kernel_size=self.kernel_size,
                               padding=self.padding),
            nn.ReLU(),
            *(
                nn.Sequential(
                    GroupConvolution(in_channels=self.hidden_channel_number, out_channels=self.hidden_channel_number,
                                     kernel_size=self.kernel_size, padding=self.padding),
                    nn.ReLU()
                )
                for _ in range(self.hidden_layer_number)
            ),
        )
        
        self.linear = nn.Linear(self.hidden_channel_number, self.out_channels)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.shape[0], self.group_order, self.hidden_channel_number, x.shape[2], x.shape[3])
        x = x.mean(dim=[1, -2, -1])
        assert x.shape == (x.shape[0], self.hidden_channel_number)
        x = self.linear(x)
        return x

class VanillaCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_channel_number = 16
        self.hidden_layer_number = 5
        self.out_channels = 10
        self.kernel_size = 3
        self.padding = 1
        self.image_size = 28

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.hidden_channel_number, kernel_size=self.kernel_size,
                      padding=self.padding),
            nn.ReLU(),
            *(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.hidden_channel_number, out_channels=self.hidden_channel_number,
                              kernel_size=self.kernel_size, padding=self.padding),
                    nn.ReLU()
                )
                for _ in range(self.hidden_layer_number)
            ),
        )

        self.linear = nn.Linear(self.hidden_channel_number, self.out_channels)

    def forward(self, x):
        x = self.convs(x)
        x = x.mean(dim=[-2, -1])
        x = self.linear(x)
        return x

class Trainer:
    def __init__(self):
        self.optimizer = None
        self.model = None
        self.train_dl = None
        self.test_dl = None
        self.root = './data'
        self.epochs = 50
        self.batch_size = 128
        self.subset_size = None
        self.checkpoint_file = os.path.join(".", "trained_models")
        self.load_checkpoint = True
        self.test_model = True
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def get_dl(self, transform, train=True):
        ds = FashionMNIST(root=self.root, train=train, transform=transform, download=True)
        if self.subset_size is not None:
            ds = Subset(ds, range(self.subset_size))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=train, pin_memory=True)
        return dl

    def run_trainer(self, use_vanilla=False):
        if use_vanilla:
            # use data augmentation for vanilla CNN
            self.train_dl = self.get_dl(transforms.Compose([transforms.ToTensor(), RandomRot90()]), train=True)
            self.checkpoint_file = os.path.join(self.checkpoint_file, "vanilla_cnn_model.pt")
        else:
            # do not use data augmentation for group CNN
            self.train_dl = self.get_dl(transforms.Compose([transforms.ToTensor()]), train=True)
            self.checkpoint_file = os.path.join(self.checkpoint_file, "model.pt")
        
        self.test_dl = self.get_dl(transforms.Compose([transforms.ToTensor(), RandomRot90()]), train=False)

        if use_vanilla:
            self.model = VanillaCNN().to(self.device)
        else:
            self.model = GroupCNN().to(self.device)

        from pathlib import Path
        if self.load_checkpoint and Path(self.checkpoint_file).exists():
            self.model.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
            if self.test_model:
                self.test()
                return

        self.optimizer = Adam(self.model.parameters(), lr=1e-2, weight_decay=1e-5)

        for epoch in range(self.epochs):
            self.model.train()
            with tqdm(self.train_dl, desc=f'Train Epoch {epoch}') as pbar:
                for inp, target in pbar:
                    self.optimizer.zero_grad()
                    accuracy, out, target = self.forward(inp, target)
                    loss = nn.CrossEntropyLoss()(out, target)
                    loss.backward()
                    self.optimizer.step()

                    pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:0%}")

            self.model.eval()
            with torch.no_grad():
                with tqdm(self.test_dl, desc=f'Test Epoch {epoch}') as pbar:
                    for inp, target in pbar:
                        accuracy, out, target = self.forward(inp, target)
                        pbar.set_postfix(accuracy=f"{accuracy:0%}")
            
            torch.save(self.model.state_dict(), self.checkpoint_file)

    def forward(self, inp, target):
        inp, target = inp.to(self.device), target.to(self.device)
        out = self.model(inp)
        correct = torch.eq(out.argmax(dim=1), target)
        accuracy = correct.float().mean().item()
        return accuracy, out, target
    def test(self):
        total_accuracy = 0
        self.model.eval()
        with torch.no_grad():
            with tqdm(self.test_dl, desc=f'Test') as pbar:
                for inp, target in pbar:
                    accuracy, _, target = self.forward(inp, target)
                    total_accuracy += accuracy*len(inp)
                    pbar.set_postfix(accuracy=f"{accuracy:0%}")
        print(f"Total accuracy: {total_accuracy/len(self.test_dl.dataset):0%}")


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run_trainer()

    vanilla_trainer = Trainer()
    vanilla_trainer.run_trainer(use_vanilla=True)