from typing import Dict

import torch
from torch import nn


class GenreClassifier(nn.Module):
    genre_map: Dict[str, int]
    
    def __init__(self) -> None:
        super().__init__()

        self.genre_map = {}

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 1, 512, padding=256),
            nn.ReLU(),
            nn.Conv1d(1, 1, 512, padding=256),
            nn.ReLU(),
        )
        self.lin1 = nn.Sequential(
            nn.Linear(21848, 16384),
            nn.ReLU(),
            nn.Linear(16384, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 16)
        )

    def add_genre(self, genre: str):
        if genre in self.genre_map:
            return
        
        self.genre_map[genre] = len(self.genre_map)

    def shape_conv_to_lin(self, x):
        return x.view(x.shape[0], -1)

    def shape_lin_to_conv(self, x):
        return x.view(x.shape[0], 1, -1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.shape_conv_to_lin(x)
        logits = self.lin1(x)

        return logits


def train(dataloader, model, loss_fn, optimizer, device):    
    """
    Trains a model using the given dataloader, loss function, optimizer, and device.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader object that provides the training data.
        model (torch.nn.Module): The model to be trained.
        loss_fn (callable): The loss function used to calculate the loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        device (torch.device): The device on which the training will be performed.
    
    Returns:
        None
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 2 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            # print(f"loss: {loss:>7f}, [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device, verbose: bool=False):
    """
    Calculates the test loss and accuracy of a given model on a given test dataset.

    Parameters:
    - dataloader (torch.utils.data.DataLoader): The data loader for the test dataset.
    - model (torch.nn.Module): The model to be evaluated.
    - loss_fn (torch.nn.Module): The loss function used to calculate the test loss.
    - device (torch.device): The device on which the model and data will be processed.

    Returns:
    - correct (float): The accuracy of the model on the test dataset.
    - test_loss (float): The average loss of the model on the test dataset.
    """
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * len(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if verbose:
                for result, true_result in zip(pred, y):
                    print(result, true_result)

    test_loss /= size
    correct /= size
    return correct, test_loss
