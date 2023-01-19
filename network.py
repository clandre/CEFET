import torch
from  torch import nn


def CreateModel():
    model = nn.Sequential(
        nn.Linear(187,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,16),
        nn.ReLU(),
        nn.Linear(16,5)
    )
    
    return model


