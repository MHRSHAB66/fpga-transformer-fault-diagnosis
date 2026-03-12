import torch
import torch.nn as nn

from src.config import INPUT_LENGTH, INPUT_CHANNELS, HIDDEN_DIM, NUM_CLASSES


class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = INPUT_LENGTH * INPUT_CHANNELS

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        return self.model(x)