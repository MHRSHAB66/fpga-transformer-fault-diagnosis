import torch
import torch.nn as nn

from src.config import INPUT_LENGTH, INPUT_CHANNELS, NUM_CLASSES

# HIDDEN_DIM was previously in config.py under an "MLP baseline" section.
# This baseline is archived (Stage 3 sanity check only), so the constant now
# lives here to keep config.py free of MLP-only values.
HIDDEN_DIM = 64


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