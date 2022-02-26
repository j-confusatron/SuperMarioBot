import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DQN(nn.Module):

    def __init__(self, image_depth, n_actions):
        super().__init__()
        self.image_depth = image_depth
        self.n_actions = n_actions

        # Intiially run the image through a series of convolutional layers.
        # Input image 84x84
        self.cnn = nn.Sequential(
            nn.Conv2d(image_depth, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Pass the convolved image through two linear layers to produce the output.
        self.linear = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, obs) -> torch.tensor:
        out = self.cnn(obs)
        out = self.linear(out)
        return out

    def act(self, obs, epsilon=0.0, device=torch.device('cpu')):
        if random.random() >= epsilon:
            action = torch.argmax(self.forward(obs.unsqueeze(0))).item()
        else:
            action = random.randrange(0, self.n_actions)
        return action