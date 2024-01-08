import torch
import torch.nn as nn
from torch import optim
from src.dqn.constants import INPUT_SHAPE, NUM_ACTIONS, LEARNING_RATE


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(INPUT_SHAPE[0], 32, kernel_size=8, stride=4)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.norm3 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(64 * 6 * 6, 512)
        self.output = nn.Linear(512, NUM_ACTIONS)

        # TODO: Maybe use RMSProp?
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()

    def _forward_features(self, x):
        x = torch.relu(self.norm1(self.conv1(x)))
        x = torch.relu(self.norm2(self.conv2(x)))
        x = torch.relu(self.norm3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        x = self.output(x)
        return x
