import torch.nn as nn
from torch import optim
from src.constants import INPUT_SHAPE, NUM_ACTIONS, LEARNING_RATE, DEVICE


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(INPUT_SHAPE[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 6 * 6, 512)
        self.output = nn.Linear(512, NUM_ACTIONS)

        self.relu = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # self.optimizer = optim.RMSprop(self.parameters(), lr=LEARNING_RATE, momentum=0.95, eps=0.01)

        self.loss = nn.HuberLoss()
        # self.loss = nn.MSELoss()

    def _forward_features(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.relu(self.fc(x))
        x = self.output(x)
        return x
