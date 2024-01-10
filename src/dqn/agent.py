import torch

from dqn import DQN
from replay_memory import ReplayMemory
from random import random
from src.dqn.constants import (
    POLICY_NET_PATH,
    TARGET_NET_PATH,
    MODELS_PATH,
    EPSILON_MAX,
    EPSILON_MIN,
    EPSILON_DECAY,
    GAMMA,
    BATCH_SIZE,
    DEVICE,
)
import numpy as np

from src.utils.helpers import check_if_dirs_exist


class Agent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.epsilon: float = EPSILON_MAX
        self.replay_memory: ReplayMemory = ReplayMemory()

        self.policy_net: DQN = DQN().to(DEVICE)
        self.target_net: DQN = DQN().to(DEVICE)
        self.update_target_net()

        self.policy_net.train()
        self.target_net.eval()

        self.total_loss = 0.0

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        if random() < self.epsilon:
            return self.action_space.sample()

        last3_frames = self.replay_memory.get_last3_frames()
        stacked_frames = last3_frames + [state]
        stacked_frames = torch.from_numpy(np.array(stacked_frames)).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action = torch.argmax(self.policy_net(stacked_frames))

        return action.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)

    def set_loss(self, loss):
        self.total_loss = loss

    def please_learn(self):
        if len(self.replay_memory) < BATCH_SIZE:
            return

        states, actions, rewards, dones, next_states = self.replay_memory.sample()

        predicted_qs = self.policy_net(states).gather(1, actions)
        target_qs = self.target_net(next_states)
        target_qs = torch.max(target_qs, dim=1).values.reshape(-1, 1)
        target_qs[dones] = 0.0
        target_qs = rewards + (GAMMA * target_qs)

        loss = self.policy_net.loss(predicted_qs, target_qs)
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()

        self.total_loss += loss.item()

    def save(self):
        check_if_dirs_exist([MODELS_PATH])
        torch.save(self.policy_net.state_dict(), POLICY_NET_PATH)
        torch.save(self.target_net.state_dict(), TARGET_NET_PATH)

    def load(self):
        self.policy_net.load_state_dict(torch.load(POLICY_NET_PATH))
        self.target_net.load_state_dict(torch.load(TARGET_NET_PATH))
        self.target_net.eval()
