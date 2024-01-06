from random import sample
import torch
import numpy as np

from src.dqn.constants import MEMORY_CAPACITY, BATCH_SIZE


class ReplayMemory:
    def __init__(self):
        self.capacity = MEMORY_CAPACITY
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.index: int = 0

    def store(self, state, action, reward, done, next_state):
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.next_states.append(next_state)
        else:
            self.states[self.index] = state
            self.actions[self.index] = action
            self.rewards[self.index] = reward
            self.dones[self.index] = done
            self.next_states[self.index] = next_state

        self.index = (self.index + 1) % self.capacity

    def sample(self, device):
        indices_to_sample = sample(range(len(self)), BATCH_SIZE)
        states = torch.from_numpy(np.array(self.states)[indices_to_sample]).float().to(device)
        actions = torch.from_numpy(np.array(self.actions)[indices_to_sample]).to(device).reshape((-1, 1))
        rewards = torch.from_numpy(np.array(self.rewards)[indices_to_sample]).float().to(device).reshape((-1, 1))
        dones = torch.from_numpy(np.array(self.dones)[indices_to_sample]).to(device).reshape((-1, 1))
        next_states = torch.from_numpy(np.array(self.next_states)[indices_to_sample]).float().to(device)

        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.states)
