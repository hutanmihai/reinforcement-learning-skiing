import torch
import numpy as np

from src.dqn.constants import MEMORY_CAPACITY, BATCH_SIZE, DEVICE


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

    def get_last3_frames(self):
        return [self.states[-3], self.states[-2], self.states[-1]]

    def _index_valid(self, index):
        if any(self.dones[i] for i in range(index - 3, index + 1)):
            return False
        return True

    def sample(self):
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []

        while len(states) < BATCH_SIZE:
            index = np.random.randint(4, len(self) - 1)
            if self._index_valid(index):
                states.append(
                    [self.states[index - 3], self.states[index - 2], self.states[index - 1], self.states[index]]
                )
                next_states.append(
                    [
                        self.next_states[index - 2],
                        self.next_states[index - 1],
                        self.next_states[index],
                        self.next_states[index + 1],
                    ]
                )
                actions.append(self.actions[index])
                rewards.append(self.rewards[index])
                dones.append(self.dones[index])

        states = torch.from_numpy(np.array(states)).float().to(DEVICE)
        actions = torch.from_numpy(np.array(actions)).reshape((-1, 1)).to(torch.int64).to(DEVICE)
        rewards = torch.from_numpy(np.array(rewards)).float().reshape((-1, 1)).to(DEVICE)
        dones = torch.from_numpy(np.array(dones)).reshape((-1, 1)).to(torch.bool).to(DEVICE)
        next_states = torch.from_numpy(np.array(next_states)).float().to(DEVICE)

        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.states)
