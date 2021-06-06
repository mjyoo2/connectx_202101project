import pickle as pkl
import torch.utils.data as data
import numpy as np
import os
import torch
from collections import defaultdict

from torchvision import transforms

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample = torch.from_numpy(sample).float()
        return sample

class Transitions(data.Dataset):
    def __init__(self):
        super().__init__()
        states = []
        actions = []
        rewards = []
        for i in range(10):
            with open('./replay/replay_{}.pkl'.format(i), 'rb') as f:
                data = pkl.load(f)
                states.append(data['states'][:data['idx']])
                actions.append(data['actions'][:data['idx']])
                rewards.append(data['rewards'][:data['idx']])
        self.states = np.concatenate(states)
        self.actions = np.concatenate(actions)
        self.rewards = np.concatenate(rewards)


    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.rewards[index]

    def __len__(self):
        return self.states.shape[0]

    def transform(self, sample):
        composed_transforms = transforms.Compose([ToTensor()])
        return composed_transforms(sample)