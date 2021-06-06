import numpy as np
import torch as th
from torch import nn as nn
import torch.nn.functional as F
from torch import tensor

def make_obs(obs):
    board = obs['board']
    mark = obs['mark']
    if mark == 2:
        board = np.where(board == 1, -1, board)
        board = np.where(board == 2, 1, board)
    else:
        board = np.where(board == 2, -1, board)
    return board.flatten()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.shared1 = nn.Linear(42, 64)
        self.shared2 = nn.Linear(64, 64)
        self.policy1 = nn.Linear(64, 64)
        self.action = nn.Linear(64, 7)

    def forward(self, x):
        input_data = x
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))
        x = F.relu(self.policy1(x))
        x = self.action(x)
        for i in range(7):
            if input_data[0][i] != 0:
                x[0][i] = -1e+10
        x = x.argmax()
        return x

class agent(object):
    def __init__(self, state_dict):
        state_dict = {
            'shared1.weight': state_dict['mlp_extractor.shared_net.0.weight'],
            'shared1.bias': state_dict['mlp_extractor.shared_net.0.bias'],
            'shared2.weight': state_dict['mlp_extractor.shared_net.2.weight'],
            'shared2.bias': state_dict['mlp_extractor.shared_net.2.bias'],

            'policy1.weight': state_dict['mlp_extractor.policy_net.0.weight'],
            'policy1.bias': state_dict['mlp_extractor.policy_net.0.bias'],

            'action.weight': state_dict['action_net.weight'],
            'action.bias': state_dict['action_net.bias'],
        }
        model = Net()
        model = model.float()
        model.load_state_dict(state_dict)
        model = model.to('cpu')
        self.model = model.eval()

    def __call__(self, obs, config):
        obs = tensor(make_obs(obs)).reshape(1, config.rows * config.columns).float()
        action = self.model(obs)
        return int(action)