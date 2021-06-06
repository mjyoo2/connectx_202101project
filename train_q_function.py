from model_based_connectx import CNNPolicy
from kaggle_environments import make
from collections import defaultdict
from data_loader import Transitions
from tqdm import tqdm
from torch.utils.data import DataLoader

import torch
import numpy as np

env = make("connectx", debug=True)
configuration = env.configuration
transitions = Transitions()
state_dict = defaultdict(list)
for i in tqdm(range(len(transitions))):
    states, actions, rewards = transitions[i]
    state_dict[str(states[0])].append(rewards[0])
train_loader = DataLoader(transitions, batch_size=1024, shuffle=True, )
Q_function = CNNPolicy()
optimizer = torch.optim.Adam(Q_function.parameters(), lr=0.001)
epochs = 50
Q_function.train()
for epoch in range(epochs):
    tbar = tqdm(train_loader)
    for i, sample in enumerate(tbar):
        states, actions, rewards = sample
        q_value = Q_function(states.float())
        real_q_value = q_value.clone().detach()
        for idx in range(real_q_value.shape[0]):
            real_q_value[idx][0] = np.mean(state_dict[str(states[idx][0].numpy())])

        optimizer.zero_grad()
        reg = 0.05
        loss = F.smooth_l1_loss(q_value, real_q_value) + reg
        loss.backward()
        optimizer.step()
        tbar.set_description('epochs %d: loss: %.7f'%(epoch, loss.item()))
torch.save(Q_function.state_dict(), './Q_function_2.pth')
