import numpy as np
import gym
from kaggle_environments import make

def make_obs(obs):
    board = obs['board']
    mark = obs['mark']
    if mark == 2:
        board = np.where(board == 1, -1, board)
        board = np.where(board == 2, 1, board)
    else:
        board = np.where(board == 2, -1, board)
    return board.flatten()


# ConnectX wrapper from Alexis' notebook.
# Changed shape, channel first.
# Changed obs/2.0
class ConnectFourGym(gym.Env):
    def __init__(self, opponent='random', mode=1):
        self.ks_env = make("connectx", debug=False)
        if mode == 1:
            self.env = self.ks_env.train([None, opponent])
        elif mode == 2:
            self.env = self.ks_env.train([opponent, None])
        self.opponent = opponent
        self.rows = self.ks_env.configuration.rows
        self.columns = self.ks_env.configuration.columns
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.columns,), dtype=np.float)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.rows * self.columns,), dtype=np.float)
        print(self.rows, self.columns)
        self.reward_range = (-10, 1)
        self.timesteps = 0
        self.spec = None
        self.metadata = None

    def reset(self):
        if np.random.random() < 0.5:
            self.env = self.ks_env.train([None, self.opponent])
        else:
            self.env = self.ks_env.train([self.opponent, None])
        self.obs = self.env.reset()
        obs = make_obs(self.obs)
        self.timesteps = 0
        return obs

    def step(self, action):
        self.timesteps += 1
        for i in range(7):
            if self.obs['board'][i] != 0:
                action[i] = -1e+10
        action = np.argmax(action)
        self.obs, reward, done, info = self.env.step(int(action))

        if reward == 1:
            reward = 1
            info = {'episode': {'r': reward, 'l': self.timesteps}}
        elif done:
            reward = -1
            info = {'episode': {'r': reward, 'l': self.timesteps}}
        else:
            reward = 1/42
        obs = make_obs(self.obs)
        return obs, reward, done, info

    def change_opponent(self, opponent):
        self.opponent = opponent