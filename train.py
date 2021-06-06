import numpy as np
import os
import argparse
import gym
import torch

from stable_baselines3 import PPO
from multiprocessing import Process
from reptile_callback import LowCallback, reptile
from env import ConnectFourGym
from rule_agent import rule_agent
from rule_agent2 import rule_agent2
from agent import agent

policy_kwargs = {
    'activation_fn':torch.nn.ReLU,
    'net_arch':[64, 64, dict(pi=[64], vf=[64])]
}

dummy_env = ConnectFourGym(opponent='random')
dummy_agent = PPO('MlpPolicy', dummy_env, n_steps=1024, verbose=1, policy_kwargs=policy_kwargs)

learnable_agent = agent(dummy_agent.policy.to('cpu').state_dict())
def_agent = ['random', 'random', rule_agent, rule_agent, rule_agent2, rule_agent2, learnable_agent, learnable_agent]

env_setting = lambda agent: ConnectFourGym(opponent=agent)
model_setting = lambda env: PPO('MlpPolicy', env, n_steps=1024, verbose=1, policy_kwargs=policy_kwargs)

def run(oper_num, args):
    np.random.seed()
    a = oper_num
    print('env {} setting: {}'.format(oper_num, a))
    env = env_setting(def_agent[a])
    model = model_setting(env)
    if a < 6:
        callback = LowCallback(oper_num, args.port)
    else:
        callback = LowCallback(oper_num, args.port, mode='learnable')

    model.learn(total_timesteps=args.total_timesteps, callback=callback, log_interval=10)
    print('finish')

def reptile_run(args):
    np.random.seed()
    env = env_setting(def_agent[4])
    model = model_setting(env)
    algo = reptile(num_of_operator=args.num_workers, port=args.port, alpha=args.alpha, model=model, env=env)
    algo.run()
    algo.save('./meta_model')
    print('finish')
    algo.test()
    algo.adapt(args.adapt_timesteps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment setting')
    # parser.add_argument('--path', type=str, default='./')
    # parser.add_argument('--midclass',  type=str)
    # parser.add_argument('--subclass',  type=str)
    # parser.add_argument('--description',  type=str)
    parser.add_argument('--num_workers',  type=int, default=8)
    parser.add_argument('--total_timesteps',  type=int, default=2000000)
    parser.add_argument('--adapt_timesteps',  type=int, default=10000)
    parser.add_argument('--alpha', type=float, default=0.25)
    # parser.add_argument('--pomdp', dest='pomdp', action='store_true')
    # parser.add_argument('--mdp', dest='pomdp', action='store_false')
    parser.add_argument('--port', type=int, default=33333)
    parser.set_defaults(pomdp=False)
    args = parser.parse_args()
    p_list = []

    for i in range(args.num_workers):
        p = Process(target=run, args=(i, args, ))
        p.start()
        p_list.append(p)
        print('make process')
    p = Process(target=reptile_run, args=(args, ))
    p.start()
    p_list.append(p)
    print('make reptile')
    for proc in p_list:
        proc.join()
