import torch as th
from rule_agent import rule_agent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import ConnectFourGym

env = ConnectFourGym()
env_2 = ConnectFourGym(opponent=rule_agent)
env_3 = ConnectFourGym(opponent='negamax')
env = DummyVecEnv([lambda: env])

policy_kwargs = {
    'activation_fn':th.nn.ReLU,
    'net_arch':[64, 64, dict(pi=[64], vf=[64])]
}

learner = PPO('MlpPolicy', env, verbose=1, n_steps=16, policy_kwargs=policy_kwargs)

print('learning_start')

learner.learn(total_timesteps=2000000, log_interval=10)
learner.save('./model')
# opponent = agent(learner.policy.to('cpu').state_dict())
# env.env_method('change_opponent', opponent)
