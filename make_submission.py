import torch as th
from stable_baselines3 import PPO

learner = PPO.load('./meta_model.zip')

th.set_printoptions(profile="full")

agent_path = 'submission.py'

state_dict = learner.policy.to('cpu').state_dict()
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

with open(agent_path, mode='a') as file:
    # file.write(f'\n    data = {learner.policy._get_data()}\n')
    file.write(f'    state_dict = {state_dict}\n')