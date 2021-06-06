import numpy as np
from kaggle_environments import evaluate
from stable_baselines3 import PPO
from agent import agent
from rule_agent import rule_agent
from rule_agent2 import rule_agent2

def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))

if __name__ == '__main__':
    model = PPO.load('./check_model2.zip')
    our_agent = agent(model.policy.to('cpu').state_dict())

    get_win_percentages(agent1=rule_agent, agent2=our_agent)
