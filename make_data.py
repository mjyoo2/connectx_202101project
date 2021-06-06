from model_based_connectx import MCTS_agent, make_state, ReplayBuffer
import tqdm
from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
configuration = env.configuration

replay_buffer = ReplayBuffer(max_len=10000, obs_shape=(2, 6, 7), act_shape=1)

for i in tqdm.tqdm(range(250)):
    game_data = env.run([MCTS_agent, MCTS_agent])
    mark = 1
    reward = game_data[-1][0]['reward']
    for idx, data in enumerate(game_data):
        if data[0]['status'] == 'DONE':
            break
        board = data[0]['observation']['board']
        state = make_state(np.array(board), mark)
        action = game_data[idx+1][mark-1]['action']
        if mark == 1:
            tmp_reward = reward
        else:
            tmp_reward = -reward
        replay_buffer.add({'state': state, 'action': action, 'reward': tmp_reward})
        mark = 3 - mark

replay_buffer.save('./replay_0.pkl')