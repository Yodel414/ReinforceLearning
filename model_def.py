from src.grid_world import GridWorld
# import argparse
from config import GetConfig
def GetModel():
    config = GetConfig(2)
    env = GridWorld(config)
    P = {}
    R = {}
    state_table = {}
    action_table = {}
    state_index = 0
    for state in env.state_space:
        state_table[state] = state_index
        state_index += 1
        action_index = 0
        for action in env.action_space:
            action_table[action] = action_index
            action_index += 1
            for _ in env.state_space:
                P[(state,action)] = 0
                R[(state,action)] = 0
            next_state , reward = env._get_next_state_and_reward(state,action)
            P[(state,action)] = next_state
            R[(state,action)] = reward
    return P,R,state_table,action_table,env

def GetMonteCarolModel(type):
    config = GetConfig(type)
    env = GridWorld(config)
    P = {}
    R = {}
    # state_table = {}
    # action_table = {}
    state_index = 0
    for state in env.state_space:
        # state_table[state] = state_index
        action_index = 0
        for action in env.action_space:
            # action_table[action] = action_index
            for _ in env.state_space:
                P[(state_index,action_index)] = 0
                R[(state_index,action_index)] = 0
            next_state , reward = env._get_next_state_and_reward(state,action)
            P[(state_index,action_index)] = next_state
            R[(state_index,action_index)] = reward
            action_index += 1
        state_index += 1
    return P,R,env
if __name__ == "__main__":
    # P,R,state_table,action_table,env = GetModel()/
    P,R,env = GetMonteCarolModel(1)
    env.reset((0,0))
    env.render()