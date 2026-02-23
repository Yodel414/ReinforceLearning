import numpy as np
import random
from model_def import GetMonteCarolModel
from src.grid_world import GridWorld
class TD:
    def __init__(self,env:GridWorld,P,R,gamma):
        self.env = env
        self.P = P
        self.R = R
        self.gamma = gamma
        self.num_states = env.num_states
        self.action_num = len(env.action_space)
        self.state_list = []
        self.reward_list = []
        self.vk = np.zeros((self.num_states,1))

    def update(self,N):
        state = random.randint(0,25)
        action_step = random.randint(0,4)
        next_state = (state % self.env.env_size[0],state // self.env.env_size[0])
        self.env.reset(next_state)
        state_list = []
        reward_list = []
        for index in range(N):
            state_list.append(next_state)
            state_index = self.postion2num(next_state)
            ak = 1 / (index + 10)
            next_state, reward, done, info  = self.env.step(self.env.action_space[action_step])
            next_state_index = self.postion2num(next_state)
            self.vk[state_index][0] = self.vk[state_index][0] - ak *(self.vk[state_index][0] - (reward+(self.gamma * self.vk[next_state_index][0])))
            action_step = self.policy[next_state_index]


    def GeneratePolicy(self,action_list):
        policy_matrix=np.zeros((self.num_states,self.action_num))
        for index in range(self.num_states):
            policy_matrix[index][action_list[index]] = 1
        self.policy_matrix = policy_matrix
        elementwise = []
        for index in range(len(policy_matrix)):
            for j in range(len(self.env.action_space)):
                if policy_matrix[index][j] == 1:
                    elementwise.append(j)
        self.policy = elementwise

    def postion2num(self,state):
        if isinstance(state,tuple):
            return state[1] * self.env.env_size[0] + state[0]
        elif isinstance(state,int):
            return(state % self.env.env_size[0],state // self.env.env_size[0])
def test1():
    P,R,env = GetMonteCarolModel(2)
    gamma = 0.9
    initial_policy = [4,3,3,2,4,4,2,0,0,2,0,4,2,0,2,0,1,4,3,2,4,3,0,3,4]
    td = TD(env,P,R,gamma)
    td.GeneratePolicy(initial_policy)
    td.update(100)
    state_value = [item[0] for item in td.vk]
    td.env.show_value(np.array(state_value))
if __name__ == '__main__':
    test1()