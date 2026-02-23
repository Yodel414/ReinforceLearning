# import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from model_def import GetMonteCarolModel
from src.grid_world import GridWorld
import random
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_num):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_num)
        )

    def forward(self, x):
        return self.fc(x)
class DQN:
    def __init__(self, env:GridWorld, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.action_num = len(env.action_space)
        self.state_dim = 2  # 输入坐标 (x, y)
        self.num_states = env.num_states
        # 两个网络：主网络和目标网络
        self.policy_net = QNetwork(self.state_dim, self.action_num)
        self.target_net = QNetwork(self.state_dim, self.action_num)
        self.epoch = 100
        self.eposilon = 0.3
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_update_freq = 10 # 每10个episode同步一次网络
        self.cache_beta = []
        self.cache_size = 10000
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
    def eposilon_greedy(self,state):
        num = random.random()
        if num <= 1 - self.eposilon/(self.action_num) * (self.action_num - 1):
            return self.env.action_space[self.policy[self.postion2num(state)]]
        else:
            return self.env.action_space[random.randint(0,4)]
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
    def cache_generate(self):
        state = self.env.start_state
        self.env.reset(state)
        for i in range(self.cache_size):
            acion_based_pi_b = self.eposilon_greedy(state)
            next_state, reward, done, info = self.env.step(acion_based_pi_b)
            self.cache_beta.append((self.postion2num(state),self.env.action_space.index(acion_based_pi_b),reward,self.postion2num(next_state)))
    def learn(self,iteration_times):
        for i in range(iteration_times):
            selected = random.choices(self.cache_beta, k = self.epoch)
            # yt = np.zeros((len(selected),1))
            for sample in selected:
                s = sample[0]
                a = sample[1]
                r = sample[2]
                s_prime = sample[3]
                q_list = []
                for acion in self.env.action_space:
                    q_value = self.target_net.forward(s_prime,a)
                    q_list.append(q_value)
                yt = r + self.gamma * max(q_list)
    def postion2num(self,state):
        if isinstance(state,tuple):
            return state[1] * self.env.env_size[0] + state[0]
        elif isinstance(state,int):
            return(state % self.env.env_size[0],state // self.env.env_size[0])

def test1():
    env = GetMonteCarolModel(3)
    gamma = 0.9
    dqn = DQN(env,gamma)
    N = 10000
    initial_policy = [4,3,3,2,4,4,2,0,0,2,0,4,2,0,2,0,1,4,3,2,4,3,0,3,4]
    dqn.GeneratePolicy(initial_policy)
    iteration_times = 1000
    dqn.cache_generate()
    dqn.learn(iteration_times)

    # for state_idx in range(dqn.env.num_states):
    #     dqn.update_policy(state_idx)

    dqn.GeneratePolicy(dqn.policy)
    dqn.env.show_policy(dqn.policy_matrix)

if __name__ == '__main__':

    test1()  