# Critical Formular :
# REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility
# which means REINFORCE

import numpy as np
import random
import torch
import torch.nn as nn
from model_def import GetMonteCarolModel
from src.grid_world import GridWorld
import matplotlib.pyplot as plt
def softmax(z, axis=None):

    # 减去最大值以提高数值稳定性
    z_shifted = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z_shifted)
    softmax_output = exp_z / np.sum(exp_z, axis=axis, keepdims=True)
    return softmax_output

class PolicyNetwork(nn.Module):
    def __init__(self,state_dim,action_num):
        super(PolicyNetwork,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_features=action_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)
class REINFORCE:
    def __init__(self,env:GridWorld,gamma,policy_network,optimizer):
        self.env = env
        self.optimizer = optimizer
        self.policy = policy_network
        self.gamma = gamma
        self.num_states = env.num_states
        self.action_num = len(env.action_space)
        self.q = 5
        w_dim = self.q * self.q
        # w = np.random.rand(w_dim, 5)
        self.w = np.zeros((w_dim, self.action_num))
    # def FeatureVectors(self,state):
    #     # 1. 还原二维坐标
    #     row = state[1]
    #     col = state[0]
    #     s_norm = np.array([row / (self.env.env_size[0] - 1), col / (self.env.env_size[1] - 1)]) # 假设 5x5 网格
        
    #     # 2. 生成状态特征 (二维傅里叶)
    #     state_feats = []
    #     for i in range(self.q):
    #         for j in range(self.q):
    #             c = np.array([i, j])
    #             state_feats.append(np.cos(np.pi * np.dot(c, s_norm)))
        
    #     state_feats = np.array(state_feats).reshape(-1, 1)
    #     return state_feats.reshape(-1, 1)


    def GeneratePolicy(self):
        policy_matrix=np.zeros((self.num_states,self.action_num))
        for index in range(self.num_states):
            x = float(self.postion2num(index)[0]/ 4.0) 
            y = float(self.postion2num(index)[1]/ 4.0) 
            pb = self.excute_policy(torch.tensor((x,y)).reshape(-1, 2)).flatten()
            action = torch.argmax(pb).item()
            policy_matrix[index][action] = 1
        self.policy_matrix = policy_matrix
    def excute_policy(self,state):
        possibility_distribution = self.policy(state)
        return possibility_distribution[0]

    def iteration(self):
        alpha_k = 0.1 
        state = self.env.start_state
        self.env.reset(state)

        traj_length = 30
        state_list = []
        action_list = []
        reward_list = []
        for i in range(traj_length):
            state_list.append(state)
            x = float(state[0] / 4.0) 
            y = float(state[1] / 4.0) 
            probs = self.excute_policy(torch.tensor((x,y)).reshape(-1, 2)).flatten()
            action = np.random.choice(self.action_num, p=probs.detach().numpy())
            action_list.append(action)
            next_state, reward, done, info = self.env.step(self.env.action_space[action])
            reward_list.append(reward)
            state = next_state
            if done:
                break
        G_list = []
        G = 0.0
        for r in reversed(reward_list):
            G = r + self.gamma * G
            G_list.insert(0, G)

        gamma_t = 1.0
        self.optimizer.zero_grad()
        for state_k,action,reward,G_t in zip(state_list,action_list,reward_list,G_list):
            x = float(state_k[0] / 4.0)
            y = float(state_k[1] / 4.0)
            prb = self.policy(torch.tensor((x, y)).reshape(-1, 2))[0]
            loss = -torch.log(prb[action]) * G_t
            loss.backward()
        self.optimizer.step()

    def postion2num(self,state):
        if isinstance(state,tuple):
            return state[1] * self.env.env_size[0] + state[0]
        elif isinstance(state,int):
            return(state % self.env.env_size[0],state // self.env.env_size[0])


def test1():
    env = GetMonteCarolModel(3)
    gamma = 0.9
    lr = 0.01
    policy_network = PolicyNetwork(2,len(env.action_space))
    optimizer = torch.optim.Adam(policy_network.parameters(), lr= lr)
    reinforce = REINFORCE(env,gamma,policy_network,optimizer)
    N = 1000
    for index in range(N):
        reinforce.iteration()
    reinforce.env.reset(reinforce.env.start_state)
    reinforce.GeneratePolicy()
    reinforce.env.show_policy(reinforce.policy_matrix)

if __name__ == '__main__':

    test1()  
