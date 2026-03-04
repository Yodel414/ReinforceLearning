# Critical Formular :
# REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility
# which means REINFORCE

import numpy as np
import random
from model_def import GetMonteCarolModel
from src.grid_world import GridWorld
import matplotlib.pyplot as plt
def softmax(z, axis=None):

    # 减去最大值以提高数值稳定性
    z_shifted = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z_shifted)
    softmax_output = exp_z / np.sum(exp_z, axis=axis, keepdims=True)
    return softmax_output
class REINFORCE:
    def __init__(self,env:GridWorld,gamma):
        self.env = env
        self.gamma = gamma
        self.num_states = env.num_states
        self.action_num = len(env.action_space)
        self.q = 5
        w_dim = self.q * self.q
        w = np.random.rand(w_dim, 5)
        self.w = w.reshape(-1,5)
    def FeatureVectors(self,state):
        # 1. 还原二维坐标
        row = state[1]
        col = state[0]
        s_norm = np.array([row / 4.0, col / 4.0]) # 假设 5x5 网格
        
        # 2. 生成状态特征 (二维傅里叶)
        state_feats = []
        for i in range(self.q):
            for j in range(self.q):
                c = np.array([i, j])
                state_feats.append(np.cos(np.pi * np.dot(c, s_norm)))
        
        state_feats = np.array(state_feats).reshape(-1, 1)
        return state_feats.reshape(-1, 1)


    def GeneratePolicy(self):
        policy_matrix=np.zeros((self.num_states,self.action_num))
        for index in range(self.num_states):
            action = np.argmax(self.excute_policy(self.postion2num(index)))
            policy_matrix[index][action] = 1
        self.policy_matrix = policy_matrix
    def excute_policy(self,state):
        q_s = self.FeatureVectors(state)
        possibility_distribution = q_s.transpose() @ self.w
        return softmax(possibility_distribution)

    def iteration(self):
        alpha_k = 0.001 
        state = self.env.start_state
        self.env.reset(state)

        traj_length = 100
        state_list = []
        action_list = []
        reward_list = []
        for i in range(traj_length):
            state_list.append(state)
            probs = self.excute_policy(state).flatten()
            action = np.random.choice(self.action_num, p=probs)
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
        for state_k,action,reward,G_t in zip(state_list,action_list,reward_list,G_list):
            probs = self.excute_policy(state_k).flatten() 
            for a in range(self.action_num):
                if a == action:
                    grad = (1 - probs[a]) * self.FeatureVectors(state_k)
                else:
                    grad = -probs[a] * self.FeatureVectors(state_k)

                self.w[:, a] += alpha_k * gamma_t * G_t * grad.flatten()
            gamma_t *= self.gamma
    def postion2num(self,state):
        if isinstance(state,tuple):
            return state[1] * self.env.env_size[0] + state[0]
        elif isinstance(state,int):
            return(state % self.env.env_size[0],state // self.env.env_size[0])


def test1():
    env = GetMonteCarolModel(3)
    gamma = 0.9
    reinforce = REINFORCE(env,gamma)
    N = 10
    for index in range(N):
        reinforce.iteration()
    reinforce.env.reset(reinforce.env.start_state)
    reinforce.GeneratePolicy()
    reinforce.env.show_policy(reinforce.policy_matrix)

if __name__ == '__main__':

    test1()  
