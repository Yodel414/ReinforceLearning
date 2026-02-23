import numpy as np
import random
from model_def import GetMonteCarolModel
from src.grid_world import GridWorld
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class VFASARSA:
    def __init__(self,env:GridWorld,gamma):
        self.env = env
        self.gamma = gamma
        self.num_states = env.num_states
        self.action_num = len(env.action_space)
        self.eposilon = 0.1
        self.q = 5
        w_dim = self.q * self.q * self.action_num
        w = np.random.default_rng().normal(size=w_dim)
        self.w = w.reshape(-1,1)
    def eposilon_greedy(self,state):
        if np.random.rand() < self.eposilon:
            # 随机选择动作（探索）
            return np.random.choice(self.action_num)
        else:
            # 根据Q值选择最优动作
            q_values = np.zeros(self.action_num)
            for action in range(self.action_num):
                feature_vector_sa = self.FeatureVectors(state,action)
                q_values[action] = (feature_vector_sa.transpose() @ self.w)[0][0]
            return np.argmax(q_values)

    def FeatureVectors(self,state,action):
        # 1. 还原二维坐标
        row = state // self.env.env_size[0]
        col = state % self.env.env_size[0]
        s_norm = np.array([row / 4.0, col / 4.0]) # 假设 5x5 网格
        
        # 2. 生成状态特征 (二维傅里叶)
        state_feats = []
        for i in range(self.q):
            for j in range(self.q):
                c = np.array([i, j])
                state_feats.append(np.cos(np.pi * np.dot(c, s_norm)))
        
        state_feats = np.array(state_feats).reshape(-1, 1)
        final_feats = np.zeros_like(self.w)
        start_index = action * self.q * self.q
        end_index = (action + 1) * self.q * self.q
        final_feats[start_index : end_index] = state_feats
        return final_feats.reshape(-1, 1)
    def update_policy(self, state: int):
        # 计算当前状态的所有动作的 Q 值
        q_values = np.zeros(self.action_num)
        for action in range(self.action_num):
            feature_vector_sa = self.FeatureVectors(state,action)
            q_values[action] = feature_vector_sa.transpose() @ self.w

        self.policy[state] = np.argmax(q_values)


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

    def iteration(self,k):
        alpha_k = 0.001 
        state = self.env.start_state
        self.env.reset(state)
        old_w = self.w.copy()
        action = self.eposilon_greedy(self.postion2num(state))
        if k == 499:
            print(1)
        done = False
        tmp_reward = 0
        traj_length = 0
        while not done:
            traj_length += 1
            next_state, reward, done, info = self.env.step(self.env.action_space[action])

            q_s = self.FeatureVectors(self.postion2num(state),action)
            q_list = []
            for tmp_action in self.env.action_space:
                q_s_1 = self.FeatureVectors(self.postion2num(next_state),tmp_action)
                q_list.append( q_s_1.transpose() @ self.w )
            q_star = np.argmax(q_list)
            td_error = reward + self.gamma * q_list[q_star] - q_s.transpose() @ self.w
            self.w = self.w + alpha_k * td_error * q_s

            tmp_reward += reward
            self.update_policy(self.postion2num(state))
            state = next_state
            
        # print(np.linalg.norm(self.w - old_w))  
        return tmp_reward,traj_length
    def postion2num(self,state):
        if isinstance(state,tuple):
            return state[1] * self.env.env_size[0] + state[0]
        elif isinstance(state,int):
            return(state % self.env.env_size[0],state // self.env.env_size[0])


def test1():
    env = GetMonteCarolModel(3)
    gamma = 0.9
    vfa_sarsa = VFASARSA(env,gamma)
    N = 1
    initial_policy = [4,3,3,2,4,4,2,0,0,2,0,4,2,0,2,0,1,4,3,2,4,3,0,3,4]
    vfa_sarsa.GeneratePolicy(initial_policy)
    reward_list = []
    trajlength_list = []
    for index in range(N):
        tmp_reward,tmp_length = vfa_sarsa.iteration(index)
        reward_list.append(tmp_reward)
        trajlength_list.append(tmp_length)
    plt.plot(range(N),reward_list)
    plt.show()
    vfa_sarsa.GeneratePolicy(vfa_sarsa.policy)
    vfa_sarsa.env.show_policy(vfa_sarsa.policy_matrix)

if __name__ == '__main__':

    test1()  
