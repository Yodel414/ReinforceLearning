from src.grid_world import GridWorld
import numpy as np
from model_def import GetModel

class ValueIteration:
    def __init__(self, env:GridWorld,n_states, n_actions, transition_probs, rewards, state_table,gamma=0.95, epsilon=1e-6):
        """
        初始化值迭代算法
        
        参数：
        n_states: 状态数量
        n_actions: 动作数量
        transition_probs: 转移概率矩阵，shape=(n_states, n_actions, n_states)
        rewards: 奖励矩阵，shape=(n_states, n_actions, n_states)
        gamma: 折扣因子
        epsilon: 收敛阈值
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = transition_probs  # P[s, a, s'] = P(s'|s, a)
        self.R = rewards           # R[s, a, s'] = R(s, a, s')
        self.state_table = state_table
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env
        self.iter_times = 1000
        # 初始化值函数
        self.V = {}
        self.v_star_last = np.zeros((25,1))
        for state in env.state_space:
            self.V[state] = 0
        # 初始化策略
        self.policy = np.zeros(n_states, dtype=int)
    def iteration(self):
        v_star = np.zeros((25,1))
        pi_star = {}
        done = False
        iters = 0
        while not done and iters < self.iter_times:
            iters += 1
            for state in self.env.state_space:
                q_s_a = -1
                action_star = (0,0)
                for action in self.env.action_space:
                    q0 = self.R[state,action] + self.gamma * v_star[self.state_table[self.P[state,action]]]
                    if q0[0] > q_s_a:
                        action_star = action
                        q_s_a = q0[0]
                pi_star[state] = action_star
                v_star[self.state_table[state]] = q_s_a
            if np.linalg.norm(self.v_star_last - v_star) < self.epsilon:
                done = True
            else:
                self.v_star_last = v_star.copy()
        return pi_star,v_star

class PolicyIteration:
    def __init__(self, env:GridWorld,n_states, n_actions, transition_probs, rewards, state_table,action_table,gamma=0.9, epsilon=1e-6):
        """
        初始化值迭代算法
        
        参数：
        n_states: 状态数量
        n_actions: 动作数量
        transition_probs: 转移概率矩阵，shape=(n_states, n_actions, n_states)
        rewards: 奖励矩阵，shape=(n_states, n_actions, n_states)
        gamma: 折扣因子
        epsilon: 收敛阈值
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = transition_probs  # P[s, a, s'] = P(s'|s, a)
        self.R = rewards           # R[s, a, s'] = R(s, a, s')
        self.state_table = state_table
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env
        self.iter_times = 10
        self.action_table = action_table
        self.value_iter_times = 1000
        # 初始化值函数
        self.V = {}
        self.v_k_last = np.zeros((25,1))
        for state in env.state_space:
            self.V[state] = 0
        # 初始化策略
        self.policy = np.zeros(n_states, dtype=int)
    def iteration(self):
        done = False
        
        iters = 0
        value_iter = 0
        v_star = np.zeros((25,1))
        policy_matrix=np.zeros((self.env.num_states,len(self.env.action_space)))    
        for item in policy_matrix:
            item[4] = 1
        while iters < self.iter_times:
            iters += 1
            # PE
            value_done = False
            while value_iter < self.value_iter_times and not value_done:
                value_iter += 1
                for state in self.env.state_space:
                    cur_v = 0
                    discounted_v = 0
                    for action in self.env.action_space:
                        cur_v =+ self.R[state,action] * policy_matrix[self.state_table[state]][self.action_table[action]]     
                        discounted_v += self.gamma * v_star[self.state_table[self.P[state,action]]]* policy_matrix[self.state_table[state]][self.action_table[action]]
                    v_star[self.state_table[state]] = cur_v + discounted_v
                if np.linalg.norm(self.v_k_last - v_star) < self.epsilon:
                    value_done = True
                else:
                    self.v_k_last = v_star.copy()
            # print(np.linalg.norm(self.v_k_last - v_star))
            # print(value_iter)
            # PI
            for state in self.env.state_space:
                a_star = (0,0)
                q_list = []
                for action in self.env.action_space:
                    q0 = self.R[state,action] + self.gamma * v_star[self.state_table[self.P[state,action]]]
                    # a_star = action
                    q_list.append(q0[0])
                a_star = np.argmax(q_list)
                for action in self.env.action_space:
                    if self.action_table[action] == a_star:
                        policy_matrix[self.state_table[state]][self.action_table[action]] = 1
                    else:
                        policy_matrix[self.state_table[state]][self.action_table[action]] = 0
        return policy_matrix,self.v_k_last
                

def test_value_iteration():
    P,R,state_table,action_table,env = GetModel()
    state = env.reset() 
    # env.render()
    boe = ValueIteration(env,25,5,P,R,state_table)
    pi_star,v_star=  boe.iteration()
    policy_matrix = np.zeros((env.num_states,len(env.action_space)))
    state_maxtirx = np.zeros(env.num_states)
    for state in pi_star:
        row = state_table[state]
        col = action_table[pi_star[state]]
        policy_matrix[row][col] = 1
    for i in range(len(v_star)):
        state_maxtirx[i] = v_star[i][0]
    # print(policy_matrix)
    # env.add_policy(policy_matrix)
    env.show_policy(policy_matrix)
    values = state_maxtirx
    env.show_value(values)

def test_policy_iteration():
    P,R,state_table,action_table,env = GetModel()
    state = env.reset((0,0)) 
    # env.render()
    boe = PolicyIteration(env,25,5,P,R,state_table,action_table)
    pi_star,v_star = boe.iteration()
    state_maxtirx = np.zeros(env.num_states)
    for i in range(len(v_star)):
        state_maxtirx[i] = v_star[i][0]
    env.show_policy(pi_star)
    env.show_value(state_maxtirx)
if __name__ == "__main__":  
    # test_value_iteration()
    test_policy_iteration()