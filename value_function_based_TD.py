import numpy as np
import random
from model_def import GetMonteCarolModel
from src.grid_world import GridWorld
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class BellmanEquation:
    def postion2num(self,state):
        if isinstance(state,tuple):
            return state[1] * 5 + state[0]
        elif isinstance(state,int):
            return(state % 5,state // 5)
    def ConsturctPolicy(self,env:GridWorld):
        policy_matrix=np.zeros((25,25)) 
        r_pi = np.zeros(env.num_states)
        for index,state in enumerate(env.state_space):
            for action in env.action_space:
                env.reset(state)
                next_state, reward, done, info = env.step(action)
                policy_matrix[self.postion2num(state)][self.postion2num(next_state)] += 0.2
                r_pi[index] = r_pi[index] + 0.2 * reward
        return policy_matrix,r_pi
    def GetStateValue(self,policy_matrix,r_pi,gamma):
        eps = 1e-3
        max_iter = 100
        vk = np.zeros(25)
        for index in range(max_iter):
            vk_new = r_pi + gamma * policy_matrix @ vk        
            if np.linalg.norm(vk_new - vk) < eps:
                break
            else:
                vk = vk_new
        self.vk = vk_new
        return vk_new
    def process(self):
        env = GetMonteCarolModel(3)
        gamma = 0.9
        policy_matrix,r_pi = self.ConsturctPolicy(env)
        # env.show_value(np.array(self.GetStateValue(policy_matrix,r_pi,gamma)))
        vk = self.GetStateValue(policy_matrix,r_pi,gamma)
class LinearValueFunction:
    def __init__(self,env:GridWorld,gamma,type):
        self.env = env
        self.gamma = gamma
        self.num_states = env.num_states
        self.action_num = len(env.action_space)
        self.eposilon =0.1
        self.type = type
        if type == 1:
            w_dim = 3
            self.w = np.zeros((w_dim,1))
            self.w[0] = random.gauss(0, 1)
            self.w[1] = random.gauss(0, 1)
            self.w[2] = random.gauss(0, 1)
        elif type == 2:
            w_dim = 6
            self.w = np.zeros((w_dim,1))
            self.w[0] = random.gauss(0, 1)
            self.w[1] = random.gauss(0, 1)
            self.w[2] = random.gauss(0, 1)
            self.w[3] = random.gauss(0, 1)
            self.w[4] = random.gauss(0, 1)
            self.w[5] = random.gauss(0, 1)
        elif type == 3:
            w_dim = 10
            self.w = np.zeros((w_dim,1))
            self.w[0] = random.gauss(0, 1)
            self.w[1] = random.gauss(0, 1)
            self.w[2] = random.gauss(0, 1)
            self.w[3] = random.gauss(0, 1)
            self.w[4] = random.gauss(0, 1)
            self.w[5] = random.gauss(0, 1)
            self.w[6] = random.gauss(0, 1)
            self.w[7] = random.gauss(0, 1)
            self.w[8] = random.gauss(0, 1)
            self.w[9] = random.gauss(0, 1)
    def uniform_policy(self):
        # num = random.random()
        # if num <= 1 - self.eposilon/(self.action_num) * (self.action_num - 1):
        #     return self.env.action_space[self.policy[state]]
        # else:
        return self.env.action_space[random.randint(0,4)]
    def FeatrueVectors(self,state):
        featrue_vectors = np.zeros_like(self.w)
        x = state // 5
        y = state % 5
        x_norm = 0.0
        y_norm = 0.0
        if x != 0 and y != 0:
            x_norm = x / (x + y)
            y_norm = y / (x + y)
        if self.type == 1:
            featrue_vectors[0] = 1
            featrue_vectors[1] = x_norm
            featrue_vectors[2] = y_norm
        elif self.type == 2:
            featrue_vectors[0] = 1
            featrue_vectors[1] = x_norm
            featrue_vectors[2] = y_norm
            featrue_vectors[3] = x_norm ** 2
            featrue_vectors[4] = y_norm **2
            featrue_vectors[5] = x_norm * y_norm
        elif self.type == 3:
            featrue_vectors[0] = 1
            featrue_vectors[1] = x_norm
            featrue_vectors[2] = y_norm
            featrue_vectors[3] = x_norm ** 2
            featrue_vectors[4] = y_norm ** 2
            featrue_vectors[5] = x_norm * y_norm
            featrue_vectors[6] = x_norm ** 3
            featrue_vectors[7] = y_norm ** 3
            featrue_vectors[8] = x_norm **2 * y_norm
            featrue_vectors[9] = y_norm **2 * x_norm
        return featrue_vectors
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

    def iteration(self):
        traj_length = 500
        alpha_k = 0.0005
        state = self.postion2num(random.randint(0,24))
        self.env.reset(state)
        for i in range(traj_length):
            action = self.uniform_policy()
            next_state, reward, done, info = self.env.step(action)
            phi_s = self.FeatrueVectors(self.postion2num(state))
            phi_s_1 = self.FeatrueVectors(self.postion2num(next_state))
            self.w = self.w + alpha_k * (
                reward + self.gamma * phi_s_1.transpose() @ self.w - phi_s.transpose() @ self.w) * phi_s

    def postion2num(self,state):
        if isinstance(state,tuple):
            return state[1] * self.env.env_size[0] + state[0]
        elif isinstance(state,int):
            return(state % self.env.env_size[0],state // self.env.env_size[0])
def test1():
    
    env = GetMonteCarolModel(3)
    gamma = 0.9
    linear_vfa = LinearValueFunction(env,gamma,1)
    
    N = 500
    rmse_list = []
    for index in range(N):
        value_list = []
        linear_vfa.iteration()
        for i in range(env.num_states):
            value_list.append((linear_vfa.FeatrueVectors(i).transpose() @ linear_vfa.w)[0][0])
        rmse_list.append(np.sqrt(mean_squared_error(nominal_value , value_list)))
    return rmse_list
    
def test2():
    env = GetMonteCarolModel(3)
    gamma = 0.9
    linear_vfa = LinearValueFunction(env,gamma,2)
    N = 500
    rmse_list = []
    for index in range(N):
        value_list = []
        linear_vfa.iteration()
        for i in range(env.num_states):
            value_list.append((linear_vfa.FeatrueVectors(i).transpose() @ linear_vfa.w)[0][0])
        rmse_list.append(np.sqrt(mean_squared_error(nominal_value , value_list)))
    return rmse_list

def test3():
    env = GetMonteCarolModel(3)
    gamma = 0.9
    linear_vfa = LinearValueFunction(env,gamma,3)
    N = 500
    rmse_list = []
    for index in range(N):
        value_list = []
        linear_vfa.iteration()
        for i in range(env.num_states):
            value_list.append((linear_vfa.FeatrueVectors(i).transpose() @ linear_vfa.w)[0][0])
        rmse_list.append(np.sqrt(mean_squared_error(nominal_value , value_list)))
    return rmse_list

if __name__ == '__main__':
    be = BellmanEquation()
    be.process()
    x_axis = range(500)
    nominal_value = be.vk
    dim3 = test1()
    dim6 = test2() 
    dim10 = test3()  
    plt.figure(figsize=(18, 16))
    plt.subplot(2,3,4)
    plt.plot(x_axis,dim3)
    plt.subplot(2,3,5)
    plt.plot(x_axis,dim6)
    plt.subplot(2,3,6)
    plt.plot(x_axis,dim10)
    plt.show()