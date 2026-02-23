import numpy as np
import random
from model_def import GetMonteCarolModel
from src.grid_world import GridWorld
import matplotlib.pyplot as plt
class QLearning:
    def __init__(self,env:GridWorld,gamma):
        self.env = env
        self.gamma = gamma
        self.num_states = env.num_states
        self.action_num = len(env.action_space)
        self.eposilon =0.5
        self.q_s_a = {}
        for action in env.action_space:
            for state in env.state_space:
                self.q_s_a[(state,action)] = 0.0
    def eposilon_greedy(self,state):
        num = random.random()
        if num <= 1 - self.eposilon/(self.action_num) * (self.action_num - 1):
            return self.env.action_space[self.policy[state]]
        else:
            return self.env.action_space[random.randint(0,4)]
    def on_policy_process(self,initial_state,total_steps):
        for step in range(total_steps):
            st = initial_state
            self.env.reset(initial_state)
            alpha_k = 0.1
            if step >= np.floor(total_steps / 3):
                alpha_k = 0.01
            elif step >= np.floor(2 *total_steps / 3):
                alpha_k = 0.001
            while st != self.env.target_state:
                at_next = self.eposilon_greedy(self.postion2num(st))
                st_next, reward, done, info  = self.env.step(at_next)

                qa_list = []
                for action in self.env.action_space:
                    qa_list.append(self.q_s_a[st_next,action])
                a_star = np.argmax(qa_list)
                self.q_s_a[(st,at_next)] = self.q_s_a[(st,at_next)] - alpha_k * (
                    self.q_s_a[(st,at_next)] - (
                        reward + self.gamma * self.q_s_a[(st_next,self.env.action_space[a_star])]))
                qa_list2 = []
                for action in self.env.action_space:
                    qa_list2.append(self.q_s_a[st,action])
                a_star = np.argmax(qa_list2)
                self.policy[self.postion2num(st)] = a_star
                st = st_next
    def off_policy_process(self,traj_length,total_steps):
        alpha_k = 0.1
        for step in range(total_steps):
            if step >= np.floor(total_steps / 3):
              alpha_k = 0.01
            elif step >= np.floor(2 *total_steps / 3):
                alpha_k = 0.001
            s0 = self.postion2num(random.randint(0,24))
            self.env.reset(s0)
            s_list = []
            a_list = []
            r_list = []
            for t in range(traj_length):
                s_list.append(s0)
                at = self.eposilon_greedy(self.postion2num(s0))
                st_next, reward, done, info  = self.env.step(at)
                a_list.append(at)
                r_list.append(reward)
                s0 = st_next
            for t in range(traj_length - 1):
                st = s_list[t]
                at = a_list[t]
                rt = r_list[t]
                st_1 = s_list[t + 1]
                qa_list = []
                for action in self.env.action_space:
                    qa_list.append(self.q_s_a[st_1,action])
                a_star = np.argmax(qa_list)
                self.q_s_a[(st,at)] = self.q_s_a[(st,at)] - alpha_k * (
                    self.q_s_a[(st,at)] - (
                        rt + self.gamma * self.q_s_a[(st_next,self.env.action_space[a_star])]))
                qa_list = []
                for action in self.env.action_space:
                    qa_list.append(self.q_s_a[st,action])
                a_star = np.argmax(qa_list)
                self.policy[self.postion2num(st)] = a_star
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
# def show_training_processing(ql:QLearning):
#     plt.figure(figsize=(8, 6))
#     plt.subplot(2, 1, 1) 
#     x = range(len(ql.reward_list))
#     plt.plot(x,ql.reward_list)
#     plt.subplot(2, 1, 2) 
#     plt.plot(x,ql.turn_num_list)
#     plt.show()
def on_policy_test():
    env = GetMonteCarolModel(3)
    gamma = 0.9
    initial_policy = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
    ql = QLearning(env,gamma)
    ql.GeneratePolicy(initial_policy)
    initial_state = (0,0)
    total_steps = 2000
    ql.on_policy_process(initial_state,total_steps)
    ql.GeneratePolicy(ql.policy)
    env.show_policy(ql.policy_matrix)
def off_policy_test():
    env = GetMonteCarolModel(3)
    gamma = 0.90
    initial_policy = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
    ql = QLearning(env,gamma)
    ql.GeneratePolicy(initial_policy)
    total_steps = 100000
    traj_length = 25
    ql.off_policy_process(traj_length,total_steps)
    ql.GeneratePolicy(ql.policy)
    env.show_policy(ql.policy_matrix)
if __name__ == '__main__':
    # on_policy_test()
    off_policy_test()