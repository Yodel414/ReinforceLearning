import numpy as np
import random
from model_def import GetMonteCarolModel
from src.grid_world import GridWorld
import matplotlib.pyplot as plt
class SARSA:
    def __init__(self,env:GridWorld,P,R,gamma):
        self.env = env
        self.P = P
        self.R = R
        self.gamma = gamma
        self.num_states = env.num_states
        self.action_num = len(env.action_space)
        self.eposilon =0.1
        self.q_s_a = {}
        for s,a in R:
            state = (s% self.env.env_size[0],s // self.env.env_size[0])
            action  = env.action_space[a]
            self.q_s_a[(state,action)] = 0.0
    def eposilon_greedy(self,state):
        num = random.random()
        if num <= 1 - self.eposilon/(self.action_num) * (self.action_num - 1):
            return self.env.action_space[self.policy[state]]
        else:
            return self.env.action_space[random.randint(0,4)]
    def process(self,initial_state,total_steps):
        reward_list = []
        turn_num_list = []
        for _ in range(total_steps):
            tmp_reward = 0
            turn_num = 0
            st = initial_state
            self.env.reset(initial_state)
            alpha_k = 0.1
            at = self.eposilon_greedy(self.postion2num(st))
            
            while st != self.env.target_state:
                st_next, reward, done, info  = self.env.step(at)
                at_next = self.eposilon_greedy(self.postion2num(st_next))
                self.q_s_a[(st,at)] = self.q_s_a[(st,at)] - alpha_k * (self.q_s_a[(st,at)] - (reward + self.gamma * self.q_s_a[(st_next,at_next)]))
                qa_list = []
                for action in self.env.action_space:
                    qa_list.append(self.q_s_a[st,action])
                a_star = np.argmax(qa_list)
                tmp_reward += reward
                turn_num += 1
                self.policy[self.postion2num(st)] = a_star
                st = st_next
                at = at_next
            reward_list.append(tmp_reward)
            turn_num_list.append(turn_num)
        self.reward_list = reward_list
        self.turn_num_list = turn_num_list
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
def show_training_processing(sarsa:SARSA):
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1) 
    x = range(len(sarsa.reward_list))
    plt.plot(x,sarsa.reward_list)
    plt.subplot(2, 1, 2) 
    plt.plot(x,sarsa.turn_num_list)
    plt.show()
def test1():
    P,R,env = GetMonteCarolModel(2)
    gamma = 0.8
    initial_policy = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
    sarsa = SARSA(env,P,R,gamma)
    sarsa.GeneratePolicy(initial_policy)
    initial_state = (0,0)
    total_steps = 20000
    sarsa.process(initial_state,total_steps)
    sarsa.GeneratePolicy(sarsa.policy)
    env.show_policy(sarsa.policy_matrix)
    show_training_processing(sarsa)

if __name__ == '__main__':
    test1()