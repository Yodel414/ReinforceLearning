import numpy as np
import random
from model_def import GetMonteCarolModel
from src.grid_world import GridWorld
class MonteCarol:
    def __init__(self,env:GridWorld,P,R,gamma):
        self.env = env
        self.P = P
        self.R = R
        self.gamma = gamma
        
        self.num_states = env.num_states
        self.action_num = len(env.action_space)
    def update_value(self):
        reward_matrix = [item[4] for item in self.R]
        pass
    def GetEpisode(self,state,action):
        length = 1
        q_pi_a = self.R[state,4]
        x = state % self.env.env_size[0]
        y = state // self.env.env_size[0]
        next_state = (x,y)
        self.env.reset(next_state)
        action_step = action
        for index in range(length):
            next_state, reward, done, info  = self.env.step(self.env.action_space[action_step])
            state = next_state[1] * self.env.env_size[0] + next_state[0]
            action_step = self.policy[state]
            q_pi_a += reward * self.gamma
        return q_pi_a
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
class ExploringMontrCarol:
    def __init__(self,env:GridWorld,P,R,gamma):
        self.env = env
        self.P = P
        self.R = R
        self.gamma = gamma
        self.Return_s_a = {}
        self.Number_s_a = {}
        self.q_s_a = {}
        self.num_states = env.num_states
        self.action_num = len(env.action_space)
        # self.q_s_a[(state,action)] = 0.0

        for s,a in R:
            state = (s% self.env.env_size[0],s // self.env.env_size[0])
            action  = env.action_space[a]
            self.q_s_a[(state,action)] = R[s,a]
        for state in env.state_space:
            for action in env.action_space:
                self.Return_s_a[(state,action)] = 0.0
                self.Number_s_a[(state,action)] = 0
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
    def update(self,state,action):
        length = 10
        x = state % self.env.env_size[0]
        y = state // self.env.env_size[0]
        next_state = (x,y)
        if next_state == (4,1):
            print((1))
        reward_list = []
        state_list = []
        action_list = []
        self.env.reset(next_state)
        action_step = action
        for index in range(length):
            state_list.append(next_state)
            action_list.append(action_step)
            next_state, reward, done, info  = self.env.step(self.env.action_space[action_step])
            reward_list.append(reward)
            state_index = next_state[1] * self.env.env_size[0] + next_state[0]
            action_step = self.policy[state_index]
        g = 0.0
        for t in range(length - 1, -1, -1):
            g = g * self.gamma + reward_list[t]
            state_tmp = state_list[t]
            # if state_tmp == (4,1):
            #     print(1)
            action_tmp = self.env.action_space[action_list[t]]
            self.Return_s_a[(state_tmp,action_tmp)] = self.Return_s_a[(state_tmp,action_tmp)] + g
            self.Number_s_a[(state_tmp,action_tmp)] = self.Number_s_a[(state_tmp,action_tmp)] + 1
            self.q_s_a[(state_tmp,action_tmp)] = self.Return_s_a[(state_tmp,action_tmp)] / self.Number_s_a[(state_tmp,action_tmp)]
            q_star_list = []
            for action in self.env.action_space:
                q_star_list.append(self.q_s_a[(state_tmp,action)])
            q_star = np.argmax(q_star_list)
            state_index = state_tmp[1] * self.env.env_size[0] + state_tmp[0]
            self.policy[state_index] = q_star
class EposilonGreedy_MontrCarol:
    def __init__(self,env:GridWorld,P,R,gamma,eposilon):
        self.env = env
        self.P = P
        self.R = R
        self.gamma = gamma
        self.eposilon = eposilon
        self.Return_s_a = {}
        self.Number_s_a = {}
        self.q_s_a = {}
        self.num_states = env.num_states
        self.action_num = len(env.action_space)
        # self.q_s_a[(state,action)] = 0.0

        for s,a in R:
            state = (s% self.env.env_size[0],s // self.env.env_size[0])
            action  = env.action_space[a]
            self.q_s_a[(state,action)] = R[s,a]
        for state in env.state_space:
            for action in env.action_space:
                self.Return_s_a[(state,action)] = 0.0
                self.Number_s_a[(state,action)] = 0
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
    def update(self,state,action):
        length = 10
        x = state % self.env.env_size[0]
        y = state // self.env.env_size[0]
        next_state = (x,y)
        if next_state == (4,1):
            print((1))
        reward_list = []
        state_list = []
        action_list = []
        self.env.reset(next_state)
        action_step = action
        for index in range(length):
            state_list.append(next_state)
            action_list.append(action_step)
            next_state, reward, done, info  = self.env.step(self.env.action_space[action_step])
            reward_list.append(reward)
            state_index = next_state[1] * self.env.env_size[0] + next_state[0]
            action_step = self.policy[state_index]
        g = 0.0
        for t in range(length - 1, -1, -1):
            g = g * self.gamma + reward_list[t]
            state_tmp = state_list[t]
            # if state_tmp == (4,1):
            #     print(1)
            action_tmp = self.env.action_space[action_list[t]]
            self.Return_s_a[(state_tmp,action_tmp)] = self.Return_s_a[(state_tmp,action_tmp)] + g
            self.Number_s_a[(state_tmp,action_tmp)] = self.Number_s_a[(state_tmp,action_tmp)] + 1
            self.q_s_a[(state_tmp,action_tmp)] = self.Return_s_a[(state_tmp,action_tmp)] / self.Number_s_a[(state_tmp,action_tmp)]
            q_star_list = []
            for action in self.env.action_space:
                q_star_list.append(self.q_s_a[(state_tmp,action)])
            q_star = np.argmax(q_star_list)
            state_index = state_tmp[1] * self.env.env_size[0] + state_tmp[0]
            num = random.random()
            if num >= self.eposilon:
                self.policy[state_index] = q_star
            else:
                self.policy[state_index] = random.randint(0,4)
def MC_test_1():
    P,R,env = GetMonteCarolModel(1)
    gamma = 0.9
    initial_policy = [0,2,1,1,2,2,1,1,4]
    best_action_list = []
    mc = MonteCarol(env,P,R,gamma)
    mc.GeneratePolicy(initial_policy)

    for state in range(mc.env.num_states):
        qa_list = []
        for action in range(len(mc.env.action_space)):
            # print(f"state: {state}, Action: {action}, Q: {mc.GetEpisode(state,action)}")
            qa_list.append(mc.GetEpisode(state,action))
        a_star = np.argmax(qa_list)
        best_action_list.append(a_star)
    mc.GeneratePolicy(best_action_list)
    env.show_policy(mc.policy_matrix)

def MC_test_2():
    P,R,env = GetMonteCarolModel(2)
    gamma = 0.9
    initial_policy = [4,3,3,2,4,4,2,0,0,2,0,4,2,0,2,0,1,4,3,2,4,3,0,3,4]
    # initial_policy = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
    best_action_list = []
    mc = MonteCarol(env,P,R,gamma)
    mc.GeneratePolicy(initial_policy)

    for state in range(mc.env.num_states):
        qa_list = []
        for action in range(len(mc.env.action_space)):
            # print(f"state: {state}, Action: {action}, Q: {mc.GetEpisode(state,action)}")
            qa_list.append(mc.GetEpisode(state,action))
        a_star = np.argmax(qa_list)
        best_action_list.append(a_star)
    mc.GeneratePolicy(best_action_list)
    
    env.show_policy(mc.policy_matrix)
    # env.show_value(m)
def MC_test_3():
    P,R,env = GetMonteCarolModel(2)
    gamma = 0.95
    initial_policy = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
    # initial_policy = [4,3,3,2,4,4,2,0,0,2,0,4,2,0,2,0,1,4,3,2,4,3,0,3,4]

    mc_exploring_starts = ExploringMontrCarol(env,P,R,gamma)
    mc_exploring_starts.GeneratePolicy(initial_policy)
    for _ in range(5):
        for turn_state in range(mc_exploring_starts.env.num_states):
            for turn_action in range(len(mc_exploring_starts.env.action_space)):
                mc_exploring_starts.update(turn_state,turn_action)
    mc_exploring_starts.GeneratePolicy(mc_exploring_starts.policy)
    env.show_policy(mc_exploring_starts.policy_matrix)
def MC_test_4():
    P,R,env = GetMonteCarolModel(2)
    gamma = 0.95
    eposilon = 0.001
    initial_policy = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
    # initial_policy = [4,3,3,2,4,4,2,0,0,2,0,4,2,0,2,0,1,4,3,2,4,3,0,3,4]

    mc_exploring_starts = EposilonGreedy_MontrCarol(env,P,R,gamma,eposilon)
    mc_exploring_starts.GeneratePolicy(initial_policy)
    for _ in range(2):
        for turn_state in range(mc_exploring_starts.env.num_states):
            for turn_action in range(len(mc_exploring_starts.env.action_space)):
                mc_exploring_starts.update(turn_state,turn_action)
    mc_exploring_starts.GeneratePolicy(mc_exploring_starts.policy)
    env.show_policy(mc_exploring_starts.policy_matrix)
def main():
    # MC_test_1()
    # MC_test_2()
    # MC_test_3()
    MC_test_4()
if __name__ == "__main__":
    main()
    
