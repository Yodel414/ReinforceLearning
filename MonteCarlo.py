import numpy as np
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
    # initial_policy = [4,3,3,2,4,4,2,0,0,2,0,4,2,0,2,0,1,4,3,2,4,3,0,3,4]
    initial_policy = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
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
def main():
    # MC_test_1()
    MC_test_2()
if __name__ == "__main__":
    main()
    
