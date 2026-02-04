from src.grid_world import GridWorld
import numpy as np
# (0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)  # down, right, up, left, stay
def ConsturctPolicy(num_states = 25,action_space = 25,type = 1):
    policy_matrix=np.zeros((num_states,action_space)) 
    policy_matrix[0][1] = 1
    policy_matrix[1][2] = 1
    policy_matrix[2][3] = 1
    policy_matrix[3][8] = 1
    policy_matrix[4][9] = 1
    
    policy_matrix[5][0] = 1
    policy_matrix[6][1] = 1
    policy_matrix[7][8] = 1
    policy_matrix[8][13] = 1
    policy_matrix[9][14] = 1
    
    policy_matrix[10][5] = 1
    policy_matrix[11][10] = 1
    policy_matrix[12][17] = 1
    policy_matrix[13][14] = 1
    policy_matrix[14][19] = 1
    
    policy_matrix[15][10] = 1
    policy_matrix[16][17] = 1
    policy_matrix[17][17] = 1
    policy_matrix[18][17] = 1
    policy_matrix[19][24] = 1
    
    policy_matrix[20][15] = 1
    policy_matrix[21][22] = 1
    policy_matrix[22][17] = 1
    policy_matrix[23][22] = 1
    policy_matrix[24][23] = 1
    return policy_matrix
def GetStateValue(Mtype,policy_matrix,r_pi,gamma):
    eps = 1e-3
    if Mtype == 1:
        return np.linalg.inv(I - gamma * policy_matrix) @ r_pi
    elif Mtype == 2:
        max_iter = 100
        vk = np.zeros(25)
        for index in range(max_iter):
            vk_new = r_pi + gamma * policy_matrix @ vk        
            if np.linalg.norm(vk_new - vk) < eps:
                print(index)
                break
            else:
                vk = vk_new
        return vk_new
if __name__ == "__main__":             
    env = GridWorld()
    state = env.reset()    
    r_pi = np.zeros(env.num_states)
    I = np.eye(env.num_states)
    gamma = 0.9
    policy_matrix = ConsturctPolicy()
    for i in range(env.env_size[0]):
        for j in range(env.env_size[1]):
            if (i,j) in env.forbidden_states:
                r_pi[5 * i + j] = env.reward_forbidden
            elif (i,j) == env.target_state:
                r_pi[5 * i + j] = env.reward_target
            else:
                r_pi[5 * i + j] = env.reward_step
    # print(GetStateValue(1,policy_matrix,r_pi,gamma))
    # print(GetStateValue(2,policy_matrix,r_pi,gamma))
    print(r_pi)