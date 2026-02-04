import sys
import os
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录（假设当前在 examples 目录下）
project_root = os.path.join(current_dir, '..')
# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(project_root))

from src.grid_world import GridWorld
import random
import numpy as np


# Example usage:
if __name__ == "__main__":             
    env = GridWorld()
    state = env.reset()               
    # for t in range(1):
    #     env.render()
    #     action = random.choice(env.action_space)
    #     next_state, reward, done, info = env.step(action)
    #     print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
    #     # if done:
    #     #     break
    env.render()
    # Add policy
    policy_matrix=np.random.rand(env.num_states,len(env.action_space))                                            
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1

    env.add_policy(policy_matrix)

    
    # Add state values
    # values = np.random.uniform(0,10,(env.num_states,))
    # env.add_state_values(values)

    # Render the environment
    env.render(animation_interval=60)