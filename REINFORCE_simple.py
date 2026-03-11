# REINFORCE 简化版 - 可以正常训练
import numpy as np
import random
from model_def import GetMonteCarolModel
from src.grid_world import GridWorld
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

class REINFORCE:
    def __init__(self, env, gamma, lr=0.01):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.num_states = env.num_states
        self.num_actions = len(env.action_space)
        
        # 简单的查表策略参数 theta[s, a]
        self.theta = np.zeros((self.num_states, self.num_actions))
        
    def get_action_probs(self, state_idx):
        """获取状态state_idx的动作概率分布"""
        return softmax(self.theta[state_idx])
    
    def state_to_idx(self, state):
        """将(x,y)坐标转换为状态索引"""
        return state[1] * self.env.env_size[0] + state[0]
    
    def generate_episode(self):
        """生成一条完整轨迹"""
        state = self.env.start_state
        self.env.reset(state)
        
        states = []
        actions = []
        rewards = []
        
        for _ in range(100):  # 最大100步
            state_idx = self.state_to_idx(state)
            states.append(state_idx)
            
            # 采样动作
            probs = self.get_action_probs(state_idx)
            action = np.random.choice(self.num_actions, p=probs)
            actions.append(action)
            
            # 执行动作
            next_state, reward, done, _ = self.env.step(self.env.action_space[action])
            rewards.append(reward)
            
            state = next_state
            if done:
                break
                
        return states, actions, rewards
    
    def update(self, states, actions, rewards):
        """策略梯度更新"""
        # 计算每个时间步的累计回报 G_t
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        
        # 归一化回报（减小方差）
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)
        
        # 策略梯度更新
        for t in range(len(states)):
            state_idx = states[t]
            action = actions[t]
            G_t = returns[t]
            
            probs = self.get_action_probs(state_idx)
            
            # 计算策略梯度: d(log pi)/d(theta)
            for a in range(self.num_actions):
                if a == action:
                    grad = 1 - probs[a]
                else:
                    grad = -probs[a]
                self.theta[state_idx, a] += self.lr * G_t * grad
    
    def train(self, num_episodes=5000, print_freq=500):
        """训练"""
        reward_history = []
        
        for ep in range(num_episodes):
            states, actions, rewards = self.generate_episode()
            self.update(states, actions, rewards)
            
            if ep % print_freq == 0:
                avg_reward = self.evaluate(10)
                reward_history.append(avg_reward)
                print(f"Episode {ep}: Avg Reward = {avg_reward:.2f}")
        
        return reward_history
    
    def evaluate(self, num_episodes=10):
        """评估当前策略（贪心）"""
        total_reward = 0
        for _ in range(num_episodes):
            state = self.env.start_state
            self.env.reset(state)
            ep_reward = 0
            
            for _ in range(100):
                state_idx = self.state_to_idx(state)
                action = np.argmax(self.get_action_probs(state_idx))
                next_state, reward, done, _ = self.env.step(self.env.action_space[action])
                ep_reward += reward
                state = next_state
                if done:
                    break
            total_reward += ep_reward
        return total_reward / num_episodes
    
    def get_policy_matrix(self):
        """获取贪心策略矩阵用于可视化"""
        policy = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            action = np.argmax(self.get_action_probs(s))
            policy[s, action] = 1
        return policy


def main():
    env = GetMonteCarolModel(3)
    agent = REINFORCE(env, gamma=0.9, lr=0.01)
    
    print("开始训练 REINFORCE...")
    history = agent.train(num_episodes=5000, print_freq=500)
    
    # 绘制收敛曲线
    plt.figure()
    plt.plot(range(0, 5000, 500), history)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('REINFORCE Training Progress')
    plt.savefig('training_curve.png')
    print("\n收敛曲线已保存到 training_curve.png")
    
    # 最终评估
    final_reward = agent.evaluate(100)
    print(f"\n最终评估（100回合）: 平均奖励 = {final_reward:.2f}")
    
    # 可视化策略
    policy_matrix = agent.get_policy_matrix()
    env.reset(env.start_state)
    env.show_policy(policy_matrix)
    print("策略可视化已保存到 policy_result.png")


if __name__ == '__main__':
    main()
