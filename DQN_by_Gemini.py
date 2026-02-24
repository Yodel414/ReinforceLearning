import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from model_def import GetMonteCarolModel

# 1. 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_num):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_num)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.action_num = len(env.action_space)
        self.state_dim = 2  # 输入坐标 (x, y)
        
        # 两个网络：主网络和目标网络
        self.policy_net = QNetwork(self.state_dim, self.action_num)
        self.target_net = QNetwork(self.state_dim, self.action_num)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=2000) # 经验回放池
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.target_update_freq = 10 # 每10个episode同步一次网络

    def get_state_tensor(self, state):
        # 将 ID 或 Tuple 统一转为归一化的 Tensor [x, y]
        if isinstance(state, int):
            row = state // self.env.env_size[0]
            col = state % self.env.env_size[0]
        else:
            row, col = state[1], state[0]
        # 归一化到 [0, 1] 有助于神经网络收敛
        return torch.FloatTensor([row/4.0, col/4.0])

    def choose_action(self, state_tensor):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_num)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # 随机采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_copy = states
        states = torch.stack(states)
        actions = torch.LongTensor(actions).view(-1, 1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)

        # 当前 Q 值
        current_q = self.policy_net(states).gather(1, actions)
        
        # 计算目标 Q 值 (使用 Target Network)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # 损失函数与优化
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes=500):
        reward_history = []
        for e in range(episodes):
            self.env.reset(self.env.start_state)
            state = self.env.start_state # 防护
            
            done = False
            total_reward = 0
            
            while not done:
                s_tensor = self.get_state_tensor(state)
                action = self.choose_action(s_tensor)
                
                next_state, reward, done, _ = self.env.step(self.env.action_space[action])
                ns_tensor = self.get_state_tensor(next_state)
                
                # 存入经验回放
                self.memory.append((s_tensor, action, reward, ns_tensor, done))
                
                state = next_state
                total_reward += reward
                self.learn()
                
            # 更新 Epsilon 和 目标网络
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if e % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if e % 20 == 0:
                print(f"Episode: {e}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")
        
        return reward_history
    
def test1():
    env = GetMonteCarolModel(3)
    gamma = 0.9
    dqn = DQNAgent(env,gamma)
    N = 10000
    reward_history = dqn.train()
    print(reward_history)

if __name__ == '__main__':

    test1()  