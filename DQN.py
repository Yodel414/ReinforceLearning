import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from model_def import GetMonteCarolModel
from src.grid_world import GridWorld
import random
import numpy as np
from collections import deque
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
class DQN:
    def __init__(self, env:GridWorld, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.action_num = len(env.action_space)
        self.state_dim = 2  # 输入坐标 (x, y)
        self.num_states = env.num_states
        # 两个网络：主网络和目标网络
        self.policy_net = QNetwork(self.state_dim, self.action_num)
        self.target_net = QNetwork(self.state_dim, self.action_num)
        self.epsilon = 0.1
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 100
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_update_freq = 10 # 每10个episode同步一次网络
        self.memory = deque(maxlen=1000) 
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
    def epsilon_greedy(self,state_tensor):
        if random.random() < self.epsilon:
            with torch.no_grad():
                q = self.policy_net(state_tensor)
                return torch.argmax(q).item()
        else:
            return random.randint(0,4)
    def GeneratePolicy(self):
        policy_matrix=np.zeros((self.num_states,self.action_num))
        for index in range(self.num_states):
            with torch.no_grad():
                tensor_state = self.get_tensor_state(index)
                action = np.argmax(self.target_net(tensor_state)).item()
                policy_matrix[index][action] = 1
        self.policy_matrix = policy_matrix

    def get_tensor_state(self,state):
        if isinstance(state,tuple):
            x = state[0]
            y = state[1]
        elif isinstance(state,int):
            x = state % self.env.env_size[0]
            y = state // self.env.env_size[0]          
        return torch.FloatTensor([x/4.0, y/4.0])

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0,0.0,0.0
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.stack(states)
        actions = torch.LongTensor(actions).view(-1, 1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        current_q = self.policy_net(states).gather(1,actions)
        with torch.no_grad():  #利用目标网络计算target_Q
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q
        loss = nn.MSELoss()(current_q.squeeze(),target_q)
        self.optimizer.zero_grad()
        loss.backward()
        total_norm = 0.0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5
        self.optimizer.step()
        rmse = np.sqrt(mean_squared_error(current_q.detach().cpu().numpy().squeeze(), 
                                          target_q.detach().cpu().numpy()))
        return loss.item(),grad_norm,rmse
    def train(self,iteration_times):
        reward_history = []
        loss_history = []
        grad_norm_history = []
        rmse_list = []
        for i in range(iteration_times):
            done = False
            episode = 0
            total_reward = 0
            state = self.env.start_state
            self.env.reset(state)
            while not done and episode < 1000:
                tensor_state = self.get_tensor_state(state)
                acion_based_pi_b = self.epsilon_greedy(tensor_state)
                next_state, reward, done, info = self.env.step(self.env.action_space[acion_based_pi_b])
                tensor_state_prime = self.get_tensor_state(next_state)
                self.memory.append((tensor_state,acion_based_pi_b,reward,tensor_state_prime))
                state = next_state
                loss, grad_norm,rmse = self.learn()
                if loss is not None:          # 仅当更新时才记录
                    loss_history.append(loss)
                    grad_norm_history.append(grad_norm)
                    rmse_list.append(rmse)
                episode += 1
                total_reward += reward
            self.epsilon = max(self.epsilon_min,self.epsilon * self.epsilon_decay)
            if i % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            reward_history.append(total_reward)
            if (i+1) % 100 == 0:
                avg_reward = np.mean(reward_history[-100:])
                print(f"Episode {i+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        return reward_history,loss_history,grad_norm_history,rmse_list
    
def plot_training(reward_hist, loss_hist, grad_norm_hist,rmse_hist):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 8))

        plt.subplot(4, 1, 1)
        plt.plot(reward_hist, color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward per Episode')
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(loss_hist, color='red', alpha=0.7)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Loss over Time')
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(grad_norm_hist, color='green', alpha=0.7)
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm over Time')
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.plot(rmse_hist, color='green', alpha=0.7)
        plt.xlabel('Training Step')
        plt.ylabel('RMSE')
        plt.title('RMSE over Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def test1():
    env = GetMonteCarolModel(3)
    gamma = 0.9
    dqn = DQN(env,gamma)
    iteration_times = 1000
    reward_hist, loss_hist, grad_hist,rmse_hist = dqn.train(iteration_times)
    dqn.GeneratePolicy()
    dqn.env.show_policy(dqn.policy_matrix)
    plot_training(reward_hist, loss_hist, grad_hist,rmse_hist)
if __name__ == '__main__':
    test1()  