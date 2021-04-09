import os
import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):    
    
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        distribution = F.softmax(out, dim=-1)
        distribution = Categorical(distribution)

        return distribution

class Critic(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1):

        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, state):

        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        value = self.fc4(out)
        return value

class ActorCritic():

    def __init__(self, env, episodes, max_score, hidden_size=256):
        
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n    

        self.actor = Actor(input_size, hidden_size, output_size)
        self.critic = Critic(input_size, hidden_size)
        self.actor_optim = optim.Adam(self.actor.parameters())
        self.critic_optim = optim.Adam(self.critic.parameters())

        self.episodes = episodes
        self.max_score = max_score
        self.log_probs = []
        self.values = []
        self.rewards = []

    def get_returns(self, gamma=0.99):

        returns = [0]
        R = 0
        for i in reversed(range(len(self.rewards) - 1)):
            # add the discounted returns of the next steps
            R = self.rewards[i] + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, requires_grad=True).to(device)

        return returns

    def get_action_value(self, state):

        state = torch.tensor([state], dtype=torch.float).to(device)

        self.actor.eval()
        self.critic.eval()

        distribution = self.actor(state)
        action = distribution.sample() 
        log_prob = torch.squeeze(distribution.log_prob(action))
        action = torch.squeeze(action).item()

        state_value = self.critic(state)  
        state_value = torch.squeeze(state_value)
        
        return action, log_prob, state_value

    def save_to_memory(self, log_prob, value, reward):

        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)

    def clear_memory(self):

        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()

    def optimize(self):

        returns = self.get_returns()
        values = torch.stack(self.values).to(device)
        advantage = returns - values
        advantage = advantage.detach() # required for actor_loss backprop w/o error

        log_probs = torch.stack(self.log_probs).to(device)
        actor_loss = ( -log_probs * advantage ).sum()

        loss_funct = nn.MSELoss()
        critic_loss = loss_funct(returns, values)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def train(self, env):

        game_rew_hist = []

        for ep in range(self.episodes):

            episode_rew = 0
            state = env.reset()
            done = False
    
            while not done:
                action, log_prob, value = self.get_action_value(state)
                state_next, reward, done, _ = env.step(action)
                episode_rew += reward
                self.save_to_memory(log_prob, value, reward)
                state = state_next

            self.optimize()
            self.clear_memory()

            game_rew_hist.append(episode_rew)

            if (ep) % 10 == 0:
                avg_score = np.mean(game_rew_hist[-10:])
                print(f"Episode: {ep + 1} \t ¦ \t Reward {avg_score}")

            if avg_score == self.max_score:
                print(f"Maximum score of {self.max_score} reached")
                break

def run():
    env = gym.make("CartPole-v0")   
    episodes = 5000
    max_score = 200

    actor_critic = ActorCritic(env, episodes=episodes, 
                    max_score=max_score, hidden_size=256)
    
    ## run training    
    actor_critic.train(env)
    
    '''
    x = np.linspace(1,epochs, epochs)
       
    plt.subplot(1,2,1)
    plt.plot(x, actor_loss)
    plt.title("Actor loss")
        
    plt.subplot(1,2,2)
    plt.plot(x, critic_loss)
    plt.title("critic loss")
        
    plt.show()
    '''

run()