import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

#%%
class Actor(nn.Module):    
    
    def __init__(self, input_size,layer_2, layer_3, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, layer_2)
        self.fc2 = nn.Linear(layer_2, layer_3)
        self.fc3 = nn.Linear(layer_3, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        distribution = F.softmax(out, dim=-1)
        distribution = Categorical(distribution)
        return distribution

#%%
class Critic(nn.Module):

    def __init__(self, input_size, layer_2, layer_3, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, layer_2)
        self.fc2 = nn.Linear(layer_2, layer_3)
        self.fc3 = nn.Linear(layer_3, output_size)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        value = self.fc3(out)
        return value


#%%
class ActorCritic():

    def __init__(self, epochs, actor, critic, actor_optim, critic_optim):
        super().__init__()
        self.epochs = epochs
        self.actor = actor
        self.critic = critic
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
    
    def return_count(self, next_value, rewards, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R 
            returns.insert(0, R)
        
        return returns

    
    def train(self, env):
        actor_losses = []
        critic_losses = []
        
        for epoch in range(self.epochs):
            
            log_probabilities_per_game = []
            values_per_game = []
            rewards_per_game = []
            advantage = []
            entropy = 0
            #env.reset()
            
            state = env.reset()
            
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
    
            for i in count():
                state = torch.from_numpy(state).float()
                distribution = self.actor(state)
                value = self.critic(state)
    
                action = distribution.sample()
                #action = action.item()
                new_state, new_reward, done, _ = env.step(action.item())
                

                log_probability = torch.squeeze(distribution.log_prob(action)).item()
                entropy += torch.squeeze(distribution.entropy().mean()).item()
                
                
                log_probabilities_per_game.append(log_probability)
                values_per_game.append(torch.squeeze(value).item())
                rewards_per_game.append(new_reward)
    
                state = new_state
    
                if done:
                    print(f"Iteration: {epoch + 1} \t ¦ \t Reward {sum(rewards_per_game)}")
                    break
    
    
            new_state = torch.FloatTensor(new_state)
            new_value = torch.squeeze(self.critic(new_state)).item()
            returns = self.return_count(new_value, rewards_per_game)
    
            advantage = [x - y for x,y in zip(returns, values_per_game)]            
            actor_loss = [-(x * y) for x,y in zip(log_probabilities_per_game, advantage)]
            actor_loss = sum(actor_loss)/i
            critic_loss = sum([x**2 for x in advantage]) /i
            
            
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
    
            actor_loss = torch.tensor(actor_loss, dtype=torch.float, requires_grad=True)
            critic_loss = torch.tensor(critic_loss, dtype=torch.float, requires_grad=True)

            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()
            

        env.close()
        
        return actor_losses, critic_losses
    
 #%%
def run():
    env = gym.make("CartPole-v0")   
    epochs = 100
    
    ## initializing NN layers
    input_size = env.observation_space.shape[0]
    layer_2 = 128
    layer_3 = 256
    output_size = env.action_space.n
    
    ## creating an object of Actor and Critic
    ## same structure as in PPO for comparing results
    actor = Actor(input_size, layer_2, layer_3, output_size)
    critic = Critic(input_size,layer_2, layer_3)
    
    ## optimizations
    actor_optim = optim.Adam(actor.parameters())
    critic_optim = optim.Adam(critic.parameters())
    
    actor_critic = ActorCritic(epochs, actor, critic, actor_optim, critic_optim)
    
    ## run training    
    actor_loss, critic_loss = actor_critic.train(env)
    
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