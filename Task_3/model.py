import os
import numpy as np
from memory import Memory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
            
    def forward(self, X):
        
        h1 = F.relu(self.fc1(X))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        output = F.softmax(self.fc4(h3), dim=-1)

        return output


class Critic(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
            
    def forward(self, X):

        h1 = F.relu(self.fc1(X.float()))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        output = self.fc4(h3)

        return output

class PPO():

    def __init__(self, env, T, hidden_dim= 256, gamma=0.99, gae_lambda=0.95,
                clip=0.2, no_batches=5, epochs=5, lr_actor=0.0001, lr_critic=0.0003):
                
                self.env = env
                self.state_dim = env.observation_space.shape[0]
                self.actions_dim = env.action_space.n
                self.gamma = gamma
                self.clip = clip
                self.epochs = epochs
                self.gae_lambda = gae_lambda

                self.actor = Actor(self.state_dim, hidden_dim, self.actions_dim)
                self.actor.to(device)
                self.critic = Critic(self.state_dim, hidden_dim)
                self.critic.to(device)

                self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
                self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

                self.memory = Memory(no_batches, T)
                
                self.actor_losses = []
                self.critic_losses = []

    def save_weights(self):
        
        torch.save(self.actor.state_dict(), "model_weights/actor.pth")
        torch.save(self.critic.state_dict(), "model_weights/critic.pth")
    
    def get_action_value(self, state):

        state = torch.tensor([state], dtype=torch.float).to(device)

        self.actor.eval()
        self.critic.eval()

        distribution = Categorical(self.actor(state))
        action = distribution.sample() 
        log_prob = torch.squeeze(distribution.log_prob(action)).item()

        state_value = self.critic(state)  
        state_value = torch.squeeze(state_value).item()
        action = torch.squeeze(action).item()

        return action, log_prob, state_value

    def optimize(self):
        
        for epoch in range(self.epochs):
                    
            batch_items = self.memory.get_batches()
            states, values, actions, log_probs,\
            rewards, dones = batch_items[0] 
            batch_inds = batch_items[1]
            self.actor_losses = []
            self.critic_losses = []

            # calculate the Generalized Advantage Estimation
            advantage = [0]
            gae = 0
            for i in reversed(range(self.memory.T - 1)):
                # calculate delta for current time step
                delta = rewards[i] + self.gamma * values[i + 1] * (1-int(dones[i])) - values[i]
                # add the discounted deltas of the next steps
                gae = delta + self.gamma * self.gae_lambda * gae * (1-int(dones[i]))
                advantage.insert(0, gae)
    
            # normalize advantage
            advantage = torch.tensor(advantage).to(device)
            advantage = advantage / 20

            values = torch.tensor(values).to(device)


            for batch in batch_inds:
                states_batch = torch.tensor(states[batch], dtype=torch.float).to(device)
                old_log_probs = torch.tensor(log_probs[batch]).to(device)
                actions_batch = torch.tensor(actions[batch]).to(device)
                
                self.critic.eval()
                critic_values = self.critic(states_batch)
                critic_values = torch.squeeze(critic_values)
                self.critic.train()

                self.actor.eval()
                new_distribution = Categorical(self.actor(states_batch))
                new_log_probs = new_distribution.log_prob(actions_batch)
                self.actor.train()

                ratio = (new_log_probs - old_log_probs).exp()
           
                surrogate = advantage[batch] * ratio
                clipped_surrogate = torch.clamp(ratio, 1-self.clip,
                                    1+self.clip) * advantage[batch]

                actor_loss = -torch.min(surrogate, clipped_surrogate).mean()
                #print(f"Actor loss {actor_loss}")
                
                self.actor_losses.append(actor_loss)
                

                # advantage = returns - values
                returns = (advantage[batch] + values[batch]).float()

                
                loss_funct = nn.MSELoss()
                critic_loss = loss_funct(returns, critic_values)
                
                self.critic_losses.append(critic_loss)
                #print(f"Critic loss {critic_loss}")
                

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                #print(actor_loss)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            