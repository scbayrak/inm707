""" a2c algorithm implementation.

"""

import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    """ Actor Neural Network class.

    Args:
        input_size: Network input dimension.
        hidden_size: Hidden layer dimension.
        output_size: Network output dimension.

    """
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """ Forward propogation.

        Args:
            state: Current state.

        Returns:
            distibution: Probability distribution of actions.

        """

        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        distribution = F.softmax(out, dim=-1)
        distribution = Categorical(distribution)

        return distribution

class Critic(nn.Module):
    """ Critic Neural Network class.

    Args:
        input_size: Network input dimension.
        hidden_size: Hidden layer dimension.
        output_size: Network output dimension.

    """

    def __init__(self, input_size, hidden_size, output_size=1):

        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """ Forward propogation.

        Args:
            state: Current state.

        Returns:
            value: Estimated state value.

        """

        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        value = self.fc4(out)
        return value

class ActorCritic():
    """ Actor Critic Class.

    Args:
        env: Environment to solve. 
        episodes: Max number of training episodes. 
        max_score: Maximum score possible in the environment.
        hidden_size: Hidden layer dimension.
        gamma: Discount factor for future rewards.
        save: Saves the network weights if set to True.

    """
    def __init__(self, env, episodes, max_score, hidden_size=256, gamma=0.99, save=False):
        
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n    

        self.env = env
        self.actor = Actor(input_size, hidden_size, output_size)
        self.actor.to(device)
        self.critic = Critic(input_size, hidden_size)
        self.critic.to(device)
        self.actor_optim = optim.Adam(self.actor.parameters())
        self.critic_optim = optim.Adam(self.critic.parameters())

        self.episodes = episodes
        self.max_score = max_score
        self.gamma = gamma
        self.save = save 
        self.log_probs = []
        self.values = []
        self.rewards = []

    def get_returns(self):
        """ Calculates returns from rewards.

        Args:
            None.

        Returns:
            returns: Calculated returns tensor.

        """
        returns = [0]
        R = 0
        for i in reversed(range(len(self.rewards) - 1)):
            # add the discounted returns of the next steps
            R = self.rewards[i] + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, requires_grad=True).to(device)

        return returns

    def get_action_value(self, state):
        """ Gets chosen action, log_probability and state_value from the 2 networks.

        Args:
            state: current state.

        Returns:
            action: Chosen action.
            log_prob: Log probality of the chosen action.
            state_value: Estimated value of the state. 

        """
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
        """ Saves log_probs, values and rewards to memory for optimization .

        Args:
            log_prob: Log-probability of the chosen action.
            value: Estimated value of the state. 
            reward: reward of the action. 

        Returns:
            None.  

        """
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)

    def clear_memory(self):
        """ Resets memory for the new episode.

        Args:
            None. 

        Returns:
            None.  

        """
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()

    def save_weights(self):
        """ Saves network weights.

        Args:
            None. 

        Returns:
            None.  

        """
        torch.save(self.actor.state_dict(), "model_weights/actor.pth")
        torch.save(self.critic.state_dict(), "model_weights/critic.pth")

    def optimize(self):
        """ Updates networks weights to train.

        Args:
            None.

        Returns:
            None.  

        """
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

    def train(self):
        """ Main training loop. Plays N episodes and optimizes networks after each episode.

        Args:
            None.

        Returns:
            best_score: Best score obtained. 
            ep: The number of episodes played. 
            game_rew_hist: List of rewards obtained at each episode.

        """
        game_rew_hist = []
        best_score = - np.Inf

        for ep in range(self.episodes):

            episode_rew = 0
            state = self.env.reset()
            done = False
    
            while not done:
                action, log_prob, value = self.get_action_value(state)
                state_next, reward, done, _ = self.env.step(action)
                episode_rew += reward
                self.save_to_memory(log_prob, value, reward)
                state = state_next

            self.optimize()
            self.clear_memory()

            game_rew_hist.append(episode_rew)

            if (ep) % 10 == 0:
                avg_score = np.mean(game_rew_hist[-10:])
                print(f"Episode: {ep + 1} \t ¦ \t Reward {avg_score}")

            if avg_score > best_score:
                best_score = avg_score
                if self.save:
                    self.save_weights()

            if avg_score == self.max_score:
                print(f"Maximum score of {self.max_score} reached")
                break

        return best_score, ep, game_rew_hist
