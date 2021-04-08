""" PPO model to solve the environment.

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from memory import Memory


os.environ['KMP_DUPLICATE_LIB_OK']='True' # required to run on MAC OS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """ Actor Neural Network class.

    Args:
        input_dim: Network input dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Network output dimension.

    """
    def __init__(self, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        """ Forward propogation.

        Args:
            X: Input.

        Returns:
            Output: Softmax distribution.

        """
        h1 = F.relu(self.fc1(X))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        output = F.softmax(self.fc4(h3), dim=-1)

        return output


class Critic(nn.Module):
    """ Critic Neural Network class.

    Args:
        input_dim: Network input dimension.
        hidden_dim: Hidden layer dimension.

    """
    def __init__(self, input_dim, hidden_dim):

        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, X):
        """ Forward propogation.

        Args:
            X: Input.

        Returns:
            Output: Softmax distribution.

        """
        h1 = F.relu(self.fc1(X.float()))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        output = self.fc4(h3)

        return output

class PPO():
    """ PPO algorithm class with the learning algorithm.

        Args:
            env: Environment.
            T: Steps to put into memory for every optimization.
            hidden_dim: Networks hidden layer neuron numbers
            gamma: Discount factor for future returns
            gae_lambda: GAE smoothing parameter
            clip: PPO actor loss epsilon clipping parameter
            no_batches: No of batches to split items in memory
            epochs: Number of epochs per optimization
            lr_actor: Actor network learning rate
            lr_critic: Critic network learning rate

        Attributes:
            state_dim: Dimensions of the environment observation.
            actions_dim: Dimensions of the action.
            actor: The actor network.
            critic: The critic network.
            actor_optim: Actor network optimizer.
            critic_optim: Critic network optimizer.
            memory: Memory object to keep items for optimization
            env: Environment.
            gamma: Discount factor for future returns
            gae_lambda: GAE smoothing parameter
            clip: PPO actor loss epsilon clipping parameter
            epochs: Number of epochs per optimization

    """
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

    def save_weights(self):
        """ Saves network weights for maximum scores.

        """
        torch.save(self.actor.state_dict(), "model_weights/actor.pth")
        torch.save(self.critic.state_dict(), "model_weights/critic.pth")

    def get_action_value(self, state):
        """ Gets action, probability and value from the state

        Args:
            state: Current state.

        Returns:
            action: Chosen action for the agent
            log_prob: Log-probability of the chosen action.
            state_value: The value of the state estimated by critic network.

        """
        state = torch.tensor([state], dtype=torch.float).to(device)

        self.actor.eval()
        self.critic.eval()

        distribution = Categorical(self.actor(state))
        action = distribution.sample()
        log_prob = torch.squeeze(distribution.log_prob(action)).item()
        action = torch.squeeze(action).item()

        state_value = self.critic(state)
        state_value = torch.squeeze(state_value).item()

        return action, log_prob, state_value

    def optimize(self):
        """ Optimizes the actor and critic networks.

        Returns:
            None

        """

        for _ in range(self.epochs):

            batch_items = self.memory.get_batches()
            states, values, actions, log_probs,\
            rewards, dones = batch_items[0]
            batch_inds = batch_items[1]

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
            # The simpler form of normalization below gives better results
            advantage = advantage / 10

            values = torch.tensor(values).to(device)

            for batch in batch_inds:
                states_batch = torch.tensor(states[batch], dtype=torch.float).to(device)
                old_log_probs = torch.tensor(log_probs[batch]).to(device)
                actions_batch = torch.tensor(actions[batch]).to(device)

                # when values in memory are used, pytorch gives errors during backprop.
                self.critic.eval()
                critic_values = self.critic(states_batch)
                critic_values = torch.squeeze(critic_values)
                self.critic.train()

                # get the new probability from the latest policy
                self.actor.eval()
                new_distribution = Categorical(self.actor(states_batch))
                new_log_probs = new_distribution.log_prob(actions_batch)
                self.actor.train()

                ratio = (new_log_probs - old_log_probs).exp()

                surrogate = advantage[batch] * ratio
                clipped_surrogate = torch.clamp(ratio, 1-self.clip,
                                    1+self.clip) * advantage[batch]

                actor_loss = -torch.min(surrogate, clipped_surrogate).mean()

                # advantage = returns - values
                returns = (advantage[batch] + values[batch]).float()

                loss_funct = nn.MSELoss()
                critic_loss = loss_funct(returns, critic_values)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()


