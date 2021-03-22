import os
import gym
import numpy as np
from PPO import PPO

class PpoLearning():

    def __init__(self, no_games, PPO):

        self.no_games = no_games
        self.ppo = PPO

    def train(self):

        game_rew_hist = [] # list to hold the total rewards per game
        iters = 0
        step = 0

        for game in range(1, no_games+1):

            done = False
            game_total_rew = 0
            state = env.reset()
            episode_steps = 0

            while not done:
                action, log_prob, value = self.ppo.get_action_value(state)
                next_state , reward, done, info = env.step(action)
                step += 1
                episode_steps += 1
                game_total_rew += reward
                self.ppo.memory.save_memory(state, value, action, log_prob, reward, done)
                if step % self.ppo.memory.T == 0:
                    self.ppo.optimize()
                    iters += 1
                    self.ppo.memory.reset()
                state = next_state

            print(f"episode steps: {episode_steps}")
            game_rew_hist.append(game_total_rew)

            if (game) % 10 == 0:
                avg_score = np.mean(game_rew_hist[-10:])

                print('Episode: ', game, 'average score:', avg_score, 
                'learning_iterations:', iters)

# define the environment
env = gym.make('LunarLander-v2')
env.reset()

# hyper-parameters for PPO learning
epochs = 4 # number of epochs per learning
no_batches = 2 # number of batches for splitting the timesteps
hidden_dim = 256
gamma = 0.99 # discount factor
gae_lambda = 0.95 # gae smoothing parameter
lr_actor = 0.0001
lr_critic = 0.0001
clip = 0.2 # PPO clipping epsilon parameter
T = 10 # timesteps per each learning

ppo = PPO(env=env, T=T, hidden_dim=hidden_dim, gamma=gamma, 
    gae_lambda=gae_lambda, clip=clip, no_batches=no_batches,
    epochs=epochs, lr_actor=lr_actor, lr_critic=lr_critic)

no_games = 500 

training = PpoLearning(no_games, ppo)
training.train()

