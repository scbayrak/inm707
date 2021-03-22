import os
import gym
import numpy as np


class PpoLearning():

    def __init__(self, no_games, PPO):

        self.no_games = no_games
        self.ppo = PPO
        self.env = PPO.env

    def train(self):

        game_rew_hist = [] # list to hold the total rewards per game
        iters = 0
        step = 0
        best_score = - np.Inf
        for game in range(1, self.no_games+1):

            done = False
            game_total_rew = 0
            state = self.env.reset()

            while not done:
                action, log_prob, value = self.ppo.get_action_value(state)
                next_state , reward, done, info = self.env.step(action)
                step += 1
                game_total_rew += reward
                self.ppo.memory.save_memory(state, value, action, log_prob, reward, done)
                if step % self.ppo.memory.T == 0:
                    self.ppo.optimize()
                    iters += 1
                    self.ppo.memory.reset()
                state = next_state

            game_rew_hist.append(game_total_rew)

            if (game) % 10 == 0:
                avg_score = np.mean(game_rew_hist[-10:])

                if avg_score > best_score:
                    self.ppo.save_weights()

                print('Episode: ', game, 'average score:', avg_score, 
                'learning_iterations:', iters)
