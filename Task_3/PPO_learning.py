import os
import gym
import numpy as np


class PpoLearning():

    def __init__(self, no_games, PPO, max_score, save):

        self.no_games = no_games
        self.ppo = PPO
        self.env = PPO.env
        self.max_score = max_score
        self.save = save

    def train(self):
        
        total_average = []
        total_loss_actor = []
        total_loss_critic = []
        
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
            
            total_average.append(game_total_rew)
            total_loss_actor.append(sum(self.ppo.actor_losses)/len(self.ppo.actor_losses))
            total_loss_critic.append(sum(self.ppo.critic_losses)/len(self.ppo.critic_losses))
            
            if (game) % 10 == 0:
                avg_score = np.mean(game_rew_hist[-10:])

                if avg_score > best_score:
                    best_score = avg_score
                    
                    if self.save:
                        self.ppo.save_weights()
                        
                
                print('Episode:', game, '\t | \taverage score:', avg_score, 
                '\t | \tlearning_iterations:', iters)

                if avg_score == self.max_score:
                    print(f"Maximum score of {self.max_score} reached")
                    break
                
                #total_average.append(avg_score)
                
                
            

        return best_score, game, iters, total_average , total_loss_critic, total_loss_actor, total_loss_critic