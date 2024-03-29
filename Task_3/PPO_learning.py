""" The main training loop.

"""

import numpy as np

class PpoLearning():
    """ PPO algorithm class with the learning algorithm.

        Args:
            no_games: no of games to train for.
            PPO: The PPO object.
            max_score: Max score allowed in the environment.
            save: Saves network weights if True.
            test_no: The test case no from .yaml file.

    """
    def __init__(self, no_games, PPO, max_score, save, test_no):

        self.no_games = no_games
        self.ppo = PPO
        self.env = PPO.env
        self.max_score = max_score
        self.save = save
        self.test_no = test_no

    def train(self):
        """ Samples the environment for T steps and optimizes for N games.

        Returns:
            best_score: The best score obtained.
            game: Number of games played.
            iters: Number of learning iterations conducted. 
            game_rew_hist: List of rewards for games.

        """
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
                next_state , reward, done, _ = self.env.step(action)
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
                    best_score = avg_score
                    if self.save:
                        self.ppo.save_weights()

                print('Test No:', self.test_no, '\t | \tEpisode:', game,
                '\t | \taverage score:', avg_score, '\t | \tlearning_iterations:', iters)

                if avg_score == self.max_score:
                    print(f"Maximum score of {self.max_score} reached")
                    break

        return best_score, game, iters, game_rew_hist
