""" Conducts training and evaluation per test cases in test.yaml.

"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import gym
from model import PPO
from PPO_learning import PpoLearning

FILENAME = "tests.yaml"

def plot_graphs(test_dict, rewards):
    """ Plots results graphs.

        Args:
            test_dict: Dictionary with hyper-parameters for the test case.

        Returns:
            None

    """
    g_s = gridspec.GridSpec(1, 4, wspace=0.3)
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle('Experiment Results', y=0.93)

    x_values = np.arange(1, len(rewards)+1)

    ax1 = plt.subplot(g_s[:, 1:])
    ax1.scatter(x_values, rewards, s=5)
    ax1.set_title('Rewards')
    ax1.set(xlabel='Game', ylabel='Reward')

    ax2 = plt.subplot(g_s[:, :1])
    ax2.axis('off')
    data = [[key, val] for key, val in test_dict.items()]
    ax2.table(cellText = data, cellLoc='center', bbox = [0,0,1,1])

    plt.savefig(f'results/charts/Test_{test_dict["test_no"]}.png', bbox_inches='tight')

def run_tests():
    """ Conducts training, evaluation, creates results table and plots.

        Returns:
            results: Results dataframe

    """

    with open(FILENAME) as file:

        # Loads the test hyper-parameters as dictionaries.
        tests = yaml.safe_load(file)

    # create a dataframe to keep the results
    test_dict = tests['Tests']
    results = pd.DataFrame(test_dict)
    results["Episode"] = ""
    results['Max average score'] = ""
    results['Learning Iterations'] = ""

    # run experiments:
    for i, test in enumerate(tests['Tests']):

        env = gym.make(test['env'])
        env.reset()

        ppo = PPO(env, test['T'], test['hidden_dim'], test['gamma'],
        test['gae_lambda'], test['clip'], test['no_batches'],
        test['epochs'], test['lr_actor'], test['lr_critic'])

        training = PpoLearning(test['no_games'], ppo, test['max_score'], test['save'], test['test_no'])
        avg_score, game, iters, rewards = training.train()

        results.loc[i,'Episode'] = game
        results.loc[i,'Max average score'] = avg_score
        results.loc[i,'Learning Iterations'] = iters

        plot_graphs(test, rewards)

    # save results to csv file
    filename = 'results/' + 'test_table.csv'
    results.to_csv(filename)
    return results

