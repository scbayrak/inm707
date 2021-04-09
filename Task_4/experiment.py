""" Runs training and evaluation of test cases from tests.yaml file.

"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import gym

from a2c import ActorCritic

FILENAME = "tests.yaml"

def plot_graphs(test_dict, rewards):
    """ Plots rewards graph.

        Args:
            test_dict: Dictionary with hyper-parameters for the test case.
            rewards: Rewards collected from all episodes. 

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
    """ Runs tests from .yaml file, saves results plots and .csv file.

        Args:
            None.

        Returns:
            results: Test results dataframe. 

    """
    with open(FILENAME) as file:

        # Loads the test hyper-parameters as dictionaries.
        tests = yaml.safe_load(file)
    
    # create a dataframe to keep the results
    test_dict = tests['Tests']
    results = pd.DataFrame(test_dict)
    results["Episode"] = ""
    results['Max average score'] = ""

    for i, test in enumerate(tests['Tests']):

        env = gym.make(test['env'])
        env.reset()

        actor_critic = ActorCritic(env, test['episodes'], test['max_score'], 
                            test['hidden_size'], test['gamma'], test['save'])

        ## run training    
        best_score, episode, rew_hist = actor_critic.train()

        results.loc[i,'Episode'] = episode
        results.loc[i,'Max average score'] = best_score

        plot_graphs(test, rew_hist)

        # save results to csv file
        filename = 'results/' + 'test_table.csv'
        results.to_csv(filename)

    return results

