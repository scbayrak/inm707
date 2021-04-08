import gym
from model import PPO
from PPO_learning import PpoLearning
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


FILENAME = "tests.yaml"

def plot_graphs(test_dict, rewards):

        gs = gridspec.GridSpec(1, 4, wspace=0.3)
        fig = plt.figure(figsize=(20, 6))
        fig.suptitle('Experiment Results', y=0.93)

        x = np.arange(1, len(rewards)+1)

        ax1 = plt.subplot(gs[:, 1:])
        ax1.scatter(x, rewards, s=5)
        ax1.set_title('Rewards')
        ax1.set(xlabel='Game', ylabel='Reward')
        
        ax2 = plt.subplot(gs[:, :1])
        ax2.axis('off')
        data = [[key, val] for key, val in test_dict.items()]
        ax2.table(cellText = data, cellLoc='center', bbox = [0,0,1,1])

        plt.savefig(f'results/charts/Test_{test_dict["test_no"]}.png', bbox_inches='tight')

def run_tests():
    
    with open(FILENAME) as file:

                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                tests = yaml.safe_load(file)


    # create a dataframe to keep the results 
    test_dict = tests['Tests']
    df = pd.DataFrame(test_dict)
    df["Episode"] = ""
    df['Max average score'] = ""
    df['Learning Iterations'] = ""

    # run experiments:
    for i, test in enumerate(tests['Tests']):
        
        env = gym.make(test['env'])
        env.reset()

        ppo = PPO(env, test['T'], test['hidden_dim'], test['gamma'], 
        test['gae_lambda'], test['clip'], test['no_batches'],
        test['epochs'], test['lr_actor'], test['lr_critic'])

        training = PpoLearning(test['no_games'], ppo, test['max_score'], test['save'], test['test_no'])
        avg_score, game, iters, rewards = training.train()

        df.loc[i,'Episode'] = game
        df.loc[i,'Max average score'] = avg_score
        df.loc[i,'Learning Iterations'] = iters

        plot_graphs(test, rewards)

    # save to csv file
    filename = 'results/' + 'test_table.csv'
    df.to_csv(filename)
    return df

run_tests()

