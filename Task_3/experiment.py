import gym
from model import PPO
from PPO_learning import PpoLearning
import yaml
import pandas as pd

FILENAME = "tests.yaml"

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

        training = PpoLearning(test['no_games'], ppo, test['max_score'], test['save'])
        avg_score, game, iters = training.train()

        df.loc[i,'Episode'] = game
        df.loc[i,'Max average score'] = avg_score
        df.loc[i,'Learning Iterations'] = iters

    # save to csv file
    filename = 'results/' + 'test_table.csv'
    df.to_csv(filename)
    return df

run_tests()

# # hyper-parameters for PPO learning
# epochs = 4 # number of epochs per learning
# no_batches = 2 # number of batches for splitting the timesteps
# hidden_dim = 256
# gamma = 0.99 # discount factor
# gae_lambda = 0.95 # gae smoothing parameter
# lr_actor = 0.0003
# lr_critic = 0.0005
# clip = 0.2 # PPO clipping epsilon parameter
# T = 20 # timesteps per each learning
# max_score = 200 # max score possible for the environment

# ppo = PPO(env=env, T=T, hidden_dim=hidden_dim, gamma=gamma, 
#     gae_lambda=gae_lambda, clip=clip, no_batches=no_batches,
#     epochs=epochs, lr_actor=lr_actor, lr_critic=lr_critic)

# no_games = 500

# training = PpoLearning(no_games, ppo, max_score)
# training.train()
