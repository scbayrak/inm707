
import yaml
import pandas as pd

FILENAME = "task2_tests.yaml"

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


