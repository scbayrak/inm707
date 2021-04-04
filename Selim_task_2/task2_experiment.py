import yaml
import pandas as pd
from rooms import rooms
from q_learning import Policy, q_learning, experiments

FILENAME = "task2_tests.yaml"

def run_tests():
    with open(FILENAME) as file:

                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                tests = yaml.safe_load(file)


    # create a dataframe to keep the results 
    test_dict = tests['Tests']
    df = pd.DataFrame(test_dict)
    df['Last Average Score'] = ""
    df['No of Q-Learning episodes'] = ""

    # run experiments:
    for i, test in enumerate(test_dict):
       
        grid = rooms(test["env_size"], testing=True)
        learning = q_learning(grid, test["gamma"], test["alpha"], test["agent_start_pos"])
        e_greedy = Policy("e-greedy", test["epsilon"], test["decay"])
        greedy = Policy(policy_type="greedy")
        experiment = experiments(grid, learning, greedy, test["iters"], test["agent_start_pos"])

        # max_reward = 0
        # early_stop_counter = 0
        # patience = 10

        for session in range(test["iters"]):
                    learning.run_multiple_episodes(test["batch_episodes"], e_greedy)
                    mean_reward = experiment.run_experiments(test["exp_per_batch"])

                    # if mean_reward > max_reward:
                    #     max_reward = mean_reward
                    #     early_stop_counter = 0
                    # else:
                    #     early_stop_counter +=1

                    # if early_stop_counter == patience:
                    #     print("Stopping as reward not improving")
                    #     break
        
        df.loc[i,'Last Average Score'] = mean_reward
        df.loc[i,'No of Q-Learning episodes'] = (session + 1) * test["batch_episodes"]

        # save to csv file
        filename = 'results/' + 'test_table.csv'
        df.to_csv(filename)

        # plot & save graphs

        experiment.generate_results(test["test_no"], test)

    return df

run_tests()
