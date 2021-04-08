""" Runs training and evaluation episodes per tests defined in task2_tests.yaml.

"""

import yaml
import pandas as pd
from rooms import Rooms
from q_learning import Policy, QLearning, Experiments

FILENAME = "task2_tests.yaml"

def run_tests():
    """ Runs all tests defined in task2_tests.yaml.

        Returns:
            results: Results dataframe.

        """
    with open(FILENAME) as file:
        # Loads testing parameters from the yaml file.
        tests = yaml.safe_load(file)

    # create a dataframe to keep the results
    test_dict = tests['Tests']
    results = pd.DataFrame(test_dict)
    results['Last Average Score'] = ""
    results['No of Q-Learning episodes'] = ""

    # run experiments:
    for i, test in enumerate(test_dict):
        grid = Rooms(test["env_size"], testing=True)
        learning = QLearning(grid, test["gamma"], test["alpha"], test["agent_start_pos"])
        e_greedy = Policy("e-greedy", test["epsilon"], test["decay"])
        greedy = Policy(policy_type="greedy")
        experiment = Experiments(grid, learning, greedy, test["iters"],
                                 test["agent_start_pos"], test["test_no"])

        for session in range(test["iters"]):
            learning.run_multiple_episodes(test["batch_episodes"], e_greedy)
            mean_reward = experiment.run_experiments(test["exp_per_batch"])

        results.loc[i,'Last Average Score'] = mean_reward
        results.loc[i,'No of Q-Learning episodes'] = (session + 1) * test["batch_episodes"]

        # save results to csv file
        filename = 'results/' + 'test_table.csv'
        results.to_csv(filename)

        # plot & save graphs
        experiment.generate_results(test["test_no"], test)

    return results

run_tests()