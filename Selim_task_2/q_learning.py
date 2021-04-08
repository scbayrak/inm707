""" Implementation of Q_Learning algorithm to solve the rooms game.

"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


actions_dict = {0:"up", 1:"down", 2:"left", 3:"right"}
actions_indexes = {"up":0, "down":1, "left":2, "right":3}

class Policy():
    """ This class is the policy to choose actions.

    Args:
        policy_type: Greedy or e-greedy.
        epsilon: Epsilon probability for choosing random action, default=0.99.
        decay: The decay for epsilon, default=0.99.

    Attributes:
        policy_type: Greedy or e-greedy policy type.
        epsilon: Epsilon probability for choosing random action.
        epsilon_start: Start value of epsilon.
        decay: The decay for epsilon.

    """
    def __init__(self, policy_type="greedy", epsilon=0.99, decay=0.99):

        self.policy_type = policy_type
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.decay = decay

    def __call__(self, state, q_values):
        """ When called, the policy returns chosen action.

        Args:
            state: Current state.
            q_values: Q_values table.

        Returns:
            Chosen action.

        """

        if self.policy_type == "greedy":
            is_greedy = True
        else:
            is_greedy = random.uniform(0, 1) > self.epsilon

        if is_greedy :
            # choose greedy action
            index_action = np.argmax(q_values[state])
        else:
            # get a random action
            index_action = random.randint(0,3)

        return actions_dict[index_action]

    def update_epsilon(self):
        """ When called, decays epsilon.

        Returns:
            None.

        """
        self.epsilon = self.epsilon * self.decay

    def reset(self):
        """ Resets epsilon to start value.

        Returns:
            None.

        """
        self.epsilon = self.epsilon_start

class QLearning():
    """ This class performs q_learning algorithm.

    Args:
        env: The environment.
        gamma: Discount factor
        alpha: Learning rate - step size.
        agent_start_pos: The chosen starting location for the agent.

    Attributes:
        env: The environment.
        gamma: Discount factor.
        alpha: Learning rate - step size.
        q_values: Q_values table.
        agent_start_pos: The chosen starting location for the agent.

    """
    def __init__(self, env, gamma, alpha, agent_start_pos):

        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.q_values = np.zeros((env.states.size, 4))
        self.agent_start_pos = agent_start_pos

    def update_q_values(self, s_current, action, r_next, s_next, action_next):
        """ Updates Q_values using Q_learning.

        Args:
            s_current: Current state.
            action: Current action.
            r_next: Next reward.
            s_next: Next state.
            action_next: Next action.

        Returns:
            None.

        """
        action = actions_indexes[action]
        action_next = actions_indexes[action_next]
        self.q_values[s_current, action] += self.alpha * (
            r_next + self.gamma * self.q_values[s_next, action_next] - self.q_values[s_current, action])

    def q_learning_episode(self, policy):
        """ Performs an episode of Q_learning.

        Args:
            policy: Chosen policy.

        Returns:
            None.

        """
        state = self.env.reset(self.agent_start_pos)
        done = False

        while not done:
            action = policy(state, self.q_values)
            next_cell = self.env.move(self.env.agent_position, action)
            s_next, rew, done, _ = self.env.step(next_cell)
            greedy_action_next = actions_dict[np.argmax(self.q_values[s_next])]
            self.update_q_values(state, action, rew, s_next, greedy_action_next)
            state = s_next

    def run_multiple_episodes(self, episodes_no, policy):
        """ Performs an episode of Q_learning.

        Args:
            episodes_no: Number of learning episodes.
            policy: Chosen policy.

        Returns:
            self.q_values: Q_values table.

        """
        for _ in range(episodes_no):
            self.q_learning_episode(policy)
            policy.update_epsilon()

        policy.reset()
        return self.q_values

class Experiments():
    """ This class performs experiments following learning.

    Args:
        env: The environment.
        learning: Q_learning object.
        policy: Chosen policy.
        iters: Total no of iterations for (q_learning + experiments).
        agent_start_pos: Total no of iterations for (q_learning + experiments).
        test_no: Current test no.

    Attributes:
        env: The environment.
        learning: Q_learning object.
        policy: Chosen policy.
        iters: Total no of iterations for (q_learning + experiments).
        agent_start_pos:Total no of iterations for (q_learning + experiments).
        mean_rewards: Holds mean rewards.
        std_rewards: Holds std of rewards.
        sub_goals: Holds percentage of success for sub_goals.
        opt_goals: Holds percentage of success for opt_goals.
        batch_no: Tracks the iteration batch no.
        test_no: Holds the test number.

    """
    def __init__(self, env, learning, policy, iters, agent_start_pos, test_no):

        self.env = env
        self.learning = learning
        self.policy = policy
        self.iters = iters
        self.agent_start_pos = agent_start_pos
        self.mean_rewards = np.zeros(iters)
        self.std_rewards = np.zeros(iters)
        self.sub_goals = np.zeros(iters)
        self.opt_goals = np.zeros(iters)
        self.batch_no = 0
        self.test_no = test_no

    def run_experiments(self, exp_per_batch):
        """ Runs evaluation experiments.

                Args:
                    exp_per_batch: No of experiments per batch.

                Returns:
                    mean_reward: Average reward of the experiment.

        """
        all_rewards = np.zeros(exp_per_batch)
        reasons = np.zeros(exp_per_batch)

        for experiment in range(exp_per_batch):

            state = self.env.reset(self.agent_start_pos)
            done = False
            total_reward = 0

            while not done:
                action = self.policy(state, self.learning.q_values)
                next_cell = self.env.move(self.env.agent_position, action)
                state, reward, done, reason = self.env.step(next_cell)
                total_reward += reward

            all_rewards[experiment] = total_reward
            reasons[experiment] = reason

        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        sub_goal = (np.count_nonzero(reasons == 1) / exp_per_batch) * 100
        opt_goal = (np.count_nonzero(reasons == 0) / exp_per_batch) * 100

        self.mean_rewards[self.batch_no] = mean_reward
        self.std_rewards[self.batch_no] = std_reward
        self.sub_goals[self.batch_no] = sub_goal
        self.opt_goals[self.batch_no] = opt_goal

        self.batch_no += 1

        print(f"Test No : {self.test_no} Iteration batch : {self.batch_no} Mean reward : {mean_reward}")

        return mean_reward

    def generate_results(self, test_no, test_dict):
        """ Generates plots and results table.

                Args:
                    test_no: The test number.
                    test_dict: The dictionary of parameters for testing.

                Returns:
                    None.
        """
        g_s = gridspec.GridSpec(4, 2, wspace=0.2, hspace=1.5)
        fig = plt.figure(figsize=(20, 6))
        fig.suptitle('Experiment Results', y=0.93)

        x_val = np.arange(1, self.iters+1)

        ax1 = plt.subplot(g_s[0:3, :1], label = 'Mean Rewards')
        ax1.set_title('Mean Rewards')
        ax1.scatter(x_val, self.mean_rewards, s=5)
        ax1.set(xlabel='Iteration', ylabel='Mean Reward')

        ax2 = plt.subplot(g_s[0:3, 1:])
        ax2.scatter(x_val, self.sub_goals, s=5, label='Sub-optimal Goal')
        ax2.scatter(x_val, self.opt_goals, s=5, label='Optimal Goal')
        ax2.set_title('Goal Success Percentage by Type')
        ax2.set(xlabel='Iteration', ylabel='Success Percentage (%)')
        ax2.legend(loc=0)

        cells = list(test_dict.values())
        cells = [str(i) for i in cells]
        columns = list(test_dict.keys())
        ax3 = plt.subplot(g_s[3:, :])
        ax3.axis('off')
        ax3.table(cellText=[cells], colLabels=columns, loc='center', cellLoc='center')

        plt.savefig(f'results/charts/Test_{test_no}.png', bbox_inches='tight')
