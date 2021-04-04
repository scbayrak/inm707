import numpy as np
import random
from rooms import rooms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

actions_dict = {0:"up", 1:"down", 2:"left", 3:"right"}
actions_indexes = {"up":0, "down":1, "left":2, "right":3}

class Policy():
    
    def __init__(self, policy_type="greedy", epsilon=0.95, decay=0.99):
       
        self.policy_type = policy_type
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.decay = decay

    def __call__(self, state, q_values):

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
        
        self.epsilon = self.epsilon * self.decay
        
    def reset(self):
        self.epsilon = self.epsilon_start

class q_learning():
    def __init__(self, env, gamma, alpha, agent_start_pos):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.q_values = np.zeros((env.states.size, 4))
        self.agent_start_pos = agent_start_pos

    def update_q_values(self, s_current, action, r_next, s_next, action_next):
        action = actions_indexes[action]
        action_next = actions_indexes[action_next]
        self.q_values[s_current, action] += self.alpha * (r_next + self.gamma * self.q_values[s_next, action_next] - self.q_values[s_current, action])

    def q_learning_episode(self, policy):
        state = self.env.reset(self.agent_start_pos)
        done = False
        # policy.reset()
        while not done:
            action = policy(state, self.q_values)
            next_cell = self.env.move(self.env.agent_position, action)
            s_next, r, done, _ = self.env.step(next_cell)
            greedy_action_next = actions_dict[np.argmax(self.q_values[s_next])]
            self.update_q_values(state, action, r, s_next, greedy_action_next)
            state = s_next

    def run_multiple_episodes(self, episodes_no, policy):

        for episode in range(episodes_no):
            self.q_learning_episode(policy)
            policy.update_epsilon()
            # print(f"epsilon: {policy.epsilon}") 
        policy.reset()   
        return self.q_values

class experiments():
    def __init__(self, env, learning, policy, iters, agent_start_pos):
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

    def run_experiments(self, exp_per_batch):

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

        print(f"Iteration batch : {self.batch_no} Mean reward : {mean_reward}")

        return mean_reward

    def generate_results(self, test_no, test_dict):

        from scipy.interpolate import UnivariateSpline

        gs = gridspec.GridSpec(4, 2, wspace=0.2, hspace=1.5)
        fig = plt.figure(figsize=(20, 6))
        fig.suptitle('Experiment Results', y=0.93)

        x = np.arange(1, self.iters+1)

        ax1 = plt.subplot(gs[0:3, :1], label = 'Mean Rewards')
        ax1.set_title('Mean Rewards')
        # y_1 = self.mean_rewards
        ax1.scatter(x, self.mean_rewards, s=5)
        # sns.regplot(x=x, y=y_1, order=4, scatter_kws={"s": 5}, ci=None, ax=ax1)
        ax1.set(xlabel='Iteration', ylabel='Mean Reward')

        ax2 = plt.subplot(gs[0:3, 1:])
        # y_2 = self.sub_goals
        # y_3 = self.opt_goals
        ax2.scatter(x, self.sub_goals, s=5, label='Sub-optimal Goal')
        ax2.scatter(x, self.opt_goals, s=5, label='Optimal Goal')
        # sns.regplot(x=x, y=y_2, order=3, scatter_kws={"s": 5}, line_kws={'color': 'm'},
        #                                     ci=None, ax=ax2, label='Sub-optimal Goal')
        # sns.regplot(x=x, y=y_3, order=3, scatter_kws={"s": 5}, line_kws={'color': 'r'},
        #                                     ci=None, ax=ax2, label='Optimal Goal')
        ax2.set_title('Goal Success Percentage by Type')
        ax2.set(xlabel='Iteration', ylabel='Success Percentage (%)')
        ax2.legend(loc=0)

        cells = list(test_dict.values())
        cells = [str(i) for i in cells]
        columns = list(test_dict.keys())
        ax3 = plt.subplot(gs[3:, :])
        ax3.axis('off')
        ax3.table(cellText=[cells], colLabels=columns, loc='center', cellLoc='center')

        plt.savefig(f'results/charts/Test_{test_no}.png', bbox_inches='tight')

            

    
