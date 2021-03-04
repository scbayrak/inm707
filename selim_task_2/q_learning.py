import numpy as np
import random
from rooms import rooms

actions_dict = {0:"up", 1:"down", 2:"left", 3:"right"}
actions_indexes = {"up":0, "down":1, "left":2, "right":3}

class Policy():
    
    def __init__(self, policy_type="greedy", epsilon=0.9, decay=0.99):
       
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
    def __init__(self, env, gamma, alpha):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.q_values = np.zeros((env.states.size, 4))

    def update_q_values(self, s_current, action, r_next, s_next, action_next):
        action = actions_indexes[action]
        action_next = actions_indexes[action_next]
        self.q_values[s_current, action] += self.alpha * (r_next + self.gamma * self.q_values[s_next, action_next] - self.q_values[s_current, action])

    def q_learning_episode(self, policy):
        state = self.env.reset()
        done = False
        policy.reset()
        while not done:
            action = policy(state, self.q_values)
            next_cell = self.env.move(self.env.agent_position, action)
            s_next, r, done = self.env.step(next_cell)
            greedy_action_next = actions_dict[np.argmax(self.q_values[s_next])]
            self.update_q_values(state, action, r, s_next, greedy_action_next)
            state = s_next
            policy.update_epsilon()

    def run_multiple_episodes(self, episodes_no, policy):

        for episode in range(episodes_no):
            self.q_learning_episode(policy)   
        return self.q_values

class experiments():
    def __init__(self, env, learning, policy, number_exp):
        self.env = env
        self.learning = learning
        self.policy = policy
        self.number_exp = number_exp

    def run_single_experiment(self): 
        state = self.env.reset()
        done = False
        
        total_reward = 0
        
        while not done:
            action = self.policy(state, self.learning.q_values)
            next_cell = self.env.move(self.env.agent_position, action)
            state, reward, done = self.env.step(next_cell)
            total_reward += reward
        
        return total_reward

    def run_experiments(self):
        all_rewards = []
        
        for experiment in range(self.number_exp):
            
            final_reward = self.run_single_experiment()
            all_rewards.append(final_reward)

        mean_reward = np.mean(all_rewards)
        
        return mean_reward

grid = rooms(10)
learning = q_learning(grid, 0.99, 0.1)
e_greedy = Policy("e-greedy", 0.9, 0.99)
greedy = Policy(policy_type="greedy")
experiment = experiments(grid, learning, greedy, 100)

for session in range(100):
            learning.run_multiple_episodes(100, e_greedy)
            print(f"Mean reward: {experiment.run_experiments()}")
    
