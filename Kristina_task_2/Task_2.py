#%%
import sys
sys.path.append("../Task_1")
import numpy as np
import random
from rooms import Rooms
import time
import matplotlib.pyplot as plt
#%% Exploration
def exploration(): 
        actions = ['up', 'down', 'left', 'right']
        action = random.choice(actions)
        num_action = actions.index(action)
        return num_action, action

#%%Exploitation
def exploitation(q_list, state):
    actions= ['up', 'down', 'left', 'right']
    num_action = np.argmax(q_list)
        
    return num_action, actions[num_action]

#%% Q-Learning
def q_learning(q_table, alpha, gamma, epsilon, epsilon_decay, game, rewards, epochs):
    steps_taken = []
    for epoch in range(epochs):
        state = game.reset()
        
        actions_taken = []
        done = False
        episode_reward = 0
        
        for step in range(steps):
            random_num = random.uniform(0.1,1)
               
            if random_num < epsilon:
                num_action, action = exploration()
            else:
                num_action, action = exploitation(q_table[state, :], state)
                    
            actions_taken.append(action)
            next_cell = game.move([int(str(state)[0]),int(str(state)[1])], action)
                    
            new_state, step_reward, done = game.step(next_cell)
                
                
            if q_table[state, num_action] == 0:
                #When reweard is 0 in Q_table - new explored cell/state
                q_table[state, num_action] = (1 - alpha) * q_table[state, num_action] + alpha * (step_reward + gamma * np.max(q_table[new_state, :]))
            else:
                q_table[state, num_action] = q_table[state, num_action] + alpha * (step_reward + gamma * np.max(q_table[new_state, :] - q_table[state, num_action]))
            
            
            state = new_state
            episode_reward += step_reward
            
            epsilon = epsilon * epsilon_decay
            
            if done == True:
                steps_taken.append(step + 1)
                break
        
        rewards.append(episode_reward)
    
    return q_table, rewards, steps_taken
    
#%% Test
def test(q_table, game, episodes=3):
    val_episodes = 2
    for episode in range(val_episodes):
        state = game.reset("close_to_sub")
        done = False
        
        print(f"****** EPISODE {episode + 1} ******\n\n\n")
        total_reward = 0
        for step in range(steps):
            game.display()
            time.sleep(2)
            
            num_action = np.argmax(q_table[state,:])
            actions = ['up', 'down', 'left', 'right']
            action = actions[num_action]
            
            next_cell = game.move([int(str(state)[0]),int(str(state)[1])], action)
            
            
            new_state, step_reward, done = game.step(next_cell)
            total_reward += step_reward
            
            if done:

                new_state = [int(str(new_state)[0]),int(str(new_state)[1])]
                value = game.grid[new_state[0], new_state[1]]
                
                
                game.display()
                if  step_reward >= 0:
                    
                    if value == 4:
                        print(f"\tCONGRATULATIONS!!!\n\tYou have reached the Optimal Goal\n\tYour total reward is:{total_reward}\t¦\tIn {step+1} Steps\n\tWell Done!!")
                    else:
                        print(f"\tWell Done!\n\tYou have reached the Sub-Optimal Goal\n\tYour total reward is:{total_reward}\t¦\tIn {step+1} Steps\n\tGood Job!!")
                    time.sleep(3)
                else:
                    print("\tBAD NEWS!!!\n\tSadnly, you have Died\n\tYou either collided with a tornado or you ran out of time\n\tBetter Luck next time!")
                    time.sleep(3)
                break
            state = new_state

#%%

#Create Environment
game = Rooms(10)
game.reset()

#Create Q-table
q_table = np.zeros((game.states.size, len(game.actions_dict)))

#%%
# Define Hyper-Parameters
turns = 10000
steps = 500

alpha = 0.6                        #learning rate
gamma = 0.99                      #discount rate

epsilon = 1                      #exploration_rate
epsilon_decay= 0.99                #exploration_decay_rate

rewards = []
steps_taken= []

q_table, rewards, steps_taken = q_learning(q_table, alpha, gamma, epsilon, epsilon_decay, game, rewards, turns)

#%%

rewards_per_hundred_episodes = np.split(np.array(rewards), turns/100)
count = 100
total_rewards = []

for r in rewards_per_hundred_episodes:
    #print(count, ": ", str(sum(r/100)))
    #print(count, ": ", str(sum(r)))
    total_rewards.append(sum(r/100))
    count += 100


steps_per_hundred_episodes = np.split(np.array(steps_taken), turns/100)
count = 100
total_steps = []

for r in steps_per_hundred_episodes:
    #print(count, ": ", str(sum(r/100)))
    #print(count, ": ", str(sum(r)))
    total_steps.append(sum(r/100))
    count += 100

x = np.linspace(1, turns, 100)
plt.plot(x,total_steps, label='Steps')
plt.plot(x, total_rewards, label="Rewards")
plt.xlabel('Game', fontsize=20)
plt.ylabel('Steps', fontsize=20)
plt.legend(loc='upper right')
plt.show()

#%%
test(q_table, game, 1)





