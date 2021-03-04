import numpy as np
import random


actions_dict = {0:"up", 1:"down", 2:"left", 3:"right"}

class rooms():
    def __init__(self, size):
        self.size = size
        self.grid= np.zeros((size, size), dtype = np.int8)
        self.agent_position= None
        self.time_elapsed = 0
        self.time_limit = (size ** 2) * 5
        self.tornados = round(size / 5)

        # customizable rewards
        self.rewards = {"empty":0, "obstacle":-3, "tornado":-25, 
                    "sub_opt_goal":size**2, "opt_goal":(size**2)*5}

        # create the states matrix and state_index_finder
        state_indexes = np.arange(size**2, dtype=int)
        self.state_index_finder = state_indexes.reshape(size, size)
        # states_shape = (size ^ 2) ^ (1 + number of tornados)
        states_shape = tuple([size**2] * (self.tornados + 1))
        states = np.arange((size**2) ** (self.tornados + 1), dtype=int)
        self.states = states.reshape(states_shape)
        
        # set the borders
        self.grid[[0, -1],:] = 1
        self.grid[:,[0,-1]] = 1

        # create the rooms
        self.grid[:,round(size/2)] = 1
        self.grid[round(size/2),:] = 1

        # create the doors to the rooms
        random_x, random_y = np.random.choice(range(1,round(size/2)), 2)
        self.grid[random_x,round(size/2)] = 0
        self.grid[round(size/2), random_y] = 0
        random_x, random_y = np.random.choice(range(-2,-round(size/2)+1, -1), 2)
        self.grid[random_x,round(size/2)] = 0
        self.grid[round(size/2), random_y] = 0
        
        # add the tornados & create the state_indices
        self.tornado_positions = self.get_empty_cells(self.tornados)
        self.grid[self.tornado_positions[0], self.tornado_positions[1]] = 2
        # state indices are in format ([agent_ind., tornado1_ind., tornado2_ind.,...])
        self.state_indices = np.zeros(self.tornados+1, dtype=int)
        for i, position in enumerate(zip(*self.tornado_positions)):
            tornado_index = self.state_index_finder[position]
            self.state_indices[i+1] = tornado_index

        # add the sub-optimal goal
        sub_optimal_goal_pos = self.get_empty_cells(1)
        self.grid[sub_optimal_goal_pos[0], sub_optimal_goal_pos[1]] = 3

        # add the optimal goal
        optimal_goal = self.get_empty_cells(1)
        self.grid[optimal_goal[0], optimal_goal[1]] = 4

    def get_empty_cells(self, n_cells):
        empty_cells_coord = np.where(self.grid == 0)       
        selected_indices = np.random.choice(len(empty_cells_coord[0]), n_cells)
        selected_cells = empty_cells_coord[0][selected_indices], empty_cells_coord[1][selected_indices]

        return selected_cells

    def move(self, current_cell, action):
        # find the next cell    
        if action == 'up':
            next_cell = (current_cell[0] - 1, current_cell[1])
        elif action == 'down':
            next_cell= (current_cell[0] + 1, current_cell[1])
        elif action == 'left':
            next_cell = (current_cell[0], current_cell[1] - 1)
        elif action == 'right':
            next_cell = (current_cell[0], current_cell[1] + 1)
    
        return next_cell

    def step(self, next_cell):
        done = False
        time_reward = -1

        # check the next cell type and update reward & done
        if self.grid[next_cell] == 1:
            reward = self.rewards["obstacle"]
        elif self.grid[next_cell] == 2:
            reward = self.rewards["tornado"]
        elif self.grid[next_cell] == 3:
            reward = self.rewards["sub_opt_goal"]
            done = True
        elif self.grid[next_cell] == 4:
            reward = self.rewards["opt_goal"]
            done = True
        else:
            reward = 0

        # update agent position
        if self.grid[next_cell] != 1:
            self.agent_position = next_cell

        # update time and check against the limit
        self.time_elapsed += 1
        if self.time_elapsed == self.time_limit:
            done = True

        # get the new state no
        agent_state_index = self.state_index_finder[self.agent_position]
        self.state_indices[0] = agent_state_index
        self.update_tornado_positions()
        new_state = self.states[tuple(self.state_indices)]
        return new_state, time_reward + reward, done

    def reset(self):
        # reset the time
        self.time_elapsed = 0
        # set the agent starting position
        agent_position = self.get_empty_cells(1)
        self.agent_position = (agent_position[0].item(), agent_position[1].item())
        # get the starting state
        agent_index = self.state_index_finder[self.agent_position]
        self.state_indices[0] = agent_index
        state = self.states[tuple(self.state_indices)]
        return state

    def update_tornado_positions(self):
        # moves each tornado in a random direction
        for i, position in enumerate(zip(*self.tornado_positions)):
            index_action = random.randint(0,3)
            next_cell = self.move(position, actions_dict[index_action])
            # doesnn't move if the next cell is obstacle, tornado or goal
            if self.grid[next_cell] in [1, 2, 3, 4]:
                continue
            else:              
                self.grid[position] = 0
                self.grid[next_cell] = 2
                self.tornado_positions[0][i] = next_cell[0]
                self.tornado_positions[1][i] = next_cell[1]
                self.state_indices[i+1] = self.state_index_finder[next_cell]
    
