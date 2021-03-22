import numpy as np

class Memory():

    def __init__(self, no_batches, T):

        self.no_batches = no_batches
        self.T = T
        keys_storage = ["states", "values", "actions", 
                "log_probs", "rewards", "dones"]
        self.memory = {k: [] for k in keys_storage}

    def get_batches(self):

        indices = np.arange((self.T), dtype=np.int64)      
        batch_inds = np.split(indices, self.no_batches)

        self.memory.update((k, np.array(v))for k, v in self.memory.items())

        return self.memory.values(), batch_inds

    def save_memory(self, states, values, actions, log_probs, rewards, dones):
        
        new_items = {k : v for k, v in locals().items() 
                if k not in ["self", "new_items"]}

        for k, v in new_items.items():
            self.memory[k].append(v)

    def reset(self):
        self.memory = {k : [] for k in self.memory}
